// saetta: single-model inference engine for Qwen3.8-Flash-Next on Apple M5 Max.
//   saetta info <model-dir>                  check the pack against its model's shapes and the GPU
//   saetta verify [golden-dir]                    compare with golden reference output (pack: $SAETTA_PACK or the golden's, under the models folder)
//   saetta serve <model-dir> [--port N]           OpenAI-compatible API on 127.0.0.1 (default 8080)
//   saetta render <request.json>                  print the chat prompt for an OpenAI request body
//   saetta tokenize <model-dir> <file>            print the token ids of a text file
//   saetta generate <model-dir> <n> <ids..>  greedy: run the prompt token ids, print n new ids
//   saetta download <pack>                   fetch a model pack from Hugging Face (pinned revision)
//   saetta import <file.gguf> <dir>          convert a Qwen3.8-27B GGUF into a pack (gguf.h)
//   saetta start <claude|pi|opencode>        run a coding agent against saetta (starts the server)
//   saetta monitor [--once]                  live view of the server, GPU, memory, prompt cache
//   saetta status | stop                     the server on this machine: show it, stop it
//   saetta update                            install the latest release over this binary
//   saetta version                           print the version
#include <signal.h>
#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "base.h"
#include "chat.h"
#include "control.h"
#include "download.h"
#include "gguf.h"
#include "gpu.h"
#include "json.h"
#include "launch.h"
#include "metal/shim.h"
#include "model.h"
#include "monitor.h"
#include "ops.h"
#include "attn27.h"
#include "lora.h"
#include "qgen.h"
#include "safetensors.h"
#include "server.h"
#include "spec.h"
#include "tokenizer.h"
#include "verify.h"
#include "weights.h"

// Release version; keep equal to the git tag (v<version>).
#define SAETTA_VERSION "0.10.0"

#define CFG_IN_MAX ((size_t)16384)
#define CFG_OUT_MAX ((size_t)262144)

// Zig's "{d:.N}" of an f64, in one of 16 rotating buffers.
static char fx_bufs[16][400];
static int fx_i;
static const char *fx(double x, int prec) {
    char *b = fx_bufs[fx_i++ & 15];
    fmt_fixed(b, 400, x, prec);
    return b;
}

static bool eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static Err parse_usize(const char *s, size_t *out) {
    uint64_t x;
    TRY(parse_u64(str(s), &x));
    *out = (size_t)x;
    return OK;
}

// ---- generate ----------------------------------------------------------------------------------

typedef struct {
    size_t done, steps;
    uint64_t t_draft, t_ver;
    size_t n_ver[8];
    size_t mtp_drafted, mtp_accepted; // MTP drafts verified, and accepted
} GenStats;

static bool is_eos(const Model *m, uint32_t t) {
    const ModelInfo *mi = model_info(m);
    for (size_t i = 0; i < mi->n_eos; i++)
        if (t == mi->eos[i]) return true;
    return false;
}

// The decode loop: greedy, with prompt-lookup or MTP speculation unless SAETTA_SPEC=0.
static Err decode_loop(Model *m, Session *s, float *logits, const uint32_t *toks, size_t ntoks, size_t n, GenStats *gs,
                       Lookup *lk) {
    const char *sv = getenv("SAETTA_SPEC");
    bool use_spec = (sv ? sv[0] != '0' : true) && !model_info(m)->no_spec;
    Gate gate = GATE_INIT;
    Depth depth = DEPTH_INIT;
    for (size_t i = 0; i < ntoks; i++) lookup_push(lk, toks[i]);
    uint32_t x = model_argmax(logits, model_info(m)->vocab);
    while (gs->done < n) {
        printf("%u\n", x);
        fflush(stdout);
        if (is_eos(m, x)) return OK;
        lookup_push(lk, x);
        uint32_t drafts[KMAX - 1];
        // a long prompt-lookup copy first (no head pass needed), else the MTP head
        uint64_t tr0 = now_ns();
        size_t nd = use_spec ? lookup_draft(lk, drafts, gate_depth(&gate, KMAX - 1)) : 0;
        bool from_mtp = false;
        if (use_spec && nd < 3 && model_has_mtp(m)) {
            // SAETTA_MTP_K fixes the depth; otherwise the controller picks it
            size_t k;
            const char *kv = getenv("SAETTA_MTP_K");
            if (kv) {
                if (parse_usize(kv, &k)) k = 2;
            } else k = depth_pick(&depth, KMAX - 1);
            uint64_t td0 = now_ns();
            TRY(model_draft_mtp(m, s, x, MIN(k, (size_t)(KMAX - 1)), drafts, &nd));
            gs->t_draft += now_ns() - td0;
            from_mtp = nd > 0;
        } else if (nd > 0 && !gate_allow(&gate)) nd = 0;
        gs->steps += 1;
        if (nd == 0) {
            TRY(model_step(m, s, x, &logits));
            x = model_argmax(logits, model_info(m)->vocab);
            if (use_spec && model_has_mtp(m)) depth_record(&depth, 0, 0, (float)(now_ns() - tr0) / 1e6f);
            gs->done += 1;
            continue;
        }
        uint64_t tv0 = now_ns();
        SpecResult r;
        TRY(model_spec_greedy(m, s, x, drafts, nd, &r));
        gs->t_ver += now_ns() - tv0;
        gs->n_ver[nd] += 1;
        if (from_mtp) {
            depth_record(&depth, nd, r.accepted, (float)(now_ns() - tr0) / 1e6f);
            gs->mtp_drafted += nd;
            gs->mtp_accepted += r.accepted;
        }
        else gate_record(&gate, r.accepted + 1);
        gs->done += 1;
        for (size_t i = 0; i < r.accepted; i++) {
            uint32_t d = drafts[i];
            if (gs->done >= n) break;
            printf("%u\n", d);
            if (is_eos(m, d)) return OK;
            lookup_push(lk, d);
            gs->done += 1;
        }
        x = r.next;
    }
    return OK;
}

static Err generate_in(Arena *arena, Pack *pack, Gpu *g, const char *dir, size_t n, char **ids, size_t nids) {
    (void)dir;
    uint64_t tw = now_ns();
    TRY(gpu_warm_except(g, pack, "ngram_embedding"));
    dbg("generate: warm %llu ms\n", (unsigned long long)((now_ns() - tw) / 1000000));
    Model *m;
    TRY(model_load(arena, pack, g, &m));
    // SAETTA_LORA=<adapter dir>: apply a LoRA adapter (Qwen3.8-27B)
    const char *lora = getenv("SAETTA_LORA");
    if (lora && *lora) {
        TRY(model_lora_load(arena, m, "lora", lora));
        model_lora_select(m, 0);
    }
    Session *s;
    TRY(session_create(m, g, &s));
    float *logits;
    model_prof_init();
    uint64_t t0 = now_ns();
    // ids, or @file with whitespace-separated ids (long prompts exceed the argument limit)
    VEC(uint32_t) list = {0};
    for (size_t i = 0; i < nids; i++) {
        const char *a = ids[i];
        if (strlen(a) > 1 && a[0] == '@') {
            char *text;
            size_t len;
            TRY(read_file(a + 1, 64u << 20, &text, &len));
            size_t p = 0;
            while (p < len) {
                while (p < len && strchr(" \n\t\r", text[p]) && text[p]) p++;
                size_t b = p;
                while (p < len && !(strchr(" \n\t\r", text[p]) && text[p])) p++;
                if (p > b) {
                    uint64_t v;
                    Err err = parse_u64(str_n(text + b, p - b), &v);
                    if (!err && v > UINT32_MAX) err = "Overflow";
                    if (err) {
                        free(text);
                        return err;
                    }
                    vec_push(&list, (uint32_t)v);
                }
            }
            free(text);
        } else {
            uint64_t v;
            TRY(parse_u64(str(a), &v));
            if (v > UINT32_MAX) return "Overflow";
            vec_push(&list, (uint32_t)v);
        }
    }
    const uint32_t *toks = list.ptr;
    size_t ntoks = list.len;
    const char *nll = getenv("SAETTA_NLL");
    if (nll) {
        // quality: prefill all but the last N tokens, then teacher-force those N through decode
        // steps and print their mean negative log-likelihood (the prefill path is what differs)
        size_t nn;
        TRY(parse_usize(nll, &nn));
        // SAETTA_NLL_STEP: the prefix through decode steps instead of prefill chunks
        float *lg = NULL;
        if (getenv("SAETTA_NLL_STEP") != NULL) {
            for (size_t i = 0; i < ntoks - nn; i++) TRY(model_step(m, s, toks[i], &lg));
        } else TRY(model_prefill(m, s, toks, ntoks - nn, NULL, &lg));
        double tot = 0;
        for (size_t i = ntoks - nn; i < ntoks; i++) {
            uint32_t tk = toks[i];
            float mx = -INFINITY;
            for (size_t j = 0; j < model_info(m)->vocab; j++) mx = fmaxf(mx, lg[j]);
            double se = 0;
            for (size_t j = 0; j < model_info(m)->vocab; j++) se += exp((double)(lg[j] - mx));
            tot += log(se) - (double)(lg[tk] - mx);
            TRY(model_step(m, s, tk, &lg));
        }
        dbg("nll: prefill %zu tokens, mean NLL of the next %zu: %s\n", ntoks - nn, nn, fx(tot / (double)nn, 5));
        return OK;
    }
    TRY(model_prefill(m, s, toks, ntoks, NULL, &logits));
    const char *dump = getenv("SAETTA_DUMP_LOGITS");
    if (dump) {
        // debug: the prompt's last logits as raw f32, to compare paths offline
        TRY(write_file(dump, logits, model_info(m)->vocab * sizeof(float)));
    }
    const char *only = getenv("SAETTA_CHUNK_COST");
    if (only) {
        // cost of a decode step against verification chunks of 1..KMAX tokens (state slots on)
        uint32_t tk = model_argmax(logits, model_info(m)->vocab);
        float *lg;
        for (int i = 0; i < 3; i++) {
            TRY(model_step(m, s, tk, &lg));
            tk = model_argmax(lg, model_info(m)->vocab);
        }
        model_prof_init();
        model_prof_reset();
        uint64_t ts = now_ns();
        if (!eq(only, "2"))
            for (int i = 0; i < 20; i++) {
                TRY(model_step(m, s, tk, &lg));
                tk = model_argmax(lg, model_info(m)->vocab);
            }
        dbg("chunk-cost: step %s ms\n", fx((double)(now_ns() - ts) / 20e6, 2));
        s->spec = true;
        for (size_t T = 1; T <= KMAX; T++) {
            if (eq(only, "step") || (eq(only, "2") && T != 2)) continue;
            uint32_t row[KMAX];
            for (size_t i = 0; i < KMAX; i++) row[i] = tk;
            TRY(model_verify_chunk(m, s, row, T, &lg));
            uint64_t tc = now_ns();
            for (int i = 0; i < 10; i++) TRY(model_verify_chunk(m, s, row, T, &lg));
            dbg("chunk-cost: verify T=%zu %s ms\n", T, fx((double)(now_ns() - tc) / 10e6, 2));
        }
        model_prof_report(eq(only, "2") ? 11 : 20);
        return OK;
    }
    uint64_t t1 = now_ns();
    GenStats gs = {0};
    Lookup lk = {0};
    uint64_t t_loop = now_ns();
    double gpu0 = sx_gpu_seconds();
    Err err = decode_loop(m, s, logits, toks, ntoks, n, &gs, &lk);
    lookup_free(&lk);
    fflush(stdout);
    {
        double tot = (double)(now_ns() - t_loop) / 1e6;
        double gms = (sx_gpu_seconds() - gpu0) * 1e3;
        dbg("generate: %zu tokens in %zu forward passes, %s ms (GPU busy %s ms, host %s ms = %s ms per pass)\n", gs.done,
            gs.steps, fx(tot, 1), fx(gms, 1), fx(tot - gms, 1), fx((tot - gms) / (double)MAX(gs.steps, (size_t)1), 2));
        Buf nv = {0};
        buf_adds(&nv, "{ ");
        for (size_t i = 0; i < 8; i++) buf_printf(&nv, i < 7 ? "%zu, " : "%zu", gs.n_ver[i]);
        buf_adds(&nv, " }");
        dbg("timing: drafts %s ms, verify %s ms, other %s ms, verify chunks by drafts %s\n", fx((double)gs.t_draft / 1e6, 1),
            fx((double)gs.t_ver / 1e6, 1), fx(tot - (double)(gs.t_draft + gs.t_ver) / 1e6, 1), nv.ptr);
        buf_free(&nv);
        if (gs.mtp_drafted)
            dbg("generate: MTP drafts %zu, accepted %zu (%s%%)\n", gs.mtp_drafted, gs.mtp_accepted,
                fx(100.0 * (double)gs.mtp_accepted / (double)gs.mtp_drafted, 1));
    }
    {
        uint64_t t2 = now_ns();
        double pre = (double)(t1 - t0) / 1e6, dec = (double)(t2 - t1) / 1e6;
        dbg("generate: prompt %zu tokens %s ms (%s ms/token), %zu new tokens %s ms (%s ms/token)\n", ntoks, fx(pre, 1),
            fx(pre / (double)MAX(ntoks, (size_t)1), 2), gs.done, fx(dec, 1), fx(dec / (double)MAX(gs.done, (size_t)1), 2));
    }
    model_prof_report(ntoks + gs.done);
    vec_free(&list);
    return err;
}

static Err generate(Arena *arena, const char *dir, size_t n, char **ids, size_t nids) {
    Pack pack;
    TRY(pack_open(arena, dir, &pack));
    Gpu g;
    Err err = gpu_init(arena, &pack, &g);
    if (err) {
        pack_close(&pack);
        return err;
    }
    err = generate_in(arena, &pack, &g, dir, n, ids, nids);
    gpu_deinit(&g);
    pack_close(&pack);
    return err;
}

// ---- info --------------------------------------------------------------------------------------

static Err gpu_self_test(sx_device *dev, FILE *out) {
    const char *src = "#include <metal_stdlib>\n"
                      "using namespace metal;\n"
                      "kernel void twice(device const float *a [[buffer(0)]], device float *b [[buffer(1)]], uint i "
                      "[[thread_position_in_grid]]) { b[i] = 2.0f * a[i]; }";
    char err[512] = {0};
    sx_pipeline *p = sx_pipeline_from_source(dev, src, "twice", err, sizeof err);
    if (!p) {
        fprintf(out, "gpu: compile failed: %s\n", err);
        return "CompileFailed";
    }
    enum { N = 1024 };
    Err ret = OK;
    sx_buffer *a = sx_buffer_new(dev, N * 4);
    sx_buffer *b = a ? sx_buffer_new(dev, N * 4) : NULL;
    if (!a || !b) {
        ret = "AllocFailed";
        goto out;
    }
    float *pa = sx_buffer_contents(a);
    for (size_t i = 0; i < N; i++) pa[i] = (float)i;
    sx_arg args[2] = {{a, 0, NULL, 0}, {b, 0, NULL, 0}};
    if (sx_dispatch_sync(dev, p, args, 2, N / 256, 256, err, sizeof err) != 0) {
        ret = "DispatchFailed";
        goto out;
    }
    const float *pb = sx_buffer_contents(b);
    for (size_t i = 0; i < N; i++)
        if (pb[i] != 2.0f * (float)i) {
            ret = "GpuSelfTestFailed";
            goto out;
        }
    fprintf(out, "gpu: self test ok\n");
out:
    if (b) sx_buffer_release(b);
    if (a) sx_buffer_release(a);
    sx_pipeline_release(p);
    return ret;
}

static Err info(Arena *arena, FILE *out, const char *dir) {
    Pack pack;
    TRY(pack_open(arena, dir, &pack));
    uint64_t per_dtype[DTYPE_COUNT] = {0};
    uint64_t total = 0;
    Str k;
    void *v;
    for (size_t it = 0; strmap_next(&pack.tensors, &it, &k, &v);) {
        const TensorInfo *t = v;
        per_dtype[t->dtype] += t->nbytes;
        total += t->nbytes;
    }
    fprintf(out, "pack: %zu files, %zu tensors, %llu bytes\n", pack.nfiles, pack.tensors.len, (unsigned long long)total);
    for (int d = 0; d < DTYPE_COUNT; d++)
        if (per_dtype[d] > 0) fprintf(out, "  %s: %llu bytes\n", dtype_name((DType)d), (unsigned long long)per_dtype[d]);

    // Every quantized linear of every layer, with its expected (out, in) shape.
    if (!model_info_for(arena, &pack)) {
        fprintf(out, "model: %s is not a Qwen3.8-Flash-Next, Qwen3.8-27B or DeepSeek-V4-Flash pack\n", dir);
        pack_close(&pack);
        return "UnknownModelType";
    }
    size_t bad = model_check_pack(arena, &pack, out);

    // GPU: wrap every shard as a no-copy buffer, then run one tiny kernel.
    Err ret = OK;
    sx_device *dev = sx_device_create();
    if (!dev) {
        pack_close(&pack);
        return "NoMetalDevice";
    }
    char name[128] = {0};
    sx_device_name(dev, name, sizeof name);
    fprintf(out, "gpu: %s, Apple10 %d, recommended working set %llu bytes\n", name, sx_device_supports_apple10(dev),
            (unsigned long long)sx_device_recommended_working_set(dev));
    size_t page = sx_page_size();
    uint64_t wrapped = 0;
    for (size_t i = 0; i < pack.nfiles; i++) {
        size_t len = ALIGN_UP(pack.files[i].len, page);
        sx_buffer *b = sx_buffer_wrap(dev, pack.files[i].bytes, len);
        if (!b) {
            ret = "WrapFailed";
            goto done;
        }
        wrapped += len;
        sx_buffer_release(b);
    }
    fprintf(out, "gpu: wrapped %zu files, %llu bytes, no copy\n", pack.nfiles, (unsigned long long)wrapped);
    ret = gpu_self_test(dev, out);
    if (!ret && bad != 0) ret = "PackCheckFailed";
done:
    sx_device_destroy(dev);
    pack_close(&pack);
    return ret;
}

// ---- mv-bench ----------------------------------------------------------------------------------

static Err mv_bench(Arena *arena, FILE *out, const char *dir) {
    // decode matvec bandwidth per matrix shape: each matrix kind over all its layers once, in
    // one command buffer: distinct weights, as in decode; best of 5 (the first bursts after
    // idle run at a low GPU clock)
    Pack pack;
    TRY(pack_open(arena, dir, &pack));
    Gpu g;
    TRY(gpu_init(arena, &pack, &g));
    TRY(gpu_warm_except(&g, &pack, "ngram_embedding"));
    Ops ops;
    TRY(ops_init(&g, &ops));
    static const char *const kinds[] = {
        "attn_hyper_connection.input_mix_weight_down", "attn_hyper_connection.input_mix_weight_up",
        "attn_hyper_connection.block_inject_weight",   "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",                       "linear_attn.out_proj",
        "linear_attn.in_proj_a",                       "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.down_proj",                 "mlp.gate",
    };
    sx_buffer *xb, *yb;
    TRY(gpu_new_buffer(&g, CFG_IN_MAX * 4, &xb));
    TRY(gpu_new_buffer(&g, CFG_OUT_MAX * 4, &yb));
    Linear lins[48];
    bool rows = getenv("SAETTA_MV_ROWS") != NULL;
    for (size_t ki = 0; ki < COUNT(kinds); ki++) {
        const char *k = kinds[ki];
        size_t n = 0;
        for (size_t li = 0; li < 48; li++) {
            const char *nm = arena_printf(arena, "language_model.model.layers.%zu.%s", li, k);
            if (pack_get(&pack, str(arena_printf(arena, "%s.weight", nm))) == NULL) continue;
            TRY(pack_linear(&pack, str(nm), &lins[n]));
            n += 1;
        }
        uint64_t by = 0;
        for (size_t i = 0; i < n; i++) by += lins[i].weight.nbytes + lins[i].scales.nbytes + lins[i].biases.nbytes;
        if (rows) {
            for (uint32_t T = 1; T < 6; T++) {
                double secs = INFINITY;
                for (int r = 0; r < 5; r++) {
                    double s;
                    TRY(ops_rows_timed_many(&ops, &g, lins, n, xb, yb, T, &s));
                    secs = fmin(secs, s);
                }
                fprintf(out, "%46s %6llux%6llu rows T=%u  %7s us/call  %6s GB/s\n", k, (unsigned long long)lins[0].out_features,
                        (unsigned long long)lins[0].in_features, T, fx(secs * 1e6 / (double)n, 2), fx((double)by / secs / 1e9, 1));
            }
            continue;
        }
        static const uint32_t lane_opts[4] = {0, 8, 64, 256};
        for (int li = 0; li < 4; li++) {
            uint32_t lanes = lane_opts[li];
            if (lanes != 0 && lins[0].bits != 4) continue;
            double secs = INFINITY;
            for (int r = 0; r < 5; r++) {
                double s;
                TRY(ops_qmatvec_timed_many(&ops, &g, lins, n, xb, yb, lanes, &s));
                secs = fmin(secs, s);
            }
            fprintf(out, "%46s %6llux%6llu %u-bit x%2zu lanes %3u  %7s us/call  %6s GB/s\n", k,
                    (unsigned long long)lins[0].out_features, (unsigned long long)lins[0].in_features, lins[0].bits, n,
                    lanes == 0 ? 32u : lanes, fx(secs * 1e6 / (double)n, 2), fx((double)by / secs / 1e9, 1));
        }
    }
    Linear lm;
    TRY(pack_linear(&pack, S("language_model.lm_head"), &lm));
    double lm_bytes = (double)(lm.weight.nbytes + lm.scales.nbytes + lm.biases.nbytes);
    if (rows)
        for (uint32_t T = 1; T < 5; T++) {
            double rs = INFINITY;
            for (int r = 0; r < 3; r++) {
                double s;
                TRY(ops_rows_timed_many(&ops, &g, &lm, 1, xb, yb, T, &s));
                rs = fmin(rs, s);
            }
            fprintf(out, "%46s %6llux%6llu rows T=%u  %7s us/call  %6s GB/s\n", "lm_head", (unsigned long long)lm.out_features,
                    (unsigned long long)lm.in_features, T, fx(rs * 1e6, 2), fx(lm_bytes / rs / 1e9, 1));
        }
    double ls = INFINITY;
    for (int r = 0; r < 5; r++) {
        double s;
        TRY(ops_qmatvec_timed_many(&ops, &g, &lm, 1, xb, yb, 0, &s));
        ls = fmin(ls, s);
    }
    fprintf(out, "%46s %6llux%6llu %u-bit x 1  %7s us/call  %6s GB/s\n", "lm_head", (unsigned long long)lm.out_features,
            (unsigned long long)lm.in_features, lm.bits, fx(ls * 1e6, 2), fx(lm_bytes / ls / 1e9, 1));
    fflush(out);
    return OK;
}

// ---- commands ----------------------------------------------------------------------------------

// A pack argument: an existing folder, else a `saetta download` name (coder, 27b-8bit, ...)
// resolved under the models folder ($SAETTA_MODELS, default ~/Models).
static const char *pack_arg(Arena *a, const char *arg) {
    struct stat st;
    if (stat(arg, &st) == 0) return arg;
    const DownloadPack *p = download_find(arg);
    char *dir;
    if (p && !download_pack_dir(a, p, &dir)) return dir;
    return arg;
}

static Err run(Arena *arena, int argc, char **argv) {
    FILE *out = stdout;
    const char *const *rest = (const char *const *)argv + 2;
    size_t nrest = argc >= 2 ? (size_t)argc - 2 : 0;
    const char *cmd = argc >= 2 ? argv[1] : "";
    if (argc == 2 && (eq(cmd, "version") || eq(cmd, "--version"))) {
        fprintf(out, "saetta %s\n", SAETTA_VERSION);
        fflush(out);
        return OK;
    }
    if (argc >= 2 && eq(cmd, "download")) {
        Err err = download_run(rest, nrest);
        if (err) {
            dbg("saetta download: %s\n", err);
            exit(1);
        }
        return OK;
    }
    if (argc >= 2 && eq(cmd, "import")) {
        Err err = gguf_import_run(rest, nrest);
        if (err) {
            dbg("saetta import: %s\n", err);
            exit(1);
        }
        return OK;
    }
    if (argc == 2 && eq(cmd, "update")) {
        int code;
        Err err = control_update(out, SAETTA_VERSION, &code);
        if (err) {
            dbg("saetta update: %s\n", err);
            exit(1);
        }
        fflush(out);
        exit(code);
    }
    if (argc == 2 && (eq(cmd, "status") || eq(cmd, "stop"))) {
        int code;
        Err err = eq(cmd, "stop") ? control_stop(out, &code) : control_status(out, &code);
        if (err) {
            dbg("saetta %s: %s\n", cmd, err);
            exit(1);
        }
        fflush(out);
        exit(code);
    }
    if (argc >= 2 && eq(cmd, "monitor")) {
        int code;
        Err err = monitor_run(rest, nrest, &code);
        if (err) {
            dbg("saetta monitor: %s\n", err);
            exit(1);
        }
        exit(code);
    }
    if (argc >= 2 && eq(cmd, "start")) {
        int code;
        Err err = launch_run(rest, nrest, &code);
        if (err) {
            dbg("saetta start: %s\n", err);
            exit(1);
        }
        exit(code);
    }
    if (argc >= 3 && eq(cmd, "info")) return info(arena, out, pack_arg(arena, argv[2]));
    if (argc == 3 && eq(cmd, "mv-bench")) return mv_bench(arena, out, pack_arg(arena, argv[2]));
    if (argc >= 2 && eq(cmd, "qgen-bench")) {
        // run-time quantization kernels against the 4-bit/32 templated ones (src/qgen.h)
        TRY(qgen_bench(arena, out));
        fflush(out);
        return OK;
    }
    if (argc >= 2 && eq(cmd, "attn-bench")) {
        // Qwen3.8-27B prompt attention, old against new kernel, one layer (src/attn27.h)
        TRY(attn27_bench(arena, out));
        fflush(out);
        return OK;
    }
    if (argc >= 2 && eq(cmd, "lora-bench")) {
        // run-time LoRA kernels, the Qwen3.8-27B decode set of one token (src/lora.h)
        TRY(lora_bench(arena, out));
        fflush(out);
        return OK;
    }
    if (argc == 3 && eq(cmd, "dispatch-cost")) {
        // GPU fixed cost per dispatch in a serial command buffer (no model needed)
        Pack pack_dummy = {0};
        pack_dummy.dir = "";
        Gpu g;
        TRY(gpu_init(arena, &pack_dummy, &g));
        size_t n;
        TRY(parse_usize(argv[2], &n));
        static const size_t threads[2] = {1, 10240};
        for (int i = 0; i < 2; i++) {
            double s0;
            TRY(gpu_dispatch_cost(&g, n, threads[i], &s0));
            fprintf(out, "%zu dispatches of %zu threads: %s ms, %s us each\n", n, threads[i], fx(s0 * 1e3, 3),
                    fx(s0 * 1e6 / (double)n, 2));
        }
        fflush(out);
        return OK;
    }
    if (argc >= 2 && eq(cmd, "verify")) {
        const char *golden = argc >= 3 ? argv[2] : "tests/golden";
        // the pack: $SAETTA_PACK, else the one the golden names: manifest "pack_path" under the
        // models folder, or "pack" under its mlx-community
        const char *pack = getenv("SAETTA_PACK");
        if (!pack) {
            const char *home = getenv("HOME");
            if (!home) return "NoHome";
            const char *name = "Qwen3.8-Flash-Next-4bit-mtp";
            const char *pack_path = NULL; // relative to the models folder (manifest "pack_path")
            char *mt;
            size_t ml;
            if (!read_file(path_join(arena, golden, "manifest.json"), 1u << 20, &mt, &ml)) {
                JValue *man;
                if (!json_parse(arena, str_n(mt, ml), false, &man)) {
                    const char *pn = json_str(json_get(man, "pack"));
                    if (pn) name = pn;
                    pack_path = json_str(json_get(man, "pack_path"));
                }
                free(mt);
            }
            char *md;
            TRY(download_models_dir(arena, &md));
            pack = pack_path ? path_join(arena, md, pack_path) : path_join(arena, path_join(arena, md, "mlx-community"), name);
        }
        bool ok;
        TRY(verify_run(arena, out, pack, golden, &ok));
        fflush(out);
        if (!ok) exit(1);
        return OK;
    }
    if (argc >= 3 && eq(cmd, "serve")) {
        uint16_t port = 8080;
        // --lora-modules name=path [name=path ...] (vLLM's form) or name=path,name=path
        const char **loras = ARENA_ARRAY(arena, const char *, (size_t)argc * 8);
        size_t nloras = 0;
        const char *kwargs = NULL;
        for (int i = 3; i < argc; i++) {
            if (eq(argv[i], "--port") && i + 1 < argc) {
                uint64_t p;
                TRY(parse_u64(str(argv[++i]), &p));
                if (p > 65535) return "Overflow";
                port = (uint16_t)p;
            } else if (eq(argv[i], "--default-chat-template-kwargs") && i + 1 < argc) {
                kwargs = argv[++i];
            } else if (eq(argv[i], "--lora-modules")) {
                for (; i + 1 < argc && argv[i + 1][0] != '-'; i++) {
                    char *list = arena_strndup(arena, argv[i + 1], strlen(argv[i + 1]));
                    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
                        if (nloras == (size_t)argc * 8) return "TooManyAdapters";
                        loras[nloras++] = tok;
                    }
                }
            } else {
                dbg("saetta serve: unknown argument %s\n", argv[i]);
                exit(64);
            }
        }
        Err err = server_serve(arena, pack_arg(arena, argv[2]), port, loras, nloras, kwargs);
        if (err && !strcmp(err, "AlreadyRunning")) exit(1);
        if (err && !strcmp(err, "AddressInUse")) {
            dbg("error: port %u is in use by another program; choose another with --port N\n", port);
            exit(1);
        }
        return err;
    }
    if (argc == 3 && eq(cmd, "render")) {
        char *body;
        size_t len;
        TRY(read_file(argv[2], 64u << 20, &body, &len));
        JValue *req;
        TRY(json_parse(arena, str_n(body, len), false, &req));
        Str prompt;
        TRY(chat_render(arena, req, &prompt));
        fwrite(prompt.ptr, 1, prompt.len, out);
        fflush(out);
        free(body);
        return OK;
    }
    if (argc == 4 && eq(cmd, "tokenize")) {
        Tokenizer *tok = ARENA_NEW(arena, Tokenizer);
        TRY(tokenizer_load(arena, pack_arg(arena, argv[2]), tok));
        char *text;
        size_t len;
        TRY(read_file(argv[3], 256u << 20, &text, &len));
        VecU32 ids = {0};
        TRY(tokenizer_encode(tok, str_n(text, len), &ids));
        for (size_t i = 0; i < ids.len; i++) fprintf(out, "%u\n", ids.ptr[i]);
        fflush(out);
        vec_free(&ids);
        free(text);
        return OK;
    }
    if (argc >= 5 && eq(cmd, "generate")) {
        size_t n;
        TRY(parse_usize(argv[3], &n));
        return generate(arena, pack_arg(arena, argv[2]), n, argv + 4, (size_t)argc - 4);
    }
    fprintf(out, "usage: saetta serve <model-dir> [--port N] [--lora-modules name=path ...] [--default-chat-template-kwargs JSON] | info <model-dir> | generate <model-dir> <n> <ids...> | tokenize "
                 "<model-dir> <file> | render <request.json> | verify [golden-dir] | download <pack> | import <file.gguf> <dir> | start "
                 "<claude|pi|opencode> | monitor | status | stop | update | version\n");
    fflush(out);
    exit(64);
}

static void sigpipe_noop(int sig) { (void)sig; }

int main(int argc, char **argv) {
    // a closed pipe or socket makes write return EPIPE instead of killing the process. A
    // handler, not SIG_IGN: handlers reset on exec, so spawned agents and tools get the
    // default SIGPIPE (Zig's std installs a no-op handler the same way)
    struct sigaction sa = {0};
    sa.sa_handler = sigpipe_noop;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPIPE, &sa, NULL);
    Arena arena = {0};
    Err err = run(&arena, argc, argv);
    fflush(stdout);
    if (err) {
        dbg("error: %s\n", err);
        return 1;
    }
    return 0;
}
