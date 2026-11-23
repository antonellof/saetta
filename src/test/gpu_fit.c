// The dispatch fit check (shim.h sx_pipeline_max_threads): a threadgroup above the pipeline's
// limit is not run and the command buffer fails naming the kernel, where Metal alone would skip
// it silently and leave its output stale.
#include <string.h>

#include "../gpu.h"
#include "../test.h"

static const char fit_src[] = "kernel void fit_probe(device uint *b [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
                              "    b[i] = i + 1;\n"
                              "}";

TEST(gpu_dispatch_fit) {
    Arena arena = {0};
    Pack empty = {0};
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &empty, &g));
    sx_pipeline *p;
    EXPECT_OK(gpu_pipeline(&g, fit_src, "fit_probe", &p));
    char name[64];
    sx_pipeline_name(p, name, sizeof name);
    EXPECT(strcmp(name, "fit_probe") == 0);
    const uint32_t max = sx_pipeline_max_threads(p);
    EXPECT(max >= 32);
    sx_buffer *b;
    EXPECT_OK(gpu_new_buffer(&g, (size_t)(max + 32) * 4 * 2, &b));
    sx_arg a[1] = {gpu_buf_arg(b)};
    char err[512];
    // fits: runs
    sx_cmd *c = sx_cmd_begin(g.dev);
    sx_cmd_dispatch(c, p, a, 1, 1, 1, max);
    EXPECT(sx_cmd_finish(c, err, sizeof err) == 0);
    EXPECT_EQ(((uint32_t *)sx_buffer_contents(b))[max - 1], max);
    // does not fit: not run, and the finish fails naming the kernel
    memset(sx_buffer_contents(b), 0, (size_t)(max + 32) * 4 * 2);
    c = sx_cmd_begin(g.dev);
    sx_cmd_dispatch(c, p, a, 1, 2, 1, max + 32);
    err[0] = 0;
    EXPECT(sx_cmd_finish(c, err, sizeof err) != 0);
    EXPECT(strstr(err, "fit_probe") != NULL);
    EXPECT_EQ(((uint32_t *)sx_buffer_contents(b))[0], 0);
    // the 3-D form and the split commit / wait
    c = sx_cmd_begin(g.dev);
    sx_cmd_dispatch3(c, p, a, 1, 1, 1, 1, max + 32);
    sx_cmd_commit(c);
    EXPECT(sx_cmd_wait(c, err, sizeof err) != 0);
    EXPECT(sx_dispatch_sync(g.dev, p, a, 1, 1, max + 32, err, sizeof err) != 0);
    // the static check
    EXPECT_OK(gpu_require_threads(p, max));
    EXPECT_ERR(gpu_require_threads(p, max + 32), "ThreadgroupTooLarge");
    sx_buffer_release(b);
    sx_pipeline_release(p);
    gpu_deinit(&g);
    arena_reset(&arena);
}
