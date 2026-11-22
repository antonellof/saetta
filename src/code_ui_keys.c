// Saetta Code's clipboard keys and flash messages (see code_ui_keys.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/interactive-mode.ts,
// components/custom-editor.ts, utils/clipboard.ts, utils/clipboard-image.ts,
// packages/tui/src/components/alt-screen-flash.ts, tui-alt-screen.ts. Each function names the pi
// function it ports.
#include "code_ui_keys.h"

#include "code_cmds.h"
#include "code_ui_ext.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---- AltScreenFlashContainer ----------------------------------------------------------------------

typedef struct {
    char *message;
    uint64_t until_ns;
} Flash;
static VEC(Flash) g_flashes;

// flash
void uik_flash(const char *msg, int duration_ms) {
    Flash f = {strdup(msg), now_ns() + (uint64_t)(duration_ms > 0 ? duration_ms : 0) * 1000000ull};
    vec_push(&g_flashes, f);
    uix_request_render();
}

// render: `\x1b[7m${truncateToWidth(` ${message} `, width, "")}\x1b[27m`
void uik_flash_render(int width, Lines *out) {
    for (size_t i = 0; i < g_flashes.len; i++) {
        Buf m = {0}, t = {0};
        buf_printf(&m, " %s ", g_flashes.ptr[i].message);
        tui_truncate(&t, buf_str(&m), width, "", false);
        Buf l = {0};
        buf_adds(&l, "\x1b[7m");
        buf_addstr(&l, buf_str(&t));
        buf_adds(&l, "\x1b[27m");
        lines_addb(out, &l);
        buf_free(&m);
        buf_free(&t);
    }
}

int uik_flash_wait_ms(uint64_t now) {
    int ms = -1;
    for (size_t i = 0; i < g_flashes.len; i++) {
        uint64_t u = g_flashes.ptr[i].until_ns;
        int left = u > now ? (int)((u - now) / 1000000ull) + 1 : 0;
        if (ms < 0 || left < ms) ms = left;
    }
    return ms;
}

// the flash's timer: removed once its time is up
bool uik_flash_expire(uint64_t now) {
    bool any = false;
    for (size_t i = 0; i < g_flashes.len;) {
        if (g_flashes.ptr[i].until_ns <= now) {
            free(g_flashes.ptr[i].message);
            memmove(g_flashes.ptr + i, g_flashes.ptr + i + 1, (g_flashes.len - i - 1) * sizeof(Flash));
            g_flashes.len--;
            any = true;
        } else i++;
    }
    return any;
}

static void flash_copied(const char *msg) { uik_flash(msg, 1000); }

// ---- the clipboard (utils/clipboard.ts, clipboard-image.ts) ------------------------------------------

// pi reads the macOS pasteboard through its native clipboard module (getFilePaths, getImage,
// getText); Saetta Code asks the same pasteboard through osascript (JavaScript for Automation):
// the copied files' paths, else an image (PNG, JPEG, WebP, GIF as they are; any other image as
// PNG, as pi converts it) written to argv[0] + its extension, else the text. The answer's first
// character says which: P paths (one per line), I the image file, T text, N nothing.
static const char CLIPBOARD_JXA[] =
    "function run(argv) {\n"
    "  ObjC.import('AppKit');\n"
    "  var pb = $.NSPasteboard.generalPasteboard;\n"
    "  var opts = $.NSDictionary.dictionaryWithObjectForKey($.NSNumber.numberWithBool(true), $('NSPasteboardURLReadingFileURLsOnlyKey'));\n"
    "  var urls = pb.readObjectsForClassesOptions($.NSArray.arrayWithObject($.NSURL), opts);\n"
    "  if (urls && urls.count > 0) {\n"
    "    var out = [];\n"
    "    for (var i = 0; i < urls.count; i++) out.push(urls.objectAtIndex(i).path.js);\n"
    "    return 'P' + out.join('\\n');\n"
    "  }\n"
    "  var types = ['public.png', 'public.jpeg', 'org.webmproject.webp', 'com.compuserve.gif'], exts = ['png', 'jpg', 'webp', 'gif'];\n"
    "  for (var k = 0; k < types.length; k++) {\n"
    "    var d = pb.dataForType(types[k]);\n"
    "    if (d && !d.isNil() && d.length > 0) { var p = argv[0] + '.' + exts[k]; d.writeToFileAtomically(p, true); return 'I' + p; }\n"
    "  }\n"
    "  var img = $.NSImage.alloc.initWithPasteboard(pb);\n"
    "  if (img && !img.isNil()) {\n"
    "    var tiff = img.TIFFRepresentation;\n"
    "    var rep = tiff && !tiff.isNil() ? $.NSBitmapImageRep.imageRepWithData(tiff) : null;\n"
    "    var png = rep && !rep.isNil() ? rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $({})) : null;\n"
    "    if (png && !png.isNil() && png.length > 0) { var p2 = argv[0] + '.png'; png.writeToFileAtomically(p2, true); return 'I' + p2; }\n"
    "  }\n"
    "  var t = pb.stringForType('public.utf8-plain-text');\n"
    "  if (t && !t.isNil()) return 'T' + t.js;\n"
    "  return 'N';\n"
    "}\n";

// The clipboard's content as CLIPBOARD_JXA reports it (malloc'd, the kind letter first), or NULL
// when the pasteboard cannot be read.
static char *read_clipboard(const char *image_base) {
    char *argv[] = {"osascript", "-l", "JavaScript", "-e", (char *)CLIPBOARD_JXA, (char *)image_base, NULL};
    Buf out = {0};
    if (!tui_run(argv, "/", S(""), &out, 5000) || !out.len) {
        buf_free(&out);
        return NULL;
    }
    // osascript ends what the script returned with a line break
    if (out.ptr[out.len - 1] == '\n') out.ptr[--out.len] = 0;
    char *r = strndup(out.ptr, out.len);
    buf_free(&out);
    return r;
}

// quoteIfNeeded
static void quote_if_needed(Buf *b, Str v) {
    bool plain = v.len > 0;
    for (size_t i = 0; i < v.len && plain; i++) {
        char c = v.ptr[i];
        plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || strchr("_-./~:@", c);
    }
    if (plain) {
        buf_addstr(b, v);
        return;
    }
    buf_addc(b, '\'');
    for (size_t i = 0; i < v.len; i++) {
        if (v.ptr[i] == '\'') buf_adds(b, "'\\''");
        else buf_addc(b, v.ptr[i]);
    }
    buf_addc(b, '\'');
}

// handleClipboardPaste: copied files' paths (quoted in bash mode, one per line otherwise, spaced
// from the text around the cursor), else an image attached through a temporary file, else the text.
static void clipboard_paste(void) {
    Editor *ed = uix_editor();
    const char *tmp = getenv("TMPDIR");
    Buf base = {0};
    buf_printf(&base, "%s", tmp && *tmp ? tmp : "/tmp");
    while (base.len > 1 && base.ptr[base.len - 1] == '/') base.ptr[--base.len] = 0;
    // pi-clipboard-<randomUUID>
    uint8_t r[16];
    arc4random_buf(r, sizeof r);
    r[6] = (uint8_t)((r[6] & 0x0f) | 0x40);
    r[8] = (uint8_t)((r[8] & 0x3f) | 0x80);
    buf_adds(&base, "/pi-clipboard-");
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) buf_addc(&base, '-');
        buf_printf(&base, "%02x", r[i]);
    }
    char *got = read_clipboard(base.ptr);
    buf_free(&base);
    if (!got) {
        uix_show_error("Failed to paste from clipboard: the clipboard could not be read");
        return;
    }
    Str body = str(got + 1);
    if (got[0] == 'P') {
        for (size_t i = 0; i < body.len; i++)
            if ((unsigned char)body.ptr[i] < 0x20 && body.ptr[i] != '\n') {
                uix_show_error("Failed to paste from clipboard: Clipboard file path contains control characters");
                free(got);
                return;
            }
        Str text = editor_text(ed);
        Str ts = text;
        while (ts.len && tui_is_space((unsigned char)ts.ptr[0])) ts = str_sub(ts, 1, ts.len);
        bool bash = ts.len && ts.ptr[0] == '!';
        Buf paths = {0};
        Str rest = body;
        for (bool first = true;; first = false) {
            ptrdiff_t nl = str_find_char(rest, '\n', 0);
            Str one = nl >= 0 ? str_sub(rest, 0, (size_t)nl) : rest;
            if (!first) buf_addc(&paths, bash ? ' ' : '\n');
            if (bash) quote_if_needed(&paths, one);
            else buf_addstr(&paths, one);
            if (nl < 0) break;
            rest = str_sub(rest, (size_t)nl + 1, rest.len);
        }
        // the characters around the cursor on its line
        size_t line, col;
        editor_cursor(ed, &line, &col);
        Str cur = text;
        for (size_t l = 0; l < line; l++) {
            ptrdiff_t nl = str_find_char(cur, '\n', 0);
            cur = nl >= 0 ? str_sub(cur, (size_t)nl + 1, cur.len) : S("");
        }
        ptrdiff_t end = str_find_char(cur, '\n', 0);
        if (end >= 0) cur = str_sub(cur, 0, (size_t)end);
        bool lead = col > 0 && col <= cur.len && !tui_is_space((unsigned char)cur.ptr[col - 1]);
        bool trail = col < cur.len && !tui_is_space((unsigned char)cur.ptr[col]);
        Buf ins = {0};
        if (lead) buf_addc(&ins, ' ');
        buf_addstr(&ins, buf_str(&paths));
        if (trail) buf_addc(&ins, ' ');
        editor_insert_text(ed, buf_str(&ins));
        buf_free(&ins);
        buf_free(&paths);
    } else if (got[0] == 'I' || got[0] == 'T') {
        if (body.len) editor_insert_text(ed, body);
    }
    free(got);
    uix_request_render();
}

// ---- the app keys (custom-editor.ts handleInput, interactive-mode.ts setupKeyHandlers) -----------

bool uik_key(const Key *k) {
    if (k->kind != K_CTRL || k->shift) return false;
    if (k->ch == 'v') { // app.clipboard.pasteImage (onPasteImage)
        clipboard_paste();
        return true;
    }
    if (k->ch == 'x') { // app.message.copy: handleCopyCommand({flashConfirmation, preferSelection})
        cmds_copy(flash_copied);
        return true;
    }
    return false;
}
