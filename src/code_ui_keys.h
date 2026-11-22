// Saetta Code's clipboard keys and the alternate screen's flash messages (code_ui_keys.c):
// app.message.copy (Ctrl+X, "Copied!"), app.clipboard.pasteImage (Ctrl+V: copied files' paths, an
// image saved to a temporary file, or text), and the flash stack the renderer composites at the
// top right.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/interactive-mode.ts (handleCopyCommand,
// handleClipboardPaste), components/custom-editor.ts (onPasteImage), utils/clipboard.ts and
// clipboard-image.ts (readClipboardFilePaths, readClipboardImage, readClipboardText,
// extensionForImageMimeType), packages/tui/src/components/alt-screen-flash.ts
// (AltScreenFlashContainer), tui-alt-screen.ts (flash, compositeFlashes).
#ifndef SAETTA_CODE_UI_KEYS_H
#define SAETTA_CODE_UI_KEYS_H

#include "code_tui.h"

// AltScreenFlashContainer.flash: msg for duration_ms (pi's default 1000).
void uik_flash(const char *msg, int duration_ms);
// render: one line per flash (inverse, " msg ", cut to width), the oldest first.
void uik_flash_render(int width, Lines *out);
// The main loop's timer: ms until the next flash expires (-1: none); true when one expired (redraw).
int uik_flash_wait_ms(uint64_t now_ns);
bool uik_flash_expire(uint64_t now_ns);

// The app keys of the editor (custom-editor.ts): true when k was one of them and was handled.
bool uik_key(const Key *k);

#endif
