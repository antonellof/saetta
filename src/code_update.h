// Saetta Code's update check (code_update.c): at the start of the interactive mode, the latest
// release is read off the UI thread and, when newer than this binary, announced once in the chat.
#ifndef SAETTA_CODE_UPDATE_H
#define SAETTA_CODE_UPDATE_H

#include <stdbool.h>

// checkForNewPiVersion started asynchronously (interactive-mode.ts run): once per process; the
// thread writes a byte to wake_fd when its answer is ready.
void code_update_check_start(int wake_fd);
// On the UI thread, each loop: the notification added to the chat when the answer came and is a
// newer version (showNewVersionNotification); true when something was added.
bool code_update_check_poll(void);
// Before wake_fd is closed: the thread, if it still runs, no longer writes to it.
void code_update_check_stop(void);

#endif
