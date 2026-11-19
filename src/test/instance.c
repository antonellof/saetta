#include <stdlib.h>
#include <unistd.h>
#include "../instance.h"
#include "../test.h"

// The server record is "pid port mode" padded to 48 bytes, for short and long pids alike.
TEST(instance_record_48_bytes) {
    char tmpl[] = "/tmp/saetta-instance-XXXXXX";
    EXPECT(mkdtemp(tmpl) != NULL);
    const char *old = getenv("HOME");
    char *saved = old ? strdup(old) : NULL;
    setenv("HOME", tmpl, 1);
    setenv("SAETTA_AUTOSTOP", "1", 1);
    int fd = -1;
    Err e = instance_acquire(8080, &fd);
    unsetenv("SAETTA_AUTOSTOP");
    char path[256];
    snprintf(path, sizeof path, "%s/.saetta/run/serve.lock", tmpl);
    char *body = NULL;
    size_t len = 0;
    Err r = e ? e : read_file(path, 4096, &body, &len);
    if (fd >= 0) close(fd);
    if (saved) { setenv("HOME", saved, 1); free(saved); }
    EXPECT_OK(e);
    EXPECT_OK(r);
    EXPECT_EQ(len, 48);
    char want[32];
    int n = snprintf(want, sizeof want, "%d 8080 auto ", (int)getpid());
    EXPECT(strncmp(body, want, (size_t)n) == 0);
    free(body);
}
