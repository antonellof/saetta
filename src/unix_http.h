// HTTP/1.1 over a Unix domain socket, for the sessions that use a model another Saetta Code holds
// (code_share.c; NSURLSession, behind the shim's sx_http, has no Unix sockets). The same contract
// as sx_http_cancellable (src/metal/shim.h): the response body to on_data as it arrives (chunked
// bodies decoded), an idle timeout between bytes, *cancel polled every 100 ms.
#ifndef SAETTA_UNIX_HTTP_H
#define SAETTA_UNIX_HTTP_H

#include "base.h"
#include "metal/shim.h"

// method target (an origin-form path, "/v1/chat/completions") on the socket at sock. Returns 0
// with *status set, or nonzero with err set ("timed out", "cancelled", the system's reason).
int unix_http(const char *sock, const char *method, const char *target, const sx_header *headers, size_t nheaders,
              const void *body, size_t body_len, double timeout_s, sx_http_sink on_data, void *ctx, int *status, char *err,
              size_t err_cap, volatile int *cancel);

#endif
