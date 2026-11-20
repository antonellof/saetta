// Implementation of shim.h. Objective-C with ARC; handles are retained
// Objective-C objects bridged to opaque C pointers.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "shim.h"

struct sx_device { void *dev; void *queue; };

static void copy_err(char *err, size_t cap, NSString *msg) {
    if (!err || cap == 0) return;
    const char *s = msg ? msg.UTF8String : "unknown error";
    strncpy(err, s, cap - 1);
    err[cap - 1] = 0;
}

const void *sx_map_file(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *len = (size_t)st.st_size;
    return p;
}

void sx_unmap_file(const void *ptr, size_t len) { munmap((void *)ptr, len); }

size_t sx_page_size(void) { return (size_t)getpagesize(); }

void sx_prefetch(const void *addr, size_t len) {
    const uintptr_t page = (uintptr_t)getpagesize();
    const uintptr_t a = (uintptr_t)addr & ~(page - 1);
    madvise((void *)a, (uintptr_t)addr + len - a, MADV_WILLNEED);
}

size_t sx_nfc(const char *in, size_t len, char *out, size_t cap) {
    @autoreleasepool {
        NSString *s = [[NSString alloc] initWithBytes:in length:len encoding:NSUTF8StringEncoding];
        if (s == nil) return (size_t)-1;
        NSData *d = [[s precomposedStringWithCanonicalMapping] dataUsingEncoding:NSUTF8StringEncoding];
        memcpy(out, d.bytes, d.length < cap ? d.length : cap);
        return d.length;
    }
}

sx_device *sx_device_create(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return NULL;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (!q) return NULL;
    sx_device *d = calloc(1, sizeof *d);
    d->dev = (__bridge_retained void *)dev;
    d->queue = (__bridge_retained void *)q;
    return d;
}

void sx_device_destroy(sx_device *d) {
    if (!d) return;
    (void)(__bridge_transfer id)d->queue;
    (void)(__bridge_transfer id)d->dev;
    free(d);
}

void sx_device_name(const sx_device *d, char *buf, size_t cap) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d->dev;
    copy_err(buf, cap, dev.name);
}

uint64_t sx_device_recommended_working_set(const sx_device *d) {
    return ((__bridge id<MTLDevice>)d->dev).recommendedMaxWorkingSetSize;
}

int sx_thermal_state(void) {
    return (int)[NSProcessInfo processInfo].thermalState;
}

uint64_t sx_device_allocated_size(const sx_device *d) {
    return ((__bridge id<MTLDevice>)d->dev).currentAllocatedSize;
}

int sx_device_supports_apple10(const sx_device *d) {
    return [(__bridge id<MTLDevice>)d->dev supportsFamily:MTLGPUFamilyApple10] ? 1 : 0;
}

sx_buffer *sx_buffer_wrap(sx_device *d, const void *ptr, size_t len) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d->dev;
    id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:(void *)ptr length:len
                                            options:MTLResourceStorageModeShared deallocator:nil];
    return b ? (__bridge_retained sx_buffer *)b : NULL;
}

sx_buffer *sx_buffer_new(sx_device *d, size_t len) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d->dev;
    id<MTLBuffer> b = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (!b) return NULL;
    memset(b.contents, 0, len);
    return (__bridge_retained sx_buffer *)b;
}

void sx_buffer_release(sx_buffer *b) { if (b) (void)(__bridge_transfer id)(void *)b; }

void *sx_buffer_contents(sx_buffer *b) { return ((__bridge id<MTLBuffer>)(void *)b).contents; }

size_t sx_buffer_length(const sx_buffer *b) { return ((__bridge id<MTLBuffer>)(void *)b).length; }

sx_pipeline *sx_pipeline_from_source(sx_device *d, const char *source, const char *name, char *err, size_t err_cap) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d->dev;
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:source] options:o error:&e];
    if (!lib) { copy_err(err, err_cap, e.localizedDescription); return NULL; }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { copy_err(err, err_cap, [NSString stringWithFormat:@"no function %s", name]); return NULL; }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { copy_err(err, err_cap, e.localizedDescription); return NULL; }
    return (__bridge_retained sx_pipeline *)p;
}

void sx_pipeline_release(sx_pipeline *p) { if (p) (void)(__bridge_transfer id)(void *)p; }

uint32_t sx_pipeline_thread_width(const sx_pipeline *p) {
    return (uint32_t)((__bridge id<MTLComputePipelineState>)(void *)p).threadExecutionWidth;
}

int sx_dispatch_sync(sx_device *d, sx_pipeline *p, const sx_arg *args, size_t nargs,
                     size_t groups, size_t threads, char *err, size_t err_cap) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)d->queue;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)(void *)p];
    for (size_t i = 0; i < nargs; i++) {
        if (args[i].buffer)
            [enc setBuffer:(__bridge id<MTLBuffer>)(void *)args[i].buffer offset:args[i].offset atIndex:i];
        else
            [enc setBytes:args[i].bytes length:args[i].bytes_len atIndex:i];
    }
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { copy_err(err, err_cap, cb.error.localizedDescription); return 1; }
    return 0;
}

struct sx_cmd {
    void *cb;   // id<MTLCommandBuffer>, retained
    void *enc;  // id<MTLComputeCommandEncoder>, retained
};

sx_cmd *sx_cmd_begin(sx_device *d) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)d->queue;
    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    sx_cmd *c = calloc(1, sizeof *c);
    c->cb = (void *)CFBridgingRetain(cb);
    c->enc = (void *)CFBridgingRetain(enc);
    return c;
}

void sx_cmd_dispatch(sx_cmd *c, sx_pipeline *p, const sx_arg *args, size_t nargs,
                     size_t groups_x, size_t groups_y, size_t threads) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)(void *)p];
    for (size_t i = 0; i < nargs; i++) {
        if (args[i].buffer)
            [enc setBuffer:(__bridge id<MTLBuffer>)(void *)args[i].buffer offset:args[i].offset atIndex:i];
        else
            [enc setBytes:args[i].bytes length:args[i].bytes_len atIndex:i];
    }
    [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

// Waiting for a short command buffer: spin on its status (no thread wake-up latency; the
// decode passes are 1-30 ms), falling back to a blocking wait after 50 ms. SAETTA_SPIN=0 blocks.
static int spin_on = -1;
static double gpu_busy = 0;
double sx_gpu_seconds(void) { return gpu_busy; }
static void wait_cb(id<MTLCommandBuffer> cb) {
    if (spin_on < 0) { const char *v = getenv("SAETTA_SPIN"); spin_on = !(v && v[0] == '0'); }
    if (spin_on) {
        const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        while (cb.status < MTLCommandBufferStatusCompleted) {
            if (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0 > 50000000ull) break;
        }
    }
    [cb waitUntilCompleted];
    gpu_busy += cb.GPUEndTime - cb.GPUStartTime;
}

int sx_cmd_finish(sx_cmd *c, char *err, size_t err_cap) {
    id<MTLCommandBuffer> cb = CFBridgingRelease(c->cb);
    id<MTLComputeCommandEncoder> enc = CFBridgingRelease(c->enc);
    [enc endEncoding];
    [cb commit];
    wait_cb(cb);
    free(c);
    if (cb.error) { copy_err(err, err_cap, cb.error.localizedDescription); return 1; }
    return 0;
}

void sx_cmd_commit(sx_cmd *c) {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)c->cb;
    id<MTLComputeCommandEncoder> enc = CFBridgingRelease(c->enc);
    c->enc = NULL;
    [enc endEncoding];
    [cb commit];
}

int sx_cmd_wait(sx_cmd *c, char *err, size_t err_cap) {
    id<MTLCommandBuffer> cb = CFBridgingRelease(c->cb);
    wait_cb(cb);
    free(c);
    if (cb.error) { copy_err(err, err_cap, cb.error.localizedDescription); return 1; }
    return 0;
}

void sx_cmd_discard(sx_cmd *c) {
    id<MTLCommandBuffer> cb = CFBridgingRelease(c->cb);
    id<MTLComputeCommandEncoder> enc = CFBridgingRelease(c->enc);
    [enc endEncoding];
    (void)cb;
    free(c);
}

int sx_cmd_finish_timed(sx_cmd *c, double *gpu_seconds, char *err, size_t err_cap) {
    id<MTLCommandBuffer> cb = CFBridgingRelease(c->cb);
    id<MTLComputeCommandEncoder> enc = CFBridgingRelease(c->enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    free(c);
    *gpu_seconds = cb.GPUEndTime - cb.GPUStartTime;
    if (cb.error) { copy_err(err, err_cap, cb.error.localizedDescription); return 1; }
    return 0;
}

void sx_cmd_dispatch3(sx_cmd *c, sx_pipeline *p, const sx_arg *args, size_t nargs,
                      size_t groups_x, size_t groups_y, size_t groups_z, size_t threads) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)c->enc;
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)(void *)p];
    for (size_t i = 0; i < nargs; i++) {
        if (args[i].buffer)
            [enc setBuffer:(__bridge id<MTLBuffer>)(void *)args[i].buffer offset:args[i].offset atIndex:i];
        else
            [enc setBytes:args[i].bytes length:args[i].bytes_len atIndex:i];
    }
    [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

// --- events --------------------------------------------------------------------

struct sx_event {
    void *ev;  // id<MTLSharedEvent>, retained
};

sx_event *sx_event_new(sx_device *d) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)d->dev;
    id<MTLSharedEvent> ev = [dev newSharedEvent];
    if (!ev) return NULL;
    sx_event *e = calloc(1, sizeof *e);
    if (!e) return NULL;
    e->ev = (void *)CFBridgingRetain(ev);
    return e;
}

void sx_event_release(sx_event *e) {
    if (!e) return;
    CFBridgingRelease(e->ev);
    free(e);
}

void sx_event_set(sx_event *e, uint64_t v) { ((__bridge id<MTLSharedEvent>)e->ev).signaledValue = v; }

uint64_t sx_event_value(const sx_event *e) { return ((__bridge id<MTLSharedEvent>)e->ev).signaledValue; }

int sx_event_wait(const sx_event *e, uint64_t v) {
    id<MTLSharedEvent> ev = (__bridge id<MTLSharedEvent>)e->ev;
    const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    while (ev.signaledValue < v) {
        uint64_t dt = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        if (dt > 60000000000ull) return 1;
        if (dt > 50000000ull) usleep(50);
    }
    return 0;
}

static void cmd_end_encoder(sx_cmd *c) {
    id<MTLComputeCommandEncoder> old = CFBridgingRelease(c->enc);
    [old endEncoding];
    c->enc = NULL;
}

static void cmd_open_encoder(sx_cmd *c) {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)c->cb;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    c->enc = (void *)CFBridgingRetain(enc);
}

void sx_cmd_signal(sx_cmd *c, sx_event *e, uint64_t v) {
    cmd_end_encoder(c);
    [(__bridge id<MTLCommandBuffer>)c->cb encodeSignalEvent:(__bridge id<MTLSharedEvent>)e->ev value:v];
    cmd_open_encoder(c);
}

void sx_cmd_wait_event(sx_cmd *c, sx_event *e, uint64_t v) {
    cmd_end_encoder(c);
    [(__bridge id<MTLCommandBuffer>)c->cb encodeWaitForEvent:(__bridge id<MTLSharedEvent>)e->ev value:v];
    cmd_open_encoder(c);
}

// --- HTTP client -------------------------------------------------------------

@interface SxHttpTask : NSObject <NSURLSessionDataDelegate>
@property(nonatomic) sx_http_sink sink;
@property(nonatomic) void *ctx;
@property(nonatomic) int status;
@property(nonatomic) BOOL cancelled;
@property(nonatomic, strong) NSError *error;
@property(nonatomic, copy) NSString *auth; // Authorization while redirects keep it
@property(nonatomic, strong) dispatch_semaphore_t done;
@end

@implementation SxHttpTask
- (void)URLSession:(NSURLSession *)session
              dataTask:(NSURLSessionDataTask *)task
    didReceiveResponse:(NSURLResponse *)response
     completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler {
    if ([response isKindOfClass:[NSHTTPURLResponse class]])
        self.status = (int)((NSHTTPURLResponse *)response).statusCode;
    completionHandler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)task didReceiveData:(NSData *)data {
    if (self.cancelled) return;
    __block int stop = 0;
    [data enumerateByteRangesUsingBlock:^(const void *bytes, NSRange range, BOOL *halt) {
        if (self.sink && self.sink(self.ctx, self.status, bytes, range.length) != 0) {
            stop = 1;
            *halt = YES;
        }
    }];
    if (stop) {
        self.cancelled = YES;
        [task cancel];
    }
}

// std.http.Client's redirect rule: Authorization is kept only while the scheme stays the
// same and the new host is the old one or a subdomain of it; once dropped it stays dropped.
// CFNetwork drops it on every redirect, so it is set again here when kept.
static BOOL same_parent_domain(NSString *parent, NSString *child) {
    if (!parent || !child || child.length < parent.length) return NO;
    NSString *tail = [child substringFromIndex:child.length - parent.length];
    if ([tail caseInsensitiveCompare:parent] != NSOrderedSame) return NO;
    return child.length == parent.length || [child characterAtIndex:child.length - parent.length - 1] == '.';
}

- (void)URLSession:(NSURLSession *)session
                          task:(NSURLSessionTask *)task
    willPerformHTTPRedirection:(NSHTTPURLResponse *)response
                    newRequest:(NSURLRequest *)request
             completionHandler:(void (^)(NSURLRequest *))completionHandler {
    NSURL *from = response.URL ? response.URL : task.currentRequest.URL, *to = request.URL;
    BOOL keep = from.scheme && to.scheme && [from.scheme caseInsensitiveCompare:to.scheme] == NSOrderedSame &&
                same_parent_domain(from.host, to.host);
    if (!keep) self.auth = nil;
    NSMutableURLRequest *r = [request mutableCopy];
    [r setValue:self.auth forHTTPHeaderField:@"Authorization"];
    completionHandler(r);
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didCompleteWithError:(NSError *)error {
    self.error = error;
    dispatch_semaphore_signal(self.done);
}
@end

int sx_http(const char *method, const char *url, const sx_header *headers, size_t nheaders,
            const void *body, size_t body_len, double timeout_s, sx_http_sink on_data, void *ctx,
            int *status, char *err, size_t err_cap) {
    @autoreleasepool {
        NSURL *u = [NSURL URLWithString:@(url)];
        if (!u) { copy_err(err, err_cap, @"invalid URL"); return 1; }
        NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:u];
        req.HTTPMethod = @(method);
        BOOL has_encoding = NO;
        NSString *auth = nil;
        for (size_t i = 0; i < nheaders; i++) {
            NSString *name = @(headers[i].name);
            if ([name caseInsensitiveCompare:@"Accept-Encoding"] == NSOrderedSame) has_encoding = YES;
            if ([name caseInsensitiveCompare:@"Authorization"] == NSOrderedSame) auth = @(headers[i].value);
            [req setValue:@(headers[i].value) forHTTPHeaderField:name];
        }
        if (!has_encoding) [req setValue:@"identity" forHTTPHeaderField:@"Accept-Encoding"];
        if (body && body_len) req.HTTPBody = [NSData dataWithBytes:body length:body_len];

        NSURLSessionConfiguration *cfg = [NSURLSessionConfiguration ephemeralSessionConfiguration];
        cfg.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
        cfg.URLCache = nil;
        cfg.HTTPCookieStorage = nil;
        cfg.HTTPShouldSetCookies = NO;
        cfg.HTTPCookieAcceptPolicy = NSHTTPCookieAcceptPolicyNever;
        if (timeout_s > 0) cfg.timeoutIntervalForRequest = timeout_s;

        SxHttpTask *t = [SxHttpTask new];
        t.sink = on_data;
        t.ctx = ctx;
        t.auth = auth;
        t.done = dispatch_semaphore_create(0);
        NSOperationQueue *q = [NSOperationQueue new];
        q.maxConcurrentOperationCount = 1; // serial: data arrives in order
        NSURLSession *session = [NSURLSession sessionWithConfiguration:cfg delegate:t delegateQueue:q];
        NSURLSessionDataTask *task = [session dataTaskWithRequest:req];
        [task resume];
        dispatch_semaphore_wait(t.done, DISPATCH_TIME_FOREVER);
        [session finishTasksAndInvalidate]; // releases the delegate

        if (status) *status = t.status;
        if (t.cancelled) { copy_err(err, err_cap, @"cancelled"); return 1; }
        if (t.error) { copy_err(err, err_cap, t.error.localizedDescription); return 1; }
        return 0;
    }
}
