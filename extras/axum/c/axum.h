/* axum for Bifrost: the types its C files share (see axum.c). */
#ifndef BIFROST_AXUM_H
#define BIFROST_AXUM_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "h2o.h"

/* Handles Bifrost holds: one pointer each (a C struct of one `ptr` field, as config.yaml declares them). */
typedef struct { void *raw; } AxumRequest;
typedef struct { void *raw; } AxumResponse; /* owned by Bifrost until a handler gives it back (drop: axum_response_free) */
typedef struct { void *raw; } AxumRouter;
typedef struct { void *raw; } AxumMethodRouter;
typedef struct { void *raw; } AxumLayer;
typedef struct { void *raw; } AxumNext;
typedef struct { void *raw; } AxumListener;
typedef struct { void *raw; } AxumStream;

/* What the compiler writes for a handler: extracts, calls it, puts its response in *out; a token of when it is done. */
typedef void *(*AxumHandler)(AxumRequest request, void *state, AxumResponse *out);
/* An async function of no arguments, as C starts it (a graceful shutdown's signal). */
typedef void *(*AxumTask)(void);

typedef struct {
    char *name; /* lower case */
    char *value;
} Header;

typedef struct {
    Header *items;
    size_t count, capacity;
} Headers;

struct Request;
struct Response;

typedef struct Response {
    int status;
    Headers headers;
    char *body;
    size_t len;
} Response;

typedef enum {
    LAYER_FROM_FN,      /* middleware::from_fn: the handler runs the rest with next.run(req) */
    LAYER_MAP_REQUEST,  /* map_request: the handler answers (stopping the request), or not (it goes on) */
    LAYER_MAP_RESPONSE, /* map_response: the handler is given the response the rest made */
    LAYER_BODY_LIMIT,   /* DefaultBodyLimit */
    LAYER_HEADER,       /* a header added to responses without one */
    LAYER_EXTENSION,    /* a value added to requests */
} LayerKind;

typedef struct {
    LayerKind kind;
    AxumHandler handler;
    void *state;
    int64_t limit;
    char *name, *value;
} Layer;

typedef struct {
    Layer **items;
    size_t count, capacity;
} Layers;

/* A route's handlers by method, each with the layers MethodRouter::layer gave it. */
typedef struct {
    AxumHandler handlers[9]; /* GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS, TRACE, CONNECT */
    Layers layers[9];
    AxumHandler any;
    Layers any_layers;
    AxumHandler fallback;
    Layers fallback_layers;
} MethodRouter;

/* What a route runs: its method router, or a service (every method), wrapped in the router's layers. */
typedef struct {
    MethodRouter *methods;
    AxumHandler service;
    Layers layers;
    void **state;      /* the state slot of the router it was added to */
    char *nest_prefix; /* nested: the prefix its handlers do not see */
} Endpoint;

typedef enum { SEGMENT_STATIC, SEGMENT_CAPTURE, SEGMENT_WILDCARD } SegmentKind;

typedef struct {
    SegmentKind kind;
    char *text; /* the static text, or the capture's name */
} Segment;

typedef struct {
    char *pattern;
    Segment *segments;
    size_t count;
    Endpoint *endpoint;
    bool prefix;   /* nest_service, or a nested router's fallback: the path and everything below it */
    bool fallback; /* a nested router's fallback */
} Route;

typedef struct {
    Route *routes;
    size_t count, capacity;
    Endpoint *fallback;
    Layers fallback_layers;
    Endpoint *method_not_allowed;
    void *state;
} Router;

typedef struct Cleanup {
    void (*release)(void *data);
    void *data;
    struct Cleanup *next;
} Cleanup;

typedef struct Next {
    struct Request *r;
    size_t level;
    bool ran;
    AxumResponse inner;
    struct Next *link;
} Next;

/* A request, copied out of h2o when it arrives: it lives until its response is sent (or it is abandoned). */
typedef struct Request {
    h2o_req_t *req; /* NULL once the client has gone */
    struct timespec started;
    const char *method, *path, *query, *original_uri, *version, *matched_path, *nested_path, *params, *peer, *allowed;
    bool has_query;
    const char *body;
    size_t body_len;
    Headers headers;
    Headers extensions;
    /* what serves it */
    Layers layers;
    size_t level; /* the layer being run: what Next.from_request reads */
    AxumHandler handler;
    void *state;
    int internal_status; /* without a handler: 404, or 405 */
    int64_t limit;
    AxumResponse response;
    bool finished; /* its handler is done; it is freed once h2o is done with it too */
    bool streaming;
    Next *nexts;
    Cleanup *cleanups; /* what feature libraries (ws, multipart) free with it */
    char **kept; /* strings it keeps until it is freed */
    size_t kept_count, kept_capacity;
} Request;

typedef struct {
    int fd;
    char *error;
    char *local;
    bool nodelay;
} Listener;

typedef struct {
    Listener *listener;
    Router *router;
    h2o_socket_t *socket;
    void *token; /* completes when it stops */
    bool shutting_down;
    int64_t in_flight;
} Server;

void *axum_alloc(size_t size);
void *axum_zalloc(size_t size);
char *axum_strdup(const char *text);
char *axum_strndup(const char *text, size_t len);
void *axum_done_token(void);
void axum_after(void *token, void (*then)(void *context, bool failed), void *context);
AxumResponse axum_response(int32_t status);
void axum_set_header(AxumResponse response, const char *name, const char *value);
void axum_stream_finish(Request *r);
void axum_on_release(Request *r, void (*release)(void *data), void *data);
const char *axum_keep(Request *r, char *text);
void axum_mark_streaming(Request *r);
void axum_streams_release(Request *r);

#endif
