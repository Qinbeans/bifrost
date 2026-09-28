/*
 * A fasthttp-style HTTP server for Bifrost, on libh2o.
 *
 *     let app = http.app()
 *     http.get(app, "/users/:id", get_user)       // get_user: (ctx: http.Context) => null
 *     http.run(app, 8080)   // on $PORT (or 8080): HTTP/1.1 and HTTP/2 (h2c), until killed;
 *                           // with $TLS_CERT and $TLS_KEY: HTTPS, HTTP/2 and HTTP/3
 *
 * State the handlers share (a counter, a connection) is lent to the server:
 *
 *     http.get_with(app, "/hits", hits)   // hits: (ctx: http.Context, state: mem.Weak[AppState]) => null
 *     http.run_with(app, 8080, state)     // state: mem.Unique[AppState], owned by main
 *
 * http.serve(app, port) and http.serve_tls(app, port, cert, key) do the same
 * without reading the environment.
 *
 * h2o runs the connections; each request is routed here and its handler is
 * called with a Context: a handle to the request being served. The handler
 * reads the request (param, query, header, body) and writes one response
 * (text, json, the json_* builder, error, redirect).
 *
 * Handlers run on the event loop, one at a time, and may pause: a handler that
 * calls http.sleep (or a function that does) is a coroutine, which gives the loop
 * back while it waits, so other requests are served meanwhile. Each handler
 * returns a token (Bifrost's async runtime, std/async_runtime.c) that completes
 * when it is done; its response is sent then. Handlers that never pause are
 * wrapped by the compiler, and finish within the request's own turn.
 *
 * A Context is valid until its response is sent: afterwards (or once the client
 * has gone), its functions do nothing and say so on stderr. Strings they return
 * live until the response is sent.
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "h2o.h"
#include "h2o/http1.h"
#include "h2o/http2.h"
#include "h2o/http3_server.h"
#include "picotls/openssl.h"
#include "quicly.h"
#include "quicly/defaults.h"

typedef struct {
    int32_t id;
} BifrostHttpApp;

typedef struct {
    int32_t id;
} BifrostHttpContext;

/* A handler starts serving a request and returns a token that completes when it is done. */
typedef void *(*BifrostHttpHandler)(BifrostHttpContext);
/* A handler of an app run with state (http.run_with), given that state: (ctx, state: mem.Shared[AppState]). */
typedef void *(*BifrostHttpStateHandler)(BifrostHttpContext, void *);

/* Bifrost's async runtime (std/async_runtime.c), linked into the program. */
void *mlirAsyncRuntimeCreateToken(void);
void mlirAsyncRuntimeEmplaceToken(void *token);
void mlirAsyncRuntimeAddRef(void *object, int64_t count);
void mlirAsyncRuntimeDropRef(void *object, int64_t count);
bool mlirAsyncRuntimeIsTokenError(void *token);
void mlirAsyncRuntimeAwaitTokenAndExecute(void *token, void *handle, void (*resume)(void *));
void bifrost_async_run_ready(void);
bool bifrost_async_pending(void);
int64_t bifrost_async_next_timer(void);
void bifrost_async_set_poller(void (*poll)(void *), void *context);

/* -- apps and routes -------------------------------------------------------- */

#define MAX_APPS 8
#define MAX_PARAMS 16
#define MAX_JSON_DEPTH 32

typedef struct {
    char method[16]; /* "" matches every method */
    char *pattern;
    BifrostHttpHandler handler;
    BifrostHttpStateHandler state_handler; /* instead of handler, for routes added with the *_with functions */
} Route;

typedef struct {
    Route *routes;
    size_t count, capacity;
} App;

static App apps[MAX_APPS];
static int32_t app_count;

/* -- the requests being served ---------------------------------------------- */

/* A request until its response is sent: allocated from its h2o pool, so it goes with it. */
typedef struct {
    int32_t id; /* of its Context */
    h2o_req_t *req;
    Route *route;
    struct timespec started;
    const char *param_names[MAX_PARAMS], *param_values[MAX_PARAMS];
    size_t param_count;
    int responded;
    int status;
    char *body;
    size_t body_len, body_cap;
    /* the json_* builder: for each open object/array, whether it has an entry yet, and whether it is an array */
    int json_depth, json_has_entry[MAX_JSON_DEPTH], json_is_array[MAX_JSON_DEPTH];
    const char *allowed; /* for a 405: the methods a matching path allows */
} Request;

/* Requests not yet answered, by Context id (the slot is the id modulo the size). */
#define MAX_LIVE 4096
static Request *live[MAX_LIVE];

/* The request the code below works on: set by request_of (and on_request). */
static Request *active;
#define current (*active)

static int32_t next_context_id = 1;
static void *app_state; /* lent by http.run_with for as long as it serves */
static const char *alt_svc; /* `h3=":8443"` when HTTP/3 is served */

static Request *find(int32_t id)
{
    Request *request = id > 0 ? live[id % MAX_LIVE] : NULL;
    return request != NULL && request->id == id ? request : NULL;
}

static h2o_req_t *request_of(BifrostHttpContext ctx, const char *function)
{
    Request *request = find(ctx.id);
    if (request == NULL) {
        fprintf(stderr, "http.%s: this Context's request is over; use it only until its response is sent\n", function);
        return NULL;
    }
    active = request;
    return request->req;
}

static char *pool_string(const char *text, size_t len)
{
    return h2o_strdup(&current.req->pool, text, len).base;
}

static void body_append(const char *text, size_t len)
{
    if (current.body_len + len + 1 > current.body_cap) {
        size_t capacity = current.body_cap ? current.body_cap : 256;
        while (capacity < current.body_len + len + 1)
            capacity *= 2;
        char *grown = realloc(current.body, capacity);
        if (grown == NULL)
            h2o_fatal("out of memory");
        current.body = grown;
        current.body_cap = capacity;
    }
    memcpy(current.body + current.body_len, text, len);
    current.body_len += len;
    current.body[current.body_len] = '\0';
}

static void body_text(const char *text)
{
    body_append(text, strlen(text));
}

static void add_header(const char *name, const char *value)
{
    size_t name_len = strlen(name);
    char *lower = h2o_mem_alloc_pool(&current.req->pool, char, name_len + 1);
    for (size_t i = 0; i != name_len; ++i)
        lower[i] = (char)tolower((unsigned char)name[i]);
    lower[name_len] = '\0';
    size_t value_len = strlen(value);
    h2o_add_header_by_str(&current.req->pool, &current.req->res.headers, lower, name_len, 1, NULL, pool_string(value, value_len),
                          value_len);
}

/* Start the one response; 0 if the handler already responded. */
static int respond(h2o_req_t *req, int status, const char *content_type, const char *function)
{
    if (current.responded) {
        fprintf(stderr, "http.%s: %.*s %.*s already has a response\n", function, (int)req->method.len, req->method.base,
                (int)req->path.len, req->path.base);
        return 0;
    }
    current.responded = 1;
    current.status = status;
    current.body_len = 0;
    if (content_type != NULL)
        add_header("Content-Type", content_type);
    return 1;
}

/* -- routing -------------------------------------------------------------------- */

// Match `path` against a pattern like "/users/:id" or "/files/*", recording the parameters.
static int match(const char *pattern, const char *path, size_t path_len)
{
    const char *end = path + path_len;
    current.param_count = 0;
    while (*pattern != '\0') {
        while (*pattern == '/')
            ++pattern;
        while (path != end && *path == '/')
            ++path;
        if (*pattern == '\0')
            break;
        const char *segment_end = strchr(pattern, '/');
        size_t segment_len = segment_end ? (size_t)(segment_end - pattern) : strlen(pattern);
        const char *value_end = memchr(path, '/', (size_t)(end - path));
        if (value_end == NULL)
            value_end = end;
        if (segment_len == 1 && *pattern == '*') {
            if (current.param_count == MAX_PARAMS)
                return 0;
            current.param_names[current.param_count] = "*";
            current.param_values[current.param_count++] = pool_string(path, (size_t)(end - path));
            return 1;
        }
        if (path == end)
            return 0;
        if (*pattern == ':') {
            if (current.param_count == MAX_PARAMS)
                return 0;
            current.param_names[current.param_count] = pool_string(pattern + 1, segment_len - 1);
            current.param_values[current.param_count++] = pool_string(path, (size_t)(value_end - path));
        } else if ((size_t)(value_end - path) != segment_len || memcmp(pattern, path, segment_len) != 0) {
            return 0;
        }
        pattern += segment_len;
        path = value_end;
    }
    while (path != end && *path == '/')
        ++path;
    return path == end;
}

static const char *reason(int status)
{
    switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 301: return "Moved Permanently";
    case 302: return "Found";
    case 303: return "See Other";
    case 304: return "Not Modified";
    case 307: return "Temporary Redirect";
    case 308: return "Permanent Redirect";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 422: return "Unprocessable Content";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default: return "";
    }
}

static void log_request(h2o_req_t *req, struct timespec *started)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double ms = (double)(now.tv_sec - started->tv_sec) * 1e3 + (double)(now.tv_nsec - started->tv_nsec) / 1e6;
    const char *protocol = req->version >= 0x300 ? "HTTP/3" : req->version >= 0x200 ? "HTTP/2" : "HTTP/1.1";
    fprintf(stderr, "%.*s %.*s -> %d (%.3f ms, %s)\n", (int)req->method.len, req->method.base, (int)req->path.len, req->path.base,
            current.status, ms, protocol);
}

typedef struct {
    h2o_handler_t super;
    App *app;
} Handler;

/* Send the response the handler wrote (a 500 if it wrote none), and forget the request. */
static void finish(Request *request, bool failed)
{
    active = request;
    h2o_req_t *req = request->req;
    if (request->route != NULL && (failed || !current.responded)) {
        const char *why = failed ? "failed" : "sent no response";
        fprintf(stderr, "http: the handler of %s %s %s\n", request->route->method, request->route->pattern, why);
        current.responded = 0;
        respond(req, 500, "application/json", "serve");
        body_text("{\"error\": \"internal server error\"}");
    }
    live[request->id % MAX_LIVE] = NULL;
    req->res.status = current.status;
    req->res.reason = reason(current.status);
    if (alt_svc != NULL && req->version < 0x300)
        add_header("Alt-Svc", alt_svc);
    log_request(req, &current.started);
    h2o_send_inline(req, current.body ? current.body : "", current.body_len);
    free(current.body);
    current.body = NULL;
}

/* The client went away before the response: its handler's Context stops working. */
static void on_request_disposed(void *memory)
{
    Request *request = memory;
    if (find(request->id) == request) {
        live[request->id % MAX_LIVE] = NULL;
        free(request->body);
    }
}

/* A handler's token, and its request: what to finish once the token completes. */
typedef struct {
    int32_t id;
    void *token;
} Pending;

static void on_handler_done(void *handle)
{
    Pending *pending = handle;
    Request *request = find(pending->id);
    if (request != NULL)
        finish(request, mlirAsyncRuntimeIsTokenError(pending->token));
    mlirAsyncRuntimeDropRef(pending->token, 1);
    free(pending);
}


static int on_request(h2o_handler_t *self, h2o_req_t *req)
{
    App *app = ((Handler *)self)->app;
    int32_t id = next_context_id++;
    if (next_context_id == INT32_MAX)
        next_context_id = 1;
    if (live[id % MAX_LIVE] != NULL) {
        h2o_send_error_503(req, "Service Unavailable", "too many requests in progress", 0);
        return 0;
    }
    Request *request = h2o_mem_alloc_shared(&req->pool, sizeof(Request), on_request_disposed);
    memset(request, 0, sizeof(*request));
    request->id = id;
    request->req = req;
    clock_gettime(CLOCK_MONOTONIC, &request->started);
    live[id % MAX_LIVE] = request;
    active = request;
    BifrostHttpContext ctx = {id};

    Route *found = NULL;
    char allowed[128] = "";
    for (size_t i = 0; i != app->count && found == NULL; ++i) {
        Route *route = app->routes + i;
        if (!match(route->pattern, req->path_normalized.base, req->path_normalized.len))
            continue;
        int is_head = h2o_memis(req->method.base, req->method.len, H2O_STRLIT("HEAD")) && strcmp(route->method, "GET") == 0;
        if (route->method[0] == '\0' || is_head ||
            h2o_memis(req->method.base, req->method.len, route->method, strlen(route->method))) {
            found = route;
        } else if (strlen(allowed) + strlen(route->method) + 3 < sizeof(allowed)) {
            if (allowed[0] != '\0')
                strcat(allowed, ", ");
            strcat(allowed, route->method);
        }
    }
    if (found == NULL) {
        if (allowed[0] != '\0') {
            respond(req, 405, "application/json", "serve");
            add_header("Allow", allowed);
            body_text("{\"error\": \"method not allowed\"}");
        } else {
            respond(req, 404, "application/json", "serve");
            body_text("{\"error\": \"not found\"}");
        }
        finish(request, false);
        return 0;
    }
    match(found->pattern, req->path_normalized.base, req->path_normalized.len); /* its parameters */
    request->route = found;
    if (found->state_handler != NULL && app_state == NULL) {
        fprintf(stderr, "http: %s %s takes the app's state; serve it with http.run_with\n", found->method, found->pattern);
        finish(request, false);
        return 0;
    }
    void *token = found->state_handler != NULL ? found->state_handler(ctx, app_state) : found->handler(ctx);
    Pending *pending = malloc(sizeof(Pending));
    if (pending == NULL)
        h2o_fatal("out of memory");
    *pending = (Pending){id, token};
    mlirAsyncRuntimeAwaitTokenAndExecute(token, pending, on_handler_done);
    bifrost_async_run_ready(); /* a handler that did not pause is done: respond now */
    return 0;
}

/* -- pausing ------------------------------------------------------------------------ */

static h2o_globalconf_t config;
static h2o_context_t context;
static h2o_accept_ctx_t accept_ctx;

typedef struct {
    h2o_timer_t timer; /* first, so the timer is the Sleep */
    void *token;
} Sleep;

static void on_wake(h2o_timer_t *timer)
{
    Sleep *sleep = (Sleep *)timer;
    mlirAsyncRuntimeEmplaceToken(sleep->token); /* drops the reference it kept until now */
    free(sleep);
}

/* A token that completes after `milliseconds`: the handler waiting on it pauses meanwhile. */
void *bifrost_http_sleep(int64_t milliseconds)
{
    void *token = mlirAsyncRuntimeCreateToken();
    if (context.loop == NULL) {
        fprintf(stderr, "http.sleep: the server is not running; call it from a handler\n");
        mlirAsyncRuntimeEmplaceToken(token);
        return token;
    }
    Sleep *sleep = calloc(1, sizeof(Sleep));
    if (sleep == NULL)
        h2o_fatal("out of memory");
    sleep->token = token;
    h2o_timer_init(&sleep->timer, on_wake);
    h2o_timer_link(context.loop, milliseconds > 0 ? (uint64_t)milliseconds : 0, &sleep->timer);
    return token;
}

/* -- building the app ----------------------------------------------------------- */

static App *app_of(BifrostHttpApp app, const char *function)
{
    if (app.id < 1 || app.id > app_count) {
        fprintf(stderr, "http.%s: not an app from http.app()\n", function);
        return NULL;
    }
    return apps + app.id - 1;
}

BifrostHttpApp bifrost_http_app(void)
{
    if (app_count == MAX_APPS) {
        fprintf(stderr, "http.app: at most %d apps\n", MAX_APPS);
        return (BifrostHttpApp){0};
    }
    return (BifrostHttpApp){++app_count};
}

static void add_route(BifrostHttpApp app, const char *method, const char *pattern, BifrostHttpHandler handler,
                      BifrostHttpStateHandler state_handler)
{
    App *found = app_of(app, "get");
    if (found == NULL)
        return;
    if (strlen(method) >= sizeof(found->routes->method)) {
        fprintf(stderr, "http.route: '%s' is not a method\n", method);
        return;
    }
    if (found->count == found->capacity) {
        found->capacity = found->capacity ? found->capacity * 2 : 16;
        found->routes = realloc(found->routes, found->capacity * sizeof(*found->routes));
        if (found->routes == NULL)
            h2o_fatal("out of memory");
    }
    Route *route = found->routes + found->count++;
    strcpy(route->method, method);
    route->pattern = strdup(pattern);
    route->handler = handler;
    route->state_handler = state_handler;
}

void bifrost_http_route(BifrostHttpApp app, const char *method, const char *pattern, BifrostHttpHandler handler)
{
    add_route(app, method, pattern, handler, NULL);
}

void bifrost_http_route_with(BifrostHttpApp app, const char *method, const char *pattern, BifrostHttpStateHandler handler)
{
    add_route(app, method, pattern, NULL, handler);
}

void bifrost_http_get_with(BifrostHttpApp app, const char *pattern, BifrostHttpStateHandler handler)
{
    add_route(app, "GET", pattern, NULL, handler);
}

void bifrost_http_post_with(BifrostHttpApp app, const char *pattern, BifrostHttpStateHandler handler)
{
    add_route(app, "POST", pattern, NULL, handler);
}

void bifrost_http_put_with(BifrostHttpApp app, const char *pattern, BifrostHttpStateHandler handler)
{
    add_route(app, "PUT", pattern, NULL, handler);
}

void bifrost_http_patch_with(BifrostHttpApp app, const char *pattern, BifrostHttpStateHandler handler)
{
    add_route(app, "PATCH", pattern, NULL, handler);
}

void bifrost_http_delete_with(BifrostHttpApp app, const char *pattern, BifrostHttpStateHandler handler)
{
    add_route(app, "DELETE", pattern, NULL, handler);
}

void bifrost_http_get(BifrostHttpApp app, const char *pattern, BifrostHttpHandler handler)
{
    bifrost_http_route(app, "GET", pattern, handler);
}

void bifrost_http_post(BifrostHttpApp app, const char *pattern, BifrostHttpHandler handler)
{
    bifrost_http_route(app, "POST", pattern, handler);
}

void bifrost_http_put(BifrostHttpApp app, const char *pattern, BifrostHttpHandler handler)
{
    bifrost_http_route(app, "PUT", pattern, handler);
}

void bifrost_http_patch(BifrostHttpApp app, const char *pattern, BifrostHttpHandler handler)
{
    bifrost_http_route(app, "PATCH", pattern, handler);
}

void bifrost_http_delete(BifrostHttpApp app, const char *pattern, BifrostHttpHandler handler)
{
    bifrost_http_route(app, "DELETE", pattern, handler);
}

int32_t bifrost_http_port(int32_t fallback)
{
    const char *text = getenv("PORT");
    if (text == NULL || *text == '\0')
        return fallback;
    char *end;
    long port = strtol(text, &end, 10);
    return *end == '\0' && port > 0 && port < 65536 ? (int32_t)port : fallback;
}

/* -- serving -------------------------------------------------------------------- */

static void on_accept(h2o_socket_t *listener, const char *err)
{
    h2o_socket_t *sock;
    if (err != NULL || (sock = h2o_evloop_socket_accept(listener)) == NULL)
        return;
    h2o_accept(&accept_ctx, sock);
}

static int open_socket(int type, int32_t port)
{
    struct sockaddr_in6 addr = {.sin6_family = AF_INET6, .sin6_port = htons((uint16_t)port), .sin6_addr = in6addr_any};
    int fd, on = 1, off = 0;
    if ((fd = socket(AF_INET6, type, 0)) == -1)
        return -1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off)); /* IPv4 too */
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 || (type == SOCK_STREAM && listen(fd, SOMAXCONN) != 0)) {
        close(fd);
        return -1;
    }
    return fd;
}

static int setup(BifrostHttpApp app, int32_t port, const char *function)
{
    App *found = app_of(app, function);
    if (found == NULL)
        return 0;
    signal(SIGPIPE, SIG_IGN);
    h2o_config_init(&config);
    h2o_hostconf_t *host = h2o_config_register_host(&config, h2o_iovec_init(H2O_STRLIT("default")), 65535);
    h2o_pathconf_t *path = h2o_config_register_path(host, "/", 0);
    Handler *handler = (Handler *)h2o_create_handler(path, sizeof(*handler));
    handler->super.on_req = on_request;
    handler->app = found;
    h2o_context_init(&context, h2o_evloop_create(), &config);
    accept_ctx.ctx = &context;
    accept_ctx.hosts = config.hosts;

    int fd = open_socket(SOCK_STREAM, port);
    if (fd == -1) {
        fprintf(stderr, "http.%s: cannot listen on port %d: %s\n", function, port, strerror(errno));
        return 0;
    }
    h2o_socket_t *sock = h2o_evloop_socket_create(context.loop, fd, H2O_SOCKET_FLAG_DONT_READ);
    h2o_socket_read_start(sock, on_accept);
    return 1;
}

/* How long to wait for events: not at all while coroutines are ready, and no longer
   than the runtime's next timer (std:time's sleep). */
static int32_t wait_limit(void)
{
    if (bifrost_async_pending())
        return 0;
    int64_t timer = bifrost_async_next_timer();
    return timer < 0 || timer > INT32_MAX ? INT32_MAX : (int32_t)timer;
}

/* Wait for events and handle them: how code blocked on a result lets the server run. */
static void poll_once(void *unused)
{
    (void)unused;
    h2o_evloop_run(context.loop, wait_limit());
}

static int32_t run(void)
{
    bifrost_async_set_poller(poll_once, NULL);
    for (;;) {
        bifrost_async_run_ready(); /* handlers resumed by the last events, and timers due */
        if (h2o_evloop_run(context.loop, wait_limit()) != 0)
            break;
    }
    return 1;
}

int32_t bifrost_http_serve(BifrostHttpApp app, int32_t port)
{
    if (!setup(app, port, "serve"))
        return 1;
    fprintf(stderr, "http: serving http://localhost:%d (HTTP/1.1, HTTP/2)\n", port);
    return run();
}

/* HTTP/3: QUIC on UDP `port`, with the certificate of the TLS listener. */

static ptls_openssl_sign_certificate_t quic_signer;
static ptls_context_t quic_tls;
static quicly_context_t quic;
static quicly_cid_plaintext_t next_cid;
static h2o_http3_server_ctx_t http3;

/* HTTP/3 is negotiated with ALPN ("h3"), as HTTP/2 is over TLS. */
static int on_client_hello(ptls_on_client_hello_t *self, ptls_t *tls, ptls_on_client_hello_parameters_t *params)
{
    (void)self;
    for (size_t i = 0; i != sizeof(h2o_http3_alpn) / sizeof(h2o_http3_alpn[0]); ++i)
        for (size_t j = 0; j != params->negotiated_protocols.count; ++j)
            if (h2o_memis(h2o_http3_alpn[i].base, h2o_http3_alpn[i].len, params->negotiated_protocols.list[j].base,
                          params->negotiated_protocols.list[j].len))
                return ptls_set_negotiated_protocol(tls, (const char *)h2o_http3_alpn[i].base, h2o_http3_alpn[i].len);
    return PTLS_ALERT_NO_APPLICATION_PROTOCOL;
}

static ptls_on_client_hello_t client_hello = {on_client_hello};

static h2o_quic_conn_t *on_quic_accept(h2o_quic_ctx_t *ctx, quicly_address_t *destaddr, quicly_address_t *srcaddr,
                                       quicly_decoded_packet_t *packet)
{
    h2o_http3_conn_t *conn = h2o_http3_server_accept((h2o_http3_server_ctx_t *)ctx, destaddr, srcaddr, packet, NULL,
                                                     &H2O_HTTP3_CONN_CALLBACKS);
    if (conn == NULL || conn == &h2o_http3_accept_conn_closed)
        return NULL;
    return &conn->super; /* or h2o_quic_accept_conn_decryption_failed, which h2o handles */
}

static int setup_http3(int32_t port, const char *cert, const char *key)
{
    FILE *fp = fopen(key, "r");
    EVP_PKEY *private_key = fp ? PEM_read_PrivateKey(fp, NULL, NULL, NULL) : NULL;
    if (fp)
        fclose(fp);
    if (private_key == NULL || ptls_openssl_init_sign_certificate(&quic_signer, private_key) != 0)
        return 0;
    EVP_PKEY_free(private_key);
    quic_tls = (ptls_context_t){
        .random_bytes = ptls_openssl_random_bytes,
        .get_time = &ptls_get_time,
        .key_exchanges = ptls_openssl_key_exchanges,
        .cipher_suites = ptls_openssl_cipher_suites,
        .sign_certificate = &quic_signer.super,
        .on_client_hello = &client_hello,
    };
    if (ptls_load_certificates(&quic_tls, cert) != 0)
        return 0;
    quicly_amend_ptls_context(&quic_tls);
    quic = quicly_spec_context;
    quic.tls = &quic_tls;
    uint8_t secret[PTLS_MAX_DIGEST_SIZE];
    ptls_openssl_random_bytes(secret, sizeof(secret));
    quic.cid_encryptor = quicly_new_default_cid_encryptor(&ptls_openssl_quiclb, &ptls_openssl_aes128ecb, &ptls_openssl_sha256,
                                                          ptls_iovec_init(secret, sizeof(secret)));
    h2o_http3_server_amend_quicly_context(&config, &quic);

    int fd = open_socket(SOCK_DGRAM, port);
    if (fd == -1)
        return 0;
    h2o_socket_t *sock = h2o_evloop_socket_create(context.loop, fd, H2O_SOCKET_FLAG_DONT_READ);
    h2o_http3_server_init_context(&context, &http3.super, context.loop, sock, NULL, &quic, &next_cid, on_quic_accept, NULL, 0);
    http3.accept_ctx = &accept_ctx;
    http3.qpack = (h2o_http3_qpack_context_t){.encoder_table_capacity = 4096};
    return 1;
}

int32_t bifrost_http_serve_tls(BifrostHttpApp app, int32_t port, const char *cert, const char *key)
{
    if (!setup(app, port, "serve_tls"))
        return 1;
    accept_ctx.ssl_ctx = SSL_CTX_new(TLS_server_method());
    SSL_CTX_set_min_proto_version(accept_ctx.ssl_ctx, TLS1_2_VERSION);
    if (SSL_CTX_use_certificate_chain_file(accept_ctx.ssl_ctx, cert) != 1 ||
        SSL_CTX_use_PrivateKey_file(accept_ctx.ssl_ctx, key, SSL_FILETYPE_PEM) != 1) {
        fprintf(stderr, "http.serve_tls: cannot load the certificate %s and key %s\n", cert, key);
        return 1;
    }
    h2o_ssl_register_alpn_protocols(accept_ctx.ssl_ctx, h2o_http2_alpn_protocols);
    if (setup_http3(port, cert, key)) {
        char text[32];
        snprintf(text, sizeof(text), "h3=\":%d\"", port);
        alt_svc = strdup(text);
        fprintf(stderr, "http: serving https://localhost:%d (HTTP/1.1, HTTP/2, HTTP/3)\n", port);
    } else {
        fprintf(stderr, "http: serving https://localhost:%d (HTTP/1.1, HTTP/2; HTTP/3 could not start)\n", port);
    }
    return run();
}

/* -- reading the request ---------------------------------------------------------- */

const char *bifrost_http_method(BifrostHttpContext ctx)
{
    h2o_req_t *req = request_of(ctx, "method");
    return req ? pool_string(req->method.base, req->method.len) : "";
}

const char *bifrost_http_path(BifrostHttpContext ctx)
{
    h2o_req_t *req = request_of(ctx, "path");
    return req ? pool_string(req->path_normalized.base, req->path_normalized.len) : "";
}

const char *bifrost_http_protocol(BifrostHttpContext ctx)
{
    h2o_req_t *req = request_of(ctx, "protocol");
    return req == NULL ? "" : req->version >= 0x300 ? "HTTP/3" : req->version >= 0x200 ? "HTTP/2" : "HTTP/1.1";
}

const char *bifrost_http_body(BifrostHttpContext ctx)
{
    h2o_req_t *req = request_of(ctx, "body");
    return req && req->entity.base ? pool_string(req->entity.base, req->entity.len) : "";
}

const char *bifrost_http_param(BifrostHttpContext ctx, const char *name)
{
    if (request_of(ctx, "param") == NULL)
        return "";
    for (size_t i = 0; i != current.param_count; ++i)
        if (strcmp(current.param_names[i], name) == 0)
            return current.param_values[i];
    return "";
}

static int64_t to_int(const char *text, int64_t fallback)
{
    char *end;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    return *text == '\0' || *end != '\0' || errno != 0 ? fallback : (int64_t)value;
}

int64_t bifrost_http_param_int(BifrostHttpContext ctx, const char *name, int64_t fallback)
{
    return to_int(bifrost_http_param(ctx, name), fallback);
}

static int hex(char c)
{
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

const char *bifrost_http_query(BifrostHttpContext ctx, const char *name)
{
    h2o_req_t *req = request_of(ctx, "query");
    if (req == NULL || req->query_at == SIZE_MAX)
        return "";
    const char *at = req->path.base + req->query_at + 1, *end = req->path.base + req->path.len;
    size_t name_len = strlen(name);
    while (at < end) {
        const char *pair_end = memchr(at, '&', (size_t)(end - at));
        if (pair_end == NULL)
            pair_end = end;
        const char *equals = memchr(at, '=', (size_t)(pair_end - at));
        const char *key_end = equals ? equals : pair_end;
        if ((size_t)(key_end - at) == name_len && memcmp(at, name, name_len) == 0) {
            const char *value = equals ? equals + 1 : pair_end;
            char *decoded = h2o_mem_alloc_pool(&req->pool, char, (size_t)(pair_end - value) + 1), *out = decoded;
            for (const char *p = value; p < pair_end; ++p) {
                if (*p == '+') {
                    *out++ = ' ';
                } else if (*p == '%' && pair_end - p > 2 && hex(p[1]) >= 0 && hex(p[2]) >= 0) {
                    *out++ = (char)(hex(p[1]) * 16 + hex(p[2]));
                    p += 2;
                } else {
                    *out++ = *p;
                }
            }
            *out = '\0';
            return decoded;
        }
        at = pair_end + 1;
    }
    return "";
}

int64_t bifrost_http_query_int(BifrostHttpContext ctx, const char *name, int64_t fallback)
{
    return to_int(bifrost_http_query(ctx, name), fallback);
}

const char *bifrost_http_header(BifrostHttpContext ctx, const char *name)
{
    h2o_req_t *req = request_of(ctx, "header");
    if (req == NULL)
        return "";
    size_t name_len = strlen(name);
    for (size_t i = 0; i != req->headers.size; ++i) {
        h2o_header_t *header = req->headers.entries + i;
        if (header->name->len == name_len && strncasecmp(header->name->base, name, name_len) == 0)
            return pool_string(header->value.base, header->value.len);
    }
    return "";
}

/* -- responding ----------------------------------------------------------------------- */

void bifrost_http_set_header(BifrostHttpContext ctx, const char *name, const char *value)
{
    if (request_of(ctx, "set_header") != NULL)
        add_header(name, value);
}

void bifrost_http_text(BifrostHttpContext ctx, int32_t status, const char *text)
{
    h2o_req_t *req = request_of(ctx, "text");
    if (req && respond(req, status, "text/plain; charset=utf-8", "text"))
        body_append(text, strlen(text));
}

void bifrost_http_html(BifrostHttpContext ctx, int32_t status, const char *html)
{
    h2o_req_t *req = request_of(ctx, "html");
    if (req && respond(req, status, "text/html; charset=utf-8", "html"))
        body_append(html, strlen(html));
}

void bifrost_http_json(BifrostHttpContext ctx, int32_t status, const char *json)
{
    h2o_req_t *req = request_of(ctx, "json");
    if (req && respond(req, status, "application/json", "json"))
        body_append(json, strlen(json));
}

void bifrost_http_status(BifrostHttpContext ctx, int32_t status)
{
    h2o_req_t *req = request_of(ctx, "status");
    if (req)
        respond(req, status, NULL, "status");
}

static void json_string(const char *text)
{
    body_text("\"");
    for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
        char escaped[8];
        switch (*p) {
        case '"':
            body_text("\\\"");
            break;
        case '\\':
            body_text("\\\\");
            break;
        case '\n':
            body_text("\\n");
            break;
        case '\r':
            body_text("\\r");
            break;
        case '\t':
            body_text("\\t");
            break;
        default:
            if (*p < 0x20) {
                snprintf(escaped, sizeof(escaped), "\\u%04x", *p);
                body_append(escaped, 6);
            } else {
                body_append((const char *)p, 1);
            }
        }
    }
    body_text("\"");
}

void bifrost_http_error(BifrostHttpContext ctx, int32_t status, const char *message)
{
    h2o_req_t *req = request_of(ctx, "error");
    if (req == NULL || !respond(req, status, "application/json", "error"))
        return;
    body_text("{\"error\": ");
    json_string(message);
    body_text("}");
}

void bifrost_http_redirect(BifrostHttpContext ctx, int32_t status, const char *location)
{
    h2o_req_t *req = request_of(ctx, "redirect");
    if (req && respond(req, status, NULL, "redirect"))
        add_header("Location", location);
}

/* -- the json_* builder: {"id": 7, "roles": ["reader"]} --------------------------------- */

/* Before a value: the separator, and the key inside an object. */
static int json_entry(BifrostHttpContext ctx, const char *key, const char *function)
{
    if (request_of(ctx, function) == NULL)
        return 0;
    if (!current.responded || current.json_depth == 0) {
        fprintf(stderr, "http.%s: start the JSON with http.json_begin\n", function);
        return 0;
    }
    int level = current.json_depth - 1;
    if (current.json_has_entry[level])
        body_text(", ");
    current.json_has_entry[level] = 1;
    if (!current.json_is_array[level]) {
        json_string(key);
        body_text(": ");
    }
    return 1;
}

static void json_open(int is_array)
{
    if (current.json_depth == MAX_JSON_DEPTH)
        h2o_fatal("http: JSON nested more than %d deep", MAX_JSON_DEPTH);
    body_append(is_array ? "[" : "{", 1);
    current.json_has_entry[current.json_depth] = 0;
    current.json_is_array[current.json_depth++] = is_array;
}

void bifrost_http_json_begin(BifrostHttpContext ctx, int32_t status)
{
    h2o_req_t *req = request_of(ctx, "json_begin");
    if (req && respond(req, status, "application/json", "json_begin"))
        json_open(0);
}

void bifrost_http_json_str(BifrostHttpContext ctx, const char *key, const char *value)
{
    if (json_entry(ctx, key, "json_str"))
        json_string(value);
}

void bifrost_http_json_int(BifrostHttpContext ctx, const char *key, int64_t value)
{
    char text[32];
    if (json_entry(ctx, key, "json_int"))
        body_append(text, (size_t)snprintf(text, sizeof(text), "%lld", (long long)value));
}

void bifrost_http_json_float(BifrostHttpContext ctx, const char *key, double value)
{
    char text[64];
    if (json_entry(ctx, key, "json_float"))
        body_append(text, (size_t)snprintf(text, sizeof(text), "%.17g", value));
}

void bifrost_http_json_bool(BifrostHttpContext ctx, const char *key, bool value)
{
    if (json_entry(ctx, key, "json_bool"))
        body_append(value ? "true" : "false", value ? 4 : 5);
}

void bifrost_http_json_null(BifrostHttpContext ctx, const char *key)
{
    if (json_entry(ctx, key, "json_null"))
        body_text("null");
}

void bifrost_http_json_object(BifrostHttpContext ctx, const char *key)
{
    if (json_entry(ctx, key, "json_object"))
        json_open(0);
}

void bifrost_http_json_array(BifrostHttpContext ctx, const char *key)
{
    if (json_entry(ctx, key, "json_array"))
        json_open(1);
}

void bifrost_http_json_close(BifrostHttpContext ctx)
{
    if (request_of(ctx, "json_close") == NULL)
        return;
    if (current.json_depth <= 1) {
        fprintf(stderr, "http.json_close: nothing to close; end the JSON with http.json_end\n");
        return;
    }
    body_append(current.json_is_array[--current.json_depth] ? "]" : "}", 1);
}

void bifrost_http_json_end(BifrostHttpContext ctx)
{
    if (request_of(ctx, "json_end") == NULL)
        return;
    while (current.json_depth > 0)
        body_append(current.json_is_array[--current.json_depth] ? "]" : "}", 1);
}

/* -- running from the environment ---------------------------------------------------- */

int32_t bifrost_http_run(BifrostHttpApp app, int32_t port)
{
    const char *cert = getenv("TLS_CERT"), *key = getenv("TLS_KEY");
    port = bifrost_http_port(port);
    if (cert != NULL && *cert != '\0' && key != NULL && *key != '\0')
        return bifrost_http_serve_tls(app, port, cert, key);
    return bifrost_http_serve(app, port);
}

/* Like http.run, lending `state` to the handlers added with the *_with functions while it serves. */
int32_t bifrost_http_run_with(BifrostHttpApp app, int32_t port, void *state)
{
    app_state = state;
    int32_t status = bifrost_http_run(app, port);
    app_state = NULL;
    return status;
}
