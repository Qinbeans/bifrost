/*
 * axum for Bifrost: routers of handlers, middleware, and a server, on libh2o.
 *
 * This is the core the axum package's Bifrost code (src/axum/) builds on, and what
 * `import("axum")` brings: routers, method routers and layers, requests and responses
 * as handles, and the server. Its handlers are any Bifrost functions: the compiler
 * writes, for each one passed where a `handler(axum_Request) => axum_Response` goes,
 * an adapter that extracts its parameters from the request (axum's extractors), calls
 * it, and puts what it returns into a response (see SourceUnit._handler_argument).
 * C calls the adapter with the request, the state of the router the route was added
 * to, and where to put the response; it returns a token that completes when it is done.
 *
 *     let app = axum.router()
 *         |> axum.route("/", axum.get(root))
 *         |> axum.route("/users/{id}", axum.get(show) |> axum.also(axum.delete(remove)))
 *         |> axum.layer(axum.from_fn(log))
 *         |> axum.with_state(state)
 *     await axum.serve(axum.bind("0.0.0.0:3000"), app)
 *
 * Routing follows axum: `{name}` (or 0.7's `:name`) captures a segment and `{*name}` (or
 * `*name`) the rest of the path; a static segment wins over a capture, and a capture over a
 * wildcard. Bifrost strings interpolate braces, so `{name}` is written `"{{name}}"` there,
 * and `"/users/:id"` reads better. A path
 * that matches with another method is a 405 (with Allow); none is the fallback (404).
 * `layer` wraps the routes (and fallback) added so far, outermost last; `route_layer`
 * only the routes. `nest` mounts a router under a prefix (its handlers see the URI
 * without it, OriginalUri has it); `merge` takes another's routes.
 *
 * Everything runs on the event loop's thread, one request's step at a time: a handler
 * that waits (a timer, the network) gives the loop back, and its request goes on when
 * its token completes. Requests are copied out of h2o when they arrive (the client may
 * go away while a handler waits), and live until their response is sent.
 */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
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

#include "axum.h"

/* Features (config.yaml): each one turned on links an object defining its flag as 1 (c/features/), over
   the weak 0 of c/defaults.c (in another file, so that the compiler cannot fold the 0 in here). */
extern const int axum_feature_http1, axum_feature_http2, axum_feature_tracing, axum_feature_matched_path,
    axum_feature_original_uri;

/* Bifrost's async runtime (std/async_runtime.c), linked into the program. */
void *mlirAsyncRuntimeCreateToken(void);
void mlirAsyncRuntimeEmplaceToken(void *token);
void mlirAsyncRuntimeSetTokenError(void *token);
void mlirAsyncRuntimeDropRef(void *object, int64_t count);
bool mlirAsyncRuntimeIsTokenError(void *token);
void mlirAsyncRuntimeAwaitTokenAndExecute(void *token, void *handle, void (*resume)(void *));
void bifrost_async_run_ready(void);
bool bifrost_async_pending(void);
int64_t bifrost_async_next_timer(void);
void bifrost_async_set_poller(void (*poll)(void *), void *context);

/* -- memory and tokens --------------------------------------------------------- */

void *axum_alloc(size_t size)
{
    void *memory = malloc(size > 0 ? size : 1);
    if (memory == NULL) {
        fputs("axum: out of memory\n", stderr);
        abort();
    }
    return memory;
}

void *axum_zalloc(size_t size)
{
    void *memory = axum_alloc(size);
    memset(memory, 0, size);
    return memory;
}

char *axum_strndup(const char *text, size_t len)
{
    char *copy = axum_alloc(len + 1);
    memcpy(copy, text, len);
    copy[len] = '\0';
    return copy;
}

char *axum_strdup(const char *text)
{
    return axum_strndup(text != NULL ? text : "", text != NULL ? strlen(text) : 0);
}

static void fatal(const char *function, const char *message, const char *detail)
{
    fprintf(stderr, "axum.%s: %s%s\n", function, message, detail != NULL ? detail : "");
    exit(1);
}

/* A token that is already complete: what a step that did not have to wait returns. */
void *axum_done_token(void)
{
    void *token = mlirAsyncRuntimeCreateToken();
    mlirAsyncRuntimeEmplaceToken(token);
    return token;
}

typedef void (*Then)(void *context, bool failed);

typedef struct {
    void *token;
    Then then;
    void *context;
} Waiting;

static void on_token(void *handle)
{
    Waiting *waiting = handle;
    bool failed = mlirAsyncRuntimeIsTokenError(waiting->token);
    mlirAsyncRuntimeDropRef(waiting->token, 1);
    waiting->then(waiting->context, failed);
    free(waiting);
}

/* Run then(context, failed) once `token` completes (now, if it has); its reference is let go of. */
void axum_after(void *token, Then then, void *context)
{
    Waiting *waiting = axum_alloc(sizeof(Waiting));
    *waiting = (Waiting){token, then, context};
    mlirAsyncRuntimeAwaitTokenAndExecute(token, waiting, on_token);
}

/* -- headers ---------------------------------------------------------------------- */

static void headers_add(Headers *headers, const char *name, size_t name_len, const char *value, size_t value_len)
{
    if (headers->count == headers->capacity) {
        headers->capacity = headers->capacity ? headers->capacity * 2 : 8;
        headers->items = realloc(headers->items, headers->capacity * sizeof(Header));
        if (headers->items == NULL)
            fatal("headers", "out of memory", NULL);
    }
    Header *header = headers->items + headers->count++;
    header->name = axum_strndup(name, name_len);
    for (char *c = header->name; *c; c++)
        *c = (char)tolower((unsigned char)*c);
    header->value = axum_strndup(value, value_len);
}

static void headers_remove(Headers *headers, const char *name)
{
    size_t kept = 0;
    for (size_t i = 0; i < headers->count; i++) {
        if (strcasecmp(headers->items[i].name, name) == 0) {
            free(headers->items[i].name);
            free(headers->items[i].value);
        } else {
            headers->items[kept++] = headers->items[i];
        }
    }
    headers->count = kept;
}

static const char *headers_get(const Headers *headers, const char *name)
{
    for (size_t i = 0; i < headers->count; i++)
        if (strcasecmp(headers->items[i].name, name) == 0)
            return headers->items[i].value;
    return NULL;
}

static void headers_free(Headers *headers)
{
    for (size_t i = 0; i < headers->count; i++) {
        free(headers->items[i].name);
        free(headers->items[i].value);
    }
    free(headers->items);
    *headers = (Headers){0};
}

static void headers_copy(Headers *to, const Headers *from)
{
    for (size_t i = 0; i < from->count; i++)
        headers_add(to, from->items[i].name, strlen(from->items[i].name), from->items[i].value, strlen(from->items[i].value));
}

/* -- responses -------------------------------------------------------------------- */

static Response *response_of(AxumResponse response, const char *function)
{
    if (response.raw == NULL)
        fatal(function, "this response is empty (a zero value)", NULL);
    return response.raw;
}

AxumResponse axum_response(int32_t status)
{
    Response *response = axum_zalloc(sizeof(Response));
    response->status = status;
    return (AxumResponse){response};
}

void axum_response_free(AxumResponse response)
{
    Response *r = response.raw;
    if (r == NULL)
        return;
    headers_free(&r->headers);
    free(r->body);
    free(r);
}

AxumResponse axum_response_clone(AxumResponse response)
{
    Response *from = response.raw;
    if (from == NULL)
        return (AxumResponse){NULL};
    Response *copy = axum_zalloc(sizeof(Response));
    copy->status = from->status;
    headers_copy(&copy->headers, &from->headers);
    copy->body = from->body != NULL ? axum_strndup(from->body, from->len) : NULL;
    copy->len = from->len;
    return (AxumResponse){copy};
}

void axum_set_body_bytes(AxumResponse response, const char *data, int64_t len)
{
    Response *r = response_of(response, "set_body");
    free(r->body);
    r->body = axum_strndup(data != NULL ? data : "", data != NULL && len > 0 ? (size_t)len : 0);
    r->len = data != NULL && len > 0 ? (size_t)len : 0;
}

void axum_set_body(AxumResponse response, const char *text)
{
    axum_set_body_bytes(response, text, text != NULL ? (int64_t)strlen(text) : 0);
}

void axum_set_header(AxumResponse response, const char *name, const char *value)
{
    Response *r = response_of(response, "set_header");
    headers_remove(&r->headers, name);
    headers_add(&r->headers, name, strlen(name), value, strlen(value));
}

void axum_append_header(AxumResponse response, const char *name, const char *value)
{
    Response *r = response_of(response, "append_header");
    headers_add(&r->headers, name, strlen(name), value, strlen(value));
}

void axum_remove_header(AxumResponse response, const char *name)
{
    headers_remove(&response_of(response, "remove_header")->headers, name);
}

const char *axum_response_header(AxumResponse response, const char *name)
{
    const char *value = headers_get(&response_of(response, "response_header")->headers, name);
    return value != NULL ? value : "";
}

void axum_set_status(AxumResponse response, int32_t status)
{
    response_of(response, "set_status")->status = status;
}

int32_t axum_status(AxumResponse response)
{
    return response.raw != NULL ? ((Response *)response.raw)->status : 200;
}

const char *axum_response_body(AxumResponse response)
{
    Response *r = response_of(response, "response_body");
    return r->body != NULL ? r->body : "";
}

static AxumResponse typed(int32_t status, const char *content_type, const char *body)
{
    AxumResponse response = axum_response(status);
    if (content_type != NULL)
        axum_set_header(response, "content-type", content_type);
    axum_set_body(response, body);
    return response;
}

AxumResponse axum_text(int32_t status, const char *text) { return typed(status, "text/plain; charset=utf-8", text); }
AxumResponse axum_html(int32_t status, const char *html) { return typed(status, "text/html; charset=utf-8", html); }
AxumResponse axum_json(int32_t status, const char *json) { return typed(status, "application/json", json); }

AxumResponse axum_redirect(int32_t status, const char *location)
{
    AxumResponse response = axum_response(status);
    axum_set_header(response, "location", location);
    return response;
}

/* -- routers ---------------------------------------------------------------------- */

static Layer *layer_new(LayerKind kind, AxumHandler handler, void *state)
{
    Layer *layer = axum_zalloc(sizeof(Layer));
    layer->kind = kind;
    layer->handler = handler;
    layer->state = state;
    return layer;
}

static void layers_push(Layers *layers, Layer *layer)
{
    if (layers->count == layers->capacity) {
        layers->capacity = layers->capacity ? layers->capacity * 2 : 4;
        layers->items = realloc(layers->items, layers->capacity * sizeof(Layer *));
        if (layers->items == NULL)
            fatal("layer", "out of memory", NULL);
    }
    layers->items[layers->count++] = layer;
}

static Layers layers_copy(const Layers *from)
{
    Layers copy = {0};
    for (size_t i = 0; i < from->count; i++)
        layers_push(&copy, from->items[i]);
    return copy;
}

static Router *router_of(AxumRouter router, const char *function)
{
    if (router.raw == NULL)
        fatal(function, "not a router from axum.router()", NULL);
    return router.raw;
}

static MethodRouter *methods_of(AxumMethodRouter methods, const char *function)
{
    if (methods.raw == NULL)
        fatal(function, "not a method router from axum.get(...), axum.on(...), ...", NULL);
    return methods.raw;
}

AxumRouter axum_router(void)
{
    return (AxumRouter){axum_zalloc(sizeof(Router))};
}

static const char *METHODS[] = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE", "CONNECT"};
#define METHOD_COUNT (sizeof(METHODS) / sizeof(METHODS[0]))

static int method_index(const char *method, size_t len)
{
    for (size_t i = 0; i < METHOD_COUNT; i++)
        if (strlen(METHODS[i]) == len && strncasecmp(METHODS[i], method, len) == 0)
            return (int)i;
    return -1;
}

static AxumMethodRouter methods_new(void)
{
    return (AxumMethodRouter){axum_zalloc(sizeof(MethodRouter))};
}

static void methods_set(MethodRouter *methods, int index, AxumHandler handler, const char *function)
{
    if (methods->handlers[index] != NULL)
        fatal(function, "overlapping method route: a handler for ", METHODS[index]);
    methods->handlers[index] = handler;
}

AxumMethodRouter axum_on(const char *filter, AxumHandler handler)
{
    AxumMethodRouter methods = methods_new();
    const char *at = filter;
    while (*at != '\0') {
        while (*at == ',' || *at == ' ' || *at == '|')
            at++;
        size_t len = strcspn(at, ", |");
        if (len == 0)
            break;
        int index = method_index(at, len);
        if (index < 0)
            fatal("on", "not a method: ", at);
        methods_set(methods.raw, index, handler, "on");
        at += len;
    }
    return methods;
}

#define METHOD_FUNCTION(name, method)                                                                                  \
    AxumMethodRouter axum_##name(AxumHandler handler) { return axum_on(method, handler); }
METHOD_FUNCTION(get, "GET")
METHOD_FUNCTION(post, "POST")
METHOD_FUNCTION(put, "PUT")
METHOD_FUNCTION(delete, "DELETE")
METHOD_FUNCTION(patch, "PATCH")
METHOD_FUNCTION(head, "HEAD")
METHOD_FUNCTION(options, "OPTIONS")
METHOD_FUNCTION(trace, "TRACE")
METHOD_FUNCTION(connect, "CONNECT")

AxumMethodRouter axum_any(AxumHandler handler)
{
    AxumMethodRouter methods = methods_new();
    ((MethodRouter *)methods.raw)->any = handler;
    return methods;
}

/* `get(a) |> also(post(b))`: one method router with both (MethodRouter::merge). */
AxumMethodRouter axum_and(AxumMethodRouter methods, AxumMethodRouter other)
{
    MethodRouter *into = methods_of(methods, "and"), *from = methods_of(other, "and");
    for (size_t i = 0; i < METHOD_COUNT; i++) {
        if (from->handlers[i] == NULL)
            continue;
        methods_set(into, (int)i, from->handlers[i], "and");
        into->layers[i] = layers_copy(&from->layers[i]);
    }
    if (from->any != NULL) {
        if (into->any != NULL)
            fatal("and", "both method routers handle any method", NULL);
        into->any = from->any;
        into->any_layers = layers_copy(&from->any_layers);
    }
    if (from->fallback != NULL)
        into->fallback = from->fallback;
    return methods;
}

/* What a method router answers a method it has no handler for, instead of a 405. */
AxumMethodRouter axum_method_fallback(AxumMethodRouter methods, AxumHandler handler)
{
    methods_of(methods, "method_fallback")->fallback = handler;
    return methods;
}

/* MethodRouter::layer: wraps its handlers so far (and its fallback). */
AxumMethodRouter axum_method_layer(AxumMethodRouter methods, AxumLayer layer)
{
    MethodRouter *m = methods_of(methods, "method_layer");
    for (size_t i = 0; i < METHOD_COUNT; i++)
        if (m->handlers[i] != NULL)
            layers_push(&m->layers[i], layer.raw);
    if (m->any != NULL)
        layers_push(&m->any_layers, layer.raw);
    layers_push(&m->fallback_layers, layer.raw);
    return methods;
}

/* MethodRouter::route_layer: wraps only the handlers (not a 405 or its fallback). */
AxumMethodRouter axum_method_route_layer(AxumMethodRouter methods, AxumLayer layer)
{
    MethodRouter *m = methods_of(methods, "method_route_layer");
    for (size_t i = 0; i < METHOD_COUNT; i++)
        if (m->handlers[i] != NULL)
            layers_push(&m->layers[i], layer.raw);
    if (m->any != NULL)
        layers_push(&m->any_layers, layer.raw);
    return methods;
}

/* Split a path pattern into segments: static, `{capture}` or `{*wildcard}`. */
static void parse_pattern(Route *route, const char *function)
{
    const char *at = route->pattern;
    if (*at != '/')
        fatal(function, "paths start with '/': ", route->pattern);
    route->segments = NULL;
    route->count = 0;
    size_t capacity = 0;
    while (*at == '/') {
        at++;
        size_t len = strcspn(at, "/");
        if (len == 0 && *at == '\0' && route->count > 0) {
            /* a trailing slash: an empty static segment, so `/a/` is not `/a` */
        } else if (len == 0 && *at == '\0') {
            break; /* `/` itself */
        }
        if (route->count == capacity) {
            capacity = capacity ? capacity * 2 : 8;
            route->segments = realloc(route->segments, capacity * sizeof(Segment));
        }
        Segment *segment = route->segments + route->count++;
        segment->kind = SEGMENT_STATIC;
        if (len >= 2 && at[0] == '{' && at[len - 1] == '}' && !(len >= 4 && at[1] == '{')) {
            bool wildcard = at[1] == '*';
            segment->kind = wildcard ? SEGMENT_WILDCARD : SEGMENT_CAPTURE;
            segment->text = axum_strndup(at + 1 + wildcard, len - 2 - wildcard);
            if (segment->text[0] == '\0')
                fatal(function, "a capture needs a name: ", route->pattern);
            if (wildcard && at[len] != '\0')
                fatal(function, "a wildcard ({*name}) must be the last segment: ", route->pattern);
        } else if (len >= 2 && (at[0] == ':' || at[0] == '*')) {
            /* axum 0.7's `:name` and `*name`: Bifrost strings interpolate `{...}`, so `{name}` is written `{{name}}` */
            bool wildcard = at[0] == '*';
            segment->kind = wildcard ? SEGMENT_WILDCARD : SEGMENT_CAPTURE;
            segment->text = axum_strndup(at + 1, len - 1);
            if (wildcard && at[len] != '\0')
                fatal(function, "a wildcard (*name) must be the last segment: ", route->pattern);
        } else {
            /* `{{` and `}}` are literal braces */
            char *text = axum_alloc(len + 1), *o = text;
            for (size_t i = 0; i < len; i++) {
                *o++ = at[i];
                if ((at[i] == '{' || at[i] == '}') && i + 1 < len && at[i + 1] == at[i])
                    i++;
            }
            *o = '\0';
            segment->text = text;
        }
        at += len;
    }
}

static Endpoint *endpoint_new(MethodRouter *methods, AxumHandler service, void **state)
{
    Endpoint *endpoint = axum_zalloc(sizeof(Endpoint));
    endpoint->methods = methods;
    endpoint->service = service;
    endpoint->state = state;
    return endpoint;
}

static Endpoint *endpoint_copy(const Endpoint *from, const char *prefix)
{
    Endpoint *copy = axum_zalloc(sizeof(Endpoint));
    *copy = *from;
    copy->layers = layers_copy(&from->layers);
    if (prefix != NULL) {
        size_t len = strlen(prefix) + (from->nest_prefix ? strlen(from->nest_prefix) : 0);
        copy->nest_prefix = axum_alloc(len + 1);
        snprintf(copy->nest_prefix, len + 1, "%s%s", prefix, from->nest_prefix ? from->nest_prefix : "");
    }
    return copy;
}

static Route *route_add(Router *router, const char *pattern, Endpoint *endpoint, bool prefix, const char *function)
{
    if (router->count == router->capacity) {
        router->capacity = router->capacity ? router->capacity * 2 : 16;
        router->routes = realloc(router->routes, router->capacity * sizeof(Route));
        if (router->routes == NULL)
            fatal(function, "out of memory", NULL);
    }
    Route *route = router->routes + router->count++;
    memset(route, 0, sizeof(*route));
    route->pattern = axum_strdup(pattern);
    route->endpoint = endpoint;
    route->prefix = prefix;
    parse_pattern(route, function);
    return route;
}

static Route *route_find(Router *router, const char *pattern, bool prefix)
{
    for (size_t i = 0; i < router->count; i++)
        if (router->routes[i].prefix == prefix && strcmp(router->routes[i].pattern, pattern) == 0)
            return router->routes + i;
    return NULL;
}

AxumRouter axum_route(AxumRouter router, const char *path, AxumMethodRouter methods)
{
    Router *r = router_of(router, "route");
    MethodRouter *m = methods_of(methods, "route");
    Route *existing = route_find(r, path, false);
    if (existing != NULL && existing->endpoint->methods != NULL && existing->endpoint->layers.count == 0) {
        axum_and((AxumMethodRouter){existing->endpoint->methods}, methods); /* the same path again: merged */
        return router;
    }
    if (existing != NULL)
        fatal("route", "overlapping route: ", path);
    route_add(r, path, endpoint_new(m, NULL, &r->state), false, "route");
    return router;
}

/* A handler for every method of `path` (Router::route_service). */
AxumRouter axum_route_service(AxumRouter router, const char *path, AxumHandler handler)
{
    Router *r = router_of(router, "route_service");
    if (route_find(r, path, false) != NULL)
        fatal("route_service", "overlapping route: ", path);
    route_add(r, path, endpoint_new(NULL, handler, &r->state), false, "route_service");
    return router;
}

static char *joined(const char *prefix, const char *path)
{
    size_t prefix_len = strlen(prefix);
    while (prefix_len > 0 && prefix[prefix_len - 1] == '/')
        prefix_len--;
    if (strcmp(path, "/") == 0)
        return prefix_len == 0 ? axum_strdup("/") : axum_strndup(prefix, prefix_len);
    size_t len = prefix_len + strlen(path);
    char *text = axum_alloc(len + 1);
    snprintf(text, len + 1, "%.*s%s", (int)prefix_len, prefix, path);
    return text;
}

static void check_nest_path(const char *path, const char *function)
{
    if (path[0] != '/')
        fatal(function, "paths start with '/': ", path);
    if (strcmp(path, "/") == 0 || path[0] == '\0')
        fatal(function, "nesting at the root is not supported; merge the router instead: ", path);
    if (strchr(path, '*') != NULL)
        fatal(function, "a nested path cannot have a wildcard: ", path);
}

/* Mount `other` under `path`: its routes, layers and fallback (Router::nest). */
AxumRouter axum_nest(AxumRouter router, const char *path, AxumRouter other)
{
    Router *r = router_of(router, "nest"), *inner = router_of(other, "nest");
    check_nest_path(path, "nest");
    char prefix[1024];
    snprintf(prefix, sizeof prefix, "%s", path);
    size_t len = strlen(prefix);
    while (len > 1 && prefix[len - 1] == '/')
        prefix[--len] = '\0';
    for (size_t i = 0; i < inner->count; i++) {
        Route *from = inner->routes + i;
        char *pattern = joined(prefix, from->pattern);
        if (route_find(r, pattern, from->prefix) != NULL)
            fatal("nest", "overlapping route: ", pattern);
        Endpoint *endpoint = endpoint_copy(from->endpoint, prefix);
        if (*endpoint->state == NULL && endpoint->state != &r->state && inner->state == NULL)
            endpoint->state = &r->state; /* a nested router without state takes this one's */
        route_add(r, pattern, endpoint, from->prefix, "nest");
        free(pattern);
    }
    if (inner->fallback != NULL) { /* what it does with what it does not route, under its prefix */
        Endpoint *endpoint = endpoint_copy(inner->fallback, prefix);
        char pattern[1100];
        snprintf(pattern, sizeof pattern, "%s", prefix);
        route_add(r, pattern, endpoint, true, "nest")->fallback = true;
    }
    return router;
}

/* Mount a handler under `path`: it serves the path and everything below it (Router::nest_service). */
AxumRouter axum_nest_service(AxumRouter router, const char *path, AxumHandler handler)
{
    Router *r = router_of(router, "nest_service");
    check_nest_path(path, "nest_service");
    Endpoint *endpoint = endpoint_new(NULL, handler, &r->state);
    char prefix[1024];
    snprintf(prefix, sizeof prefix, "%s", path);
    size_t len = strlen(prefix);
    while (len > 1 && prefix[len - 1] == '/')
        prefix[--len] = '\0';
    endpoint->nest_prefix = axum_strdup(prefix);
    if (route_find(r, prefix, true) != NULL)
        fatal("nest_service", "overlapping route: ", prefix);
    route_add(r, prefix, endpoint, true, "nest_service");
    return router;
}

/* Take `other`'s routes (and its fallback, if this has none) (Router::merge). */
AxumRouter axum_merge(AxumRouter router, AxumRouter other)
{
    Router *r = router_of(router, "merge"), *from = router_of(other, "merge");
    for (size_t i = 0; i < from->count; i++) {
        Route *route = from->routes + i;
        if (route_find(r, route->pattern, route->prefix) != NULL)
            fatal("merge", "overlapping route: ", route->pattern);
        Endpoint *endpoint = endpoint_copy(route->endpoint, NULL);
        if (from->state == NULL && endpoint->state == &from->state)
            endpoint->state = &r->state;
        route_add(r, route->pattern, endpoint, route->prefix, "merge")->fallback = route->fallback;
    }
    if (from->fallback != NULL) {
        if (r->fallback != NULL)
            fatal("merge", "both routers have a fallback", NULL);
        r->fallback = endpoint_copy(from->fallback, NULL);
    }
    if (from->method_not_allowed != NULL && r->method_not_allowed == NULL)
        r->method_not_allowed = from->method_not_allowed;
    return router;
}

/* Wrap the routes added so far, and the fallback, in a layer: outermost last (Router::layer). */
AxumRouter axum_layer(AxumRouter router, AxumLayer layer)
{
    Router *r = router_of(router, "layer");
    for (size_t i = 0; i < r->count; i++)
        layers_push(&r->routes[i].endpoint->layers, layer.raw);
    layers_push(&r->fallback_layers, layer.raw);
    return router;
}

/* Wrap the routes added so far, not the fallback, and only once one matches (Router::route_layer). */
AxumRouter axum_route_layer(AxumRouter router, AxumLayer layer)
{
    Router *r = router_of(router, "route_layer");
    if (r->count == 0)
        fatal("route_layer", "there are no routes yet to wrap; add them first", NULL);
    for (size_t i = 0; i < r->count; i++)
        if (!r->routes[i].fallback)
            layers_push(&r->routes[i].endpoint->layers, layer.raw);
    return router;
}

AxumRouter axum_fallback(AxumRouter router, AxumHandler handler)
{
    Router *r = router_of(router, "fallback");
    r->fallback = endpoint_new(NULL, handler, &r->state);
    return router;
}

/* Forget the fallback set so far: unmatched paths are 404s again (Router::reset_fallback). */
AxumRouter axum_reset_fallback(AxumRouter router)
{
    router_of(router, "reset_fallback")->fallback = NULL;
    return router;
}

/* What answers a path that matches with another method, instead of a 405 (Router::method_not_allowed_fallback). */
AxumRouter axum_method_not_allowed_fallback(AxumRouter router, AxumHandler handler)
{
    Router *r = router_of(router, "method_not_allowed_fallback");
    r->method_not_allowed = endpoint_new(NULL, handler, &r->state);
    return router;
}

/* The state handlers of this router's routes are given (a mem.Shared or mem.Atomic, lent). */
AxumRouter axum_with_state(AxumRouter router, void *state)
{
    router_of(router, "with_state")->state = state;
    return router;
}

bool axum_has_routes(AxumRouter router)
{
    return router_of(router, "has_routes")->count > 0;
}

/* -- layers (middleware) --------------------------------------------------------------- */

AxumLayer axum_from_fn(AxumHandler handler) { return (AxumLayer){layer_new(LAYER_FROM_FN, handler, NULL)}; }
AxumLayer axum_from_fn_with_state(void *state, AxumHandler handler)
{
    return (AxumLayer){layer_new(LAYER_FROM_FN, handler, state)};
}
AxumLayer axum_map_request(AxumHandler handler) { return (AxumLayer){layer_new(LAYER_MAP_REQUEST, handler, NULL)}; }
AxumLayer axum_map_request_with_state(void *state, AxumHandler handler)
{
    return (AxumLayer){layer_new(LAYER_MAP_REQUEST, handler, state)};
}
/* Middleware that runs extractors (its handler's parameters) and stops the request if one rejects it. */
AxumLayer axum_from_extractor(AxumHandler handler) { return axum_map_request(handler); }
AxumLayer axum_map_response(AxumHandler handler) { return (AxumLayer){layer_new(LAYER_MAP_RESPONSE, handler, NULL)}; }
AxumLayer axum_map_response_with_state(void *state, AxumHandler handler)
{
    return (AxumLayer){layer_new(LAYER_MAP_RESPONSE, handler, state)};
}

/* The largest body a request may have (bytes); larger ones get a 413 (DefaultBodyLimit::max). */
AxumLayer axum_body_limit(int64_t bytes)
{
    Layer *layer = layer_new(LAYER_BODY_LIMIT, NULL, NULL);
    layer->limit = bytes;
    return (AxumLayer){layer};
}

/* No limit on bodies (DefaultBodyLimit::disable). */
AxumLayer axum_disable_body_limit(void)
{
    return axum_body_limit(-1);
}

/* Add a header to every response that has none of that name (a common middleware, as a layer). */
AxumLayer axum_default_header(const char *name, const char *value)
{
    Layer *layer = layer_new(LAYER_HEADER, NULL, NULL);
    layer->name = axum_strdup(name);
    layer->value = axum_strdup(value);
    return (AxumLayer){layer};
}

/* A text value requests carry, for extractors to read (Extension). */
AxumLayer axum_extension(const char *key, const char *value)
{
    Layer *layer = layer_new(LAYER_EXTENSION, NULL, NULL);
    layer->name = axum_strdup(key);
    layer->value = axum_strdup(value);
    return (AxumLayer){layer};
}

/* -- requests -------------------------------------------------------------------------------- */

#define DEFAULT_BODY_LIMIT (2 * 1024 * 1024) /* axum's: 2 MB */

static Request *request_of(AxumRequest request, const char *function)
{
    if (request.raw == NULL)
        fatal(function, "not a request (a zero value)", NULL);
    return request.raw;
}

/* A string the request keeps until it is done. */
static const char *keep(Request *r, char *text)
{
    if (r->kept_count == r->kept_capacity) {
        r->kept_capacity = r->kept_capacity ? r->kept_capacity * 2 : 16;
        r->kept = realloc(r->kept, r->kept_capacity * sizeof(char *));
    }
    r->kept[r->kept_count++] = text;
    return text;
}

const char *axum_method(AxumRequest request) { return request_of(request, "method")->method; }
const char *axum_path(AxumRequest request) { return request_of(request, "path")->path; }
const char *axum_query(AxumRequest request) { return request_of(request, "query")->query; }
bool axum_has_query(AxumRequest request) { return request_of(request, "has_query")->has_query; }
const char *axum_version(AxumRequest request) { return request_of(request, "version")->version; }
const char *axum_body(AxumRequest request) { return request_of(request, "body")->body; }
int64_t axum_body_len(AxumRequest request) { return (int64_t)request_of(request, "body_len")->body_len; }
const char *axum_path_params(AxumRequest request) { return request_of(request, "path_params")->params; }
const char *axum_peer_addr(AxumRequest request) { return request_of(request, "peer_addr")->peer; }

const char *axum_uri(AxumRequest request)
{
    Request *r = request_of(request, "uri");
    if (!r->has_query)
        return r->path;
    size_t len = strlen(r->path) + strlen(r->query) + 2;
    char *uri = axum_alloc(len);
    snprintf(uri, len, "%s?%s", r->path, r->query);
    return keep(r, uri);
}

/* The URI as it came, before nesting stripped its prefix (OriginalUri). */
const char *axum_original_uri(AxumRequest request) { return request_of(request, "original_uri")->original_uri; }

/* The pattern of the route that matched (MatchedPath): "/users/{id}"; "" for a fallback. */
const char *axum_matched_path(AxumRequest request) { return request_of(request, "matched_path")->matched_path; }

/* The prefix nesting stripped (NestedPath): "/api"; "" when not nested. */
const char *axum_nested_path(AxumRequest request) { return request_of(request, "nested_path")->nested_path; }

const char *axum_header(AxumRequest request, const char *name)
{
    const char *value = headers_get(&request_of(request, "header")->headers, name);
    return value != NULL ? value : "";
}

bool axum_has_header(AxumRequest request, const char *name)
{
    return headers_get(&request_of(request, "has_header")->headers, name) != NULL;
}

int64_t axum_header_count(AxumRequest request) { return (int64_t)request_of(request, "header_count")->headers.count; }

const char *axum_header_name(AxumRequest request, int64_t index)
{
    Request *r = request_of(request, "header_name");
    return index >= 0 && (size_t)index < r->headers.count ? r->headers.items[index].name : "";
}

const char *axum_header_value(AxumRequest request, int64_t index)
{
    Request *r = request_of(request, "header_value");
    return index >= 0 && (size_t)index < r->headers.count ? r->headers.items[index].value : "";
}

void axum_set_request_header(AxumRequest request, const char *name, const char *value)
{
    Request *r = request_of(request, "set_request_header");
    headers_remove(&r->headers, name);
    headers_add(&r->headers, name, strlen(name), value, strlen(value));
}

void axum_remove_request_header(AxumRequest request, const char *name)
{
    headers_remove(&request_of(request, "remove_request_header")->headers, name);
}

void axum_set_method(AxumRequest request, const char *method)
{
    Request *r = request_of(request, "set_method");
    r->method = keep(r, axum_strdup(method));
}

void axum_set_body_text(AxumRequest request, const char *body)
{
    Request *r = request_of(request, "set_body_text");
    r->body = keep(r, axum_strdup(body));
    r->body_len = strlen(body);
}

/* A value an extension layer gave the request ("" when none did). */
const char *axum_extension_value(AxumRequest request, const char *key)
{
    const char *value = headers_get(&request_of(request, "extension_value")->extensions, key);
    return value != NULL ? value : "";
}

bool axum_has_extension(AxumRequest request, const char *key)
{
    return headers_get(&request_of(request, "has_extension")->extensions, key) != NULL;
}

void axum_insert_extension(AxumRequest request, const char *key, const char *value)
{
    Request *r = request_of(request, "insert_extension");
    headers_remove(&r->extensions, key);
    headers_add(&r->extensions, key, strlen(key), value, strlen(value));
}

/* -- matching ------------------------------------------------------------------------------- */

/* A growing string, for a request's path parameters as a form. */
typedef struct {
    char *text;
    size_t len, capacity;
} Builder;

static void build_reserve(Builder *b, size_t more)
{
    if (b->len + more + 1 <= b->capacity)
        return;
    size_t capacity = b->capacity ? b->capacity : 64;
    while (capacity < b->len + more + 1)
        capacity *= 2;
    b->text = realloc(b->text, capacity);
    if (b->text == NULL)
        fatal("serve", "out of memory", NULL);
    b->capacity = capacity;
}

static void build_char(Builder *b, char c)
{
    build_reserve(b, 1);
    b->text[b->len++] = c;
    b->text[b->len] = '\0';
}

/* Path text in a form value: `&`, `=`, `+` and `#` escaped (other `%`-escapes kept), so form.decode reads it back. */
static void build_form(Builder *b, const char *text, size_t text_len)
{
    build_reserve(b, text_len * 3);
    for (size_t i = 0; i < text_len; i++) {
        char c = text[i];
        if (c == '&' || c == '=' || c == '+' || c == '#')
            b->len += (size_t)snprintf(b->text + b->len, 4, "%%%02X", (unsigned char)c);
        else
            b->text[b->len++] = c;
    }
    b->text[b->len] = '\0';
}

typedef struct {
    const char *start[32];
    size_t len[32];
    size_t count;
    int score[32]; /* per segment: 3 static, 2 capture, 1 wildcard */
    size_t consumed; /* how much of the path a prefix route took */
} Match;

/* Match `path` against a route; record what its captures take. */
static bool match_route(const Route *route, const char *path, size_t path_len, Match *match)
{
    match->count = 0;
    size_t at = 0;
    size_t segment_index = 0;
    for (; segment_index < route->count; segment_index++) {
        const Segment *segment = route->segments + segment_index;
        if (at >= path_len || path[at] != '/')
            return false;
        at++;
        size_t end = at;
        while (end < path_len && path[end] != '/')
            end++;
        if (segment->kind == SEGMENT_WILDCARD) {
            if (match->count == 32 || at > path_len)
                return false;
            match->start[match->count] = path + at;
            match->len[match->count] = path_len - at;
            match->score[match->count] = 1;
            match->count++;
            match->consumed = path_len;
            return true;
        }
        if (segment->kind == SEGMENT_CAPTURE) {
            if (end == at || match->count == 32)
                return false; /* a capture takes at least one character */
            match->start[match->count] = path + at;
            match->len[match->count] = end - at;
            match->score[match->count] = 2;
            match->count++;
        } else if (strlen(segment->text) != end - at || memcmp(segment->text, path + at, end - at) != 0) {
            return false;
        }
        at = end;
    }
    match->consumed = at;
    if (route->prefix)
        return at == path_len || path[at] == '/';
    if (route->count == 0)
        return path_len == 1 && path[0] == '/';
    return at == path_len;
}

/* Whether route `a` (as matched) is more specific than `b`: segment by segment, static > capture > wildcard. */
static bool more_specific(const Route *a, const Route *b)
{
    size_t count = a->count < b->count ? a->count : b->count;
    for (size_t i = 0; i < count; i++) {
        int sa = 3 - (int)a->segments[i].kind, sb = 3 - (int)b->segments[i].kind;
        if (sa != sb)
            return sa > sb;
    }
    if (a->prefix != b->prefix)
        return !a->prefix; /* a route beats a nested prefix */
    return a->count > b->count;
}

static char *params_of(const Route *route, const Match *match)
{
    Builder b = {0};
    build_reserve(&b, 0);
    b.text[0] = '\0';
    size_t capture = 0;
    for (size_t i = 0; i < route->count && capture < match->count; i++) {
        const Segment *segment = route->segments + i;
        if (segment->kind == SEGMENT_STATIC)
            continue;
        if (b.len > 0)
            build_char(&b, '&');
        build_form(&b, segment->text, strlen(segment->text));
        build_char(&b, '=');
        build_form(&b, match->start[capture], match->len[capture]);
        capture++;
    }
    return b.text;
}

/* -- running a request --------------------------------------------------------------------- */

static void request_release(Request *r);
static void *run_level(Request *r, size_t level, AxumResponse *out);

static AxumRequest handle(Request *r) { return (AxumRequest){r}; }

static AxumResponse plain_response(int status, const char *body)
{
    return body != NULL ? axum_text(status, body) : axum_response(status);
}

/* The layers a request goes through: the route's, then its method's. */
static void choose_layers(Request *r, const Layers *outer, const Layers *inner)
{
    r->layers.count = 0;
    /* added in order innermost first (each layer() call wraps what was there): run the last added first */
    for (size_t i = outer->count; i > 0; i--)
        layers_push(&r->layers, outer->items[i - 1]);
    for (size_t i = inner != NULL ? inner->count : 0; i > 0; i--)
        layers_push(&r->layers, inner->items[i - 1]);
}

/* Pick the route for the request, its handler and layers; or the 404 / 405 answers. */
static void dispatch(Request *r, Router *router)
{
    const char *path = r->path;
    size_t path_len = strlen(path);
    Route *best = NULL;
    Match match = {0}, found = {0};
    for (size_t i = 0; i < router->count; i++) {
        Route *route = router->routes + i;
        if (!match_route(route, path, path_len, &match))
            continue;
        if (best == NULL || more_specific(route, best)) {
            best = route;
            found = match;
        }
    }
    int method = method_index(r->method, strlen(r->method));
    r->handler = NULL;
    r->internal_status = 404;
    if (best == NULL || best->fallback) {
        Endpoint *fallback = best != NULL ? best->endpoint : router->fallback;
        r->matched_path = "";
        if (best != NULL && best->endpoint->nest_prefix != NULL) {
            r->nested_path = best->endpoint->nest_prefix;
            const char *rest = r->path + strlen(best->endpoint->nest_prefix);
            r->path = keep(r, axum_strdup(*rest == '\0' ? "/" : rest));
        }
        choose_layers(r, &router->fallback_layers, fallback != NULL ? &fallback->layers : NULL);
        if (fallback != NULL) {
            r->handler = fallback->service;
            r->state = fallback->state != NULL ? *fallback->state : router->state;
        }
        return;
    }
    Endpoint *endpoint = best->endpoint;
    r->params = keep(r, params_of(best, &found));
    if (axum_feature_matched_path)
        r->matched_path = best->pattern;
    r->state = endpoint->state != NULL && *endpoint->state != NULL ? *endpoint->state : router->state;
    if (endpoint->nest_prefix != NULL) {
        r->nested_path = endpoint->nest_prefix;
        const char *rest = r->path + strlen(endpoint->nest_prefix);
        r->path = keep(r, axum_strdup(*rest == '\0' ? "/" : rest));
    }
    if (endpoint->service != NULL) {
        r->handler = endpoint->service;
        choose_layers(r, &endpoint->layers, NULL);
        return;
    }
    MethodRouter *m = endpoint->methods;
    AxumHandler handler = method >= 0 ? m->handlers[method] : NULL;
    const Layers *method_layers = method >= 0 ? &m->layers[method] : NULL;
    if (handler == NULL && method == 5 /* HEAD */ && m->handlers[0] != NULL) {
        handler = m->handlers[0]; /* GET answers HEAD (without its body) */
        method_layers = &m->layers[0];
    }
    if (handler == NULL && m->any != NULL) {
        handler = m->any;
        method_layers = &m->any_layers;
    }
    if (handler == NULL && m->fallback != NULL) {
        handler = m->fallback;
        method_layers = &m->fallback_layers;
    }
    if (handler == NULL) { /* 405, with the methods it allows */
        char allowed[128] = "";
        for (size_t i = 0; i < METHOD_COUNT; i++) {
            if (m->handlers[i] == NULL)
                continue;
            if (allowed[0] != '\0')
                strcat(allowed, ",");
            strcat(allowed, METHODS[i]);
            if (i == 0 && m->handlers[5] == NULL)
                strcat(allowed, ",HEAD");
        }
        r->allowed = keep(r, axum_strdup(allowed));
        r->internal_status = 405;
        if (router->method_not_allowed != NULL) {
            handler = router->method_not_allowed->service;
            r->state = *router->method_not_allowed->state != NULL ? *router->method_not_allowed->state : router->state;
        }
        method_layers = &m->fallback_layers;
    }
    r->handler = handler;
    choose_layers(r, &endpoint->layers, method_layers);
}

typedef struct {
    Request *r;
    size_t level;
    AxumResponse *out;
    void *token;
} Step;

static void step_done(void *context, bool failed)
{
    Step *step = context;
    if (failed)
        mlirAsyncRuntimeSetTokenError(step->token);
    else
        mlirAsyncRuntimeEmplaceToken(step->token);
    free(step);
}

static void *call_handler(Request *r, AxumResponse *out)
{
    r->level = r->layers.count;
    if (r->limit >= 0 && (int64_t)r->body_len > r->limit) {
        *out = plain_response(413, "length limit exceeded");
        return axum_done_token();
    }
    if (r->handler == NULL) {
        *out = axum_response(r->internal_status);
        if (r->internal_status == 405)
            axum_set_header(*out, "allow", r->allowed);
        return axum_done_token();
    }
    return r->handler(handle(r), r->state, out);
}

static void after_map_request(void *context, bool failed)
{
    Step *step = context;
    if (failed || step->out->raw != NULL) { /* it answered: the request stops here */
        step_done(step, failed);
        return;
    }
    axum_after(run_level(step->r, step->level + 1, step->out), step_done, step);
}

static void after_inner_for_map_response(void *context, bool failed)
{
    Step *step = context;
    if (failed) {
        step_done(step, true);
        return;
    }
    Layer *layer = step->r->layers.items[step->level];
    if (step->out->raw == NULL)
        *step->out = axum_response(200);
    step->r->level = step->level;
    axum_after(layer->handler(handle(step->r), layer->state, step->out), step_done, step);
}

static void after_inner_for_header(void *context, bool failed)
{
    Step *step = context;
    Layer *layer = step->r->layers.items[step->level];
    if (!failed) {
        if (step->out->raw == NULL)
            *step->out = axum_response(200);
        Response *response = step->out->raw;
        if (headers_get(&response->headers, layer->name) == NULL)
            headers_add(&response->headers, layer->name, strlen(layer->name), layer->value, strlen(layer->value));
    }
    step_done(step, failed);
}

/* Run the request from layer `level` inward, putting the response in `out`; return the token of when it is. */
static void *run_level(Request *r, size_t level, AxumResponse *out)
{
    if (level >= r->layers.count)
        return call_handler(r, out);
    Layer *layer = r->layers.items[level];
    Step *step;
    switch (layer->kind) {
    case LAYER_FROM_FN:
        r->level = level; /* what Next.from_request reads */
        return layer->handler(handle(r), layer->state, out);
    case LAYER_MAP_REQUEST:
        step = axum_alloc(sizeof(Step));
        *step = (Step){r, level, out, mlirAsyncRuntimeCreateToken()};
        r->level = level;
        void *token = step->token;
        axum_after(layer->handler(handle(r), layer->state, out), after_map_request, step);
        return token;
    case LAYER_MAP_RESPONSE:
    case LAYER_HEADER:
        step = axum_alloc(sizeof(Step));
        *step = (Step){r, level, out, mlirAsyncRuntimeCreateToken()};
        token = step->token;
        axum_after(run_level(r, level + 1, out), layer->kind == LAYER_HEADER ? after_inner_for_header : after_inner_for_map_response,
                   step);
        return token;
    case LAYER_BODY_LIMIT:
        r->limit = layer->limit;
        return run_level(r, level + 1, out);
    case LAYER_EXTENSION:
        headers_remove(&r->extensions, layer->name);
        headers_add(&r->extensions, layer->name, strlen(layer->name), layer->value, strlen(layer->value));
        return run_level(r, level + 1, out);
    }
    return call_handler(r, out);
}

/* -- middleware: Next ------------------------------------------------------------------------ */

/* The rest of the stack, below the from_fn middleware being run (what Next.from_request takes). */
AxumNext axum_next_of(AxumRequest request)
{
    Request *r = request_of(request, "next_of");
    Next *next = axum_zalloc(sizeof(Next));
    next->r = r;
    next->level = r->level;
    next->link = r->nexts;
    r->nexts = next;
    return (AxumNext){next};
}

/* Run the rest of the stack: its token completes when its response is ready (axum_next_response). */
void *axum_next_run(AxumNext handle_, AxumRequest request)
{
    Next *next = handle_.raw;
    (void)request; /* the same request (middleware changes it in place) */
    if (next == NULL)
        fatal("next_run", "not a Next", NULL);
    if (next->ran)
        fatal("next_run", "this Next already ran; call next.run(req) once", NULL);
    next->ran = true;
    next->inner = (AxumResponse){NULL};
    return run_level(next->r, next->level + 1, &next->inner);
}

/* The response the rest of the stack gave: the caller owns it now. */
AxumResponse axum_next_response(AxumNext handle_)
{
    Next *next = handle_.raw;
    AxumResponse response = next->inner;
    next->inner = (AxumResponse){NULL};
    return response.raw != NULL ? response : axum_response(200);
}

/* -- serving ---------------------------------------------------------------------------------- */

static h2o_globalconf_t config;
static h2o_context_t context;
static h2o_accept_ctx_t accept_ctx;
static bool started;
static Server *server;
static const char *alt_svc; /* `h3=":8443"` when HTTP/3 is served */

static const char *reason(int status)
{
    switch (status) {
    case 100: return "Continue";
    case 101: return "Switching Protocols";
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 206: return "Partial Content";
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
    case 406: return "Not Acceptable";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 410: return "Gone";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 422: return "Unprocessable Entity";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    default: return "";
    }
}

static void log_request(Request *r, int status)
{
    if (!axum_feature_tracing)
        return;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double ms = (double)(now.tv_sec - r->started.tv_sec) * 1e3 + (double)(now.tv_nsec - r->started.tv_nsec) / 1e6;
    fprintf(stderr, "%s %s -> %d (%.3f ms, %s)\n", r->method, r->original_uri, status, ms, r->version);
}

/* The request's h2o side went away (sent, or the client left). */
static void on_disposed(void *memory)
{
    Request *r = *(Request **)memory;
    r->req = NULL;
    if (r->finished)
        request_release(r);
}

/* Run release(data) when the request is freed (feature libraries keep their state with it). */
void axum_on_release(Request *r, void (*release)(void *data), void *data)
{
    Cleanup *cleanup = axum_alloc(sizeof(Cleanup));
    *cleanup = (Cleanup){release, data, r->cleanups};
    r->cleanups = cleanup;
}

const char *axum_keep(Request *r, char *text) { return keep(r, text); }

/* The request's response goes out another way (a websocket): what its handler returns is not sent. */
void axum_mark_streaming(Request *r) { r->streaming = true; }

static void request_release(Request *r)
{
    while (r->cleanups != NULL) {
        Cleanup *cleanup = r->cleanups;
        r->cleanups = cleanup->next;
        cleanup->release(cleanup->data);
        free(cleanup);
    }
    axum_streams_release(r);
    for (size_t i = 0; i < r->kept_count; i++)
        free(r->kept[i]);
    free(r->kept);
    headers_free(&r->headers);
    headers_free(&r->extensions);
    free(r->layers.items);
    for (Next *next = r->nexts; next != NULL;) {
        Next *link = next->link;
        axum_response_free(next->inner);
        free(next);
        next = link;
    }
    axum_response_free(r->response);
    free(r);
}

static void send_response(Request *r)
{
    h2o_req_t *req = r->req;
    Response *response = r->response.raw;
    int status = response != NULL ? response->status : 200;
    log_request(r, status);
    req->res.status = status;
    req->res.reason = reason(status);
    if (response != NULL) {
        for (size_t i = 0; i < response->headers.count; i++) {
            Header *header = response->headers.items + i;
            size_t name_len = strlen(header->name), value_len = strlen(header->value);
            h2o_add_header_by_str(&req->pool, &req->res.headers, h2o_strdup(&req->pool, header->name, name_len).base, name_len,
                                  1, NULL, h2o_strdup(&req->pool, header->value, value_len).base, value_len);
        }
    }
    if (alt_svc != NULL && req->version < 0x300)
        h2o_add_header_by_str(&req->pool, &req->res.headers, H2O_STRLIT("alt-svc"), 0, NULL, alt_svc, strlen(alt_svc));
    bool head = strcmp(r->method, "HEAD") == 0;
    const char *body = response != NULL && response->body != NULL ? response->body : "";
    size_t len = response != NULL ? response->len : 0;
    if (head)
        len = 0;
    h2o_send_inline(req, body, len);
}

static void finish_request(void *data, bool failed)
{
    Request *r = data;
    if (failed) {
        fprintf(stderr, "axum: the handler of %s %s failed\n", r->method, r->original_uri);
        axum_response_free(r->response);
        r->response = plain_response(500, NULL);
    }
    r->finished = true;
    if (server != NULL)
        server->in_flight--;
    if (r->streaming) { /* its head went out already: what it returned is not sent */
        axum_response_free(r->response);
        r->response = (AxumResponse){NULL};
        axum_stream_finish(r);
        if (r->req == NULL)
            request_release(r);
        return;
    }
    if (r->req != NULL) {
        send_response(r); /* h2o disposes of the request once sent: request_release then */
    } else {
        request_release(r);
    }
}

static const char *protocol_of(h2o_req_t *req)
{
    return req->version >= 0x300 ? "HTTP/3.0" : req->version >= 0x200 ? "HTTP/2.0" : req->version == 0x100 ? "HTTP/1.0" : "HTTP/1.1";
}

static int on_request(h2o_handler_t *self, h2o_req_t *req)
{
    (void)self;
    if (server == NULL || server->router == NULL) {
        h2o_send_error_503(req, "Service Unavailable", "not serving", 0);
        return 0;
    }
    if ((req->version < 0x200 && !axum_feature_http1) || (req->version >= 0x200 && req->version < 0x300 && !axum_feature_http2)) {
        /* the features the app was built with leave this protocol out */
        req->res.status = 505;
        req->res.reason = "HTTP Version Not Supported";
        h2o_send_inline(req, H2O_STRLIT(""));
        return 0;
    }
    if (server->shutting_down)
        h2o_add_header(&req->pool, &req->res.headers, H2O_TOKEN_CONNECTION, NULL, H2O_STRLIT("close"));
    Request *r = axum_zalloc(sizeof(Request));
    r->req = req;
    clock_gettime(CLOCK_MONOTONIC, &r->started);
    r->limit = DEFAULT_BODY_LIMIT;
    r->method = keep(r, axum_strndup(req->method.base, req->method.len));
    const char *path = req->path_normalized.base;
    size_t path_len = req->path_normalized.len;
    r->path = keep(r, axum_strndup(path_len > 0 ? path : "/", path_len > 0 ? path_len : 1));
    if (req->query_at != SIZE_MAX) {
        r->has_query = true;
        r->query = keep(r, axum_strndup(req->path.base + req->query_at + 1, req->path.len - req->query_at - 1));
    } else {
        r->query = "";
    }
    r->original_uri = keep(r, axum_strndup(req->path.base, req->path.len));
    r->version = protocol_of(req);
    r->nested_path = "";
    r->matched_path = "";
    r->params = "";
    for (size_t i = 0; i < req->headers.size; i++) {
        h2o_header_t *header = req->headers.entries + i;
        headers_add(&r->headers, header->name->base, header->name->len, header->value.base, header->value.len);
    }
    if (req->upgrade.base != NULL) { /* h2o keeps these hop-by-hop headers apart: handlers (websockets) see them */
        if (headers_get(&r->headers, "connection") == NULL)
            headers_add(&r->headers, "connection", 10, "upgrade", 7);
        if (headers_get(&r->headers, "upgrade") == NULL)
            headers_add(&r->headers, "upgrade", 7, req->upgrade.base, req->upgrade.len);
    }
    if (req->authority.len > 0 && headers_get(&r->headers, "host") == NULL)
        headers_add(&r->headers, "host", 4, req->authority.base, req->authority.len);
    r->body = keep(r, axum_strndup(req->entity.base != NULL ? req->entity.base : "", req->entity.base != NULL ? req->entity.len : 0));
    r->body_len = req->entity.base != NULL ? req->entity.len : 0;
    struct sockaddr_storage peer;
    socklen_t peer_len = req->conn->callbacks->get_peername(req->conn, (struct sockaddr *)&peer);
    char host[INET6_ADDRSTRLEN] = "", text[INET6_ADDRSTRLEN + 16] = "";
    if (peer_len > 0 && getnameinfo((struct sockaddr *)&peer, peer_len, host, sizeof host, NULL, 0, NI_NUMERICHOST) == 0) {
        int port = peer.ss_family == AF_INET6 ? ntohs(((struct sockaddr_in6 *)&peer)->sin6_port)
                                              : ntohs(((struct sockaddr_in *)&peer)->sin_port);
        snprintf(text, sizeof text, peer.ss_family == AF_INET6 ? "[%s]:%d" : "%s:%d", host, port);
    }
    r->peer = keep(r, axum_strdup(text));
    Request **link = h2o_mem_alloc_shared(&req->pool, sizeof(Request *), on_disposed);
    *link = r;

    dispatch(r, server->router);
    server->in_flight++;
    axum_after(run_level(r, 0, &r->response), finish_request, r);
    bifrost_async_run_ready(); /* a handler that did not pause is done: respond now */
    return 0;
}

/* -- listeners ---------------------------------------------------------------------------------- */

static Listener *listener_of(AxumListener listener, const char *function)
{
    if (listener.raw == NULL)
        fatal(function, "not a listener from axum.bind(...)", NULL);
    return listener.raw;
}

/* Bind a TCP address ("0.0.0.0:3000", "[::]:8080", "localhost:0"); `listener_error` says why one failed. */
AxumListener axum_bind(const char *address)
{
    Listener *listener = axum_zalloc(sizeof(Listener));
    listener->fd = -1;
    char host[256] = "";
    const char *port_text = NULL;
    if (address[0] == '[') {
        const char *close = strchr(address, ']');
        if (close != NULL && close[1] == ':') {
            snprintf(host, sizeof host, "%.*s", (int)(close - address - 1), address + 1);
            port_text = close + 2;
        }
    } else {
        const char *colon = strrchr(address, ':');
        if (colon != NULL) {
            snprintf(host, sizeof host, "%.*s", (int)(colon - address), address);
            port_text = colon + 1;
        }
    }
    if (port_text == NULL || *port_text == '\0') {
        listener->error = axum_strdup("an address is host:port, like 0.0.0.0:3000");
        return (AxumListener){listener};
    }
    struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM, .ai_flags = AI_PASSIVE}, *found = NULL;
    int failed = getaddrinfo(host[0] != '\0' ? host : NULL, port_text, &hints, &found);
    if (failed != 0) {
        listener->error = axum_strdup(gai_strerror(failed));
        return (AxumListener){listener};
    }
    for (struct addrinfo *at = found; at != NULL; at = at->ai_next) {
        int fd = socket(at->ai_family, at->ai_socktype, 0), on = 1;
        if (fd == -1)
            continue;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        if (bind(fd, at->ai_addr, at->ai_addrlen) == 0 && listen(fd, SOMAXCONN) == 0) {
            listener->fd = fd;
            break;
        }
        close(fd);
    }
    if (listener->fd == -1)
        listener->error = axum_strdup(strerror(errno));
    freeaddrinfo(found);
    return (AxumListener){listener};
}

/* Why binding failed; "" if it did not. */
const char *axum_listener_error(AxumListener listener)
{
    Listener *l = listener_of(listener, "listener_error");
    return l->error != NULL ? l->error : "";
}

/* The address it listens on: "0.0.0.0:3000" (the port the system chose, for :0). */
const char *axum_local_addr(AxumListener listener)
{
    Listener *l = listener_of(listener, "local_addr");
    if (l->fd == -1)
        return "";
    struct sockaddr_storage address;
    socklen_t len = sizeof address;
    char host[INET6_ADDRSTRLEN] = "";
    if (getsockname(l->fd, (struct sockaddr *)&address, &len) != 0 ||
        getnameinfo((struct sockaddr *)&address, len, host, sizeof host, NULL, 0, NI_NUMERICHOST) != 0)
        return "";
    int port = address.ss_family == AF_INET6 ? ntohs(((struct sockaddr_in6 *)&address)->sin6_port)
                                             : ntohs(((struct sockaddr_in *)&address)->sin_port);
    free(l->local);
    l->local = axum_alloc(sizeof host + 16);
    snprintf(l->local, sizeof host + 16, address.ss_family == AF_INET6 ? "[%s]:%d" : "%s:%d", host, port);
    return l->local;
}

/* The port it listens on (the one the system chose, for :0). */
int32_t axum_local_port(AxumListener listener)
{
    Listener *l = listener_of(listener, "local_port");
    struct sockaddr_storage address;
    socklen_t len = sizeof address;
    if (l->fd == -1 || getsockname(l->fd, (struct sockaddr *)&address, &len) != 0)
        return 0;
    return address.ss_family == AF_INET6 ? ntohs(((struct sockaddr_in6 *)&address)->sin6_port)
                                         : ntohs(((struct sockaddr_in *)&address)->sin_port);
}

/* Set TCP_NODELAY on the connections it accepts (serve(...).tcp_nodelay(true)). */
AxumListener axum_tcp_nodelay(AxumListener listener, bool on)
{
    listener_of(listener, "tcp_nodelay")->nodelay = on;
    return listener;
}

static void on_accept(h2o_socket_t *socket, const char *err)
{
    h2o_socket_t *sock;
    if (err != NULL || (sock = h2o_evloop_socket_accept(socket)) == NULL)
        return;
    if (server != NULL && server->listener->nodelay) {
        int on = 1;
        setsockopt(h2o_socket_get_fd(sock), IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
    }
    h2o_accept(&accept_ctx, sock);
}

static int32_t wait_limit(void)
{
    if (bifrost_async_pending())
        return 0;
    int64_t timer = bifrost_async_next_timer();
    if (server != NULL && server->shutting_down)
        timer = timer < 0 || timer > 50 ? 50 : timer; /* look at the requests in flight again soon */
    return timer < 0 || timer > INT32_MAX ? INT32_MAX : (int32_t)timer;
}

static void check_shutdown(void);

/* Wait for events and handle them: how code waiting on a token lets the server run. */
static void poll_once(void *unused)
{
    (void)unused;
    h2o_evloop_run(context.loop, wait_limit());
    check_shutdown();
}

/* -- shutdown ------------------------------------------------------------------------------------ */

static volatile sig_atomic_t signalled;

static void on_signal(int number)
{
    (void)number;
    signalled = 1;
}

static void *signal_tokens[64];
static size_t signal_token_count;

/* A token that completes on Ctrl+C (SIGINT) or SIGTERM: the usual graceful shutdown signal. */
void *axum_shutdown_signal(void)
{
    struct sigaction action = {0};
    action.sa_handler = on_signal;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    void *token = mlirAsyncRuntimeCreateToken();
    if (signal_token_count == sizeof signal_tokens / sizeof signal_tokens[0])
        fatal("shutdown_signal", "too many waiting for the shutdown signal", NULL);
    signal_tokens[signal_token_count++] = token;
    return token;
}

/* Stop: no new connections; finish the requests in flight, then complete the serve token. */
static void begin_shutdown(Server *s)
{
    if (s->shutting_down)
        return;
    s->shutting_down = true;
    if (s->socket != NULL) {
        h2o_socket_read_stop(s->socket);
        h2o_socket_close(s->socket);
        s->socket = NULL;
    }
    h2o_context_request_shutdown(&context);
}

static void check_shutdown(void)
{
    if (signalled && signal_token_count > 0) {
        signalled = 0;
        for (size_t i = 0; i < signal_token_count; i++)
            mlirAsyncRuntimeEmplaceToken(signal_tokens[i]);
        signal_token_count = 0;
    }
    if (server != NULL && server->shutting_down && server->in_flight == 0 && server->token != NULL) {
        void *token = server->token;
        server->token = NULL;
        server = NULL;
        bifrost_async_set_poller(NULL, NULL);
        mlirAsyncRuntimeEmplaceToken(token);
    }
}

static void on_shutdown_signal(void *context_, bool failed)
{
    (void)failed;
    begin_shutdown(context_);
    check_shutdown();
}

static bool setup(Listener *listener, Router *router, const char *function)
{
    if (listener->fd == -1)
        fatal(function, "the listener did not bind: ", listener->error);
    if (server != NULL)
        fatal(function, "a server is running already; serve one router at a time", NULL);
    if (!started) {
        signal(SIGPIPE, SIG_IGN);
        h2o_config_init(&config);
        h2o_hostconf_t *host = h2o_config_register_host(&config, h2o_iovec_init(H2O_STRLIT("default")), 65535);
        h2o_pathconf_t *path = h2o_config_register_path(host, "/", 0);
        h2o_handler_t *handler = h2o_create_handler(path, sizeof(*handler));
        handler->on_req = on_request;
        h2o_context_init(&context, h2o_evloop_create(), &config);
        accept_ctx.ctx = &context;
        accept_ctx.hosts = config.hosts;
        started = true;
    }
    server = axum_zalloc(sizeof(Server));
    server->listener = listener;
    server->router = router;
    server->socket = h2o_evloop_socket_create(context.loop, listener->fd, H2O_SOCKET_FLAG_DONT_READ);
    h2o_socket_read_start(server->socket, on_accept);
    bifrost_async_set_poller(poll_once, NULL);
    return true;
}

static void *serve(AxumListener listener, AxumRouter router, AxumTask shutdown, const char *function)
{
    Listener *l = listener_of(listener, function);
    Router *r = router_of(router, function);
    setup(l, r, function);
    server->token = mlirAsyncRuntimeCreateToken();
    void *token = server->token;
    if (shutdown != NULL)
        axum_after(shutdown(), on_shutdown_signal, server);
    if (axum_feature_tracing)
        fprintf(stderr, "axum: listening on %s\n", axum_local_addr(listener));
    return token;
}

/* Serve `router` on `listener` until the program ends: a token that completes when the server stops. */
void *axum_serve(AxumListener listener, AxumRouter router)
{
    return serve(listener, router, NULL, "serve");
}

/* Serve until `signal` (an async function, e.g. waiting for axum.shutdown_signal()) completes, then finish
   the requests in flight and stop (with_graceful_shutdown). */
void *axum_serve_with_graceful_shutdown(AxumListener listener, AxumRouter router, AxumTask signal_)
{
    return serve(listener, router, signal_, "serve_with_graceful_shutdown");
}

/* HTTP/3: QUIC on UDP, on the TCP listener's port, with the certificate of the TLS listener. */

static ptls_openssl_sign_certificate_t quic_signer;
static ptls_context_t quic_tls;
static quicly_context_t quic;
static quicly_cid_plaintext_t next_cid;
static h2o_http3_server_ctx_t http3;

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
    return &conn->super;
}

static bool setup_http3(int32_t port, const char *cert, const char *key)
{
    FILE *fp = fopen(key, "r");
    EVP_PKEY *private_key = fp ? PEM_read_PrivateKey(fp, NULL, NULL, NULL) : NULL;
    if (fp)
        fclose(fp);
    if (private_key == NULL || ptls_openssl_init_sign_certificate(&quic_signer, private_key) != 0)
        return false;
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
        return false;
    quicly_amend_ptls_context(&quic_tls);
    quic = quicly_spec_context;
    quic.tls = &quic_tls;
    uint8_t secret[PTLS_MAX_DIGEST_SIZE];
    ptls_openssl_random_bytes(secret, sizeof(secret));
    quic.cid_encryptor = quicly_new_default_cid_encryptor(&ptls_openssl_quiclb, &ptls_openssl_aes128ecb, &ptls_openssl_sha256,
                                                          ptls_iovec_init(secret, sizeof(secret)));
    h2o_http3_server_amend_quicly_context(&config, &quic);
    struct sockaddr_in6 addr = {.sin6_family = AF_INET6, .sin6_port = htons((uint16_t)port), .sin6_addr = in6addr_any};
    int fd = socket(AF_INET6, SOCK_DGRAM, 0), on = 1, off = 0;
    if (fd == -1)
        return false;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return false;
    }
    h2o_socket_t *sock = h2o_evloop_socket_create(context.loop, fd, H2O_SOCKET_FLAG_DONT_READ);
    h2o_http3_server_init_context(&context, &http3.super, context.loop, sock, NULL, &quic, &next_cid, on_quic_accept, NULL, 0);
    http3.accept_ctx = &accept_ctx;
    http3.qpack = (h2o_http3_qpack_context_t){.encoder_table_capacity = 4096};
    return true;
}

/* Serve HTTPS (HTTP/1.1 and HTTP/2 over TLS, and HTTP/3 over QUIC) with a certificate and its key (PEM files). */
void *axum_serve_tls(AxumListener listener, AxumRouter router, const char *cert, const char *key)
{
    Listener *l = listener_of(listener, "serve_tls");
    setup(l, router_of(router, "serve_tls"), "serve_tls");
    accept_ctx.ssl_ctx = SSL_CTX_new(TLS_server_method());
    SSL_CTX_set_min_proto_version(accept_ctx.ssl_ctx, TLS1_2_VERSION);
    if (SSL_CTX_use_certificate_chain_file(accept_ctx.ssl_ctx, cert) != 1 ||
        SSL_CTX_use_PrivateKey_file(accept_ctx.ssl_ctx, key, SSL_FILETYPE_PEM) != 1)
        fatal("serve_tls", "cannot load the certificate and key: ", cert);
    if (axum_feature_http2)
        h2o_ssl_register_alpn_protocols(accept_ctx.ssl_ctx, h2o_http2_alpn_protocols);
    int32_t port = axum_local_port(listener);
    if (setup_http3(port, cert, key)) {
        char text[32];
        snprintf(text, sizeof(text), "h3=\":%d\"", port);
        alt_svc = axum_strdup(text);
    }
    server->token = mlirAsyncRuntimeCreateToken();
    return server->token;
}

/* -- JSON syntax, for the Json extractor's 400 (bad syntax) and 422 (wrong shape) ------------------ */

static const char *json_skip(const char *at, int depth);

static const char *json_space(const char *at)
{
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r')
        at++;
    return at;
}

static const char *json_string(const char *at)
{
    if (*at != '"')
        return NULL;
    for (at++; *at != '"'; at++) {
        if (*at == '\0' || (unsigned char)*at < 0x20)
            return NULL;
        if (*at == '\\') {
            at++;
            if (*at == 'u') {
                for (int i = 0; i < 4; i++)
                    if (!isxdigit((unsigned char)*++at))
                        return NULL;
            } else if (strchr("\"\\/bfnrt", *at) == NULL || *at == '\0') {
                return NULL;
            }
        }
    }
    return at + 1;
}

static const char *json_number(const char *at)
{
    if (*at == '-')
        at++;
    if (*at == '0')
        at++;
    else if (isdigit((unsigned char)*at))
        while (isdigit((unsigned char)*at))
            at++;
    else
        return NULL;
    if (*at == '.') {
        at++;
        if (!isdigit((unsigned char)*at))
            return NULL;
        while (isdigit((unsigned char)*at))
            at++;
    }
    if (*at == 'e' || *at == 'E') {
        at++;
        if (*at == '+' || *at == '-')
            at++;
        if (!isdigit((unsigned char)*at))
            return NULL;
        while (isdigit((unsigned char)*at))
            at++;
    }
    return at;
}

static const char *json_skip(const char *at, int depth)
{
    if (depth > 512)
        return NULL;
    at = json_space(at);
    switch (*at) {
    case '"':
        return json_string(at);
    case '{':
        at = json_space(at + 1);
        if (*at == '}')
            return at + 1;
        for (;;) {
            at = json_string(json_space(at));
            if (at == NULL || *(at = json_space(at)) != ':')
                return NULL;
            at = json_skip(at + 1, depth + 1);
            if (at == NULL)
                return NULL;
            at = json_space(at);
            if (*at == '}')
                return at + 1;
            if (*at != ',')
                return NULL;
            at++;
        }
    case '[':
        at = json_space(at + 1);
        if (*at == ']')
            return at + 1;
        for (;;) {
            at = json_skip(at, depth + 1);
            if (at == NULL)
                return NULL;
            at = json_space(at);
            if (*at == ']')
                return at + 1;
            if (*at != ',')
                return NULL;
            at++;
        }
    case 't':
        return strncmp(at, "true", 4) == 0 ? at + 4 : NULL;
    case 'f':
        return strncmp(at, "false", 5) == 0 ? at + 5 : NULL;
    case 'n':
        return strncmp(at, "null", 4) == 0 ? at + 4 : NULL;
    default:
        return json_number(at);
    }
}

/* Whether `text` is JSON at all (its shape aside). */
bool axum_json_valid(const char *text)
{
    const char *end = json_skip(text, 0);
    return end != NULL && *json_space(end) == '\0';
}

/* Whether a request's content type is `essence` (application/json, ...), or `+json`-style when `suffix`. */
bool axum_content_type_is(AxumRequest request, const char *essence, const char *suffix)
{
    const char *value = axum_header(request, "content-type");
    size_t len = strcspn(value, ";");
    while (len > 0 && value[len - 1] == ' ')
        len--;
    if (len == strlen(essence) && strncasecmp(value, essence, len) == 0)
        return true;
    size_t suffix_len = strlen(suffix);
    return suffix_len > 0 && len > suffix_len && strncasecmp(value + len - suffix_len, suffix, suffix_len) == 0 &&
           strncasecmp(value, "application/", 12) == 0;
}
