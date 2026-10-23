/*
 * Streamed bodies: server-sent events, and bodies written bit by bit (Body::from_stream).
 *
 * A handler opens a stream on its request, which sends the response's head at once;
 * then it writes to it, awaiting each write (its token completes once h2o has taken
 * the bytes, so a slow client slows the writer down), and closes it, or returns:
 *
 *     let stream = sse.start(req)
 *     await sse.send(stream, sse.Event(data: "hello"))
 *     return sse.end(stream)
 *
 * What the handler returns is not sent (the head went out already), so layers that
 * change responses do not change a streamed one. A stream whose client went away is
 * `closed`: writes then complete at once, writing nothing.
 */

#include <stdlib.h>
#include <string.h>

#include "axum.h"

void *mlirAsyncRuntimeCreateToken(void);
void mlirAsyncRuntimeEmplaceToken(void *token);

typedef struct Chunk {
    char *data;
    size_t len;
    void *token; /* completes once h2o has taken it */
    struct Chunk *next;
} Chunk;

typedef struct OutStream {
    h2o_generator_t generator; /* first: h2o hands it back */
    Request *r;
    Chunk *head, *tail;
    Chunk *in_flight; /* given to h2o_send, until proceed */
    bool closed;      /* the client went away, or it was ended and sent */
    bool ending;      /* end asked for: sent once what is queued is */
    h2o_timer_t keep_alive;
    uint64_t keep_alive_ms;
    char *keep_alive_text;
    struct OutStream *link;
} OutStream;

static OutStream *streams; /* those open: freed when their request is */

static h2o_req_t *req_of(OutStream *s)
{
    return s->r != NULL ? s->r->req : NULL;
}

static void complete(Chunk *chunk)
{
    if (chunk->token != NULL)
        mlirAsyncRuntimeEmplaceToken(chunk->token);
    free(chunk->data);
    free(chunk);
}

static void drop_all(OutStream *s)
{
    while (s->head != NULL) {
        Chunk *next = s->head->next;
        complete(s->head);
        s->head = next;
    }
    s->tail = NULL;
}

/* Hand the next queued chunk to h2o (one at a time); end the response when asked and all is sent. */
static void pump(OutStream *s)
{
    h2o_req_t *req = req_of(s);
    if (req == NULL) {
        s->closed = true;
        drop_all(s);
        return;
    }
    if (s->in_flight != NULL || s->closed)
        return;
    if (s->head != NULL) {
        Chunk *chunk = s->head;
        s->head = chunk->next;
        if (s->head == NULL)
            s->tail = NULL;
        s->in_flight = chunk;
        h2o_iovec_t buffer = h2o_iovec_init(chunk->data, chunk->len);
        h2o_send(req, &buffer, 1, H2O_SEND_STATE_IN_PROGRESS);
        return;
    }
    if (s->ending) {
        s->closed = true;
        if (h2o_timer_is_linked(&s->keep_alive))
            h2o_timer_unlink(&s->keep_alive);
        h2o_send(req, NULL, 0, H2O_SEND_STATE_FINAL);
    }
}

static void on_proceed(h2o_generator_t *generator, h2o_req_t *req)
{
    (void)req;
    OutStream *s = (OutStream *)generator;
    if (s->in_flight != NULL) {
        complete(s->in_flight);
        s->in_flight = NULL;
    }
    pump(s);
}

static void on_stop(h2o_generator_t *generator, h2o_req_t *req)
{
    (void)req;
    OutStream *s = (OutStream *)generator;
    s->closed = true;
    if (s->in_flight != NULL) {
        complete(s->in_flight);
        s->in_flight = NULL;
    }
    drop_all(s);
    if (h2o_timer_is_linked(&s->keep_alive))
        h2o_timer_unlink(&s->keep_alive);
}

static void queue(OutStream *s, char *data, size_t len, void *token)
{
    Chunk *chunk = axum_zalloc(sizeof(Chunk));
    chunk->data = data;
    chunk->len = len;
    chunk->token = token;
    if (s->tail != NULL)
        s->tail->next = chunk;
    else
        s->head = chunk;
    s->tail = chunk;
    pump(s);
}

static OutStream *stream_of(AxumStream stream, const char *function)
{
    if (stream.raw == NULL) {
        fprintf(stderr, "axum.%s: not a stream (a zero value)\n", function);
        exit(1);
    }
    return stream.raw;
}

/* Start the response now, with `status` and `content_type`, its body to be written: a stream. */
AxumStream axum_stream_open(AxumRequest request, int32_t status, const char *content_type)
{
    Request *r = request.raw;
    OutStream *s = axum_zalloc(sizeof(OutStream));
    s->generator.proceed = on_proceed;
    s->generator.stop = on_stop;
    s->r = r;
    s->link = streams;
    streams = s;
    h2o_timer_init(&s->keep_alive, NULL);
    if (r->streaming) {
        fprintf(stderr, "axum.stream_open: %s %s already streams its response\n", r->method, r->original_uri);
        s->closed = true;
        return (AxumStream){s};
    }
    r->streaming = true;
    h2o_req_t *req = r->req;
    if (req == NULL) {
        s->closed = true;
        return (AxumStream){s};
    }
    req->res.status = status;
    req->res.reason = status == 200 ? "OK" : "";
    size_t len = strlen(content_type);
    h2o_add_header(&req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE, NULL, h2o_strdup(&req->pool, content_type, len).base,
                   len);
    if (strncmp(content_type, "text/event-stream", 17) == 0) {
        h2o_add_header(&req->pool, &req->res.headers, H2O_TOKEN_CACHE_CONTROL, NULL, H2O_STRLIT("no-cache"));
        h2o_add_header_by_str(&req->pool, &req->res.headers, H2O_STRLIT("x-accel-buffering"), 0, NULL, H2O_STRLIT("no"));
    }
    h2o_start_response(req, &s->generator);
    s->in_flight = axum_zalloc(sizeof(Chunk)); /* the head goes out now: nothing more until h2o proceeds */
    h2o_send(req, NULL, 0, H2O_SEND_STATE_IN_PROGRESS);
    return (AxumStream){s};
}

/* Write `text` to the stream: the token completes once it is taken (at once, if the client has gone). */
void *axum_stream_write(AxumStream stream, const char *text)
{
    OutStream *s = stream_of(stream, "stream_write");
    void *token = mlirAsyncRuntimeCreateToken();
    if (s->closed || s->ending) {
        mlirAsyncRuntimeEmplaceToken(token);
        return token;
    }
    size_t len = strlen(text);
    queue(s, axum_strndup(text, len), len, token);
    return token;
}

/* Whether the client has gone (or the stream was ended): writing to it does nothing. */
bool axum_stream_closed(AxumStream stream)
{
    OutStream *s = stream_of(stream, "stream_closed");
    if (req_of(s) == NULL)
        s->closed = true;
    return s->closed;
}

/* End the stream once what was written is sent. */
void axum_stream_end(AxumStream stream)
{
    OutStream *s = stream_of(stream, "stream_end");
    s->ending = true;
    pump(s);
}

static void on_keep_alive(h2o_timer_t *timer)
{
    OutStream *s = H2O_STRUCT_FROM_MEMBER(OutStream, keep_alive, timer);
    if (s->closed || req_of(s) == NULL)
        return;
    if (s->head == NULL && s->in_flight == NULL)
        queue(s, axum_strdup(s->keep_alive_text), strlen(s->keep_alive_text), NULL);
    h2o_req_t *req = req_of(s);
    if (req != NULL && !s->closed)
        h2o_timer_link(req->conn->ctx->loop, s->keep_alive_ms, &s->keep_alive);
}

/* Write `text` (an SSE comment, ":\n\n") every `milliseconds` the stream is idle (KeepAlive). */
void axum_stream_keep_alive(AxumStream stream, int64_t milliseconds, const char *text)
{
    OutStream *s = stream_of(stream, "stream_keep_alive");
    h2o_req_t *req = req_of(s);
    if (req == NULL || milliseconds <= 0)
        return;
    free(s->keep_alive_text);
    s->keep_alive_text = axum_strdup(text);
    s->keep_alive_ms = (uint64_t)milliseconds;
    s->keep_alive.cb = on_keep_alive;
    if (h2o_timer_is_linked(&s->keep_alive))
        h2o_timer_unlink(&s->keep_alive);
    h2o_timer_link(req->conn->ctx->loop, s->keep_alive_ms, &s->keep_alive);
}

/* One server-sent event: `event:` (if named), `id:` (if given), then `data:` for each line of data. */
void *axum_sse_send(AxumStream stream, const char *event, const char *data, const char *id, int64_t retry)
{
    size_t size = strlen(event) + strlen(data) * 2 + strlen(id) + 64;
    for (const char *c = data; *c; c++)
        size += *c == '\n' ? 6 : 0;
    char *text = axum_alloc(size), *o = text;
    if (*event != '\0')
        o += sprintf(o, "event: %s\n", event);
    if (*id != '\0')
        o += sprintf(o, "id: %s\n", id);
    if (retry > 0)
        o += sprintf(o, "retry: %lld\n", (long long)retry);
    const char *line = data;
    do {
        size_t len = strcspn(line, "\n");
        o += sprintf(o, "data: %.*s\n", (int)len, line);
        line += len;
    } while (*line++ == '\n');
    *o++ = '\n';
    *o = '\0';
    OutStream *s = stream_of(stream, "sse_send");
    void *token = mlirAsyncRuntimeCreateToken();
    if (s->closed || s->ending) {
        free(text);
        mlirAsyncRuntimeEmplaceToken(token);
        return token;
    }
    queue(s, text, (size_t)(o - text), token);
    return token;
}

/* What a handler streaming its response returns: the response, which is not sent (the stream is). */
AxumResponse axum_stream_response(AxumStream stream)
{
    axum_stream_end(stream);
    return axum_response(200);
}

/* Free the streams of a request that is done (called when it is freed). */
void axum_streams_release(Request *r)
{
    OutStream **at = &streams;
    while (*at != NULL) {
        OutStream *s = *at;
        if (s->r != r) {
            at = &s->link;
            continue;
        }
        *at = s->link;
        if (h2o_timer_is_linked(&s->keep_alive))
            h2o_timer_unlink(&s->keep_alive);
        if (s->in_flight != NULL)
            complete(s->in_flight);
        drop_all(s);
        free(s->keep_alive_text);
        free(s);
    }
}

/* A request done streaming (its handler returned): end its stream if the handler did not. */
void axum_stream_finish(Request *r)
{
    for (OutStream *s = streams; s != NULL; s = s->link)
        if (s->r == r && !s->ending) {
            s->ending = true;
            pump(s);
        }
}
