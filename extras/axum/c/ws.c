/*
 * WebSockets (axum's `ws` feature): RFC 6455 over the socket h2o hands over after an
 * HTTP/1.1 upgrade.
 *
 * A handler takes a WebSocketUpgrade (its extractor checks the handshake), accepts it,
 * which sends the 101, and then talks over the socket until it is done:
 *
 *     let socket = upgrade.accept()
 *     while await ws.recv(socket) {
 *         await ws.send(socket, "you said: {ws.text(socket)}")
 *     }
 *     return ws.done(socket)
 *
 * recv's token completes when a message arrives (read it with ws.text and ws.kind) or the
 * socket closes (ws.closed); pings are answered, and fragments put together, here. send's
 * completes once the frame is written. What the handler returns is not sent (the 101 was).
 */

#include <stdlib.h>
#include <string.h>

#include <openssl/sha.h>

#include "axum.h"
#include "h2o/http1.h"

void *mlirAsyncRuntimeCreateToken(void);
void mlirAsyncRuntimeEmplaceToken(void *token);

typedef struct { void *raw; } AxumSocket;

enum { OP_CONTINUATION = 0, OP_TEXT = 1, OP_BINARY = 2, OP_CLOSE = 8, OP_PING = 9, OP_PONG = 10 };

#define MAX_MESSAGE (16 * 1024 * 1024) /* axum's (tungstenite's) default limit: 64 MB; 16 here */

typedef struct Message {
    int kind;
    char *data;
    size_t len;
    struct Message *next;
} Message;

typedef struct Frame {
    char *data;
    size_t len;
    void *token;
    bool closing; /* the close frame: the socket closes once it is written */
    struct Frame *next;
} Frame;

typedef struct Socket {
    Request *r;
    h2o_socket_t *sock; /* once the upgrade is done */
    bool closed;        /* the connection is gone (or closing) */
    bool close_sent;
    Message *head, *tail;  /* received, not yet taken */
    Message *current;      /* the one the last recv took */
    void *waiting;         /* a recv's token, until a message (or the end) */
    Frame *out_head, *out_tail;
    Frame *writing;
    h2o_iovec_t writing_buffer;
    char *fragments;
    size_t fragments_len;
    int fragments_kind;
    int close_code;
    bool orphan; /* its request is freed: it goes once its last frames are written */
} Socket;

static void socket_free(void *data);
static void free_now(Socket *s);

static void wake(Socket *s)
{
    if (s->waiting != NULL && (s->head != NULL || s->closed)) {
        void *token = s->waiting;
        s->waiting = NULL;
        mlirAsyncRuntimeEmplaceToken(token);
    }
}

static void deliver(Socket *s, int kind, char *data, size_t len)
{
    Message *message = axum_zalloc(sizeof(Message));
    *message = (Message){kind, data, len, NULL};
    if (s->tail != NULL)
        s->tail->next = message;
    else
        s->head = message;
    s->tail = message;
    wake(s);
}

static void write_next(Socket *s);

static void finish_frames(Socket *s)
{
    while (s->out_head != NULL) {
        Frame *frame = s->out_head;
        s->out_head = frame->next;
        if (frame->token != NULL)
            mlirAsyncRuntimeEmplaceToken(frame->token);
        free(frame->data);
        free(frame);
    }
    s->out_tail = NULL;
}

static void gone(Socket *s)
{
    if (s->sock != NULL) {
        s->sock->data = NULL;
        h2o_socket_close(s->sock);
        s->sock = NULL;
    }
    s->closed = true;
    if (s->writing != NULL) {
        if (s->writing->token != NULL)
            mlirAsyncRuntimeEmplaceToken(s->writing->token);
        free(s->writing->data);
        free(s->writing);
        s->writing = NULL;
    }
    finish_frames(s);
    wake(s);
    if (s->orphan)
        free_now(s);
}

static void on_written(h2o_socket_t *sock, const char *err)
{
    Socket *s = sock->data;
    if (s == NULL)
        return;
    Frame *frame = s->writing;
    s->writing = NULL;
    bool closing = frame != NULL && frame->closing;
    if (frame != NULL) {
        if (frame->token != NULL)
            mlirAsyncRuntimeEmplaceToken(frame->token);
        free(frame->data);
        free(frame);
    }
    if (err != NULL || closing) {
        gone(s);
        return;
    }
    write_next(s);
    if (s->orphan && s->writing == NULL && s->out_head == NULL)
        gone(s); /* its request is done, and so is the last frame */
}

static void write_next(Socket *s)
{
    if (s->sock == NULL || s->writing != NULL || s->out_head == NULL)
        return;
    Frame *frame = s->out_head;
    s->out_head = frame->next;
    if (s->out_head == NULL)
        s->out_tail = NULL;
    s->writing = frame;
    s->writing_buffer = h2o_iovec_init(frame->data, frame->len);
    h2o_socket_write(s->sock, &s->writing_buffer, 1, on_written);
}

/* Queue a frame (server frames are not masked); `token` completes once it is written. */
static void send_frame(Socket *s, int opcode, const char *payload, size_t len, void *token)
{
    if (s->close_sent) {
        if (token != NULL)
            mlirAsyncRuntimeEmplaceToken(token);
        return;
    }
    char *data = axum_alloc(len + 10), *o = data;
    *o++ = (char)(0x80 | opcode);
    if (len < 126) {
        *o++ = (char)len;
    } else if (len < 65536) {
        *o++ = 126;
        *o++ = (char)(len >> 8);
        *o++ = (char)len;
    } else {
        *o++ = 127;
        for (int i = 7; i >= 0; i--)
            *o++ = (char)((uint64_t)len >> (8 * i));
    }
    memcpy(o, payload, len);
    Frame *frame = axum_zalloc(sizeof(Frame));
    *frame = (Frame){data, (size_t)(o - data) + len, token, opcode == OP_CLOSE, NULL};
    if (opcode == OP_CLOSE)
        s->close_sent = true;
    if (s->out_tail != NULL)
        s->out_tail->next = frame;
    else
        s->out_head = frame;
    s->out_tail = frame;
    write_next(s);
}

static void send_close(Socket *s, int code, const char *reason)
{
    char payload[125];
    size_t len = strlen(reason);
    if (len > sizeof payload - 2)
        len = sizeof payload - 2;
    payload[0] = (char)(code >> 8);
    payload[1] = (char)code;
    memcpy(payload + 2, reason, len);
    send_frame(s, OP_CLOSE, payload, code > 0 ? len + 2 : 0, NULL);
}

/* Read the complete frames in the socket's input; whether the connection goes on. */
static bool read_frames(Socket *s)
{
    h2o_buffer_t *input = s->sock->input;
    for (;;) {
        const unsigned char *at = (const unsigned char *)input->bytes;
        size_t size = input->size;
        if (size < 2)
            return true;
        bool fin = at[0] & 0x80;
        int opcode = at[0] & 0x0F;
        bool masked = at[1] & 0x80;
        uint64_t len = at[1] & 0x7F;
        size_t header = 2;
        if (len == 126) {
            if (size < 4)
                return true;
            len = ((uint64_t)at[2] << 8) | at[3];
            header = 4;
        } else if (len == 127) {
            if (size < 10)
                return true;
            len = 0;
            for (int i = 0; i < 8; i++)
                len = (len << 8) | at[2 + i];
            header = 10;
        }
        if (!masked || len > MAX_MESSAGE) { /* clients mask every frame */
            send_close(s, masked ? 1009 : 1002, masked ? "message too big" : "frames from clients are masked");
            return false;
        }
        if (size < header + 4 + len)
            return true;
        const unsigned char *mask = at + header;
        char *payload = axum_alloc(len + 1);
        for (uint64_t i = 0; i < len; i++)
            payload[i] = (char)(at[header + 4 + i] ^ mask[i % 4]);
        payload[len] = '\0';
        h2o_buffer_consume(&s->sock->input, header + 4 + len);
        input = s->sock->input;
        switch (opcode) {
        case OP_PING:
            send_frame(s, OP_PONG, payload, len, NULL);
            free(payload);
            break;
        case OP_PONG:
            free(payload);
            break;
        case OP_CLOSE:
            s->close_code = len >= 2 ? ((unsigned char)payload[0] << 8) | (unsigned char)payload[1] : 1005;
            if (!s->close_sent)
                send_close(s, s->close_code == 1005 ? 1000 : s->close_code, "");
            free(payload);
            s->closed = true;
            deliver(s, OP_CLOSE, axum_strdup(""), 0);
            return true; /* the close frame's write closes the connection */
        case OP_TEXT:
        case OP_BINARY:
        case OP_CONTINUATION:
            if (opcode != OP_CONTINUATION) {
                free(s->fragments);
                s->fragments = NULL;
                s->fragments_len = 0;
                s->fragments_kind = opcode;
            }
            if (fin && s->fragments == NULL) {
                deliver(s, opcode, payload, len);
                break;
            }
            s->fragments = realloc(s->fragments, s->fragments_len + len + 1);
            memcpy(s->fragments + s->fragments_len, payload, len);
            s->fragments_len += len;
            s->fragments[s->fragments_len] = '\0';
            free(payload);
            if (fin) {
                deliver(s, s->fragments_kind, s->fragments, s->fragments_len);
                s->fragments = NULL;
                s->fragments_len = 0;
            }
            break;
        default:
            free(payload);
            send_close(s, 1002, "unknown opcode");
            return false;
        }
    }
}

static void on_read(h2o_socket_t *sock, const char *err)
{
    Socket *s = sock->data;
    if (s == NULL)
        return;
    if (err != NULL) {
        gone(s);
        return;
    }
    if (!read_frames(s))
        s->closed = true;
}

static void on_upgraded(void *data, h2o_socket_t *sock, size_t reqsize)
{
    Socket *s = data;
    if (sock == NULL) {
        s->closed = true;
        wake(s);
        return;
    }
    s->sock = sock;
    sock->data = s;
    h2o_buffer_consume(&sock->input, reqsize);
    write_next(s); /* what was sent before the upgrade was done */
    if (sock->input->size > 0)
        on_read(sock, NULL);
    if (s->sock != NULL)
        h2o_socket_read_start(sock, on_read);
}

/* Why `req` is not a websocket handshake (axum's rejection), or "" if it is. */
const char *axum_ws_rejection(AxumRequest request)
{
    Request *r = request.raw;
    if (strcmp(r->method, "GET") != 0)
        return "Request method must be `GET`";
    if (r->req != NULL && r->req->version >= 0x200)
        return "WebSockets over HTTP/2 are not supported here; connect with HTTP/1.1";
    const char *connection = NULL, *upgrade = NULL, *version = NULL, *key = NULL;
    for (size_t i = 0; i < r->headers.count; i++) {
        const Header *h = r->headers.items + i;
        if (strcmp(h->name, "connection") == 0)
            connection = h->value;
        else if (strcmp(h->name, "upgrade") == 0)
            upgrade = h->value;
        else if (strcmp(h->name, "sec-websocket-version") == 0)
            version = h->value;
        else if (strcmp(h->name, "sec-websocket-key") == 0)
            key = h->value;
    }
    if (connection == NULL || strcasestr(connection, "upgrade") == NULL)
        return "Connection header did not include 'upgrade'";
    if (upgrade == NULL || strcasecmp(upgrade, "websocket") != 0)
        return "`Upgrade` header did not include 'websocket'";
    if (version == NULL || strcmp(version, "13") != 0)
        return "`Sec-WebSocket-Version` header did not include '13'";
    if (key == NULL)
        return "`Sec-WebSocket-Key` header missing";
    return "";
}

/* Accept the handshake: the 101 goes out, and the socket is the request's from now on. */
AxumSocket axum_ws_accept(AxumRequest request)
{
    Request *r = request.raw;
    Socket *s = axum_zalloc(sizeof(Socket));
    s->r = r;
    axum_on_release(r, socket_free, s);
    axum_mark_streaming(r);
    h2o_req_t *req = r->req;
    const char *key = NULL;
    for (size_t i = 0; i < r->headers.count; i++)
        if (strcmp(r->headers.items[i].name, "sec-websocket-key") == 0)
            key = r->headers.items[i].value;
    if (req == NULL || key == NULL) {
        s->closed = true;
        return (AxumSocket){s};
    }
    char source[128];
    unsigned char digest[SHA_DIGEST_LENGTH];
    snprintf(source, sizeof source, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
    SHA1((const unsigned char *)source, strlen(source), digest);
    char *accept = h2o_mem_alloc_pool(&req->pool, char, 32);
    h2o_base64_encode(accept, digest, sizeof digest, 0);
    req->res.status = 101;
    req->res.reason = "Switching Protocols";
    h2o_add_header(&req->pool, &req->res.headers, H2O_TOKEN_UPGRADE, NULL, H2O_STRLIT("websocket"));
    h2o_add_header_by_str(&req->pool, &req->res.headers, H2O_STRLIT("sec-websocket-accept"), 0, NULL, accept, strlen(accept));
    h2o_http1_upgrade(req, NULL, 0, on_upgraded, s);
    return (AxumSocket){s};
}

static Socket *socket_of(AxumSocket socket)
{
    if (socket.raw == NULL) {
        fputs("axum.ws: not a websocket (a zero value)\n", stderr);
        exit(1);
    }
    return socket.raw;
}

/* Wait for the next message: the token completes when one came (ws_kind > 0) or the socket closed. */
void *axum_ws_recv(AxumSocket socket)
{
    Socket *s = socket_of(socket);
    if (s->current != NULL) {
        free(s->current->data);
        free(s->current);
        s->current = NULL;
    }
    void *token = mlirAsyncRuntimeCreateToken();
    if (s->waiting != NULL) {
        fputs("axum.ws_recv: already waiting for a message on this socket\n", stderr);
        mlirAsyncRuntimeEmplaceToken(token);
        return token;
    }
    s->waiting = token;
    wake(s);
    return token;
}

/* Take the message recv waited for: whether there was one (false once the socket has closed). */
bool axum_ws_take(AxumSocket socket)
{
    Socket *s = socket_of(socket);
    if (s->current != NULL) {
        free(s->current->data);
        free(s->current);
        s->current = NULL;
    }
    if (s->head == NULL)
        return false;
    s->current = s->head;
    s->head = s->head->next;
    if (s->head == NULL)
        s->tail = NULL;
    return s->current->kind != OP_CLOSE;
}

/* The message taken: its text (or bytes), its kind (1 text, 2 binary, 8 close), its length. */
const char *axum_ws_text(AxumSocket socket)
{
    Socket *s = socket_of(socket);
    return s->current != NULL ? s->current->data : "";
}

int32_t axum_ws_kind(AxumSocket socket)
{
    Socket *s = socket_of(socket);
    return s->current != NULL ? s->current->kind : 0;
}

int64_t axum_ws_len(AxumSocket socket)
{
    Socket *s = socket_of(socket);
    return s->current != NULL ? (int64_t)s->current->len : 0;
}

/* The close code the client sent (1000 normal, 1001 going away, ...; 1005 none; 0 not closed). */
int32_t axum_ws_close_code(AxumSocket socket)
{
    return socket_of(socket)->close_code;
}

bool axum_ws_closed(AxumSocket socket)
{
    return socket_of(socket)->closed;
}

static void *send_kind(AxumSocket socket, int opcode, const char *data, size_t len)
{
    Socket *s = socket_of(socket);
    void *token = mlirAsyncRuntimeCreateToken();
    if (s->closed && s->sock == NULL) {
        mlirAsyncRuntimeEmplaceToken(token);
        return token;
    }
    send_frame(s, opcode, data, len, token);
    return token;
}

void *axum_ws_send(AxumSocket socket, const char *text) { return send_kind(socket, OP_TEXT, text, strlen(text)); }

void *axum_ws_send_binary(AxumSocket socket, const char *data, int64_t len)
{
    return send_kind(socket, OP_BINARY, data, len > 0 ? (size_t)len : 0);
}

void *axum_ws_ping(AxumSocket socket, const char *data) { return send_kind(socket, OP_PING, data, strlen(data)); }

/* Close the socket with a code (1000: normal) and a reason. */
void axum_ws_close(AxumSocket socket, int32_t code, const char *reason)
{
    Socket *s = socket_of(socket);
    if (!s->close_sent && !(s->closed && s->sock == NULL))
        send_close(s, code, reason);
}

/* What a websocket handler returns: closes the socket (if it is open), and is not sent. */
AxumResponse axum_ws_done(AxumSocket socket)
{
    axum_ws_close(socket, 1000, "");
    return axum_response(101);
}

static void socket_free(void *data)
{
    Socket *s = data;
    if (s->sock != NULL && (s->out_head != NULL || s->writing != NULL)) {
        s->orphan = true; /* frames still going out (the close): freed once they are written */
        if (s->waiting != NULL) {
            mlirAsyncRuntimeEmplaceToken(s->waiting);
            s->waiting = NULL;
        }
        return;
    }
    if (s->sock != NULL) {
        s->sock->data = NULL;
        h2o_socket_close(s->sock);
        s->sock = NULL;
    }
    free_now(s);
}

static void free_now(Socket *s)
{
    if (s->waiting != NULL)
        mlirAsyncRuntimeEmplaceToken(s->waiting);
    finish_frames(s);
    if (s->writing != NULL) {
        free(s->writing->data);
        free(s->writing);
    }
    while (s->head != NULL) {
        Message *next = s->head->next;
        free(s->head->data);
        free(s->head);
        s->head = next;
    }
    if (s->current != NULL) {
        free(s->current->data);
        free(s->current);
    }
    free(s->fragments);
    free(s);
}
