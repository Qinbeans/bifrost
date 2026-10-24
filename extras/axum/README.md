# axum

[axum](https://docs.rs/axum/latest/axum/) for Bifrost: routers of handlers that take extractors, middleware written as functions, and a server, on [h2o](https://github.com/h2o/h2o) (HTTP/1.1, HTTP/2, HTTP/3). It is a Bifrost [package](../../README.md#packages); [`examples/axum`](../../examples/axum) is an API built with it.

```bifrost
let axum = import("axum")
let extract = import("axum.extract:extract")
let json = import("axum.json:json")
let mem = import("std:mem")

let User = struct { let id: i64, let name: mem.Unique[str] }
let NewUser = struct { let name: mem.Unique[str] }

let root = [axum.text] () => axum.Response axum.text(200, "Hello, World!")

let show = [axum.json] (id: extract.Path[i64]) => axum.Response axum.json(200, #{id: id.value})

let create = [] (body: json.Json[NewUser]) => json.Json[User] {
    return json.Json(value: User(id: 1337, name: copy(body.value.name)))
}

let app = [axum.router, axum.route, axum.get, axum.post, root, show, create] () => axum.Router {
    return axum.router()
        |> axum.route("/", axum.get(root))
        |> axum.route("/users", axum.post(create))
        |> axum.route("/users/:id", axum.get(show))
}

let main = [app, axum.bind, axum.serve] async () => null {
    await axum.serve(axum.bind("0.0.0.0:3000"), app())
}
```

```yaml
# config.yaml
package:
  type: async            # axum.serve is awaited: main is async
packages:
  axum: 0.3.0            # or {path: ../extras/axum}, or {version: 0.3.0, features: [ws]}
```

## How axum's API reads here

Rust chains methods on values; Bifrost pipes a value into a call as its first argument, so `Router::new().route(...).layer(...)` is `axum.router() |> axum.route(...) |> axum.layer(...)`. The router, method routers and layers are handles (`axum.Router`, `axum.MethodRouter`, `axum.Layer`): each call changes the one value and gives it back.

| axum | Bifrost |
|---|---|
| `Router::new()` | `axum.router()` |
| `.route("/users/{id}", get(h))` | `\|> axum.route("/users/:id", axum.get(h))` |
| `get(a).post(b)` | `axum.get(a) \|> axum.also(axum.post(b))` (or `route` the path twice) |
| `get`, `post`, `put`, `delete`, `patch`, `head`, `options`, `trace`, `connect`, `any`, `on(MethodFilter, h)` | `axum.get(h)`, ..., `axum.any(h)`, `axum.on("GET,POST", h)` |
| `MethodRouter::fallback`, `::layer`, `::route_layer` | `axum.method_fallback`, `axum.method_layer`, `axum.method_route_layer` |
| `.route_service(path, s)` | `axum.route_service(path, h)` (a handler for every method) |
| `get_service(s)`, `any_service(s)`, `on_service(filter, s)`, `.fallback_service(s)` | `axum.get_service(s)`, `axum.any_service(s)`, `axum.on_service("GET,POST", s)`, `axum.fallback_service(s)`: route an `axum.Service` (see [Serving files](#serving-files)) |
| `.nest(path, router)`, `.nest_service(path, s)` | `axum.nest`, `axum.nest_service` |
| `.merge(other)` | `axum.merge(other)` |
| `.layer(l)`, `.route_layer(l)` | `axum.layer(l)`, `axum.route_layer(l)` |
| `.fallback(h)`, `.method_not_allowed_fallback(h)`, `.reset_fallback()` | `axum.fallback`, `axum.method_not_allowed_fallback`, `axum.reset_fallback` |
| `.with_state(state)` | `axum.with_state(state)`: a `mem.Shared` or `mem.Atomic` main owns |
| `axum::serve(listener, app)` | `await axum.serve(axum.bind("0.0.0.0:3000"), app)` |
| `.with_graceful_shutdown(signal)` | `await axum.serve_with_graceful_shutdown(listener, app, signal)`; `axum.shutdown_signal()` completes on Ctrl+C or SIGTERM |
| `tokio::net::TcpListener::bind`, `local_addr` | `axum.bind(address)` (`axum.listener_error` says if it failed), `axum.local_addr`, `axum.local_port`, `axum.tcp_nodelay` |

Paths: `:id` captures a segment and `*rest` the rest of the path (axum 0.7's syntax). axum 0.8's `{id}` and `{*rest}` work too, but Bifrost strings interpolate braces, so they are written `"/users/{{id}}"`. A static segment wins over a capture, and a capture over a wildcard. A path that matches with another method is a 405 with `Allow`; `HEAD` is answered by `GET`'s handler without its body.

### Handlers and extractors

A handler is any function. A parameter that takes one is declared `handler(axum_Request) => axum_Response` in `config.yaml`, and for each function passed there the compiler writes the adapter that gives its parameters their values, by their types, as axum's extractors do:

| A parameter of type | is given |
|---|---|
| `axum.Request` | the request (read with `axum.method`, `axum.header`, `axum.body`, ...) |
| `mem.Shared[T]` or `mem.Atomic[T]` | the state (axum's `State`) |
| an object with `static let from_request = (req: axum.Request) => X` | what it returns: an extractor. Returning `Result[X, axum.Response]`, its error is the response, and the handler is not called (a rejection) |
| `Result[X, axum.Response]` | the extractor's Result, to look at (axum's `Result<Json<T>, JsonRejection>`) |
| `axum.Response` | in `map_response` middleware, the response so far |

An extractor may pause (`async`): it is awaited. What the handler returns is the response: an `axum.Response` (`axum.text`, `axum.html`, `axum.json`, `axum.redirect`, `axum.response`), an object with an `into_response` method, a `Result` of those (whichever side it holds), or nothing (200, empty). A handler that pauses is awaited; other requests are served meanwhile.

The extractors:

| axum | Bifrost | |
|---|---|---|
| `Path<T>` | `extract.Path[T]` | `T` a number or string for one capture, a record or object for several; 400 when they do not fit |
| `RawPathParams` | `extract.RawPathParams` | the captures as a form, `id=7&name=x` |
| `Query<T>`, `RawQuery` | `query.Query[T]`, `query.RawQuery` | `query` feature |
| `Json<T>` | `json.Json[T]` | `json` feature: 415 without `Content-Type: application/json`, 400 when the body is not JSON, 422 when it does not fit `T` |
| `Form<T>`, `RawForm` | `form.Form[T]`, `form.RawForm` | `form` feature: a POST's body, or a GET's query |
| `State<S>` | a `mem.Shared[S]` or `mem.Atomic[S]` parameter | |
| `Extension<T>` | `extract.Extension` (`ext.get("key")`), given by the `axum.extension(key, value)` layer | values are text |
| `HeaderMap`, `Method`, `Uri`, `Version` | `extract.HeaderMap` (`get`, `contains`, `count`, `name(i)`, `value(i)`), `extract.Method`, `extract.Uri`, `extract.Version` | |
| `String`, `Bytes` | `extract.Text` | the body |
| `Host` (axum-extra) | `extract.Host` | |
| `NestedPath` | `extract.NestedPath` | |
| `MatchedPath` | `matched_path.MatchedPath` | `matched-path` feature |
| `OriginalUri` | `original_uri.OriginalUri` | `original-uri` feature |
| `ConnectInfo<SocketAddr>` | `connect_info.ConnectInfo` | `tokio` feature |
| `Multipart` | `multipart.Multipart` (`while form.next_field() { form.name() ... }`) | `multipart` feature |
| `WebSocketUpgrade` | `ws.WebSocketUpgrade` | `ws` feature (HTTP/1.1) |
| `Request` | `axum.Request` | |

`T` is decoded by the compiler from its fields (`json.decode` and `form.decode`): numbers, bools, owned strings (`mem.Unique[str]`), and, from JSON, lists and nested records and objects. Strings a request carries are lent to the handler for the request, as `str`; strings decoded are new, so a type decoded holds `mem.Unique[str]`.

### Responses

`axum.response` (the module, `import("axum.response:response")`) has axum's `IntoResponse` types as objects: `Html(body: ...)`, `Text(body: ...)` (axum's `Html<T>`: `Html[str]` lends a literal, `Html[mem.Unique[str]]` owns text made at run time, like a file's), `Redirect.to(uri)` (303) / `.temporary` (307) / `.permanent` (308), `StatusCode(code: 404)`, `NoContent.new()`, `WithStatus(status: 201, inner: json.Json(value: user))` (axum's `(StatusCode, T)`), `WithHeader(name: ..., value: ..., inner: ...)` (`AppendHeaders`), and constants (`response.NOT_FOUND`). An `axum.Response` is a handle Bifrost owns: it is freed where its owner's scope ends unless it is returned, and `axum.set_status`, `axum.set_header`, `axum.append_header`, `axum.remove_header` and `axum.set_body` change it.

Your own error type is an object with `into_response`, and a handler returning `Result[T, AppError]` answers with either side, as in axum:

```bifrost
let AppError = struct {
    let status: i32,
    let message: str,
    let into_response = [super, axum.json] () => axum.Response axum.json(super.status, #{error: super.message})
}

let show = [find] (state: mem.Shared[AppState], id: extract.Path[i64]) => Result[json.Json[User], AppError] { ... }
```

### Middleware

| axum | Bifrost |
|---|---|
| `middleware::from_fn(f)`, `from_fn_with_state` | `axum.from_fn(f)`, `axum.from_fn_with_state(state, f)`: `f` takes extractors, the request and a `middleware.Next`, and runs the rest with `await next.run(req)` |
| `map_request(f)`, `_with_state` | `axum.map_request(f)`: `f` changes the request in place (`axum.set_request_header`, `axum.set_method`, ...) and returns nothing to go on, or a response (or `Result[null, axum.Response]`'s error) to stop |
| `map_response(f)`, `_with_state` | `axum.map_response(f)`: `f` takes an `axum.Response` (the response so far) and changes it in place, or returns another |
| `from_extractor::<E>()` | `axum.from_extractor(f)`: `f`'s parameters are extracted, and a rejection stops the request |
| `DefaultBodyLimit::max(n)`, `::disable()` | `axum.body_limit(n)` (2 MB by default), `axum.disable_body_limit()` |
| `Extension(value)` as a layer | `axum.extension(key, value)` |
| tower-http's `SetResponseHeader::if_not_present` | `axum.default_header(name, value)` |
| tower-http's `ServeFile::new(path)`, `::new_with_mime` | `axum.serve_file(path)`, `axum.serve_file_with_mime(path, mime)` (see [Serving files](#serving-files)) |

```bifrost
let timing = [time.now, axum.append_header] async (req: axum.Request, next: middleware.Next) => axum.Response {
    let started = time.now()
    let res = await next.run(req)
    axum.append_header(res, "x-response-time", "{time.now() - started}ms")
    return res
}

let auth = [axum.header, axum.text] (req: axum.Request) => Result[null, axum.Response] {
    if axum.header(req, "authorization") != "Bearer secret" {
        return err(axum.text(401, "unauthorized"))
    }
    return ok()
}

app |> axum.route_layer(axum.map_request(auth)) |> axum.layer(axum.from_fn(timing))
```

`layer` wraps the routes added so far (and the fallback); the layer added last runs first. `route_layer` wraps only the routes, and only once one matches, so a 404 is not, say, a 401.

### Streaming: server-sent events, bodies, websockets

A handler that streams sends the response's head at once, then writes, awaiting each write (a slow client slows the writer down):

```bifrost
let ticks = [sse.start, sse.send, sse.keep_alive, sse.end, sse.Event.of, time.sleep] async (req: axum.Request) => axum.Response {
    let stream = sse.start(req)
    sse.keep_alive(stream, 15000)
    forall i in #[1...4] {
        await sse.send(stream, sse.Event.of("tick {i}"))
        await time.sleep(1000)
    }
    return sse.end(stream)
}

let echo = [ws.recv, ws.text, ws.send, ws.done] async (upgrade: ws.WebSocketUpgrade) => axum.Response {
    let socket = upgrade.accept()
    while await ws.recv(socket) {
        await ws.send(socket, "echo: {ws.text(socket)}")
    }
    return ws.done(socket)
}
```

`axum.stream_open(req, status, content_type)` and `axum.stream_write` stream any body (`Body::from_stream`). Since the head goes out before the handler returns, layers that change responses do not change a streamed one.

### Serving files

`axum.serve_file(path)` is tower-http's `ServeFile`: an `axum.Service` answering with one file, routed with `axum.get_service`, `axum.any_service` or `axum.fallback_service`:

```bifrost
axum.router()
    |> axum.route("/", axum.get_service(axum.serve_file("index.html")))
    |> axum.route("/favicon.ico", axum.get_service(axum.serve_file("static/favicon.ico")))
    |> axum.fallback_service(axum.serve_file("404.html"))
```

It answers GET and HEAD with the file, its type from its extension (`text/html`, `image/png`, ..., or `application/octet-stream`; `serve_file_with_mime` gives one), `Last-Modified` and `Accept-Ranges`. `If-Modified-Since` gets a 304 when the file has not changed and `If-Unmodified-Since` a 412 when it has; one `Range: bytes=...` gets a 206 of that part (a 416 past the end; several ranges get the whole file). A missing file is a 404, another method a 405. The path is relative to where the server runs, and the file is read on each request (whole, into memory), so a change shows at once. Layers wrap a service as they wrap handlers. To make a page from a file's text instead, read it with `std:fs` in a handler and return it as a `response.Html[mem.Unique[str]]`.

## Features

As axum's, chosen where a project lists the package (`features: [ws]`, `default-features: false`):

| Feature | Default | What it brings |
|---|---|---|
| `json` | yes | `axum.json`: `Json[T]` |
| `form` | yes | `axum.form`: `Form[T]`, `RawForm` |
| `query` | yes | `axum.query`: `Query[T]`, `RawQuery` |
| `matched-path` | yes | `axum.matched_path`: `MatchedPath` (the router records the matched pattern) |
| `original-uri` | yes | `axum.original_uri`: `OriginalUri` |
| `tokio` | yes | `axum.connect_info`: `ConnectInfo` |
| `http1` | yes | HTTP/1.1 (without it, HTTP/1.1 requests are 505s) |
| `http2` | no | HTTP/2: h2c, and h2 over TLS |
| `tracing` | yes | a line on stderr per request: method, URI, status, time, protocol |
| `tower-log`, `macros` | yes, no | nothing more here (Bifrost's compiler checks handlers as `#[debug_handler]` would) |
| `ws` | no | `axum.ws`: websockets (links `libaxum_ws.a`) |
| `multipart` | no | `axum.multipart`: `Multipart` (links `libaxum_multipart.a`) |

## Build and package

h2o needs OpenSSL and zlib (`dnf install openssl-devel zlib-devel`, or `apt install libssl-dev zlib1g-dev`), CMake and a C compiler. Build the C side once; projects using this folder link `build/lib/`:

```bash
cmake -B build -G Ninja && cmake --build build      # h2o (pinned), c/axum.c, and the features' libraries
bfc package                                         # dist/axum-0.3.0-<target>.bifpkg
```

The C side is `c/axum.c` (routers, middleware, requests, responses, the server), `c/stream.c` (streamed bodies), `c/ws.c` and `c/multipart.c` (their features), and `c/features/` (one flag each for the features the C side reads). `tests/test_axum.py` drives a server built from all of it over HTTP.

Not here (yet): tower's own services and layers beyond those above (CORS, compression, timeouts), `Extension` of values other than text, `FromRef` substates, and websockets over HTTP/2.
