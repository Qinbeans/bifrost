# http

A JSON API on [h2o](https://github.com/h2o/h2o): HTTP/1.1, HTTP/2 and HTTP/3. Each endpoint's handler has a file of its own in `src/http/api/`; `src/http/main.bif` creates the app and routes to them, fasthttp style:

```bifrost
let app = http.app()
http.get(app, "/hello/:name", hello.hello)     // hello: (ctx: http.Context) => null
http.post(app, "/echo", echo.echo)
return http.run(app, 8080)
```

Handlers that share state, like the `/hits` counter, take it as a `mem.Shared` and reach it through a guard. `main` owns the state; `http.run_with` passes it to the `*_with` handlers, which borrow it for each request. h2o runs every handler on one thread, so `mem.Shared` is enough: with threads, it would be a `mem.Atomic`, whose guard locks a mutex.

```bifrost
// api/hits.bif
let hits = [http.json] (ctx: http.Context, app: mem.Shared[state.AppState]) => null {
    let s <- app
    s.hits = s.hits + 1
    http.json(ctx, 200, #{hits: s.hits})
    s -> app
}

// main.bif
http.get_with(app, "/hits", hits.hits)
let shared: mem.Shared[state.AppState] = state.AppState(hits: 0)
return http.run_with(app, 8080, shared)
```

Handlers respond with records, `http.json(ctx, 200, #{sum: a + b})`: `http.json`'s body is declared `json` in `config.yaml`, so the compiler encodes the record at the call.

`http` is not part of Bifrost: it comes from the [`http_server`](../../extras/http-server) package, which this project's `config.yaml` uses by path. The package's `c/http.c` wraps h2o, and its `config.yaml` declares it for Bifrost (the `http` extern module) and links it; its `http_server.handlers` module has ready-made handlers, like `/health`'s.

## Build and run

h2o needs OpenSSL and zlib (`dnf install openssl-devel zlib-devel`, or `apt install libssl-dev zlib1g-dev`), CMake and a C compiler.

```bash
(cd ../../extras/http-server && cmake -B build -G Ninja && cmake --build build)   # once
bfc build
./build/http                 # http://localhost:8080 ($PORT to change it)
```

For HTTPS and HTTP/3, give it a certificate:

```bash
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 30 \
    -keyout key.pem -out cert.pem -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost"
TLS_CERT=cert.pem TLS_KEY=key.pem ./build/http    # https://localhost:8080, TCP and UDP
```

## Try it

```bash
curl localhost:8080/health                      # ok
curl localhost:8080/hello/World                 # {"message": "hello", "name": "World", "protocol": "HTTP/1.1"}
curl 'localhost:8080/add?a=20&b=22'             # {"sum": 42}
curl localhost:8080/hits                        # {"hits": 1}, then 2, 3, ...
curl 'localhost:8080/slow?ms=500'               # {"slept": 500}, while other requests are served
curl -d 'hi' -H 'Content-Type: text/plain' localhost:8080/echo   # 201 {"received": "hi", ...}
curl -i localhost:8080/echo                     # 405, allow: POST
curl --http2-prior-knowledge localhost:8080/hello/h2c   # "protocol": "HTTP/2"

# With TLS_CERT and TLS_KEY (-k: the certificate is self-signed)
curl -k --http2 https://localhost:8080/hello/two        # "protocol": "HTTP/2"
curl -k --http3-only https://localhost:8080/hello/three # "protocol": "HTTP/3"
```

Each request is logged to stderr: `GET /hello/World -> 200 (0.005 ms, HTTP/2)`.

## Notes

- Handlers run on h2o's event loop, one at a time. An `async` handler that `await`s `http.sleep` (or any call that pauses) gives the loop back there, and other requests are served until it resumes; its response is sent when it finishes. Blocking C calls (`sleep`, slow file reads) still stop the whole server.
- A `Context` is valid until its response is sent; strings read from it live until then.
- A handler that sends no response gets a 500; a path that matches another method's route gets a 405 with `Allow`; `HEAD` is served by `GET` routes.
