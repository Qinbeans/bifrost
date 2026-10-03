# http_server

A fasthttp-style HTTP server for Bifrost, on [h2o](https://github.com/h2o/h2o): HTTP/1.1, HTTP/2 and, with a certificate, HTTP/3. It is a Bifrost [package](../../README.md#packages): `examples/http` uses it.

It gives a project:

- the `http` module (`c/http.c`, declared in `config.yaml`): `http.app()`, routes (`http.get(app, "/users/:id", handler)`, `http.post`, ... and the `*_with` ones that pass shared state), the request (`http.param`, `http.query_int`, `http.body`), responses (`http.text`, `http.json`, `http.error`), and `http.run` / `http.run_with`;
- `http_server.handlers`, Bifrost handlers ready to route to: `handlers.health` (200 "ok").

```yaml
# a project's config.yaml
packages:
  http_server:
    path: ../http-server      # this folder; or `http_server: 0.1.0`, from the index (see the main README)
```

```bifrost
let http = import("http")
let handlers = import("http_server.handlers:handlers")

let main = [http.app, http.get, http.run, handlers.health] () => i32 {
    let app = http.app()
    http.get(app, "/health", handlers.health)
    return http.run(app, 8080)      // $PORT, or 8080; HTTPS and HTTP/3 with $TLS_CERT and $TLS_KEY
}
```

## Build and package

h2o needs OpenSSL and zlib (`dnf install openssl-devel zlib-devel`, or `apt install libssl-dev zlib1g-dev`), CMake and a C compiler. Build the C side once; projects using this folder link `build/lib/`:

```bash
cmake -B build -G Ninja && cmake --build build                         # h2o, fetched at a pinned commit, and c/http.c
bfc package                                                            # dist/http_server-0.1.0-<target>.bifpkg
```
