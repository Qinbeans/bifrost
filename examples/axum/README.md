# axum_example

A JSON API with [axum](../../extras/axum): a users resource, a search, server-sent events, an async handler, middleware and a graceful shutdown. `src/axum_example/main.bif` builds the routers with pipes, as axum chains methods:

```bifrost
let app = [...] (state: mem.Shared[AppState]) => axum.Router {
    return axum.router()
        |> axum.route("/", axum.get(root))
        |> axum.route("/events", axum.get(ticks))
        |> axum.nest("/api", users())
        |> axum.fallback(not_found)
        |> axum.layer(axum.from_fn(timing))
        |> axum.with_state(state)
}
```

Each handler takes what it needs as parameters, by type: `state: mem.Shared[AppState]` (axum's `State`), `id: extract.Path[i64]`, `body: json.Json[NewUser]`, `found: query.Query[Search]`. Bad input never reaches it: `/api/users/abc` is a 400, a POST without `Content-Type: application/json` a 415, a body without `name` a 422. What it returns is the response: `show_user` returns `Result[json.Json[User], ApiError]`, and `ApiError`'s `into_response` makes the 404.

## Build and run

Build the axum package's C side once (it fetches h2o; see its README), then the example:

```bash
(cd ../../extras/axum && cmake -B build -G Ninja && cmake --build build)
bfc build
./build/axum_example                 # listening on http://0.0.0.0:3000; Ctrl+C finishes requests in flight, then stops
```

```bash
curl localhost:3000/
curl -X POST localhost:3000/api/users -H 'content-type: application/json' -d '{"name": "Ada"}'   # 201 {"id": 1, "name": "Ada"}
curl localhost:3000/api/users                                                                      # [{"id": 1, "name": "Ada"}]
curl localhost:3000/api/users/1
curl -X DELETE localhost:3000/api/users/1                                                          # 204
curl "localhost:3000/api/search?q=hello+world&limit=5"
curl localhost:3000/slow/500          # waits without blocking other requests
curl -N localhost:3000/events         # data: tick 1 ... (server-sent events)
curl --http2-prior-knowledge localhost:3000/    # HTTP/2: the `http2` feature, turned on in config.yaml
```

Every response has an `x-response-time` header (the `timing` middleware), and each request is logged on stderr (the `tracing` feature).
