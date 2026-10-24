"""The axum package (extras/axum): a server built from it, driven over HTTP.

Needs the package's C side built (``cmake -B build -G Ninja && cmake --build build`` in
extras/axum); skipped otherwise.
"""

import http.client
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from bifrost.configs import ConfigBuilder, source_root
from bifrost.lowering import lower_file
from bifrost.project import Project

AXUM = Path(__file__).parents[1] / "extras" / "axum"

pytestmark = pytest.mark.skipif(
    not (AXUM / "build" / "lib" / "libbifrost_axum.a").is_file(), reason="extras/axum's C side is not built"
)

SERVER = r"""
let axum = import("axum")
let extract = import("axum.extract:extract")
let json = import("axum.json:json")
let form = import("axum.form:form")
let query = import("axum.query:query")
let response = import("axum.response:response")
let middleware = import("axum.middleware:middleware")
let matched_path = import("axum.matched_path:matched_path")
let original_uri = import("axum.original_uri:original_uri")
let connect_info = import("axum.connect_info:connect_info")
let sse = import("axum.sse:sse")
let ws = import("axum.ws:ws")
let multipart = import("axum.multipart:multipart")
let mem = import("std:mem")
let io = import("std:stdio")

let Counter = struct { let hits: i64 }
let Login = struct { let user: mem.Unique[str], let remember: bool }
let Page = struct { let page: i64 }

let hit = [axum.text] (counter: mem.Shared[Counter]) => axum.Response {
    let c <- counter
    c.hits = c.hits + 1
    let n = c.hits
    c -> counter
    return axum.text(200, "hits {n}")
}

let whoami = [
    axum.json
] (path: matched_path.MatchedPath, uri: original_uri.OriginalUri, inner: extract.Uri, nested: extract.NestedPath, method: extract.Method) => axum.Response {
    return axum.json(200, #{
        matched: path.value,
        original: uri.value,
        uri: inner.path,
        nested: nested.value,
        method: method.name
    })
}

let peer = [
    axum.text
] (who: connect_info.ConnectInfo, headers: extract.HeaderMap, host: extract.Host) => axum.Response {
    let name = headers.get("x-name")
    let none = headers.contains("x-none")
    let has_host = host.value != ""
    let has_peer = who.addr != ""
    return axum.text(200, "{name} {none} {has_host} {has_peer}")
}

let login = [axum.text] (body: form.Form[Login]) => axum.Response {
    return axum.text(200, "{body.value.user} {body.value.remember}")
}

let pages = [
    axum.text
] (page: query.Query[Page], raw: query.RawQuery) => axum.Response {
    return axum.text(200, "page {page.value.page} of {raw.value}")
}

let echo = [axum.text] (body: extract.Text, ext: extract.Extension) => axum.Response {
    return axum.text(200, "{body.value} [{ext.get("tenant")}]")
}

let wild = [axum.text] (rest: extract.Path[mem.Unique[str]]) => axum.Response axum.text(
    200,
    "rest {rest.value}"
)
let static_route = [axum.text] () => axum.Response axum.text(200, "static")
let service = [axum.text] (method: extract.Method) => axum.Response axum.text(
    200,
    "service {method.name}"
)
let files = [axum.text] (uri: extract.Uri) => axum.Response axum.text(
    200,
    "files {uri.path}"
)
let teapot = [axum.text] () => axum.Response axum.text(418, "only GET here")
let nested_missing = [axum.text] () => axum.Response axum.text(404, "no such api")
let created = [] () => response.WithStatus[response.Text] response.WithStatus(
    status: 201,
    inner: response.Text(body: "made")
)
let moved = [response.Redirect.permanent] () => response.Redirect response.Redirect.permanent(
    "/static"
)

let guard = [
    axum.header,
    axum.text
] (req: axum.Request) => Result[null, axum.Response] {
    if axum.header(req, "authorization") != "secret" {
        return err(axum.text(401, "unauthorized"))
    }
    return ok()
}

let stamp = [axum.set_header] (res: axum.Response) => null {
    axum.set_header(res, "x-stamp", "yes")
}

let around = [
    axum.append_header
] async (req: axum.Request, next: middleware.Next) => axum.Response {
    let res = await next.run(req)
    axum.append_header(res, "x-around", "1")
    return res
}

let events = [
    sse.start,
    sse.send,
    sse.end,
    sse.Event.of,
    sse.Event.named
] async (req: axum.Request) => axum.Response {
    let stream = sse.start(req)
    await sse.send(stream, sse.Event.of("one\ntwo"))
    await sse.send(stream, sse.Event.named("done", "three"))
    return sse.end(stream)
}

let chat = [
    ws.recv,
    ws.text,
    ws.send,
    ws.done
] async (upgrade: ws.WebSocketUpgrade) => axum.Response {
    let socket = upgrade.accept()
    while await ws.recv(socket) {
        let said = ws.text(socket)
        await ws.send(socket, "echo: {said}")
    }
    return ws.done(socket)
}

let upload = [axum.text] (form: multipart.Multipart) => axum.Response {
    let found: mem.Unique[str][] = #[]
    let total = 0
    while form.next_field() {
        let found = #[
            ...found,
            "{form.name()}={form.file_name()}/{form.content_type()}:{form.text()}"
        ]
        let total = total + form.size()
    }
    return axum.text(200, "{found} {total}")
}

let api = [
    axum.router,
    axum.route,
    axum.get,
    axum.fallback,
    whoami,
    nested_missing
] () => axum.Router {
    return axum.router()
        |> axum.route("/who/:id", axum.get(whoami))
        |> axum.fallback(nested_missing)
}

let private = [
    axum.router,
    axum.route,
    axum.get,
    axum.route_layer,
    axum.map_request,
    static_route,
    guard
] () => axum.Router {
    return axum.router()
        |> axum.route("/private", axum.get(static_route))
        |> axum.route_layer(axum.map_request(guard))
}

let extra = [axum.router, axum.route, axum.get, static_route] () => axum.Router axum.router()
    |> axum.route("/extra", axum.get(static_route))

let app = [
    axum.router,
    axum.route,
    axum.get,
    axum.post,
    axum.method_fallback,
    axum.route_service,
    axum.nest,
    axum.nest_service,
    axum.merge,
    axum.layer,
    axum.map_response,
    axum.from_fn,
    axum.body_limit,
    axum.default_header,
    axum.extension,
    axum.with_state,
    axum.get_service,
    axum.any_service,
    axum.fallback_service,
    axum.serve_file,
    axum.serve_file_with_mime,
    api,
    extra,
    hit,
    peer,
    login,
    pages,
    echo,
    wild,
    static_route,
    service,
    files,
    teapot,
    created,
    moved,
    stamp,
    around,
    events,
    private,
    chat,
    upload
] (counter: mem.Shared[Counter]) => axum.Router {
    let open = axum.router()
        |> axum.route("/hit", axum.get(hit))
        |> axum.route("/peer", axum.get(peer))
        |> axum.route("/login", axum.post(login))
        |> axum.route("/pages", axum.get(pages))
        |> axum.route("/echo", axum.post(echo) |> axum.method_fallback(teapot))
        |> axum.route("/wild/*rest", axum.get(wild))
        |> axum.route("/wild/static", axum.get(static_route))
        |> axum.route("/created", axum.get(created))
        |> axum.route("/moved", axum.get(moved))
        |> axum.route("/static", axum.get(static_route))
        |> axum.route("/events", axum.get(events))
        |> axum.route_service("/service", service)
        |> axum.nest("/api", api())
        |> axum.nest_service("/files", files)
        |> axum.merge(extra())
        |> axum.layer(axum.body_limit(32))
        |> axum.layer(axum.extension("tenant", "acme"))
        |> axum.layer(axum.map_response(stamp))
    return open
        |> axum.route("/chat", axum.get(chat))
        |> axum.route("/upload", axum.post(upload))
        |> axum.route("/page", axum.get_service(axum.serve_file("page.html")))
        |> axum.route("/logo", axum.get_service(axum.serve_file("logo.png")))
        |> axum.route("/notes", axum.any_service(axum.serve_file_with_mime("notes.md", "text/markdown; charset=utf-8")))
        |> axum.route("/gone", axum.get_service(axum.serve_file("missing.txt")))
        |> axum.nest("/docs", axum.router() |> axum.fallback_service(axum.serve_file("page.html")))
        |> axum.merge(private())
        |> axum.layer(axum.from_fn(around))
        |> axum.layer(axum.default_header("x-server", "bifrost"))
        |> axum.with_state(counter)
}

let stop = [axum.shutdown_signal] async () => null {
    await axum.shutdown_signal()
}

let main = [
    app,
    axum.bind,
    axum.listener_error,
    axum.serve_with_graceful_shutdown,
    io.puts,
    stop
] async () => i32 {
    let counter: mem.Shared[Counter] = Counter(hits: 0)
    let listener = axum.bind("127.0.0.1:PORT")
    if axum.listener_error(listener) != "" {
        io.puts(axum.listener_error(listener))
        return 1
    }
    await axum.serve_with_graceful_shutdown(listener, app(counter), stop)
    io.puts("stopped")
    return 0
}
"""  # noqa: E501 - Bifrost's formatter keeps long parameter lists and match arms whole

CONFIG = f"""
package:
  name: axum_test
  version: 0.1.0
  description: tests axum
  entry: src/axum_test/main.bif
  type: async
flags:
  optimization: 0
  linker: clang
packages:
  axum:
    path: {AXUM}
    features: [http2, ws, multipart]
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


PAGE = "<!doctype html><h1>served</h1>\n"
LOGO = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # bytes past a NUL: not text


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):
    folder = tmp_path_factory.mktemp("axum")
    (folder / "src" / "axum_test").mkdir(parents=True)
    port = _free_port()
    (folder / "config.yaml").write_text(CONFIG)
    (folder / "src" / "axum_test" / "main.bif").write_text(SERVER.replace("PORT", str(port)))
    config = ConfigBuilder(folder / "config.yaml").build()
    config.path = folder / "build"
    config.path.mkdir()
    project = Project(config)
    unit = lower_file(project, folder / "src" / "axum_test" / "main.bif", root=source_root(folder))
    with unit.errors():
        executable = project.build()
    (folder / "page.html").write_text(PAGE)
    (folder / "notes.md").write_text("# notes\n")
    (folder / "logo.png").write_bytes(LOGO)
    process = subprocess.Popen(  # noqa: S603 - served files are read from where it runs
        [executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=folder
    )
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    yield port, process
    if process.poll() is None:
        process.kill()


def _request(port: int, method: str, path: str, body: str | None = None, headers: dict[str, str] | None = None):  # noqa: ANN202
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    answer = connection.getresponse()
    text = answer.read().decode()
    connection.close()
    return answer.status, dict(answer.getheaders()), text


@pytest.mark.parametrize(
    ("method", "path", "body", "headers", "status", "text"),
    [
        ("GET", "/hit", None, {}, 200, "hits 1"),
        ("GET", "/hit", None, {}, 200, "hits 2"),
        ("GET", "/peer", None, {"x-name": "ada"}, 200, "ada false true true"),
        (
            "POST",
            "/login",
            "user=ada+l&remember=true",
            {"content-type": "application/x-www-form-urlencoded"},
            200,
            "ada l true",
        ),
        (
            "POST",
            "/login",
            "user=ada",
            {"content-type": "application/x-www-form-urlencoded"},
            422,
            "Failed to deserialize form body: missing field `remember`",
        ),
        (
            "POST",
            "/login",
            "user=ada",
            {},
            415,
            "Form requests must have `Content-Type: application/x-www-form-urlencoded`",
        ),
        ("GET", "/pages?page=3", None, {}, 200, "page 3 of page=3"),
        ("POST", "/echo", "hello", {}, 200, "hello [acme]"),
        ("POST", "/echo", "x" * 40, {}, 413, "length limit exceeded"),
        ("PUT", "/echo", None, {}, 418, "only GET here"),
        ("GET", "/wild/a/b%2Bc", None, {}, 200, "rest a/b+c"),
        ("GET", "/wild/static", None, {}, 200, "static"),
        ("GET", "/created", None, {}, 201, "made"),
        ("DELETE", "/service", None, {}, 200, "service DELETE"),
        ("GET", "/files/a/b.txt", None, {}, 200, "files /a/b.txt"),
        ("GET", "/extra", None, {}, 200, "static"),
        ("GET", "/api/nope", None, {}, 404, "no such api"),
        ("GET", "/private", None, {}, 401, "unauthorized"),
        ("GET", "/private", None, {"authorization": "secret"}, 200, "static"),
        ("GET", "/nowhere", None, {}, 404, ""),
    ],
)
def test_routes(server, *, method, path, body, headers, status, text) -> None:  # noqa: ANN001, PLR0913
    found = _request(server[0], method, path, body, headers)
    assert (found[0], found[2]) == (status, text)


def test_extractors_of_a_nested_route(server) -> None:  # noqa: ANN001
    status, _, text = _request(server[0], "GET", "/api/who/7?x=1")
    assert status == 200
    assert (
        text == '{"matched": "/api/who/:id", "original": "/api/who/7?x=1", "uri": "/who/7", "nested": "/api", '
        '"method": "GET"}'
    )


def test_layers(server) -> None:  # noqa: ANN001
    # layer wraps what was added before it (map_response: x-stamp), the last added runs first;
    # route_layer runs only for matched routes (the 404 is not guarded); default_header fills in.
    status, headers, _ = _request(server[0], "GET", "/static")
    assert status == 200
    assert headers["x-stamp"] == "yes"
    assert headers["x-around"] == "1"
    assert headers["x-server"] == "bifrost"
    status, headers, _ = _request(server[0], "GET", "/private", headers={"authorization": "secret"})
    assert "x-stamp" not in headers  # added after the map_response layer
    assert headers["x-around"] == "1"


def test_method_not_allowed_and_head(server) -> None:  # noqa: ANN001
    status, headers, _ = _request(server[0], "DELETE", "/static")
    assert status == 405
    assert headers["allow"] == "GET,HEAD"
    status, _, text = _request(server[0], "HEAD", "/static")
    assert (status, text) == (200, "")


def test_redirect(server) -> None:  # noqa: ANN001
    status, headers, _ = _request(server[0], "GET", "/moved")
    assert (status, headers["location"]) == (308, "/static")


def test_server_sent_events(server) -> None:  # noqa: ANN001
    status, headers, text = _request(server[0], "GET", "/events")
    assert status == 200
    assert headers["content-type"] == "text/event-stream"
    assert text == "data: one\ndata: two\n\nevent: done\ndata: three\n\n"


def test_multipart(server) -> None:  # noqa: ANN001
    boundary = "XyZ"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="title"\r\n\r\nhello\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\nline 1\nline 2\r\n--{boundary}--\r\n"
    )
    headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
    status, _, text = _request(server[0], "POST", "/upload", body, headers)
    assert (status, text) == (200, '["title=/:hello", "file=a.txt/text/plain:line 1\\nline 2"] 18')
    status, _, text = _request(server[0], "POST", "/upload", "x", {"content-type": "text/plain"})
    assert (status, text) == (400, "Invalid `boundary` for `multipart/form-data` request")


def _frame(opcode: int, payload: bytes) -> bytes:
    """Make a client's frame: masked, as RFC 6455 asks."""
    mask = b"\x01\x02\x03\x04"
    head = bytes([0x80 | opcode, 0x80 | len(payload)])
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _read_frame(conn: socket.socket) -> tuple[int, bytes]:
    head = conn.recv(2)
    length = head[1] & 0x7F
    data = b""
    while len(data) < length:
        data += conn.recv(length - len(data))
    return head[0] & 0x0F, data


def test_websocket(server) -> None:  # noqa: ANN001
    port = server[0]
    conn = socket.create_connection(("127.0.0.1", port), timeout=5)
    conn.sendall(
        b"GET /chat HTTP/1.1\r\nHost: x\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n"
        b"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
    )
    head = b""
    while b"\r\n\r\n" not in head:
        head += conn.recv(1)
    assert head.startswith(b"HTTP/1.1 101")
    assert b"s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in head  # RFC 6455's example key and answer
    conn.sendall(_frame(1, b"hi"))
    assert _read_frame(conn) == (1, b"echo: hi")
    conn.sendall(_frame(9, b"p"))  # a ping is answered with a pong
    assert _read_frame(conn) == (10, b"p")
    conn.sendall(_frame(1, b"again"))
    assert _read_frame(conn) == (1, b"echo: again")
    conn.sendall(_frame(8, b"\x03\xe8"))  # close (1000): the server closes too
    assert _read_frame(conn)[0] == 8
    conn.close()
    status, _, text = _request(port, "GET", "/chat")
    assert (status, text) == (400, "Connection header did not include 'upgrade'")


def _raw(port: int, method: str, path: str, headers: dict[str, str] | None = None):  # noqa: ANN202
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, headers=headers or {})
    answer = connection.getresponse()
    body = answer.read()
    connection.close()
    return answer.status, {name.lower(): value for name, value in answer.getheaders()}, body


def test_serve_file(server) -> None:  # noqa: ANN001
    port = server[0]
    status, headers, body = _raw(port, "GET", "/page")
    assert (status, body.decode()) == (200, PAGE)
    assert headers["content-type"] == "text/html"
    assert headers["accept-ranges"] == "bytes"
    assert headers["x-server"] == "bifrost"  # the router's layers wrap a service too
    modified = headers["last-modified"]
    assert modified.endswith(" GMT")
    assert _raw(port, "GET", "/logo") == (200, _raw(port, "GET", "/logo")[1], LOGO)
    assert _raw(port, "GET", "/logo")[1]["content-type"] == "image/png"
    assert _raw(port, "GET", "/notes")[1]["content-type"] == "text/markdown; charset=utf-8"
    assert _raw(port, "GET", "/docs/anything")[2].decode() == PAGE  # fallback_service
    assert _raw(port, "GET", "/gone")[0] == 404


def test_serve_file_conditions_and_ranges(server) -> None:  # noqa: ANN001
    port = server[0]
    modified = _raw(port, "GET", "/page")[1]["last-modified"]
    assert _raw(port, "GET", "/page", {"if-modified-since": modified})[::2] == (304, b"")
    old = "Sun, 06 Nov 1994 08:49:37 GMT"
    assert _raw(port, "GET", "/page", {"if-modified-since": old})[0] == 200
    assert _raw(port, "GET", "/page", {"if-unmodified-since": old})[0] == 412
    status, headers, body = _raw(port, "GET", "/page", {"range": "bytes=2-8"})
    assert (status, body, headers["content-range"]) == (206, PAGE.encode()[2:9], f"bytes 2-8/{len(PAGE)}")
    assert _raw(port, "GET", "/page", {"range": "bytes=-6"})[2] == PAGE.encode()[-6:]
    assert _raw(port, "GET", "/page", {"range": "bytes=10-"})[2] == PAGE.encode()[10:]
    status, headers, _ = _raw(port, "GET", "/page", {"range": "bytes=9999-"})
    assert (status, headers["content-range"]) == (416, f"bytes */{len(PAGE)}")
    assert _raw(port, "GET", "/page", {"range": "bytes=0-1,4-5"})[::2] == (200, PAGE.encode())  # several: all


def test_serve_file_methods(server) -> None:  # noqa: ANN001
    port = server[0]
    status, headers, body = _raw(port, "HEAD", "/page")
    assert (status, body) == (200, b"")
    assert headers["content-type"] == "text/html"
    assert _raw(port, "POST", "/page")[0] == 405  # get_service: the router's 405
    status, headers, _ = _raw(port, "POST", "/notes")  # any_service: the file's own
    assert (status, headers["allow"]) == (405, "GET,HEAD")


def test_graceful_shutdown(server) -> None:  # noqa: ANN001
    _, process = server
    process.send_signal(signal.SIGINT)
    out, _ = process.communicate(timeout=10)
    assert process.returncode == 0
    assert out.splitlines()[-1] == "stopped"
