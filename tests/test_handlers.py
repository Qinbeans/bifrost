"""Handlers: any function passed where C takes a ``handler(Request) => Response`` (axum's extractors).

A small C "server" here stands in for one: it keeps the handlers, then calls each with
a request, waits for its token, and prints the response it was given. Responses are
handles C allocates (``drop:``), which Bifrost owns until it hands one over.
"""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

SERVER = r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { void *raw; } Req;
typedef struct { void *raw; } Res;
typedef struct { int status; char *body; } Response;
typedef void *(*Handler)(Req, void *, Res *);

void mlirAsyncRuntimeAwaitTokenAndExecute(void *token, void *handle, void (*resume)(void *));
void mlirAsyncRuntimeDropRef(void *object, int64_t count);
void bifrost_async_run_ready(void);

static Handler handlers[8];
static int count;
static long live;  /* responses made and not yet freed */

void server_add(Handler handler) { handlers[count++] = handler; }

Res server_text(int32_t status, const char *body)
{
    Response *r = malloc(sizeof(Response));
    r->status = status;
    r->body = strdup(body);
    live++;
    return (Res){r};
}

void server_free(Res res)
{
    Response *r = res.raw;
    if (r == NULL)
        return;
    free(r->body);
    free(r);
    live--;
}

void server_set_status(Res res, int32_t status) { ((Response *)res.raw)->status = status; }
const char *server_body(Req req) { return req.raw; }

static bool done;
static void on_done(void *unused) { (void)unused; done = true; }

void server_run(const char *body, int64_t state)
{
    for (int i = 0; i < count; i++) {
        Res out = {0};
        int64_t shared[3] = {1, 0, state};  /* a mem.Shared cell: its count, whether it is locked, its value */
        void *token = handlers[i]((Req){(void *)body}, &shared[2], &out);
        done = false;
        mlirAsyncRuntimeAwaitTokenAndExecute(token, NULL, on_done);
        while (!done)
            bifrost_async_run_ready();
        mlirAsyncRuntimeDropRef(token, 1);
        Response *r = out.raw;
        if (r == NULL)
            printf("%d: 200 (empty)\n", i);
        else
            printf("%d: %d %s\n", i, r->status, r->body);
        server_free(out);
    }
    printf("live %ld\n", live);
}
"""

EXTERNS = [
    {
        "module": "server",
        "description": "a stand-in for an HTTP server",
        "declarations": [
            {"name": "Req", "type": "struct", "fields": {"raw": "ptr"}},
            {"name": "Res", "type": "struct", "fields": {"raw": "ptr"}, "drop": "server_free"},
            {
                "name": "server_free",
                "as": "free",
                "type": "function",
                "parameters": {"res": "server_Res"},
                "return": "None",
            },
            {
                "name": "server_add",
                "as": "add",
                "type": "function",
                "parameters": {"handler": "handler(server_Req) => server_Res"},
                "return": "None",
            },
            {
                "name": "server_text",
                "as": "text",
                "type": "function",
                "parameters": {"status": "i32", "body": "cstr"},
                "return": "server_Res",
            },
            {
                "name": "server_set_status",
                "as": "set_status",
                "type": "function",
                "parameters": {"res": "server_Res", "status": "i32"},
                "return": "None",
            },
            {
                "name": "server_body",
                "as": "body",
                "type": "function",
                "parameters": {"req": "server_Req"},
                "return": "cstr",
            },
            {
                "name": "server_run",
                "as": "run",
                "type": "function",
                "parameters": {"body": "cstr", "state": "i64"},
                "return": "None",
            },
        ],
    }
]

PROGRAM = r"""
let server = import("server")
let json = import("std:json")
let mem = import("std:mem")
let time = import("std:time")

// Extractors and responses, as a web framework's package would write them.
let Json[T] = struct {
    let value: T,
    static let from_request = [server.body, server.text, json.decode] (req: server.Req) => Result[Json[T], server.Res] {
        let parsed: Result[T, mem.Unique[str]] = json.decode(server.body(req))
        match parsed {
            ok(v): return ok(Json(value: copy(v))),
            err(e): return err(server.text(422, e))
        }
    },
    let into_response = [super, server.text, json.encode] () => server.Res server.text(200, json.encode(super.value))
}

let Body = struct {
    let text: str,
    static let from_request = [server.body] async (req: server.Req) => Body Body(text: server.body(req))
}

let Created = struct {
    let id: i64,
    let into_response = [super, server.text] () => server.Res server.text(201, "created {super.id}")
}

let AppError = struct {
    let message: mem.Unique[str],
    let into_response = [super, server.text] () => server.Res server.text(500, super.message)
}

let User = struct { let name: mem.Unique[str], let age: i64 }

let echo = [] (user: Json[User]) => Json[User] user |> copy

let create = [time.sleep] async (user: Json[User], state: mem.Shared[i64]) => Result[Created, AppError] {
    await time.sleep(1)
    let n <- state
    let id = n + user.value.age
    n -> state
    if user.value.age > 100 {
        return err(AppError(message: "too old: {user.value.age}"))
    }
    return ok(Created(id: id))
}

let peek = [server.text, server.set_status] (body: Body, user: Result[Json[User], server.Res]) => server.Res {
    let res = server.text(200, "body {body.text}, user ok: {user.is_ok()}")
    server.set_status(res, 203)
    return res
}

let nothing = [] (req: server.Req) => null {
}

let main = [server.add, server.run, echo, create, peek, nothing] () => null {
    server.add(echo)
    server.add(create)
    server.add(peek)
    server.add(nothing)
    server.add([server.text] (state: mem.Shared[i64]) => server.Res {
        let n <- state
        let text = "state {n}"
        n -> state
        return server.text(200, text)
    })
    server.run("{{\"name\": \"Ada\", \"age\": 36}}", 1000)
    server.run("{{\"name\": \"Old\", \"age\": 120}}", 0)
    server.run("not json", 0)
}
"""


def _project(tmp_path: Path, library: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "handlers", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
            libraries=[library],
            externs=EXTERNS,
        )
    )


def _compile(tmp_path: Path, source: str) -> Project:
    (tmp_path / "server.c").write_text(SERVER)
    library = tmp_path / "server.o"
    subprocess.run(["clang", "-c", "-fPIC", "-o", library, tmp_path / "server.c"], check=True)  # noqa: S603, S607
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = _project(tmp_path, library)
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def test_handlers(tmp_path: Path) -> None:
    # Each parameter is extracted by its type (an extractor's from_request, the state, the
    # request); a rejection is the response; results turn into responses through
    # into_response or a Result's side; nothing extracted or returned is left alive.
    executable = _compile(tmp_path, PROGRAM).build()
    result = subprocess.run([executable], capture_output=True, text=True, check=True)  # noqa: S603
    assert result.stdout.splitlines() == [
        '0: 200 {"name": "Ada", "age": 36}',
        "1: 201 created 1036",
        '2: 203 body {"name": "Ada", "age": 36}, user ok: true',
        "3: 200 (empty)",
        "4: 200 state 1000",
        "live 0",
        '0: 200 {"name": "Old", "age": 120}',
        "1: 500 too old: 120",
        '2: 203 body {"name": "Old", "age": 120}, user ok: true',
        "3: 200 (empty)",
        "4: 200 state 0",
        "live 0",
        "0: 422 expected an object at 0, near 'not json'",
        "1: 422 expected an object at 0, near 'not json'",
        "2: 203 body not json, user ok: false",
        "3: 200 (empty)",
        "4: 200 state 0",
        "live 0",
    ]


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        ("[] (n: i64) => null {\n}", "each parameter is given from the request, but n is an i64"),
        ("[] (req: server.Req) => i64 1", "returns i64, which is not a response"),
    ],
)
def test_errors(tmp_path: Path, handler: str, message: str) -> None:
    source = f'let server = import("server")\nlet main = [server.add] () => null {{\n    server.add({handler})\n}}\n'
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source)
    assert message in error.value.msg
