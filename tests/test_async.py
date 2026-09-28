"""Functions that pause: ``async`` functions, and ``await`` at the calls that pause.

A call pauses if it calls a C function returning a ``token``, or an ``async``
function; the compiler checks each is awaited, in an async function. Async
functions compile to coroutines; others run while one waits. ``main`` is async
only when config.yaml says so (``package.type: async``).

The C library here is a small event loop, as a server's would be: ``later(n)``
completes after ``n`` turns, and ``serve`` starts a handler per request, then
turns the loop until every handler is done.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.configs.schema import _Extern, _Function
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

LOOP_C = r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void *mlirAsyncRuntimeCreateToken(void);
void mlirAsyncRuntimeEmplaceToken(void *);
void mlirAsyncRuntimeDropRef(void *, int64_t);
void mlirAsyncRuntimeAddRef(void *, int64_t);
void bifrost_async_run_ready(void);
void bifrost_async_set_poller(void (*)(void *), void *);

static struct { void *token; int64_t turns; } timers[64];
static int timer_count;

/* One turn: every timer gets closer; those done complete their token. */
static void turn(void *context) {
    (void)context;
    for (int i = 0; i < timer_count; ++i)
        if (--timers[i].turns <= 0) {
            void *token = timers[i].token;
            timers[i--] = timers[--timer_count];
            mlirAsyncRuntimeEmplaceToken(token); /* drops the reference it kept until now */
        }
}

__attribute__((constructor)) static void start(void) { bifrost_async_set_poller(turn, NULL); }

void *later(int64_t turns) {
    void *token = mlirAsyncRuntimeCreateToken(); /* the caller's reference, and one kept until complete */
    timers[timer_count].token = token;
    timers[timer_count++].turns = turns;
    return token;
}

static int64_t order;
void record(int64_t request) { order = order * 10 + request; }

bool mlirAsyncRuntimeIsTokenError(void *);
typedef struct { int64_t refs; bool ready; } Head; /* the runtime's object starts so */

int64_t serve(void *(*handler)(int64_t), int64_t requests) {
    void *tokens[16];
    order = 0;
    for (int64_t i = 0; i < requests; ++i) tokens[i] = handler(i + 1);
    for (;;) {
        bifrost_async_run_ready();
        bool done = true;
        for (int64_t i = 0; i < requests; ++i) done &= ((Head *)tokens[i])->ready;
        if (done) break;
        turn(NULL);
    }
    for (int64_t i = 0; i < requests; ++i) mlirAsyncRuntimeDropRef(tokens[i], 1);
    return order;
}
"""

PROGRAM = """\
let io = import("std:stdio")
let loop = import("loop")

let pause = [loop.later] async (turns: i64) => null {
    await loop.later(turns)
}

let handle = [pause, loop.record] async (request: i64) => null {
    await pause(4 - request)
    loop.record(request)
}

let quick = [loop.record] (request: i64) => null {
    loop.record(request)
}

let doubled = [pause] async (n: i64) => i64 {
    await pause(n)
    return n * 2
}

let main = [loop.serve, handle, quick, doubled, io.printf] async () => i32 {
    let order = loop.serve(handle, 3)
    io.printf("%lld\\n", order)
    let quick_order = loop.serve(quick, 3)
    io.printf("%lld\\n", quick_order)
    let lambda_order = loop.serve([loop.later, loop.record] async (request: i64) => null {
        await loop.later(request)
        loop.record(request)
    }, 2)
    io.printf("%lld\\n", lambda_order)
    let value = await doubled(5)
    io.printf("%lld\\n", value)
    return 0
}
"""


def _project(tmp_path: Path, library: Path | None = None, kind: str = "sync") -> Project:
    declarations = [
        _Function.model_validate(
            {"name": "later", "type": "function", "parameters": {"turns": "i64"}, "return": "token"}
        ),
        _Function.model_validate(
            {"name": "record", "type": "function", "parameters": {"request": "i64"}, "return": "None"}
        ),
        _Function.model_validate(
            {
                "name": "serve",
                "type": "function",
                "parameters": {"handler": "(i64) => token", "requests": "i64"},
                "return": "i64",
            }
        ),
    ]
    return Project(
        Config(
            path=tmp_path,
            package={"name": "loop_app", "version": "0", "description": "", "type": kind},
            flags={"optimization": 0, "linker": "clang"},
            libraries=[library] if library is not None else [],
            externs=[_Extern(module="loop", description="a small event loop", declarations=declarations)],
        )
    )


def test_pausing_functions_run_on_the_event_loop(tmp_path: Path) -> None:
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("needs clang")
    (tmp_path / "loop.c").write_text(LOOP_C)
    library = tmp_path / "libloop.a"
    subprocess.run([compiler, "-c", "-fPIC", tmp_path / "loop.c", "-o", tmp_path / "loop.o"], check=True)  # noqa: S603
    subprocess.run(["ar", "rcs", library, tmp_path / "loop.o"], check=True)  # noqa: S603, S607
    source = tmp_path / "main.bif"
    source.write_text(PROGRAM)
    project = _project(tmp_path, library, kind="async")  # main is async
    unit = lower_file(project, source)
    with unit.errors():
        executable = project.build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output.splitlines() == [
        "321",  # request 1 waits 3 turns, 2 waits 2, 3 waits 1: they finish in reverse
        "123",  # handlers that never pause run to the end at once, in order
        "12",  # a lambda handler that pauses
        "10",  # main awaits a function that pauses; the loop turns meanwhile
    ]


PRELUDE = """\
let mem = import("std:mem")
let loop = import("loop")
let Counter = struct { let hits: i64 }
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "let f = [loop.later] async (c: mem.Weak[Counter]) => null {\n"
            "    let g <- c\n    await loop.later(1)\n    g -> c\n}",
            "g holds c across loop.later(...), which can pause",
        ),
        (
            "let f = [loop.later] async () => null {\n    loop.later(1)\n}",
            "loop.later(...) is async and is not awaited. Did you forget `await loop.later(...)`?",
        ),
        (
            "let f = [loop.later] () => null {\n    await loop.later(1)\n}",
            "`await` is only allowed in an async function; mark f `async`",
        ),
        (
            "let f = [loop.record] async () => null {\n    await loop.record(1)\n}",
            "loop.record(...) does not pause, so there is nothing to await",
        ),
        (
            "let main = [loop.later] () => null {\n    await loop.later(1)\n}",
            "mark main `async` (with `type: async` in config.yaml's package)",
        ),
        (
            "let main = [] async () => null {}",
            "main is async, but config.yaml's package.type is sync",
        ),
    ],
)
def test_errors(tmp_path: Path, body: str, message: str) -> None:
    source = tmp_path / "bad.bif"
    source.write_text(PRELUDE + body + "\n")
    with pytest.raises(BifrostError) as error:
        _compile(_project(tmp_path), source)
    assert message in error.value.msg


def _compile(project: Project, source: Path) -> None:
    unit = lower_file(project, source)
    with unit.errors():
        _ = project.program.mlir


TIMED = r"""
let io = import("std:stdio")
let time = import("std:time")

let nap = [time.sleep] async (ms: i64) => null {
    await time.sleep(ms)
}

let main = [io.printf, time.now, nap] async () => null {
    let started = time.now()
    await nap(50)
    await nap(30)
    io.printf("%lld\n", time.now() - started)
}
"""


def test_std_time_sleeps_without_an_event_loop(tmp_path: Path) -> None:
    source = tmp_path / "main.bif"
    source.write_text(TIMED)
    project = _project(tmp_path, kind="async")
    unit = lower_file(project, source)
    with unit.errors():
        executable = project.build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert 80 <= int(output) < 1000  # the runtime slept until each timer was due
