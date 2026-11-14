"""Threads: ``spawn(f(x))`` runs ``f(x)`` on a new OS thread, and ``await t`` gives its result.

The programs are built and run. Memory is checked by preloading a library that
counts ``malloc`` and ``free`` (as in ``test_sync``), and with valgrind where it is
installed. In async code, ``await t`` pauses only the task waiting: another task
ticks while the thread works.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document
from tests.test_sync import RUNS, _counting, _leftover, _valgrind

HEADER = """\
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let sync = import("std:sync")
let tasks = import("std:tasks")
let time = import("std:time")

let square = (n: i64) => i64 {
    return n * n
}

let total = (xs: i64[]) => i64 {
    let sum = 0
    forall x in xs {
        let sum = sum + x
    }
    return sum
}

let label = [fmt.format] (n: i64) => mem.Unique[str] {
    return fmt.format("n=%ld", n)
}

let produce = (jobs: sync.Channel[i64], start: i64, count: i64) => null {
    let i = 0
    while i < count {
        start + i -> jobs
        let i = i + 1
    }
}

let consume = (jobs: sync.Channel[i64], count: i64, seen: mem.Atomic[i64]) => i64 {
    let i = 0
    let mine = 0
    while i < count {
        let job <- jobs
        let g <- seen
        g = g + job
        g -> seen
        let mine = mine + job
        let i = i + 1
    }
    return mine
}
"""


def _project(tmp_path: Path, kind: str = "sync") -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "threads", "version": "0", "description": "", "type": kind},
            flags={"optimization": 0, "linker": "clang"},
        )
    )


def _build(tmp_path: Path, body: str, kind: str = "sync") -> Path:
    source = tmp_path / "main.bif"
    source.write_text(HEADER + body)
    project = _project(tmp_path, kind)
    unit = lower_file(project, source)
    with unit.errors():
        return project.build()


def _lower(tmp_path: Path, body: str) -> None:
    source = tmp_path / "main.bif"
    source.write_text(HEADER + body)
    project = _project(tmp_path)
    unit = lower_file(project, source)
    with unit.errors():
        _ = project.program.mlir


def _run(executable: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([executable], capture_output=True, text=True, check=True, timeout=60, env=env)  # noqa: S603


def test_await_gives_each_threads_result(tmp_path: Path) -> None:
    program = _build(
        tmp_path,
        """
let main = [spawn, square, io.printf] () => null {
    let a = spawn(square(7))
    let b = spawn(square(8))
    let x = await a
    io.printf("%ld %ld %ld\\n", x, await b, await spawn(square(9)))
}
""",
    )
    assert _run(program).stdout == "49 64 81\n"


OWNED = """
let main = [spawn, total, label, io.puts, io.printf] () => null {
    let xs = #[1, 2, 3, 4]
    let t = spawn(total(xs))
    let made = spawn(label(await t))
    let name = await made
    io.puts(name)
    io.puts(await spawn(label(5)))
    let ys = #[10, 20]
    let ignored = spawn(total(ys))
    io.printf("%ld\\n", await ignored)
}
"""


def test_owned_values_move_in_and_results_move_out_once(tmp_path: Path) -> None:
    program = _build(tmp_path, OWNED)
    result = _run(program, _counting(tmp_path))
    assert result.stdout == "n=10\nn=5\n30\n"
    assert _leftover(result.stderr) <= 1


@pytest.mark.skipif(shutil.which("valgrind") is None, reason="needs valgrind")
def test_threads_with_owned_values_leave_nothing_behind(tmp_path: Path) -> None:
    _valgrind(_build(tmp_path, OWNED))


def test_a_thread_started_and_awaited_in_each_iteration(tmp_path: Path) -> None:
    program = _build(
        tmp_path,
        """
let main = [spawn, total, io.printf] () => null {
    let i = 0
    let sum = 0
    while i < 3 {
        let xs = #[i, i + 1]
        let t = spawn(total(xs))
        let sum = sum + await t
        let i = i + 1
    }
    io.printf("%ld\\n", sum)
}
""",
    )
    result = _run(program, _counting(tmp_path))
    assert result.stdout == "9\n"
    assert _leftover(result.stderr) <= 1


SHARED = """
let main = [spawn, produce, consume, channel, io.printf] () => null {
    let jobs: sync.Channel[i64] = channel(4)
    let seen: mem.Atomic[i64] = 0
    let p1 = spawn(produce(jobs, 0, 500))
    let p2 = spawn(produce(jobs, 500, 500))
    let c1 = spawn(consume(jobs, 400, seen))
    let c2 = spawn(consume(jobs, 600, seen))
    await p1
    await p2
    let mine = await c1 + await c2
    let g <- seen
    io.printf("%ld %ld\\n", mine, g)
    g -> seen
}
"""


def test_threads_share_a_channel_and_an_atomic(tmp_path: Path) -> None:
    program = _build(tmp_path, SHARED)
    env = _counting(tmp_path)
    for _ in range(RUNS):
        result = _run(program, env)
        assert result.stdout == "499500 499500\n"  # every message received once: 0 + 1 + ... + 999
        assert _leftover(result.stderr) <= 1


@pytest.mark.skipif(shutil.which("valgrind") is None, reason="needs valgrind")
def test_threads_sharing_a_channel_leave_nothing_behind(tmp_path: Path) -> None:
    _valgrind(_build(tmp_path, SHARED))


def test_an_ignored_thread_runs_on(tmp_path: Path) -> None:
    program = _build(
        tmp_path,
        """
let bump = (seen: mem.Atomic[i64]) => i64 {
    let g <- seen
    g = g + 1
    g -> seen
    return 0
}

let main = [spawn, bump, tasks.ignore, io.puts] () => null {
    let seen: mem.Atomic[i64] = 0
    let t = spawn(bump(seen))
    tasks.ignore(t)
    tasks.ignore(spawn(bump(seen)))
    let done = false
    while !done {
        let g <- seen
        let done = g == 2
        g -> seen
    }
    io.puts("both ran")
}
""",
    )
    for _ in range(RUNS):
        assert _run(program).stdout == "both ran\n"


def test_in_async_code_await_pauses_only_the_task_waiting(tmp_path: Path) -> None:
    program = _build(
        tmp_path,
        """
let slow = [time.now] (n: i64) => i64 {
    let start = time.now()
    while time.now() - start < 200 {}
    return n * 2
}

let tick = [time.sleep, io.puts] async () => null {
    let i = 0
    while i < 3 {
        await time.sleep(20)
        io.puts("tick")
        let i = i + 1
    }
}

let main = [spawn, slow, tick, tasks.ignore, io.printf] async () => null {
    tasks.ignore(tick())
    let t = spawn(slow(21))
    io.printf("got %ld\\n", await t)
}
""",
        "async",
    )
    assert _run(program).stdout == "tick\ntick\ntick\ngot 42\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "let main = [spawn, square] () => null {\n    let t = spawn(square(1))\n}\n",
            "the thread t is never awaited here. Did you forget `await t`? "
            "If not waiting is deliberate, let it go with `tasks.ignore(t)`",
        ),
        (
            "let main = [spawn, square] () => null {\n    let t = spawn(square(1))\n"
            "    if true {\n        return\n    }\n    await t\n}\n",
            "the thread t is never awaited here",
        ),
        (
            "let main = [spawn, square] () => null {\n    spawn(square(1))\n}\n",
            "spawn(...) gives the thread running square(...); wait for it, `await spawn(...)`",
        ),
        (
            "let main = [spawn, square] () => null {\n    let t = spawn(square(1))\n    let n = square(t)\n}\n",
            "t is a sync.Thread: wait for its result with `await t`",
        ),
        (
            "let main = [spawn, square] () => null {\n    let t = spawn(square(1))\n    let i = 0\n"
            "    while i < 2 {\n        await t\n        let i = i + 1\n    }\n}\n",
            "t is moved inside a loop",
        ),
        (
            "let main = [spawn, total] () => null {\n    let xs = #[1]\n    let t = spawn(total(xs))\n"
            "    let n = xs[0]\n    await t\n}\n",
            "xs was moved on line",
        ),
        (
            "let poke = (c: mem.Shared[i64]) => null {}\n"
            "let main = [spawn, poke] () => null {\n    let c: mem.Shared[i64] = 1\n    await spawn(poke(c))\n}\n",
            "poke takes a mem.Shared, which is one thread's",
        ),
        (
            "let lend = (c: mem.Weak[i64]) => null {}\n"
            "let main = [spawn, lend] () => null {\n    let c: mem.Unique[i64] = 1\n    await spawn(lend(c))\n}\n",
            "lend borrows c (a mem.Weak), which a thread could still use after it is gone",
        ),
        (
            "let bump = (c: mem.Atomic[i64]) => null {}\n"
            "let main = [spawn, bump] () => null {\n    let c: mem.Atomic[i64] = 1\n    let g <- c\n"
            "    await spawn(bump(c))\n    g -> c\n}\n",
            "c is locked by g (line 48); use g until `g -> c`",
        ),
        (
            "let fetch = [] async (n: i64) => i64 {\n    return n\n}\n"
            "let main = [spawn, fetch] () => null {\n    await spawn(fetch(1))\n}\n",
            "fetch is async, and spawn runs a function that is not async on a thread of its own",
        ),
        (
            "let names = (n: i64) => i64[] {\n    return #[n]\n}\n"
            "let main = [spawn, names, tasks.ignore] () => null {\n    tasks.ignore(spawn(names(1)))\n}\n",
            "this thread's function returns i64[], which tasks.ignore would leak",
        ),
        (
            "let main = [spawn] () => null {\n    let t = spawn(1 + 2)\n    await t\n}\n",
            "spawn runs a call of a Bifrost function on a new thread, like spawn(work(jobs))",
        ),
        (
            "let main = [square] () => null {\n    await spawn(square(1))\n}\n",
            "main calls 'spawn' without depending on it; add it to the dependency list: [spawn]",
        ),
        ("let main = [spawn] () => null {\n    let x = 1\n}\n", "'spawn' is a dependency of main but never called"),
    ],
)
def test_thread_errors(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, body)
    assert message in error.value.msg


def test_a_name_of_the_file_hides_spawn(tmp_path: Path) -> None:
    program = _build(
        tmp_path,
        """
let spawn = (n: i64) => i64 {
    return n + 1
}

let main = [spawn, io.printf] () => null {
    io.printf("%ld\\n", spawn(1))
}
""",
    )
    assert _run(program).stdout == "2\n"


def test_hover_shows_a_threads_type(tmp_path: Path) -> None:
    source = HEADER + (
        "let main = [spawn, square, io.printf] () => null {\n"
        "    let t = spawn(square(7))\n"
        '    io.printf("%ld\\n", await t)\n'
        "}\n"
    )
    document = Document.open(tmp_path / "main.bif", source)
    row = source.splitlines().index("    let t = spawn(square(7))")
    hover = document.hover((row, 8))
    assert hover is not None
    assert "sync.Thread[i64]" in hover
    spawned = document.hover((row, 13))
    assert spawned is not None
    assert "spawn = (f(...): R) => sync.Thread[R]" in spawned
