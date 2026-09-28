"""``std:tasks``: ``await tasks.gather(...)`` runs async calls at the same time.

Each call sleeps (``std:time``), so running them together takes as long as the
longest, not the sum; named calls give a record of their results.
``tasks.ignore(...)`` starts a call without waiting for it.
"""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

PRELUDE = """\
let io = import("std:stdio")
let mem = import("std:mem")
let time = import("std:time")
let tasks = import("std:tasks")

let User = struct { let id: i64, let score: f64 }

let fetch_user = [time.sleep] async (id: i64) => User {
    await time.sleep(100)
    return User(id: id, score: 1.5)
}

let fetch_count = [time.sleep] async (id: i64) => i64 {
    await time.sleep(80)
    return id * 10
}

let log = [time.sleep, io.printf] async (text: str) => null {
    await time.sleep(50)
    io.printf("%s\\n", text)
}

let plain = [] (x: i64) => i64 x

let peek = [time.sleep] async (user: mem.Weak[User]) => null {
    await time.sleep(1)
}
"""

PROGRAM = (
    PRELUDE
    + """
let main = [tasks.gather, fetch_user, fetch_count, log, time.now, io.printf] async () => null {
    let started = time.now()
    let found = await tasks.gather(user: fetch_user(7), count: fetch_count(4), log("logged"))
    io.printf("user %lld score %.1f count %lld\\n", found.user.id, found.user.score, found.count)
    await tasks.gather(log("a"), log("b"))
    io.printf("%lld\\n", time.now() - started)
}
"""
)


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "tasks_app", "version": "0", "description": "", "type": "async"},
            flags={"optimization": OptLevel.O2, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _lower(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = _project(tmp_path)
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def test_gather_runs_calls_at_the_same_time(tmp_path: Path) -> None:
    executable = _lower(tmp_path, PROGRAM).build()
    lines = subprocess.run([executable], capture_output=True, text=True, check=True).stdout.splitlines()  # noqa: S603
    assert lines[:4] == ["logged", "user 7 score 1.5 count 40", "a", "b"]
    # The longest call of each gather (100 ms, then 50 ms), not the sum of all of them (330 ms).
    assert 150 <= int(lines[4]) < 300


IGNORING = (
    PRELUDE
    + """
let after = [time.sleep, io.printf] async (text: str, milliseconds: i64) => null {
    await time.sleep(milliseconds)
    io.printf("%s\\n", text)
}

let main = [tasks.ignore, after, time.sleep, io.printf] async () => null {
    tasks.ignore(after("started first, done second", 20))
    tasks.ignore(after("cut off by the end of main", 500))
    io.printf("main goes on\\n")
    await time.sleep(70)
    io.printf("main is done\\n")
}
"""
)


def test_ignore_starts_a_call_without_waiting(tmp_path: Path) -> None:
    executable = _lower(tmp_path, IGNORING).build()
    lines = subprocess.run([executable], capture_output=True, text=True, check=True).stdout.splitlines()  # noqa: S603
    assert lines == ["main goes on", "started first, done second", "main is done"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('log("a")', "log(...) is async and is not awaited. Did you forget `await log(...)`?"),
        ('tasks.ignore(log("a"))\n    await tasks.ignore(log("b"))', "tasks.ignore(...) does not pause"),
        ("tasks.ignore(plain(1))", "tasks.ignore starts a call that pauses, like log(text); plain is not one"),
        ("tasks.ignore(fetch_count(1))", "fetch_count(...) returns a value, which tasks.ignore would drop"),
        ('tasks.ignore(log("a"), log("b"))', "tasks.ignore takes one call to start"),
        ('let text = "a"\n    tasks.ignore(log(text))', "text could be gone before a call nobody waits for is done"),
        (
            "let u: mem.Unique[User] = User(id: 1, score: 0.0)\n    tasks.ignore(peek(u))",
            "peek borrows a value, which could be gone",
        ),
        ("let n = tasks.gather(count: fetch_count(1))", "tasks.gather(...) pauses; wait for it with `await"),
        ("let n = await tasks.gather(count: plain(1))", "tasks.gather runs calls that pause, like fetch_user(id)"),
        ("await tasks.gather(fetch_count(1))", "fetch_count(...) returns a value; name it"),
        ('await tasks.gather(done: log("a"))', "log(...) returns nothing; pass it without a name"),
        ("let n = await tasks.gather(a: fetch_count(1), a: fetch_count(2))", "already has a result named 'a'"),
        ("await tasks.gather()", "tasks.gather takes the calls to run"),
        ('let n = await tasks.gather(log("a"), log("b"))', "tasks.gather(...) gives nothing here"),
        ('let tasks = await tasks.gather(log("a"))', "'tasks' is a module of this file (an import)"),
        (
            "let u: mem.Unique[User] = User(id: 1, score: 0.0)\n    await tasks.gather(peek(u))",
            "peek borrows a mem.Weak, which tasks.gather cannot lend",
        ),
    ],
)
def test_errors(tmp_path: Path, body: str, message: str) -> None:
    main = "\nlet main = [tasks.gather, tasks.ignore, fetch_count, log, plain, peek] async () => null {\n    "
    main += body + "\n}\n"
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, PRELUDE + main)
    assert message in error.value.msg
