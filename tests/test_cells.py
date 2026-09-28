"""``mem.Shared`` and ``mem.Atomic``: values with several owners, counted at run time.

The programs are built and run. Memory is checked by preloading a library that
counts ``malloc`` and ``free``: every cell is freed once its last owner is gone.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.configs.schema import _Extern, _Function
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

SHARED = r"""
let io = import("std:stdio")
let mem = import("std:mem")

let Counter = struct { let hits: i64 }

let make_counter = [] (start: i64) => mem.Shared[Counter] {
    let counter: mem.Shared[Counter] = Counter(hits: start)
    return counter
}

let bump = [] (counter: mem.Shared[Counter]) => null {
    let g <- counter
    g.hits = g.hits + 1
    g -> counter
}

let read = [] (counter: mem.Weak[Counter]) => i64 {
    let g <- counter
    let hits = g.hits
    g -> counter
    return hits
}

let same = [] (counter: mem.Shared[Counter]) => mem.Shared[Counter] counter

let main = [io.printf, make_counter, bump, read, same] () => null {
    let counter = make_counter(40)
    let other = counter
    bump(counter)
    bump(other)
    let again = same(other)
    bump(again)
    let i = 0
    while i < 3 {
        let copy = counter
        bump(copy)
        let i = i + 1
    }
    let hits = read(counter)
    io.printf("%lld\n", hits)
    let fresh: mem.Shared[Counter] = Counter(hits: 7)
    let g <- fresh
    io.printf("%lld\n", g.hits)
    g -> fresh
}
"""

ATOMIC = r"""
let io = import("std:stdio")
let mem = import("std:mem")
let pthread = import("pthread")

let Counter = struct { let hits: i64 }
let Thread = struct { let id: u64 }

let work = [] (counter: mem.Atomic[Counter]) => null {
    let i = 0
    while i < 1000000 {
        let g <- counter
        g.hits = g.hits + 1
        g -> counter
        let i = i + 1
    }
}

let read = [] (counter: mem.Weak[Counter]) => i64 {
    let g <- counter
    let hits = g.hits
    g -> counter
    return hits
}

let main = [io.printf, pthread.create, pthread.join, work, read] () => null {
    let counter: mem.Atomic[Counter] = Counter(hits: 0)
    let a: mem.Unique[Thread] = Thread(id: 0)
    let b: mem.Unique[Thread] = Thread(id: 0)
    pthread.create(a, 0, work, counter)
    pthread.create(b, 0, work, counter)
    let ga <- a
    pthread.join(ga.id, 0)
    ga -> a
    let gb <- b
    pthread.join(gb.id, 0)
    gb -> b
    let hits = read(counter)
    io.printf("%lld\n", hits)
}
"""

CONFLICT = """
let mem = import("std:mem")

let Counter = struct { let hits: i64 }

let main = [] () => null {
    let counter: mem.Shared[Counter] = Counter(hits: 0)
    let other = counter
    let g <- counter
    let h <- other
    h -> other
    g -> counter
}
"""

# Counts allocations; reports how many are still live when the program exits.
COUNTER_C = r"""
#include <stdio.h>
#include <stddef.h>
extern void *__libc_malloc(size_t);
extern void __libc_free(void *);
static long live;
void *malloc(size_t size) { __atomic_add_fetch(&live, 1, __ATOMIC_SEQ_CST); return __libc_malloc(size); }
void free(void *pointer) { if (pointer) __atomic_sub_fetch(&live, 1, __ATOMIC_SEQ_CST); __libc_free(pointer); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "live allocations: %ld\n", live); }
"""


def _project(tmp_path: Path) -> Project:
    thread = {"thread": "ptr", "attributes": "i64", "start": "(ptr) => None", "argument": "ptr"}
    declarations = [
        _Function.model_validate(
            {"name": "pthread_create", "as": "create", "type": "function", "parameters": thread, "return": "i32"}
        ),
        _Function.model_validate(
            {
                "name": "pthread_join",
                "as": "join",
                "type": "function",
                "parameters": {"thread": "u64", "result": "i64"},
                "return": "i32",
            }
        ),
    ]
    return Project(
        Config(
            path=tmp_path,
            package={"name": "cells", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            externs=[_Extern(module="pthread", description="POSIX threads", declarations=declarations)],
        )
    )


def _build(tmp_path: Path, program: str) -> Path:
    source = tmp_path / "main.bif"
    source.write_text(program)
    project = _project(tmp_path)
    unit = lower_file(project, source)
    with unit.errors():
        return project.build()


def _counting(tmp_path: Path) -> dict[str, str]:
    """Return an environment that preloads the allocation counter, built with the C compiler at hand."""
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler to build the allocation counter")
    (tmp_path / "count.c").write_text(COUNTER_C)
    library = tmp_path / "libcount.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-o", library, tmp_path / "count.c"], check=True)  # noqa: S603
    return {**os.environ, "LD_PRELOAD": str(library)}


def test_shared_values_have_several_owners(tmp_path: Path) -> None:
    executable = _build(tmp_path, SHARED)
    result = subprocess.run([executable], capture_output=True, text=True, check=True, env=_counting(tmp_path))  # noqa: S603
    assert result.stdout.splitlines() == ["46", "7"]  # 40, bumped by three owners, then three copies in a loop
    assert "live allocations: 1" in result.stderr  # only stdout's buffer: both cells were freed


def test_atomic_values_are_shared_across_threads(tmp_path: Path) -> None:
    executable = _build(tmp_path, ATOMIC)
    result = subprocess.run([executable], capture_output=True, text=True, check=True)  # noqa: S603
    # No increment lost: each held the mutex. (Through a mem.Weak, unlocked, about 40% are lost.)
    assert result.stdout.splitlines() == ["2000000"]


def test_a_second_guard_on_a_shared_value_stops_the_program(tmp_path: Path) -> None:
    executable = _build(tmp_path, CONFLICT)
    result = subprocess.run([executable], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode != 0
    assert "main.bif:10: other is already locked by another guard; a mem.Shared allows one at a time" in result.stderr


VALUES = r"""
let io = import("std:stdio")
let mem = import("std:mem")

let bump = [] (count: mem.Shared[i64]) => null {
    let g <- count
    g = g * 10 + 1
    g -> count
}

let main = [io.printf, bump] () => null {
    let count: mem.Shared[i64] = 4
    bump(count)
    let g <- count
    io.printf("%lld\n", g)
    g = g + 1
    io.printf("%lld\n", g)
    g -> count
    let total: mem.Unique[i64] = 2
    let t <- total
    t = t * 21
    io.printf("%lld\n", t)
    t -> total
}
"""


def test_a_guard_on_a_whole_value_reads_and_writes_it(tmp_path: Path) -> None:
    executable = _build(tmp_path, VALUES)
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output.splitlines() == ["41", "42", "42"]


PRELUDE = """\
let mem = import("std:mem")
let Counter = struct { let hits: i64 }
let Other = struct { let hits: i64 }
let bump = [] (counter: mem.Shared[Counter]) => null {}
let make = [] () => mem.Shared[Counter] {
    let counter: mem.Shared[Counter] = Counter(hits: 0)
    return counter
}
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "let f = [bump] (c: mem.Weak[Counter]) => null { bump(c) }",
            "this parameter takes a mem.Shared[Counter]; c is not one",
        ),
        (
            "let f = [bump] (c: mem.Atomic[Counter]) => null { bump(c) }",
            "c is a mem.Atomic, but this parameter takes a mem.Shared",
        ),
        (
            "let f = [bump] (c: mem.Shared[Other]) => null { bump(c) }",
            "c holds a Other, not a Counter",
        ),
        (
            "let f = [] (c: mem.Shared[Counter]) => null {\n    let d: mem.Atomic[Counter] = c\n}",
            "c is a mem.Shared, not a mem.Atomic",
        ),
        (
            "let f = [] (c: mem.Shared[Counter]) => i64 c.hits",
            "c is a mem.Shared; lock it (`let guard <- c`) to reach its fields",
        ),
        (
            "let f = [make] () => null {\n    make()\n}",
            "make(...) returns a mem.Shared or mem.Atomic, which nothing would free",
        ),
        (
            "let f = [] () => null {\n    let x = 1\n    x = 2\n}",
            "x is not a guard; give a new value a name with `let x = ...`",
        ),
        (
            "let f = [] (c: mem.Shared[Counter]) => null {\n    let g <- c\n    g = Counter(hits: 1)\n    g -> c\n}",
            "g holds an object; change its fields (g.x = ...) instead",
        ),
        (
            "let Holder = struct { let counter: mem.Shared[Counter] }",
            "an object holding a mem.Shared (a mem.Shared field) is not supported yet",
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
