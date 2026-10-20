"""The pipe operator: ``x |> f(a)`` is ``f(x, a)``, checked and compiled as that call."""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

PROGRAM = r"""
let io = import("std:stdio")
let mem = import("std:mem")

module maths = {
    let add = [] (a: i64, b: i64) => i64 a + b
}

let double = [] (x: i64) => i64 x * 2
let add = [] (a: i64, b: i64) => i64 a + b
let total = [] (xs: i64[]) => i64 {
    let sum = 0
    forall x in xs {
        let sum = sum + x
    }
    return sum
}
let make = [] (n: i64) => i64[] #[n, n, n]

let Team = struct {
    let size: i64,
    let grow = [super] (by: i64) => null {
        let s <- super
        s.size = s.size + by
        s -> super
    }
}
let Holder = struct { let items: i64[] }

let bump = [] (c: mem.Weak[Team], by: i64) => null {
    let g <- c
    g.size = g.size + by
    g -> c
}

let main = [io.printf, double, add, total, make, maths.add, bump] () => null {
    io.printf("%d\n", 3 |> double)
    io.printf("%d\n", 3 |> add(4))
    io.printf("%d\n", 3 |> double |> add(1) |> double)
    io.printf("%d\n", 1 + 2 |> maths.add(10))
    io.printf("%d\n", 5 |> [] (y: i64) => i64 y * 100)
    let n = 7
    io.printf("%d\n", 1 |> [n] (y: i64) => i64 y + n) // ignore: copy
    let xs = #[3, 1, 2]
    io.printf("%d\n", xs |> total)
    io.printf("%d\n", 4 |> make |> total)
    let sorted = xs |> .sorted()
    io.printf("%d %d\n", sorted[0], xs[0])
    io.printf("%d\n", #[9, 8] |> .sorted() |> total)
    io.printf("%d\n", xs |> .map([] (x: i64) => i64 x * 10) |> total)
    io.printf("%d\n", xs |> len())
    let team = Team(size: 2)
    team |> .grow(3)
    io.printf("%d\n", team.size)
    let shared: mem.Shared[Team] = Team(size: 1)
    let g <- shared
    g |> bump(2)
    io.printf("%d\n", g.size)
    g -> shared
    let ys = #[1, 2]
    let holder = ys |> Holder
    io.printf("%d\n", len(holder.items))
}
"""

COUNTER = r"""
#include <stdio.h>
#include <stdlib.h>
void *__libc_malloc(size_t);
void *__libc_realloc(void *, size_t);
void __libc_free(void *);
static long allocations, frees;
void *malloc(size_t n) { allocations++; return __libc_malloc(n); }
void *realloc(void *p, size_t n) { if (!p) allocations++; return __libc_realloc(p, n); }
void free(void *p) { if (p) frees++; __libc_free(p); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "%ld %ld\n", allocations, frees); }
"""


def _compile(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = Project(
        Config(
            path=tmp_path,
            package={"name": "pipes", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def test_pipes(tmp_path: Path) -> None:
    # Into a function or a call (the value first), a module's function, a lambda, a method of the
    # value; owned values lend and move as in the call, and nothing leaks.
    executable = _compile(tmp_path, PROGRAM).build()
    (tmp_path / "count.c").write_text(COUNTER)
    counter = tmp_path / "count.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    assert result.stdout.split("\n")[:-1] == [
        "6",  # double(3)
        "7",  # add(3, 4)
        "14",  # double(add(double(3), 1))
        "13",  # maths.add(1 + 2, 10)
        "500",
        "8",
        "6",
        "12",  # total(make(4))
        "1 3",  # a new list, sorted; xs as it was
        "17",
        "60",
        "3",
        "5",  # team.grow(3)
        "3",  # bump(g, 2)
        "2",  # Holder(ys): moved in
    ]
    allocations, frees = map(int, result.stderr.split())
    assert allocations - frees == 1  # stdout keeps its buffer


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("let a = 3 |> triple", "main calls 'triple' without depending on it; add it to the dependency list: [triple]"),
        ("let a = 3 |> 4", "the right side of |> is a function, a call or a lambda, given the value first"),
        (
            "let a = 3 |> [] (x: i64, y: i64) => i64 x",
            "a lambda a pipe calls takes one parameter, the value piped in; this one takes 2",
        ),
        (
            "let xs = #[1, 2]\n    let h = xs |> Holder\n    let n = len(xs)",
            "xs was moved on line 5, so it is no longer here",
        ),
    ],
)
def test_errors(tmp_path: Path, body: str, message: str) -> None:
    source = (
        "let triple = [] (x: i64) => i64 x * 3\nlet Holder = struct { let items: i64[] }\n"
        f"let main = [] () => null {{\n    {body}\n}}\n"
    )
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source)
    assert message in error.value.msg


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "20 |> time.sleep",
            "time.sleep(...) is async and is not awaited. Did you forget `await (20 |> time.sleep)`? "
            "If not waiting is deliberate, start it with `tasks.ignore(20 |> time.sleep)`",
        ),
        ("await 20 |> time.sleep", "await binds tighter than |>, so this awaits only the value piped in"),
    ],
)
def test_awaiting_a_pipe(tmp_path: Path, body: str, message: str) -> None:
    # `await (x |> f)` waits for the call; `await x |> f` is `(await x) |> f`.
    head = 'let time = import("std:time")\nlet main = [] () => null {\n}\n'
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, head + f"let f = [time.sleep] async () => null {{\n    {body}\n}}\n")
    assert message in error.value.msg
    _compile(tmp_path, head + "let f = [time.sleep] async () => null {\n    await (20 |> time.sleep)\n}\n")


def test_formatting() -> None:
    # Spaced like other operators; a chain too long for its line breaks before each |>.
    source = (
        "let f = [g, total] () => i64 {\n    let a = 3|>g( 4 )\n"
        "    return #[1, 2, 3] |> .map([] (x: i64) => i64 x * 10) |> .filter([] (x: i64) => bool x > 10)"
        " |> .sorted() |> total\n}\n"
    )
    formatted = format_source(source)
    assert formatted == (
        "let f = [g, total] () => i64 {\n    let a = 3 |> g(4)\n    return #[1, 2, 3]\n"
        "        |> .map([] (x: i64) => i64 x * 10)\n        |> .filter([] (x: i64) => bool x > 10)\n"
        "        |> .sorted()\n        |> total\n}\n"
    )
    assert format_source(formatted) == formatted


def test_editor(tmp_path: Path) -> None:
    # The value piped in fills the first parameter, so hints start at the second; after `|> .`, methods.
    path = tmp_path / "main.bif"
    source = PROGRAM.replace("    let sorted = xs |> .sorted()\n", "    let sorted = xs |> .sorted()\n    team |> .\n")
    document = Document.open(path, source)
    lines = source.splitlines()
    row = lines.index('    io.printf("%d\\n", 3 |> add(4))')
    hints = [(hint.position, hint.label) for hint in document.inlay_hints() if hint.position[0] == row]
    assert hints == [((row, 14), "format:"), ((row, 31), "b:")]
    row = lines.index("    team |> .")
    assert [completion.label for completion in document.completions((row, len(lines[row])))] == ["grow"]
    assert document.hover((lines.index('    io.printf("%d\\n", 3 |> double)'), 27)) == (
        "```bifrost\nlet double = (x: i64) => i64\n```"
    )
