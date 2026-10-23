"""Generics: ``let first[T] = ...`` and ``let Box[T] = struct {...}``, compiled once for each set of types.

Also what functional code built on them needs: ``copy(x)``, numeric conversions (``i32(x)``),
lists of function values, and objects built only to be lent.
"""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

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

FUNCTIONS = r"""
let io = import("std:stdio")

let first[T] = [] (xs: T[]) => T xs[0]
let pick[T] = [] (a: T, b: T, left: bool) => T {
    if left {
        return a
    }
    return b
}
let apply[T, U] = [] (f: (x: T) => U, x: T) => U f(x)
let count[T] = [this] (xs: T[], n: i64) => i64 {
    if n >= len(xs) {
        return n
    }
    return count(xs, n + 1)
}
let empty[T] = [] () => T[] #[]

let Pair[A, B] = struct { let left: A, let right: B }
let swap[A, B] = [] (p: Pair[A, B]) => Pair[B, A] Pair(left: p.right, right: p.left)

module m = {
    let twice[T] = [] (f: (x: T) => T, x: T) => T f(f(x))
}

let double = [] (x: i64) => i64 x * 2
let half = [] (x: f64) => f64 x / 2.0

let main = [io.printf, first, pick, apply, count, empty, swap, m.twice, double, half] () => null {
    io.printf("%d %.1f\n", first(#[7, 8]), first(#[2.5]))
    io.printf("%d %.1f %.1f\n", pick(1, 2, false), pick(1.5, 2, true), pick(1, 2.5, true))
    io.printf("%d %.2f\n", apply(double, 21), apply(half, 3.0))
    io.printf("%d\n", count(#[1, 2, 3], 0))
    let q = swap(Pair(left: 1, right: 2.5))
    io.printf("%.1f %d\n", q.left, q.right)
    let b: Pair[f64, i64] = Pair(left: 3, right: 4)
    io.printf("%.1f\n", b.left)
    io.printf("%d %.2f\n", m.twice(double, 5), m.twice(half, 8.0))
    let none: i64[] = empty()
    io.printf("%d %d\n", #[4, 5] |> first, len(none))
}
"""

# The shape of the game engine: an app of handlers over any world, built with pipes.
ENGINE = r"""
let io = import("std:stdio")

let App[W] = struct {
    let world: W,
    let updates: ((n: i64, w: W) => W)[]
}

let app[W] = [] (world: W) => App[W] App(world: copy(world), updates: #[])

let on_update[W] = [] (a: App[W], handler: (n: i64, w: W) => W) => App[W] {
    return App(world: copy(a.world), updates: #[...a.updates, copy(handler)])
}

let run[W] = [] (a: App[W], frames: i64) => W {
    let world = copy(a.world)
    forall n in #[0...frames] {
        forall handler in a.updates {
            let world = handler(n, world)
        }
    }
    return world
}

let World = struct { let score: i64, let log: i64[] }

let tick = [] (n: i64, w: World) => World World(score: w.score + n, log: #[...w.log, n])
let bonus = [] (_n: i64, w: World) => World World(score: w.score + 100, log: copy(w.log))

let main = [io.printf, app, on_update, run, tick, bonus] () => null {
    let game = app(World(score: 0, log: #[])) |> on_update(tick) |> on_update(bonus)
    let done = run(game, 4)
    io.printf("%d %d\n", done.score, len(done.log))
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "generics", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _compile(tmp_path: Path, source: str, name: str = "main.bif") -> Project:
    path = tmp_path / name
    path.write_text(source)
    project = _project(tmp_path)
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def _run(tmp_path: Path, source: str) -> tuple[list[str], int]:
    """Build and run ``source``; return its output's lines, and how many allocations it never freed."""
    executable = _compile(tmp_path, source).build()
    (tmp_path / "count.c").write_text(COUNTER)
    counter = tmp_path / "count.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    allocations, frees = map(int, result.stderr.split())
    return result.stdout.splitlines(), allocations - frees - 1  # stdout keeps its buffer


def test_generic_functions_and_objects(tmp_path: Path) -> None:
    # Each call compiles the function for its arguments' types (or the type written where its
    # value goes); numbers written as arguments take the type the others give (f64 if one is).
    lines, leaked = _run(tmp_path, FUNCTIONS)
    assert lines == ["7 2.5", "2 1.5 1.0", "42 1.50", "3", "2.5 1", "3.0", "20 2.00", "4 0"]
    assert leaked == 0


def test_an_engine_of_pure_handlers(tmp_path: Path) -> None:
    # A generic object holding the world and a list of function values, copied as it is built
    # with pipes; running it replaces the world with what each handler returns. Nothing leaks.
    lines, leaked = _run(tmp_path, ENGINE)
    assert lines == ["406 4"]
    assert leaked == 0


def test_generics_from_another_file(tmp_path: Path) -> None:
    # A generic in an imported module is compiled there, for the importing file's types.
    (tmp_path / "boxes.bif").write_text(
        "module boxes = {\n"
        "    let Box[T] = struct { let value: T }\n"
        "    let wrap[T] = [] (x: T) => Box[T] Box(value: x)\n"
        "    let unwrap[T] = [] (b: Box[T]) => T b.value\n"
        "}\n\nexport(boxes)\n"
    )
    source = (
        'let io = import("std:stdio")\nlet boxes = import(".boxes:boxes")\n\n'
        "let Point = struct { let x: i64, let y: i64 }\n\n"
        "let main = [io.printf, boxes.wrap, boxes.unwrap] () => null {\n"
        "    let p = Point(x: 1, y: 2) |> boxes.wrap |> boxes.unwrap\n"
        "    let b: boxes.Box[f64] = boxes.wrap(2.5)\n"
        '    io.printf("%d %.1f\\n", p.y, b.value)\n'
        "}\n"
    )
    lines, leaked = _run(tmp_path, source)
    assert lines == ["2 2.5"]
    assert leaked == 0


HEAD = """let io = import("std:stdio")
let pick[T] = [] (a: T, b: T) => T a
let empty[T] = [] () => T[] #[]
let Box[T] = struct { let value: T }
let bad[T] = [] (x: T) => T x.missing
let apply = [] (f: (x: i64) => i64, x: i64) => i64 f(x)
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('io.printf("%d\\n", pick(1.5, "a"))', "pick's T is f64 (from 1.5), but this is str"),
        ("let xs = empty()", "cannot tell what T is in this call of empty[T]: no argument's type says it"),
        (
            'io.printf("%d\\n", apply(pick, 1))',
            "pick is generic, so it is compiled for the types a call gives it; call it, or pass a function",
        ),
        ("let b: Box = Box(value: 1)", "Box is generic: write the types it holds, Box[T], as in Box[i64]"),
        ("let b: Box[i64, i64] = Box(value: 1)", "Box[T] takes 1 types, not 2"),
        ('io.printf("%d\\n", pick(1))', "pick takes 2 arguments, got 1"),
        ('io.printf("%d\\n", bad(1))', "in bad[i64]: i64 has no fields"),
        ('io.printf("%d\\n", i32("a"))', "i32(...) converts a number, not a str"),
    ],
)
def test_errors(tmp_path: Path, body: str, message: str) -> None:
    used = [name for name in ("pick", "empty", "bad", "apply") if f"{name}(" in body]
    source = f"{HEAD}\nlet main = [{', '.join(['io.printf', *used])}] () => null {{\n    {body}\n}}\n"
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source)
    assert message in error.value.msg


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "let Thing = struct { let a: i64 }\nlet id[Thing] = [] (x: Thing) => Thing x\n",
            "Thing already names a type here; give the type parameter another name",
        ),
        ("let twice[T, T] = [] (x: T) => T x\n", "twice already has a type parameter T"),
    ],
)
def test_definition_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source + "let main = [] () => null {\n}\n")
    assert message in error.value.msg


def test_copy_and_conversions(tmp_path: Path) -> None:
    # copy(x) is a new owner of a deep copy; i32(x) and the rest convert numbers (toward zero).
    source = r"""
let io = import("std:stdio")
let Team = struct { let name: str, let scores: i64[] }
let H = struct { let make: (x: i64) => i64[] }

let main = [io.printf] () => null {
    let a = Team(name: "a", scores: #[1, 2])
    let b = copy(a)
    let c = copy(b.scores)
    let n = 7
    io.printf("%d %d %d %.1f %d\n", len(b.scores), len(c), i32(n / 2), f64(n), i64(2.9))
    let h = H(make: [] (x: i64) => i64[] #[x, x])
    let xs = h.make(3)
    let xs = h.make(4)
    io.printf("%d %d\n", len(xs), len(a.scores))
}
"""
    lines, leaked = _run(tmp_path, source)
    assert lines == ["2 2 3 7.0 2", "2 2"]
    assert leaked == 0


def test_copy_is_what_the_checker_asks_for(tmp_path: Path) -> None:
    source = (
        "let Team = struct { let scores: i64[] }\nlet Club = struct { let team: Team }\n"
        "let join = [] (t: Team) => Club Club(team: t)\nlet main = [] () => null {\n}\n"
    )
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source)
    assert error.value.msg == "t is a parameter, lent by the caller, so it cannot be kept here; keep a copy: copy(t)"


def test_an_object_built_only_to_be_lent_holds_its_fields(tmp_path: Path) -> None:
    # `f(World(log: #[5]))`: the object is freed after the statement, and the list with it, once.
    source = r"""
let io = import("std:stdio")
let World = struct { let score: i64, let log: i64[] }
let f = [] (w: World) => i64 w.score + len(w.log)
let main = [io.printf, f] () => null {
    io.printf("%d\n", f(World(score: 1, log: #[5])))
    let xs = #[1, 2]
    io.printf("%d\n", f(World(score: 1, log: xs)))
}
"""
    lines, leaked = _run(tmp_path, source)
    assert lines == ["2", "3"]
    assert leaked == 0
    # What fills it moves in, as into a kept object.
    source_after = source.replace(
        '    io.printf("%d\\n", f(World(score: 1, log: xs)))',
        '    io.printf("%d\\n", f(World(score: 1, log: xs)))\n    io.printf("%d\\n", len(xs))',
    )
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source_after)
    assert error.value.msg == "xs was moved on line 8, so it is no longer here"


def test_editor(tmp_path: Path) -> None:
    # Hints write generic objects as Bifrost does; the names in a function type are not parameters.
    path = tmp_path / "main.bif"
    source = FUNCTIONS.replace("(f: (x: T) => U, x: T) => U f(x)", "(f: (x: T) => U, value: T) => U f(value)")
    path.write_text(source)
    document = Document.open(path, source)
    assert [d.message for d in document.diagnostics() if "never used" in d.message] == []
    assert ": Pair[f64, i64]" in [hint.label for hint in document.inlay_hints()]
    assert format_source(FUNCTIONS).count("let pick[T] = ") == 1
    assert format_source(ENGINE) == format_source(format_source(ENGINE))
    assert "let updates: ((n: i64, w: W) => W)[]" in format_source(ENGINE)


def test_a_generic_objects_functions(tmp_path: Path) -> None:
    # Each object type made from a generic one gets its functions, compiled for its types:
    # methods that read it, change it (lent), and make another of the same type.
    source = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Box[T] = struct {
    let value: T,
    let items: T[],
    let get = [super] () => T super.value,
    let bump = [super] (by: T) => null {
        let b <- super
        b.value = b.value + by
        b.items = #[...b.items, by]
        b -> super
    },
    let mapped = [super] (f: (x: T) => T) => Box[T] Box(value: f(super.value), items: super.items.map(f)),
    let describe = [super, fmt.format] () => mem.Unique[str] fmt.format("Box(%v, %v)", super.value, super.items)
}

let double = [] (x: i64) => i64 x * 2
let half = [] (x: f64) => f64 x / 2.0

let main = [io.printf, io.puts, double, half] () => null {
    let b = Box(value: 5, items: #[1])
    b.bump(2)
    io.printf("%d\n", b.get())
    let c = b.mapped(double)
    io.puts(c.describe())
    let f = Box(value: 3.0, items: #[1.0])
    io.puts(f.mapped(half).describe())
}
"""
    lines, leaked = _run(tmp_path, source)
    assert lines == ["7", "Box(14, [2, 4])", "Box(1.5, [0.5])"]
    assert leaked == 0
