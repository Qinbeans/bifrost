"""Lists: ``#[1, 2, 3]`` values, ``T[]`` types, indexing, ``len``, ``forall``, and spreading.

A list holds any type (numbers, strings, records, objects, other lists), and
has one owner, which frees it (see ``bifrost.ownership``); records and objects
can hold lists. ``#[a...b]`` is the range from ``a`` up to ``b`` (not
included): as a value, a list of those numbers; in ``forall``, counting, with
no list made. ``#[...xs, x]`` makes a new list, or, when it replaces ``xs``,
grows it in place.
"""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

PROGRAM = r"""
let io = import("std:stdio")
let time = import("std:time")

let total = [] (xs: i64[]) => i64 {
    let sum = 0
    forall x in xs {
        let sum = sum + x
    }
    return sum
}

let upto = [] (n: i64) => i64[] #[0...n]

let listed = [] () => i64[] #[4, 8, 15, 16, 23, 42]

let ticks = [time.sleep, io.printf] async (count: i64) => null {
    forall i in #[0...count] {
        io.printf("tick %lld\n", i)
        await time.sleep(1)
    }
}

let main = [io.printf, total, upto, listed, ticks] async () => null {
    await ticks(2)
    let scores = listed()
    io.printf("%lld %lld %lld\n", len(scores), scores[0], scores[-1])
    io.printf("%lld %lld\n", total(scores), total(upto(5)))
    let ratios: f64[] = #[0.5, 1.5]
    io.printf("%.1f\n", ratios[1])
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "lists", "version": "0", "description": "", "type": "async"},
            flags={"optimization": OptLevel.O2, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _compile(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = _project(tmp_path)
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def test_lists(tmp_path: Path) -> None:
    executable = _compile(tmp_path, PROGRAM).build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output.splitlines() == ["tick 0", "tick 1", "6 4 42", "108 10", "1.5"]


# Counts the program's calls to the allocator, to show every list is freed once.
COUNTER = r"""
#include <stdio.h>
#include <stdlib.h>
/* glibc's own allocator, under names that are not replaced (dlsym itself calls calloc).
   Coroutine frames come from aligned_alloc. */
void *__libc_malloc(size_t);
void *__libc_calloc(size_t, size_t);
void *__libc_realloc(void *, size_t);
void __libc_free(void *);
void *__libc_memalign(size_t, size_t);
static long allocations, frees;
void *malloc(size_t n) { allocations++; return __libc_malloc(n); }
void *calloc(size_t n, size_t size) { allocations++; return __libc_calloc(n, size); }
void *realloc(void *p, size_t n) { if (!p) allocations++; return __libc_realloc(p, n); }
void *aligned_alloc(size_t alignment, size_t n) { allocations++; return __libc_memalign(alignment, n); }
void free(void *p) { if (p) frees++; __libc_free(p); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "%ld %ld\n", allocations, frees); }
"""

OWNED = r"""
let io = import("std:stdio")
let json = import("std:json")

let Team = struct { let name: str, let members: str[] }

let squares = [] (n: i64) => i64[] {
    let xs: i64[] = #[]
    forall i in #[0...n] {
        let xs = #[...xs, i * i]
    }
    return xs
}

let rows = [squares] () => #{name: str, scores: i64[]}[] {
    let found: #{name: str, scores: i64[]}[] = #[]
    let found = #[...found, #{name: "ada", scores: #[3, 4]}]
    let extra = #{name: "bob", scores: squares(3)}
    let found = #[...found, extra]
    return found
}

let total = [] (xs: i64[]) => i64 {
    let sum = 0
    forall x in xs {
        let sum = sum + x
    }
    return sum
}

let main = [io.printf, io.puts, json.encode, squares, rows, total] () => null {
    let big = squares(1000)
    io.printf("%lld %lld %lld\n", len(big), big[999], big[-2])
    let small = squares(4)
    let both = #[0, ...small, ...small, 99]
    io.printf("%lld %lld %lld\n", len(both), total(both), both[-1])
    let copy = #[...small]
    let small = #[100, ...small]
    io.printf("%lld %lld %lld\n", len(small), small[0], copy[0])
    let found = rows()
    forall row in found {
        io.printf("%s %lld %lld\n", row.name, len(row.scores), total(row.scores))
    }
    let team = Team(name: "core", members: #["ada"])
    let guard <- team
    guard.members = #[...guard.members, "bob"]
    guard -> team
    io.printf("%s %s %lld\n", team.name, team.members[1], len(team.members))
    let grid = #[#[1, 2], #[3, 4, 5]]
    io.printf("%lld %lld\n", len(grid[1]), grid[1][2])
    let text = json.encode(#{rows: found, tags: #["x", "y"], grid: grid, empty: squares(0)})
    io.puts(text)
    io.printf("%lld\n", total(#[1, 2, 3]) + len(squares(5)))
}
"""

ASYNC = r"""
let io = import("std:stdio")
let time = import("std:time")
let tasks = import("std:tasks")
let json = import("std:json")

let fetch = [time.sleep] async (n: i64) => Record {
    let ids: i64[] = #[]
    forall i in #[0...n] {
        await time.sleep(1)
        let ids = #[...ids, i * 10]
    }
    return #{count: n, ids: ids}
}

let only_ids = [fetch] async (n: i64) => i64[] {
    let found = await fetch(n)
    return #[...found.ids]
}

let main = [io.printf, io.puts, json.encode, tasks.gather, fetch, only_ids] async () => null {
    let both = await tasks.gather(a: fetch(3), b: fetch(2))
    io.printf("%lld %lld %lld\n", both.a.ids[2], len(both.b.ids), both.b.count)
    io.printf("%lld\n", len(await only_ids(2)))
    let text = json.encode(both)
    io.puts(text)
}
"""


def _counted(tmp_path: Path, source: str) -> tuple[list[str], int, int]:
    """Build and run ``source``, counting its allocations and frees."""
    executable = _compile(tmp_path, source).build()
    counter = tmp_path / "count.so"
    (tmp_path / "count.c").write_text(COUNTER)
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    allocations, frees = map(int, result.stderr.split())
    return result.stdout.splitlines(), allocations, frees


def test_lists_hold_records_and_records_hold_lists(tmp_path: Path) -> None:
    lines, allocations, frees = _counted(tmp_path, OWNED)
    assert lines == [
        "1000 998001 996004",
        "10 127 99",
        "5 100 0",
        "ada 2 7",
        "bob 3 5",
        "core bob 2",
        "3 5",
        '{"rows": [{"name": "ada", "scores": [3, 4]}, {"name": "bob", "scores": [0, 1, 4]}], '
        '"tags": ["x", "y"], "grid": [[1, 2], [3, 4, 5]], "empty": []}',
        "11",
    ]
    # Every list freed once (libc keeps one buffer, for stdout); growing to 1000 items one at a time
    # grows in place, so the whole program allocates a few dozen times, not a thousand.
    assert allocations - frees == 1
    assert allocations < 50


def test_lists_in_async_functions(tmp_path: Path) -> None:
    lines, allocations, frees = _counted(tmp_path, ASYNC)
    assert lines == ["20 2 2", "2", '{"a": {"count": 3, "ids": [0, 10, 20]}, "b": {"count": 2, "ids": [0, 10]}}']
    assert allocations - frees == 1


STRINGS = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let json = import("std:json")
let mem = import("std:mem")

let User = struct { let id: i64, let name: mem.Unique[str] }

let names = [fmt.format] (count: i64) => mem.Unique[str][] {
    let found: mem.Unique[str][] = #[]
    forall i in #[0...count] {
        let found = #[...found, fmt.format("user %lld", i)]
    }
    return found
}

let label = [fmt.format] (id: i64) => mem.Unique[str] fmt.format("#%lld", id)

let main = [io.printf, io.puts, fmt.format, json.encode, names, label] () => null {
    let all = names(100)
    io.printf("%lld %s %s\n", len(all), all[0], all[-1])
    let some = all[1...3]
    let copied = #[...some]
    io.printf("%s %s %lld %lld\n", some[1], copied[0], len(all[...-98]), len(all[200...]))
    let text = fmt.format("guest")
    let users: User[] = #[User(id: 1, name: text)]
    let users = #[...users, User(id: 2, name: label(2))]
    io.printf("%s %lld\n", users[1].name, users[-1].id)
    let encoded = json.encode(#{users: users, names: some, row: #{title: label(3), tags: #["a", "b"]}})
    io.puts(encoded)
}
"""

CELLS = r"""
let io = import("std:stdio")
let mem = import("std:mem")

let Board = struct { let title: str, let scores: i64[] }

let record = [] (board: mem.Shared[Board], score: i64) => null {
    let guard <- board
    guard.scores = #[...guard.scores, score]
    guard -> board
}

let bump = [] (board: mem.Weak[Board]) => null {
    let guard <- board
    guard.scores = #[...guard.scores, 1000]
    guard -> board
}

let main = [io.printf, record, bump] () => null {
    let shared: mem.Shared[Board] = Board(title: "shared", scores: #[])
    let other = shared
    forall i in #[0...100] {
        record(shared, i)
    }
    let guard <- other
    io.printf("%lld %lld\n", len(guard.scores), guard.scores[99])
    guard -> other
    let atomic: mem.Atomic[Board] = Board(title: "atomic", scores: #[1, 2])
    let local: mem.Unique[Board] = Board(title: "unique", scores: #[5])
    bump(local)
    let reader <- local
    io.printf("%lld %lld\n", len(reader.scores), reader.scores[1])
    reader -> local
}
"""


def test_owned_strings_and_slices(tmp_path: Path) -> None:
    lines, allocations, frees = _counted(tmp_path, STRINGS)
    assert lines == [
        "100 user 0 user 99",
        "user 2 user 1 2 0",
        "#2 2",
        '{"users": [{"id": 1, "name": "guest"}, {"id": 2, "name": "#2"}], "names": ["user 1", "user 2"], '
        '"row": {"title": "#3", "tags": ["a", "b"]}}',
    ]
    assert allocations - frees == 1


def test_lists_in_shared_values(tmp_path: Path) -> None:
    # The last owner of a mem.Shared frees the list its value holds; a guard grows it in place.
    lines, allocations, frees = _counted(tmp_path, CELLS)
    assert lines == ["100 99", "2 1000"]
    assert allocations - frees == 1
    assert allocations < 30


ITEMS = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let User = struct { let name: mem.Unique[str], let age: i64 }
let Team = struct { let users: User[], let tags: mem.Unique[str][] }

let main = [io.printf, fmt.format] () => null {
    let grid = #[#[1, 2], #[3, 4]]
    let g <- grid
    g[0][1] = 20
    g[1] = #[30, 40, 50]
    g[0] = #[...g[0], 99]
    g[-1][-1] = g[0][0] + 1
    g -> grid
    io.printf("%v\n", grid)

    let team = Team(users: #[User(name: fmt.format("ada"), age: 36)], tags: #[fmt.format("a"), fmt.format("b")])
    let t <- team
    t.users[0].age = t.users[0].age + 1
    t.users[0].name = fmt.format("ada %d", 2)
    t.tags[-1] = fmt.format("z")
    t.users = #[...t.users, User(name: fmt.format("bob"), age: 5)]
    t.users[1] = User(name: fmt.format("cy"), age: 7)
    t -> team
    let first = team.users[0]
    io.printf("%s %d %s %s %d\n", first.name, first.age, team.tags[1], team.users[1].name, len(team.users))

    let shared: mem.Shared[Team] = Team(users: #[], tags: #[fmt.format("x")])
    let s <- shared
    s.tags[0] = fmt.format("y")
    s -> shared
    let r <- shared
    io.printf("%s\n", r.tags[0])
    r -> shared
}
"""


def test_items_change_through_a_guard(tmp_path: Path) -> None:
    # An item (or a field of one) replaced through a guard frees what it held, once.
    lines, allocations, frees = _counted(tmp_path, ITEMS)
    assert lines == ["[[1, 20, 99], [30, 40, 2]]", "ada 2 37 z cy 2", "y"]
    assert allocations - frees == 1


HEAD = (
    'let io = import("std:stdio")\nlet mem = import("std:mem")\nlet fmt = import("std:fmt")\n'
    "let Team = struct { let members: i64[] }\nlet User = struct { let name: mem.Unique[str] }\n"
    "let take = (t: mem.Unique[str]) => null {}\n"
)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("let f = [] () => null {\n    let xs = #[]\n}", "give an empty list its type: let xs: i64[] = #[]"),
        ("let f = [] () => null {\n    let xs = #[...3]\n}", "only a list can be spread into a list, not 3"),
        ("let f = [] () => null {\n    let n = len(1, 2)\n}", "len takes one list: len(xs)"),
        ("let f = [] () => null {\n    let n = len(3)\n}", "len takes a list, not a i64"),
        ("let f = [] () => null {\n    let xs = #[1, 2]\n    let n = 3[1...]\n}", "only a list can be sliced, not i64"),
        (
            "let f = [] (xs: i64[]) => i64[] {\n    return xs\n}",
            "xs is a parameter, lent by the caller, so it cannot be kept here; keep a copy: #[...xs]",
        ),
        (
            "let f = [] () => i64[] {\n    let t = Team(members: #[1])\n    return t.members\n}",
            "t.members is part of t, so it cannot be kept here; keep a copy: #[...t.members]",
        ),
        (
            "let f = [io.printf] () => null {\n    let xs = #[1]\n    let t = Team(members: xs)\n"
            '    io.printf("%lld", len(xs))\n}',
            "xs was moved on line 9, so it is no longer here",
        ),
        (
            "let f = [] () => null {\n    let xs = #[1, 2]\n    forall x in xs {\n"
            "        let xs = #[...xs, x]\n    }\n}",
            "xs is read by x (line 9) until the end of its block, so it cannot be replaced here",
        ),
        (
            "let f = [] () => null {\n    let xs = #[1]\n    let all: i64[][] = #[]\n    forall i in #[0...3] {\n"
            "        let all = #[...all, xs]\n    }\n}",
            "xs is moved inside a loop",
        ),
        (
            "let f = [] () => i64[] {\n    let grid = #[#[1], #[2]]\n    let row = grid[0]\n    return row\n}",
            "row reads from grid, so it cannot be kept here; keep a copy: #[...row]",
        ),
        (
            'let f = [] () => null {\n    let xs: mem.Unique[str][] = #["a"]\n}',
            'this list\'s strings are owned (mem.Unique[str]), but "a" is not',
        ),
        (
            'let f = [fmt.format] () => null {\n    let t = fmt.format("x")\n    let xs: str[] = #[t]\n}',
            "this list's strings are not owned (str), but t is; write the type as mem.Unique[str]",
        ),
        ('let f = [] () => null {\n    let u = User(name: "x")\n}', "User's name is owned (mem.Unique[str])"),
        (
            "let f = [] (users: User[]) => mem.Unique[str] {\n    return users[0].name\n}",
            'users[0].name is part of users, so it cannot be kept here; keep a copy: fmt.format("%s", users[0].name)',
        ),
        (
            "let f = [take] (users: User[]) => null {\n    take(users[0].name)\n}",
            "users[0].name is part of users, so it cannot be kept here",
        ),
        (
            "let f = [] () => null {\n    let xs = #[1]\n    xs[0] = 2\n}",
            "items change only through a guard: lock xs first (`let guard <- xs`)",
        ),
        (
            "let f = [] () => null {\n    let xs = #[1]\n    let g <- xs\n    xs[0] = 2\n    g -> xs\n}",
            "xs is locked by g (line 9); change it through g",
        ),
        (
            "let f = [] (xs: i64[]) => null {\n    let g <- xs\n    g[0] = 2\n    g -> xs\n}",
            "xs is lent to this function, so the items of its lists cannot change here (they are its caller's)",
        ),
        (
            "let f = [] () => null {\n    let grid = #[#[1]]\n    forall row in grid {\n        let g <- row\n"
            "        g[0] = 5\n        g -> row\n    }\n}",
            "row is a view of grid, so its items cannot change; change them through grid",
        ),
        (
            "let f = [] () => null {\n    let xs = #[1, 2]\n    let g <- xs\n    g[0...1] = #[3]\n    g -> xs\n}",
            "a slice is a new list, so changing it changes nothing; change items: g[i] = x",
        ),
        (
            'let f = [fmt.format] () => null {\n    let xs = #[fmt.format("a")]\n    let g <- xs\n    g[0] = "b"\n'
            "    g -> xs\n}",
            'g[0] is owned (mem.Unique[str]), but "b" is not; keep an owned copy: fmt.format("%s", "b")',
        ),
        (
            'let f = [fmt.format] () => null {\n    let u = User(name: fmt.format("a"))\n    let g <- u\n'
            '    g.name = "b"\n    g -> u\n}',
            'g.name is owned (mem.Unique[str]), but "b" is not',
        ),
        (
            "let f = [io.printf] () => null {\n    let rows = #[#[1], #[2]]\n    let first = rows[0]\n"
            '    let g <- rows\n    g[0] = #[9]\n    g -> rows\n    io.printf("%v", first)\n}',
            "rows is read by first (line 9) until the end of its block, so it cannot be changed here",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, HEAD + source + "\n")
    assert message in error.value.msg


def test_lists_format() -> None:
    source = (
        "let f = [] (xs: i64[]) => #{name: str, ids: i64[]}[] {\n    forall x in #[0...len(xs)] {\n"
        '        let y = xs[x]\n    }\n    return #[#{name: "a", ids: #[...xs, 1]}]\n}\n'
    )
    assert format_source(source) == source
