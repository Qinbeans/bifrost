"""Methods: an object's functions without ``static``, called on a value, which they read as ``super``.

``%v`` prints an object with its ``to_string`` method, and any other value as its type calls for.
"""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

PROGRAM = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Team = struct {
    let name: str,
    let members: str[],
    let size = [super] () => i64 len(super.members),
    let to_string = [super, fmt.format] () => mem.Unique[str] {
        return fmt.format("%s (%d members)", super.name, super.size())
    },
    let greet = [super, io.printf] (who: str) => null {
        io.printf("%s welcomes %s\n", super.name, who)
    }
}

let main = [io.printf, io.puts] () => null {
    let team = Team(name: "core", members: #["ada", "bob"])
    io.printf("%v\n", team)
    team.greet("cy")
    let teams = #[team, Team(name: "ops", members: #[])]
    io.printf("%v | %d | %v\n", teams, teams[1].size(), #{lead: teams[0], count: len(teams)})
    let text = teams[0].to_string()
    io.puts(text)
    let a = #[1...3]
    io.printf("%v %v %v %v %-6v| %v\n", a, 5000000000, 2.5, true, "text", #{x: 1.5, tags: #["a"]})
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


def _lower(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = Project(
        Config(
            path=tmp_path,
            package={"name": "methods", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O1, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


CHANGING = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Counter = struct {
    let hits: i64,
    let names: mem.Unique[str][],
    let size = [super] () => i64 len(super.names),
    let bump = [super] (by: i64) => null {
        let c <- super
        c.hits = c.hits + by
        c -> super
    },
    let add = [super, fmt.format] (name: str) => i64 {
        let c <- super
        c.names = #[...c.names, fmt.format("%s", name)]
        let n = c.size()
        c -> super
        return n
    },
    let twice = [super] () => null {
        let c <- super
        c.bump(1)
        c.bump(1)
        c -> super
    }
}

let tick = [] (counter: mem.Weak[Counter]) => null {
    counter.bump(10)
}

let main = [io.printf, tick] () => null {
    let counter = Counter(hits: 0, names: #[])
    counter.bump(2)
    counter.twice()
    let n = counter.add("ada")
    io.printf("%d %d %d\n", counter.hits, n, counter.size())

    let unique: mem.Unique[Counter] = Counter(hits: 0, names: #[])
    tick(unique)
    unique.bump(1)
    let u <- unique
    io.printf("%d\n", u.hits)
    u -> unique

    let counters = #[Counter(hits: 1, names: #[]), Counter(hits: 5, names: #[])]
    let g <- counters
    g[1].bump(100)
    g[0].add("bob")
    g -> counters
    io.printf("%d %d %d\n", counters[0].hits, counters[1].hits, counters[0].size())

    let shared: mem.Shared[Counter] = Counter(hits: 0, names: #[])
    shared.bump(7)
    shared.add("cy")
    let s <- shared
    io.printf("%d %d\n", s.hits, s.size())
    s -> shared
}
"""


def test_methods_that_change_their_object(tmp_path: Path) -> None:
    # A method that locks super is lent its object: a local, a list's item through a guard,
    # a mem.Weak (passed on) or a mem.Shared (locked for the call) sees the change.
    executable = _lower(tmp_path, CHANGING).build()
    (tmp_path / "count.c").write_text(COUNTER)
    counter = tmp_path / "count.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    assert result.stdout.splitlines() == ["4 1 1", "11", "1 105 1", "7 1"]
    allocations, frees = map(int, result.stderr.split())
    assert allocations - frees == 1


def test_methods_and_printing_values(tmp_path: Path) -> None:
    executable = _lower(tmp_path, PROGRAM).build()
    (tmp_path / "count.c").write_text(COUNTER)
    counter = tmp_path / "count.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    assert result.stdout.splitlines() == [
        "core (2 members)",
        "core welcomes cy",
        '[core (2 members), ops (0 members)] | 0 | {"lead": core (2 members), "count": 2}',
        "core (2 members)",
        '[1, 2] 5000000000 2.5 true text  | {"x": 1.5, "tags": ["a"]}',
    ]
    allocations, frees = map(int, result.stderr.split())
    assert allocations - frees == 1  # each to_string's text freed once printed; stdout keeps its buffer


# An object with a method that changes it (`bump`) and one that also returns (`add`).
_C = (
    "let C = struct { let hits: i64, let names: i64[], let bump = [super] () => null {\n    let c <- super\n"
    "    c.hits = c.hits + 1\n    c -> super\n}, let add = [super] () => i64 {\n    let c <- super\n"
    "    c.names = #[]\n    c -> super\n    return 1\n} }\n"
)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            'let P = struct { let x: i64 }\nlet f = [io.printf] () => null {\n    io.printf("%v", P(x: 1))\n}',
            "%v prints an object with its to_string, which P does not have; add one",
        ),
        (
            "let P = struct { let x: i64 }\nlet f = [] () => null {\n    let p = P(x: 1)\n    let y = p.nope()\n}",
            "P has no method nope",
        ),
        ("let P = struct { let x: i64, let get = [] () => i64 super.x }", "P.get reads super without depending on it"),
        (
            "let P = struct { let x: i64, static let get = [super] () => i64 1 }",
            "P.get is not called on an object, so it has no super to depend on",
        ),
        ("let f = [] () => i64 super.x", "super is the object a method is called on"),
        (
            "let P = struct { let x: i64, let get = [super] () => i64 super.x }\nlet f = [] () => i64 P.get()",
            "P.get is a method, called on a P (value.get(...)); call it on one",
        ),
        (
            "let P = struct { let x: i64, let to_string = [super] () => i64 super.x }\n"
            'let f = [io.printf] () => null {\n    io.printf("%v", P(x: 1))\n}',
            "P's to_string returns i64; it must return a string",
        ),
        (
            f"{_C}let D = struct {{ let hits: i64, let bad = [super] () => i64 {{\n    let c <- super\n    c -> super\n"
            "    return super.hits\n} }",
            "super is a mem.Weak; lock it (`let guard <- super`) to reach its fields",
        ),
        (
            "let D = struct { let hits: i64, let bad = [] () => null {\n    let c <- super\n    c.hits = 1\n"
            "    c -> super\n} }",
            "D.bad reads super without depending on it; add it: [super]",
        ),
        (
            f"{_C}let f = [] (c: C) => null {{\n    c.bump()\n}}",
            "bump changes its object, but c is a copy given to this function, so its caller would not see the change",
        ),
        (
            f"{_C}let f = [] () => null {{\n    let cs = #[C(hits: 1)]\n    forall c in cs {{\n        c.bump()\n"
            "    }\n}",
            "bump changes its object, but c is a view of cs (a copy of an item); change it through a guard on cs",
        ),
        (
            f"{_C}let D = struct {{ let c: C, let bad = [super] () => null {{\n    super.c.bump()\n}} }}",
            "bump changes its object, but this method reads super as a copy; lock super here to change it",
        ),
        (
            f'{_C}let f = [io.printf] () => null {{\n    let c = C(hits: 1)\n    io.printf("%d", c.add())\n}}',
            "call add in a statement of its own: it lends c for the call",
        ),
        (
            f"{_C}let f = [io.printf] () => null {{\n    let c = C(hits: 1)\n    let ns = c.names\n    c.bump()\n"
            '    io.printf("%v", ns)\n}',
            "c is read by ns",
        ),
        (
            f"{_C}let f = [] () => null {{\n    C(hits: 1).bump()\n}}",
            "bump changes its object, so call it on a name or through a guard (g[i].bump()), not on C(hits: 1)",
        ),
        (
            f"{_C}let f = [] () => null {{\n    let cs = #[C(hits: 1)]\n    cs[0].bump()\n}}",
            "bump changes cs[0]; change it through a guard: `let g <- cs`",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, 'let io = import("std:stdio")\n' + source + "\n")
    assert message in error.value.msg
