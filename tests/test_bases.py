"""Objects built on others: ``struct { ... } & Person`` has Person's fields and methods, and is a Person."""

import shutil
import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

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


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "bases", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
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


BUILT = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let Person = struct {
    let name: mem.Unique[str],
    let age: i64,
    let greet = [super, fmt.format] () => mem.Unique[str] fmt.format("hi %s", super.name)
}
let Employee = struct { let employee_id: mem.Unique[str] } & Person
let Manager = struct { let reports: i64[] } & Employee
let describe = [io.printf] (p: Person) => null {
    io.printf("%s is %d\n", p.name, p.age)
}
let promote = [fmt.format] (n: i64) => Person {
    let e = Employee(fmt.format("e%d", n), fmt.format("worker"), 30 + n)
    return e
}
let main = [io.printf, fmt.format, describe, promote] () => null {
    let e = Employee(fmt.format("e1"), fmt.format("ada"), 36)
    describe(e)
    describe(Employee(fmt.format("e2"), fmt.format("bo"), 20))
    let m = Manager(#[1, 2], fmt.format("m1"), fmt.format("cy"), 50)
    describe(m)
    io.printf("%s %s\n", m.greet(), m.employee_id)
    let p: Person = e
    describe(p)
    describe(promote(3))
    let people: Person[] = #[
        Employee(fmt.format("e3"), fmt.format("di"), 1),
        Person(name: fmt.format("ed"), age: 2)
    ]
    forall q in people {
        describe(q)
    }
    let kept: Person = Manager(#[3], fmt.format("m2"), fmt.format("fay"), 60)
    describe(kept)
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_an_object_is_one_it_is_built_on(tmp_path: Path) -> None:
    # An Employee (& Person) and a Manager (& Employee) are lent as a Person to a parameter, and
    # move into one where it is kept (a let, a return, a list): their own fields are freed then.
    lines, leaked = _run(tmp_path, BUILT)
    assert lines == [
        "ada is 36",
        "bo is 20",
        "cy is 50",
        "hi cy m1",
        "ada is 36",
        "worker is 33",
        "di is 1",
        "ed is 2",
        "fay is 60",
    ]
    assert leaked == 0


METHODS = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let Person = struct {
    let name: mem.Unique[str],
    let age: i64,
    let greet = [super, fmt.format] (greeting: str) => mem.Unique[str] fmt.format("%s %s", greeting, super.name),
    let years = [super] () => i64 super.age
}
let Employee = struct { let type: "employee", let employee_id: mem.Unique[str] } & Person
let Service = struct {
    let type: "service",
    let name: str,
    let years = [] () => i64 0,
    let greet = [super, fmt.format] (g: str) => mem.Unique[str] fmt.format("%s, service %s", g, super.name)
}
let Actor = Employee | Service on "type" {
    let kind = [super] () => i64 match super { Employee(_): 1, Service(_): 2 },
    static let make = [] () => Actor Service(name: "api")
}
let main = [io.printf, fmt.format, Actor.make] () => null {
    let e = Employee(fmt.format("e1"), fmt.format("ada"), 36)
    io.printf("%s %d\n", e.greet("hello"), e.years())
    let a: Actor = e
    io.printf("%s %d %d\n", a.greet("hey"), a.years(), a.kind())
    let s = Actor.make()
    io.printf("%s %d %d\n", s.greet("yo"), s.years(), s.kind())
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_methods_from_what_an_object_is_made_of(tmp_path: Path) -> None:
    # An Employee has Person's methods; a union has its own functions (methods and static ones),
    # and the methods each of its objects has alike, called for the one it holds.
    lines, leaked = _run(tmp_path, METHODS)
    assert lines == ["hello ada 36", "hey ada 36 1", "yo, service api 0 2"]
    assert leaked == 0


HEAD = """
let mem = import("std:mem")
let Person = struct { let name: str, let age: i64 }
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("let E = struct { let name: str } & Person", "E and Person both have a member named 'name'"),
        ("let limit = 3\nlet E = struct { let id: i64 } & limit", "E is built on objects, and 'limit' is not one"),
        ("let E = struct { let id: i64 } & Missing", "'Missing' is not defined"),
        (
            'let C = struct { let type: "c", let n: i64 }\nlet D = struct { let type: "d", let m: i64 }\n'
            'let CD = C | D on "type"\nlet E = struct { let id: i64 } & CD',
            "E cannot be built on a union (CD); build it on one of its objects",
        ),
        ("let Box[T] = struct { let v: T }\nlet E = struct { let id: i64 } & Box", "building on a generic object"),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, HEAD + source + "\nlet main = [] () => null {\n}\n")
    assert message in error.value.msg
