"""Unions: ``let Actor = Employee | Service``, made by putting an object where one goes, read with match."""

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

PROGRAM = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Employee = struct {
    let type: "employee",
    let name: mem.Unique[str],
    let age: i64
}

let Service = struct {
    let type: "service",
    let name: str,
    let replicas: i64[]
}

let Robot = struct { let serial: i64, let type: "robot" }

let Actor = Employee | Service | Robot on "type"

let describe = [io.printf] (actor: Actor) => null {
    match actor {
        Employee(e): io.printf("employee %s, %d\n", e.name, e.age),
        Service(s): io.printf("service %s x%d\n", s.name, len(s.replicas)),
        Robot(r): io.printf("robot %d\n", r.serial)
    }
}

let hire = [fmt.format] (n: i64) => Actor {
    if n > 1 {
        return Service(name: "api", replicas: #[1, 2, n])
    }
    return Employee(name: fmt.format("worker %d", n), age: 30 + n)
}

let label = [] (actor: Actor) => i64 {
    match actor {
        Employee(_): {
            return 1
        },
        Service(_): {
            return 2
        },
        Robot(_): {
            return 3
        }
    }
}

let serial = [] (actor: Actor) => i64 {
    let found = match actor { Robot(r): r.serial, _: -1 }
    return found
}

let main = [io.printf, fmt.format, describe, hire, label, serial] () => null {
    let first: Actor = Employee(name: fmt.format("ada"), age: 36)
    describe(first)
    describe(hire(1))
    let service = hire(3)
    describe(service)
    describe(Robot(serial: 7))
    let robot = Robot(serial: 9)
    let held: Actor = robot
    io.printf("%d %d %d\n", label(first), label(service), label(held))
    io.printf("%d %d\n", serial(held), serial(first))
    let copied = copy(first)
    describe(copied)
    let n = serial(held)
    let picked: Actor = match n { 9: Robot(serial: 1), _: Service(name: "x", replicas: #[]) }
    describe(picked)
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "unions", "version": "0", "description": ""},
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


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_unions(tmp_path: Path) -> None:
    # An object put where a union goes joins it (a let's type, an argument, a returned value, a
    # match's arm); match reads the one it holds. Owned objects are freed with their union.
    lines, leaked = _run(tmp_path, PROGRAM)
    assert lines == [
        "employee ada, 36",
        "employee worker 1, 31",
        "service api x3",
        "robot 7",
        "1 2 3",
        "9 -1",
        "employee ada, 36",
        "robot 1",
    ]
    assert leaked == 0


ALONGSIDE = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Employee = struct { let type: "employee", let name: mem.Unique[str], let age: i64 }
let Service = struct { let type: "service", let name: str }
let Actor = Employee | Service on "type"

module zoo = {
    let Cat = struct { let kind: "cat", let lives: i64 }
    let Dog = struct { let kind: "dog", let good: bool }
    let Pet = Cat | Dog on "kind"
}

let Team = struct { let lead: Actor, let size: i64 }

let show = [io.printf] (a: Actor) => null {
    match a { Employee(e): io.printf("employee %s\n", e.name), Service(s): io.printf("service %s\n", s.name) }
}

let find = [fmt.format] (n: i64) => Result[Actor, str] {
    if n > 0 {
        return ok(Employee(name: fmt.format("bo"), age: n))
    }
    return err("none")
}

let keep[T] = [] (x: T) => T {
    let y: T = x
    return y
}

let promote = [] (c: mem.Weak[Team]) => null {
    let g <- c
    g.lead = Service(name: "lead")
    g -> c
}

let main = [io.printf, fmt.format, show, find, keep, promote] () => null {
    let team = Team(lead: Employee(name: fmt.format("ann"), age: 1), size: 2)
    show(team.lead)
    promote(team)
    show(team.lead)
    let r = find(2)
    match r { ok(a): show(a), err(e): io.printf("%s\n", e) }
    let xs: Actor[] = #[Employee(name: fmt.format("cy"), age: 1), Service(name: "db")]
    forall x in xs {
        show(x)
    }
    let p: zoo.Pet = zoo.Cat(lives: 9)
    match p { zoo.Cat(c): io.printf("cat %d\n", c.lives), zoo.Dog(_): io.printf("dog\n") }
    let kept: zoo.Pet = keep(zoo.Dog(good: true))
    match kept { zoo.Cat(_): io.printf("cat\n"), zoo.Dog(d): io.printf("dog %d\n", d.good) }
    io.printf("%d\n", keep(41) + 1)
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_unions_alongside(tmp_path: Path) -> None:
    # A union in an object's field (set through a guard too), a Result, a list, a module and a
    # generic (compiled once with a union and once without) holds and frees what it is given.
    lines, leaked = _run(tmp_path, ALONGSIDE)
    assert lines == [
        "employee ann",
        "service lead",
        "employee bo",
        "employee cy",
        "service db",
        "cat 9",
        "dog 1",
        "42",
    ]
    assert leaked == 0


BRANCHES = r"""
let io = import("std:stdio")
let Cat = struct { let kind: "cat", let lives: i64 }
let Dog = struct { let kind: "dog", let good: bool }
let Pet = Cat | Dog on "kind"
let pick = [] (n: i64) => Pet if n > 1 { Cat(lives: n) } else if n == 0 { Dog(good: true) } else { Dog(good: false) }
let main = [pick, io.printf] () => null {
    forall n in #[0...3] {
        let p = pick(n)
        let q: Pet = if n == 1 { Cat(lives: 1) } else { p }
        match q { Cat(c): io.printf("cat %d\n", c.lives), Dog(d): io.printf("dog %d\n", d.good) }
    }
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_if_branches_join_a_union(tmp_path: Path) -> None:
    # Each branch of an `if` used as a value joins the union it goes to (one owning nothing: an
    # `if` cannot yet give a value that owns memory); a branch that is the union already stays.
    lines, leaked = _run(tmp_path, BRANCHES)
    assert lines == ["dog 1", "cat 1", "cat 2"]
    assert leaked == 0


JSON = r"""
let io = import("std:stdio")
let json = import("std:json")
let mem = import("std:mem")
let fmt = import("std:fmt")

let Employee = struct { let type: "employee", let name: mem.Unique[str], let age: i64 }
let Service = struct { let name: mem.Unique[str], let type: "service" }
let Actor = Employee | Service on "type"

let show = [io.printf] (a: Actor) => null {
    match a {
        Employee(e): io.printf("employee %s %d (%s)\n", e.name, e.age, e.type),
        Service(s): io.printf("service %s (%s)\n", s.name, s.type)
    }
}

let read = [json.decode, show, io.printf] (text: str) => null {
    let r: Result[Actor, mem.Unique[str]] = json.decode(text)
    match r { ok(a): show(a), err(e): io.printf("error: %s\n", e) }
}

let main = [fmt.format, io.printf, json.encode, json.decode, read] () => null {
    let a: Actor = Employee(fmt.format("ada"), 36)
    let text = json.encode(a)
    io.printf("%s\n", text)
    read(text)
    read(
        "{{\"name\": \"api\", \"extra\": [1, {{\"type\": \"x\"}}], \"type\": \"service\"}}"
    )
    read("{{\"name\": \"api\", \"type\": \"robot\"}}")
    read("{{\"name\": \"api\"}}")
    read("{{\"type\": \"employee\", \"name\": \"bo\"}}")
    let xs: Actor[] = #[Service(name: fmt.format("db")), a]
    io.printf("%s\n", json.encode(xs))
    let r: Result[Employee, mem.Unique[str]] = json.decode(
        "{{\"type\": \"service\", \"name\": \"x\", \"age\": 1}}"
    )
    match r { ok(e): io.printf("%s\n", e.name), err(e): io.printf("error: %s\n", e) }
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_unions_as_json(tmp_path: Path) -> None:
    # A union is written as the object it holds, its literal field with it; decoding reads that
    # field first (wherever it is) to know which object follows, and checks an object's own.
    lines, leaked = _run(tmp_path, JSON)
    assert lines == [
        '{"type": "employee", "name": "ada", "age": 36}',
        "employee ada 36 (employee)",
        "service api (service)",
        'error: no object has `type` "robot"',
        "error: missing field `type`",
        "error: missing field `age`",
        '[{"name": "db", "type": "service"}, {"type": "employee", "name": "ada", "age": 36}]',
        'error: expected "employee" at 9, near \'"service", "\'',
    ]
    assert leaked == 0


SHARED = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let Cat = struct { let kind: "cat", let name: mem.Unique[str], let lives: i64 }
let Dog = struct { let kind: "dog", let name: mem.Unique[str], let good: bool }
let Pet = Cat | Dog on "kind"
let show = [io.printf] (p: Pet) => null {
    io.printf("%s the %s\n", p.name, p.kind)
}
let main = [io.printf, fmt.format, show] () => null {
    let p: Pet = Dog(name: fmt.format("rex"), good: true)
    show(p)
    let n = p.name
    io.printf("%s\n", n)
    let q: Pet = Cat(fmt.format("tom"), 9)
    show(q)
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_what_each_object_has_alike(tmp_path: Path) -> None:
    # A field each object of a union has, of one type, is read from the one it holds (a view of it).
    lines, leaked = _run(tmp_path, SHARED)
    assert lines == ["rex the dog", "rex", "tom the cat"]
    assert leaked == 0


PRINTED = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let Cat = struct {
    let kind: "cat",
    let lives: i64,
    let to_string = [super, fmt.format] () => mem.Unique[str] fmt.format("cat(%d)", super.lives)
}
let Dog = struct {
    let kind: "dog",
    let good: bool,
    let to_string = [super, fmt.format] () => mem.Unique[str] fmt.format("dog(%d)", super.good)
}
let Pet = Cat | Dog on "kind"
let main = [io.printf] () => null {
    let p: Pet = Cat(9)
    let q: Pet = Dog(true)
    io.printf("%v and %v\n", p, q)
    let pets: Pet[] = #[p, q]
    io.printf("%v\n", pets)
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_printing_a_union(tmp_path: Path) -> None:
    # %v prints a union with its to_string: here each object's, called for the one it holds.
    lines, leaked = _run(tmp_path, PRINTED)
    assert lines == ["cat(9) and dog(1)", "[cat(9), dog(1)]"]
    assert leaked == 0


BRANCHING = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let Employee = struct { let type: "employee", let name: mem.Unique[str] }
let Service = struct { let type: "service", let name: str }
let Actor = Employee | Service on "type"
let loud = [io.printf, fmt.format] (s: str) => mem.Unique[str] {
    io.printf("making %s\n", s)
    return fmt.format("%s!", s)
}
let pick = [loud, fmt.format] (n: i64) => Actor if n > 1 {
    Employee(name: loud("emp"))
} else if n == 1 {
    Service(name: "svc")
} else {
    Employee(name: fmt.format("zero"))
}
let main = [io.printf, loud, pick] () => null {
    io.printf("%s\n", if true {
        loud("a")
    } else {
        loud("b")
    })
    forall n in #[0...3] {
        let a = pick(n)
        match a {
            Employee(e): io.printf("%s\n", e.name),
            Service(s): io.printf("%s\n", s.name)
        }
    }
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to count allocations")
def test_if_branches_make_an_owning_union(tmp_path: Path) -> None:
    # An `if` used as a value makes only the branch it takes (its side effects too), and gives what
    # that branch made: here a union owning memory, from an `else if` chain.
    lines, leaked = _run(tmp_path, BRANCHING)
    assert lines == ["making a", "a!", "zero", "svc", "making emp", "emp!"]
    assert leaked == 0


HEAD = """
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let Employee = struct { let type: "employee", let name: mem.Unique[str], let age: i64 }
let Service = struct { let type: "service", let name: str }
let Robot = struct { let type: "robot", let serial: i64 }
let Actor = Employee | Service on "type"
let make = [fmt.format] () => Actor Employee(name: fmt.format("a"), age: 1)
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "let f = [] (a: Actor) => i64 {\n    match a {\n        Employee(e): {\n            return 1\n        }\n"
            "    }\n}",
            "f must return i64, but its body can end without a `return`",
        ),
        (
            'let f = [io.puts] (a: Actor) => null {\n    match a { Robot(r): io.puts("r"), _: io.puts("x") }\n}',
            "Actor is Employee | Service; Robot is not one of them",
        ),
        (
            "let f = [io.puts] (a: Actor) => null {\n"
            '    match a { Service(s): io.puts("s"), Service(t): io.puts("t"), _: io.puts("x") }\n}',
            "this match already has a Service(...) arm",
        ),
        (
            'let f = [io.puts] (a: Actor) => null {\n    match a { 1: io.puts("one"), _: io.puts("x") }\n}',
            "a match on Actor has the arms Employee(...), Service(...), and _",
        ),
        (
            "let f = [] (a: Actor) => i64 a.age",
            "Actor is Employee | Service, so it has no 'age'; read what it holds with match",
        ),
        (
            "let f = [] (a: Actor) => i64 a.tag",
            "Actor is Employee | Service, so it has no 'tag'",
        ),
        (
            "let f = [] (a: Actor) => i64 a.size()",
            "Actor is Employee | Service, so it has no 'size'",
        ),
        (
            "let f = [] () => null {\n    let a: Actor = Robot(serial: 1)\n}",
            "Actor is Employee | Service, but this is Robot",
        ),
        ("let f = [] () => null {\n    let a: Actor = 3\n}", "Actor is Employee | Service, but this is i64"),
        ('let Twice = Robot | Robot on "type"', "Robot is in Twice twice"),
        ('let Odd = Robot | make on "type"', "a union is of objects, and 'make' is not one"),
        ('let Nested = Actor | Robot on "type"', "a union holding a union (Actor) is not supported yet"),
        (
            'let f = [make, io.puts] () => null {\n    match make() { Employee(e): io.puts("e"), _: io.puts("x") }\n}',
            "name this Actor first, since it owns what it holds: let a = make()",
        ),
        (
            "let C = struct { let a: Actor }\n"
            "let f = [] (c: mem.Weak[C]) => null {\n    let g <- c\n    g.a.tag = 1\n    g -> c\n}",
            "Actor is Employee | Service, so it has no 'tag'",
        ),
        (
            'let f = [io.printf] () => null {\n    let a: Actor = Service(name: "q")\n    io.printf("%v", a)\n}',
            "%v prints a union with its to_string: give each of Employee | Service one returning a string",
        ),
        # A union's own functions, and what it reads of its objects.
        ('let Odd = Robot | Service on "type" {\n    let n: i64\n}', "Odd holds no fields of its own"),
        (
            "let f = [] (a: Actor) => str a.name",
            "Actor reads only what each of its objects has alike, and their names differ "
            "(Employee's is mem.Unique[str], Service's is str)",
        ),
        (
            'let A = struct { let type: "a", let f = [] (x: i64) => i64 x }\n'
            'let B = struct { let type: "b", let f = [] (x: str) => i64 1 }\n'
            'let AB = A | B on "type"\nlet g = [] (v: AB) => i64 v.f(1)',
            "AB calls f only when each of its objects has one taking and giving the same",
        ),
        (
            'let form = import("std:form")\nlet f = [form.decode] () => null {\n'
            '    let r: Result[Actor, mem.Unique[str]] = form.decode("")\n}',
            "a form holds no unions (Actor)",
        ),
        # What tells a union's objects apart: named, a literal field of each, a different text.
        (
            "let Bare = Robot | Service",
            'a union names the literal field telling its objects apart: let Bare = Robot | Service on "type"',
        ),
        (
            'let Plain = struct { let id: i64 }\nlet NoTag = Robot | Plain on "type"',
            'NoTag tells its objects apart by type, which Plain does not have: let type: "plain"',
        ),
        (
            'let Loose = struct { let type: str }\nlet Free = Robot | Loose on "type"',
            "Free tells its objects apart by type, which Loose can set to any text; fix it",
        ),
        (
            'let Clone = struct { let type: "robot", let id: i64 }\nlet Same = Robot | Clone on "type"',
            'Robot and Clone both have type "robot"; each object of Same needs its own',
        ),
        ('let Odd = Robot | 3 on "type"', 'a union is of objects joined by |: let Odd = A | B on "type"'),
        # A literal field holds its text, always.
        ('let f = [] () => Robot Robot(type: "robot", serial: 1)', 'Robot\'s type is always "robot"; leave it out'),
        ("let f = [] () => Robot Robot(1, 2)", "Robot takes 1 values (serial), got 2"),
        (
            'let f = [] (r: mem.Weak[Robot]) => null {\n    let g <- r\n    g.type = "robot"\n    g -> r\n}',
            'Robot\'s type is always "robot"; it cannot change',
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, HEAD + source + "\nlet main = [] () => null {\n}\n")
    assert message in error.value.msg
