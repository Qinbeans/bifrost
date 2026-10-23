"""``Result[T, E]``: ``ok(x)`` and ``err(e)``, ``r?``, and ``match r { ok(v): ..., err(e): ... }``."""

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

let parse_age = [] (n: i64) => Result[i64, str] {
    if n < 0 {
        return err("negative")
    }
    return ok(n)
}

let older = [parse_age] (n: i64) => Result[i64, str] {
    let age = parse_age(n)?
    return ok(age + 1)
}

let check = [parse_age] (n: i64) => Result[null, str] {
    parse_age(n)?
    return ok()
}

let names = [fmt.format] (count: i64) => Result[mem.Unique[str][], mem.Unique[str]] {
    if count > 3 {
        return err(fmt.format("too many: %d", count))
    }
    let found: mem.Unique[str][] = #[]
    forall i in #[0...count] {
        let found = #[...found, fmt.format("name %d", i)]
    }
    return ok(found)
}

let first_name = [
    names,
    fmt.format
] (count: i64) => Result[mem.Unique[str], mem.Unique[str]] {
    let kept = #[1, 2, 3]
    let all = names(count)?
    return ok(fmt.format("%s of %d (%d)", all[0], len(all), len(kept)))
}

let describe = [io.printf] (r: Result[i64, str]) => null {
    match r { ok(v): io.printf("ok %d\n", v), err(e): io.printf("err %s\n", e) }
}

let main = [io.printf, older, check, names, first_name, describe] () => null {
    describe(older(41))
    describe(older(-1))
    let checked = check(-5)
    io.printf("%d %d\n", checked.is_ok(), checked.is_err())
    let some = names(2)
    match some {
        ok(list): io.printf("%d names, %s\n", len(list), list[1]),
        err(e): io.printf("%s\n", e)
    }
    let none = names(9)
    match none { ok(_): io.printf("unexpected\n"), err(e): io.printf("%s\n", e) }
    let a = first_name(3)
    let b = first_name(5)
    let shown = match a { ok(text): text, err(e): e }
    io.printf("%s\n", shown)
    match b { err(e): io.printf("failed: %s\n", e), _: io.printf("unexpected\n") }
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "results", "version": "0", "description": ""},
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


def test_results(tmp_path: Path) -> None:
    # ok and err make the Result written where they go; `?` passes an error on; match reads either
    # side. Owned values in a Result are freed with it, on whichever side it holds, and `?` frees
    # what is alive when it returns early.
    lines, leaked = _run(tmp_path, PROGRAM)
    assert lines == [
        "ok 42",
        "err negative",
        "0 1",
        "2 names, name 1",
        "too many: 9",
        "name 0 of 3 (3)",
        "failed: too many: 5",
    ]
    assert leaked == 0


HEAD = """
let io = import("std:stdio")
let parse = [] (n: i64) => Result[i64, str] {
    if n < 0 {
        return err("negative")
    }
    return ok(n)
}
let mem = import("std:mem")
"""


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("let f = [parse] () => i64 parse(1)?", "f must return a Result[..., str]"),
        (
            "let f = [parse] () => Result[i64, i64] {\n    return ok(parse(1)?)\n}",
            "parse(1)'s error is str, but f returns Result[i64, i64]",
        ),
        ("let f = [] () => null {\n    let r = ok(1)\n}", "cannot tell which Result ok(...) makes"),
        ("let f = [] () => i64 {\n    let x = 3\n    return x?\n}", "? takes a Result"),
        ("let f = [] () => Result[i64, null] ok(1)", "a Result's error has a type"),
        (
            'let f = [parse, io.puts] () => null {\n    match parse(1) {\n        ok(v): io.puts("a"),\n'
            '        nope: io.puts("b")\n    }\n}',
            "a match on a Result has the arms",
        ),
        (
            "let f = [parse] () => i64 {\n    let r = parse(1)\n    return r.unwrap()\n}",
            "a Result's methods are is_ok and is_err",
        ),
        (
            "let C = struct { let n: i64 }\n"
            "let f = [parse] (c: mem.Weak[C]) => Result[i64, str] {\n    let g <- c\n    let n = parse(g.n)?\n"
            "    g -> c\n    return ok(n)\n}",
            "`?` may return here while g still holds c",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, HEAD + source + "\nlet main = [] () => null {\n}\n")
    assert message in error.value.msg


DECODING = r"""
let io = import("std:stdio")
let json = import("std:json")
let form = import("std:form")
let mem = import("std:mem")

let Address = struct { let city: mem.Unique[str], let zip: i32 }
let User = struct {
    let name: mem.Unique[str],
    let age: i64,
    let admin: bool,
    let score: f64,
    let tags: mem.Unique[str][],
    let home: Address
}

let show = [io.printf, json.decode] (text: str) => null {
    let found: Result[User, mem.Unique[str]] = json.decode(text)
    match found {
        ok(u): io.printf("%s %d %d %.1f %d %s %s %d\n", u.name, u.age, u.admin, u.score, len(u.tags), u.tags[-1], u.home.city, u.home.zip),
        err(e): io.printf("error: %s\n", e)
    }
}

let Search = struct { let q: mem.Unique[str], let page: i64, let exact: bool }

let search = [io.printf, form.decode] (text: str) => null {
    let found: Result[Search, mem.Unique[str]] = form.decode(text)
    match found {
        ok(s): io.printf("%s|%d|%d\n", s.q, s.page, s.exact),
        err(e): io.printf("error: %s\n", e)
    }
}

let numbers = [json.decode] (text: str) => Result[i64[], mem.Unique[str]] json.decode(
    text
)

let main = [io.printf, show, search, numbers, form.decode] () => null {
    show(
        "{{\"name\": \"Ada \\u00e9\\n\", \"age\": 36, \"admin\": true, \"score\": 9.5, \"tags\": [\"a\", \"b\"], \"extra\": {{\"x\": [1, null]}}, \"home\": {{\"city\": \"London\", \"zip\": 12}}}}"
    )
    show("{{\"name\": \"Bob\", \"age\": \"old\"}}")
    show(
        "{{\"name\": \"Bob\", \"tags\": [\"x\"], \"age\": 3, \"admin\": false, \"score\": 1}}"
    )
    show("{{\"name\": \"Bob\", \"name\": \"Rob\"}}")
    show("[1, 2]")
    show("{{\"name\": \"Bob\"}} x")
    search("q=hello+world%21&page=2&exact=true")
    search("q=x&page=two&exact=false")
    search("page=1")
    let ns = numbers("[4, 8, 15]")
    match ns {
        ok(xs): io.printf("%d\n", xs[0] + xs[1] + xs[2]),
        err(e): io.printf("%s\n", e)
    }
    let trailing = numbers("[1] x")
    match trailing { ok(_): io.printf("unexpected\n"), err(e): io.printf("%s\n", e) }
    let id: Result[i64, mem.Unique[str]] = form.decode("id=42")
    match id { ok(n): io.printf("%d\n", n), err(e): io.printf("%s\n", e) }
}
"""  # noqa: E501 - Bifrost's formatter keeps long parameter lists and match arms whole


def test_decoding(tmp_path: Path) -> None:
    # json.decode and form.decode make the type written where they go; errors say what is
    # wrong; what a failed decoding made is freed.
    lines, leaked = _run(tmp_path, DECODING)
    assert lines == [
        "Ada é",
        " 36 1 9.5 2 b London 12",
        "error: expected an integer at 23, near '\"old\"}'",
        "error: missing field `home`",
        "error: duplicate field `name`",
        "error: expected an object at 0, near '[1, 2]'",
        "error: missing field `age`",
        "hello world!|2|1",
        "error: field `page`: expected an integer, not 'two'",
        "error: missing field `q`",
        "27",
        "trailing characters at 4, near 'x'",
        "42",
    ]
    assert leaked == 0
