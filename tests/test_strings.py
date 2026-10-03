"""Interpolated strings: ``"Hello, {name}!"`` is a new owned string, formatted and checked as ``fmt.format`` is."""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document
from bifrost.syntax import syntax_errors

PROGRAM = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Team = struct {
    let name: str,
    let size: i64,
    let to_string = [super] () => mem.Unique[str] "{super.name} ({super.size})"
}

let main = [io.printf, io.puts, fmt.format] () => null {
    let name = "ada"
    let price = 2.5
    let xs = #[1, 2, 3]
    io.puts("Hello, {name}! {len(xs)} items: {xs}, {{braces}} and 100%")
    io.printf("{name} costs {price:.2f}, and %d more\n", 7)
    let team = Team(name: "core", size: 2)
    let message = "team {team}, record {#{a: 1}}, sum {xs[0] + xs[1]}"
    io.puts(message)
    io.puts(fmt.format("%d and {name}", 3))
    let i = 0
    while i < 3 {
        io.puts("line {i} of a text long enough to grow past the sixty-four bytes it starts with")
        let i = i + 1
    }
    io.puts("plain\x41")
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


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "strings", "version": "0", "description": ""},
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


def test_strings_interpolate(tmp_path: Path) -> None:
    # Each {value} prints as %v does (a list or record as JSON, an object with its to_string);
    # {value:spec} with a printf conversion; in a printf pattern, interpolated values go where
    # the pattern reads them. Every new string is freed: kept by its owner, or after the statement.
    executable = _compile(tmp_path, PROGRAM).build()
    (tmp_path / "count.c").write_text(COUNTER)
    counter = tmp_path / "count.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    assert result.stdout.splitlines() == [
        "Hello, ada! 3 items: [1, 2, 3], {braces} and 100%",
        "ada costs 2.50, and 7 more",
        'team core (2), record {"a": 1}, sum 3',
        "3 and ada",
        *(f"line {i} of a text long enough to grow past the sixty-four bytes it starts with" for i in range(3)),
        "plainA",
    ]
    allocations, frees = map(int, result.stderr.split())
    assert allocations - frees == 1  # stdout keeps its buffer


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('    let p = 2.5\n    io.puts("{p:d}")\n', "%d prints integers, but p is an f64: use %f"),
        ('    io.puts("{nope}")\n', "'nope' is not defined"),
        ('    let n = 1\n    io.printf("{n} %d %d\\n", 2)\n', "the pattern formats 3 values, but 2 follow it"),
        (
            '    let n = 1\n    let g = [n] () => i64 n\n    io.puts("{g}")\n',
            "g is a function value, which %v cannot print",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    head = 'let io = import("std:stdio")\nlet f = [io.puts, io.printf] () => null {\n'
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, head + source + "}\n")
    assert message in error.value.msg


def test_a_new_string_is_owned(tmp_path: Path) -> None:
    source = (
        'let N = struct { let name: str }\nlet f = [] () => null {\n    let x = 1\n    let n = N(name: "n {x}")\n}\n'
    )
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, source)
    assert error.value.msg == 'N\'s name is not owned (str), but "n {x}" is; write the type as mem.Unique[str]'


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('io.puts("a } b")', "a '}' in a string closes nothing; write }} for a brace"),
        ('io.puts("a { b")', "a '{' in a string starts a value ({name}) that is never closed; write {{ for a brace"),
    ],
)
def test_stray_braces(line: str, message: str) -> None:
    source = f"let f = [] () => null {{\n    {line}\n}}\n"
    document = Document.open(Path("scratch.bif"), source)
    assert [problem.message for problem in syntax_errors(document.root)] == [message]


def test_strings_in_the_editor(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    )
    source = (
        'let io = import("std:stdio")\nlet main = [io.puts] () => null {\n'
        '    let name = "ada"\n    let message = "Hello, {name}!"\n    io.puts(message)\n}\n'
    )
    document = Document.open(tmp_path / "main.bif", source)
    assert [d.message for d in document.diagnostics()] == []
    assert [hint.label for hint in document.inlay_hints() if not hint.parameter] == [": str", ": mem.Unique[str]"]
    line = source.splitlines()[3]
    assert document.hover((3, line.index("{name}") + 2)) == '```bifrost\nlet name: str = "ada"\n```'
