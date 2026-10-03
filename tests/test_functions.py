"""Functions as values: function types, passing and calling them, and lambdas."""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.configs.schema import _Extern, _Function
from bifrost.lowering import BifrostError, display_types, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

EXAMPLES = Path(__file__).parents[1] / "examples" / "raylib" / "src"

PROGRAM = """\
let io = import("std:stdio")

let Handlers = struct {
    let on_value: (x: i32) => i32,
    let scale: i32
}

let double = [] (x: i32) => i32 x * 2

let square = [] (x: i32) => i32 x * x

let apply = [] (f: (x: i32) => i32, x: i32) => i32 f(x)

let twice = [apply] (f: (i32) => i32, x: i32) => i32 apply(f, apply(f, x))

let main = [io.printf, apply, twice, double, square] () => null {
    let g: (i32) => i32 = double
    io.printf("%d\\n", apply(g, 21))
    io.printf("%d\\n", twice(square, 3))
    let add_one = [] (x: i32) => i32 x + 1
    io.printf("%d\\n", add_one(41))
    io.printf("%d\\n", apply([] (x: i32) => i32 { return x * 10 }, 4))
    let h = Handlers(on_value: square, scale: 2)
    io.printf("%d\\n", h.on_value(7) * h.scale)
}
"""

PRELUDE = """\
let double = [] (x: i32) => i32 x * 2
let apply = [] (f: (x: i32) => i32, x: i32) => i32 f(x)
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "functions", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


CLOSURES = """\
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")

let Handlers = struct {
    let on_value: (x: i64) => i64,
    let label: mem.Unique[str]
}

let double = [] (x: i64) => i64 x * 2

let apply = [] (f: (x: i64) => i64, x: i64) => i64 f(x)

let adder = [] (n: i64) => (i64) => i64 {
    return [n] (x: i64) => i64 x + n
}

let main = [io.printf, apply, double, adder, fmt.format] () => null {
    let scale = 3
    let times = [scale] (x: i64) => i64 x * scale
    let scale = 100
    io.printf("%d %d %d\\n", times(2), apply(times, 5), apply(double, 21))
    let add3 = adder(3)
    io.printf("%d %d\\n", add3(4), apply(adder(10), 1))
    let numbers = #[1, 2, 3]
    let total = [numbers] (start: i64) => i64 {
        let sum = start
        forall n in numbers {
            let sum = sum + n
        }
        return sum
    }
    io.printf("%d\\n", total(100))
    let name = fmt.format("tripled")
    let h = Handlers(on_value: [scale] (x: i64) => i64 x * scale, label: name)
    io.printf("%s %d\\n", h.label, h.on_value(2))
    let fs = #[adder(1), times, double]
    let sum = 0
    forall f in fs {
        let sum = sum + f(1)
    }
    io.printf("%d %d\\n", sum, len(fs))
    io.printf("%d\\n", apply([scale] (x: i64) => i64 x + scale, 1))
}
"""

COUNTER = r"""
#include <stdio.h>
#include <stdlib.h>
void *__libc_malloc(size_t);
void __libc_free(void *);
static long allocations, frees;
void *malloc(size_t n) { allocations++; return __libc_malloc(n); }
void free(void *p) { if (p) frees++; __libc_free(p); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "%ld %ld\n", allocations, frees); }
"""


def test_closures_capture_copies_and_are_freed(tmp_path: Path) -> None:
    # A lambda's dependency list names the locals it captures: copies (`scale` is 3 for `times`,
    # though a later `let scale` gives 100), an owned one moved in (`numbers`). A closure can be
    # returned, stored in an object or a list, passed and called; its owner frees what it captured,
    # and one only lent (`apply([scale] ..., 1)`) is freed after the statement.
    source = tmp_path / "main.bif"
    source.write_text(CLOSURES)
    project = _project(tmp_path)
    lower_file(project, source)
    executable = project.build()
    (tmp_path / "count.c").write_text(COUNTER)
    counter = tmp_path / "count.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c"], check=True)  # noqa: S603, S607
    result = subprocess.run(  # noqa: S603
        [executable], capture_output=True, text=True, check=True, env={"LD_PRELOAD": str(counter)}
    )
    assert result.stdout.splitlines() == ["6 15 42", "7 11", "106", "tripled 200", "7 3", "101"]
    allocations, frees = map(int, result.stderr.split())
    assert allocations - frees == 1  # stdout keeps its buffer


def test_functions_are_values(tmp_path: Path) -> None:
    source = tmp_path / "main.bif"
    source.write_text(PROGRAM)
    project = _project(tmp_path)
    lower_file(project, source)
    executable = project.build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output.split() == ["42", "81", "42", "40", "98"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "let main = [apply] () => null {\n    apply(double, 1)\n}\n",
            "main uses 'double' without depending on it; add it to the dependency list: [double]",
        ),
        (
            "let main = [apply] () => null {\n    apply([] (x: i32) => i32 double(x), 1)\n}\n",
            "the lambda on line 4 calls 'double' without depending on it",
        ),
        (
            "let main = [apply] () => null {\n    let n = 3\n    apply([] (x: i32) => i32 x + n, 1)\n}\n",
            "a lambda uses 'n' of the function around it; capture a copy by listing it: [n]",
        ),
        (
            'let fmt = import("std:fmt")\nlet main = [] () => null {\n    let f = fmt.format\n}\n',
            "fmt.format is built into the compiler, so it cannot be a function value",
        ),
        (
            'let mem = import("std:mem")\nlet f = [] (g: (x: mem.Weak[i32]) => null) => null {}\n',
            "a mem container in a function type is not supported yet",
        ),
        ("let main = [] () => null {\n    let n = 3\n    n(1)\n}\n", "'n' is i64, not a function value"),
        ("let main = [apply] () => null {\n    apply(1, 1)\n}\n", "expected (i32) => i32, got the int 1"),
        (
            "let main = [double] () => null {\n    let f: (i32) => f64 = double\n}\n",
            "expected (i32) => f64, got (i32) => i32",
        ),
        (
            "let main = [] () => null {\n    let grid = #[#[1], #[2]]\n    forall row in grid {\n"
            "        let g = [row] () => i64 len(row)\n    }\n}\n",
            "row reads from grid, so a lambda cannot capture it; capture a copy: `let mine = #[...row]`, then [mine]",
        ),
        (
            "let f = [] (xs: i64[]) => null {\n    let g = [xs] () => i64 len(xs)\n}\n",
            "xs is a parameter, lent by the caller, so a lambda cannot capture it",
        ),
        (
            "let main = [] () => i64 {\n    let xs = #[1]\n    let g = [xs] () => i64 len(xs)\n    return len(xs)\n}\n",
            "xs was moved on line 5, so it is no longer here",
        ),
        (
            'let mem = import("std:mem")\nlet f = [] (c: mem.Weak[i64]) => null {\n    let g = [c] () => i64 1\n}\n',
            "a lambda cannot capture c, a mem.Weak: it would outlive the lock or loan",
        ),
        (
            "let main = [] () => null {\n    let n = 1\n    let g = [n] () => i64 2\n}\n",
            "'n' is a dependency of the lambda on line 5 but never used",
        ),
        (
            "let f = [] () => null {\n    let xs = #[1]\n    let g = [xs] () => i64[] xs\n}\n",
            "xs is what this lambda captured, which its closure keeps, so it cannot be kept here",
        ),
        (
            'let io = import("std:stdio")\nlet main = [io.printf] () => null {\n    let n = 1\n'
            '    let g = [n] () => i64 n\n    io.printf("%v", g)\n}\n',
            "g is a function value, which %v cannot print",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    path = tmp_path / "bad.bif"
    path.write_text(PRELUDE + source)
    with pytest.raises(BifrostError) as error:
        _compile(_project(tmp_path), path)
    assert message in error.value.msg


def _compile(project: Project, path: Path) -> None:
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir


def test_display_types() -> None:
    assert display_types("Fn[[Fn[[i32, cstr], None], i32], f64]") == "((i32, str) => null, i32) => f64"
    assert display_types("expected cstr") == "expected cstr"  # only function types are rewritten


def test_editor_shows_function_types() -> None:
    source = (
        PRELUDE + "let main = [apply, double] () => null {\n"
        "    let g = double\n"
        "    let h = [] (x: i32) => i32 {\n"
        "        let y = x + 1\n"
        "        return y\n"
        "    }\n"
        "    apply(g, 1)\n"
        "    apply(h, 1)\n"
        "}\n"
    )
    document = Document.open(EXAMPLES / "scratch.bif", source)
    assert [d.message for d in document.diagnostics()] == []
    hints = {
        row: label for (row, _), label in ((h.position, h.label) for h in document.inlay_hints() if not h.parameter)
    }
    assert hints == {3: ": (x: i32) => i32", 4: ": (x: i32) => i32", 5: ": i32"}
    assert "x: i32" in (document.hover((5, 16)) or "")  # the lambda's own parameter


def test_lambda_locals_belong_to_the_lambda() -> None:
    source = "let f = [] () => null {\n    let used = 1\n    let g = [] (x: i32) => i32 used\n    g(1)\n}\n"
    unused = Document.open(EXAMPLES / "scratch.bif", source).unused()
    # `used` is read only inside the lambda, which cannot see it: unused in f; `x` is unused in g.
    assert sorted(d.message.split(" ")[0] for d in unused) == ["used", "x"]


def test_c_calls_back_into_bifrost(tmp_path: Path) -> None:
    # An extern takes a function type, written in config.yaml as `() => None`.
    source = tmp_path / "main.bif"
    source.write_text(
        'let io = import("std:stdio")\n'
        'let c = import("c")\n'
        'let goodbye = [io.puts] () => null {\n    io.puts("bye")\n}\n'
        'let main = [c.atexit, goodbye, io.puts] () => null {\n    c.atexit(goodbye)\n    io.puts("main")\n}\n'
    )
    project = _project(tmp_path)
    project.config.externs = [
        _Extern(
            module="c",
            description="libc",
            declarations=[
                _Function.model_validate(
                    {"name": "atexit", "type": "function", "parameters": {"callback": "() => None"}, "return": "i32"}
                )
            ],
        )
    ]
    project = Project(project.config)
    lower_file(project, source)
    output = subprocess.run([project.build()], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output == "main\nbye\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "    let n = 1\n    c.atexit([n] () => null {\n        let m = n\n    })\n",
            "C takes a plain function, but this lambda captures n",
        ),
        (
            "    let g = [] () => null {}\n    c.atexit(g)\n",
            "C takes a plain function, and g is a function value (a closure); name a function instead",
        ),
    ],
)
def test_c_takes_plain_functions(tmp_path: Path, body: str, message: str) -> None:
    source = tmp_path / "main.bif"
    source.write_text(f'let c = import("c")\nlet main = [c.atexit] () => null {{\n{body}}}\n')
    project = _project(tmp_path)
    project.config.externs = [
        _Extern(
            module="c",
            description="libc",
            declarations=[
                _Function.model_validate(
                    {"name": "atexit", "type": "function", "parameters": {"callback": "() => None"}, "return": "i32"}
                )
            ],
        )
    ]
    with pytest.raises(BifrostError) as error:
        _compile(Project(project.config), source)
    assert message in error.value.msg


def test_editor_warns_a_capture_is_a_copy() -> None:
    source = (
        "let main = [] () => null {\n"
        "    let n = 1\n"
        "    let f = [n] () => i64 n\n"
        "    // ignore: copy\n"
        "    let g = [n] () => i64 n + 1\n"
        "    let h = [n] () => i64 n + 2 // ignore: copy\n"
        "    let total = f() + g() + h()\n"
        "}\n"
    )
    document = Document.open(EXAMPLES / "scratch.bif", source)
    warnings = [d for d in document.diagnostics() if d.message.startswith("n is captured")]
    assert [d.range[0][0] for d in warnings] == [2]  # the others accept it
    assert "Add `// ignore: copy`" in warnings[0].message
    assert document.hover((2, 26)) == "```bifrost\nlet n: i64 = 1\n// captured: a copy, made where the lambda is\n```"


def test_owned_locals_are_lent_to_c_pointers(tmp_path: Path) -> None:
    # memset's `ptr` (void *) parameter borrows `pair` for the call; the change is copied back.
    source = tmp_path / "main.bif"
    source.write_text(
        'let io = import("std:stdio")\n'
        'let mem = import("std:mem")\n'
        'let c = import("c")\n'
        "let Pair = struct { let a: i32, let b: i32 }\n"
        "let main = [c.memset, io.printf] () => null {\n"
        "    let pair: mem.Unique[Pair] = Pair(a: 1, b: 2)\n"
        "    c.memset(pair, 0, 8)\n"
        "    let g <- pair\n"
        '    io.printf("%d %d\\n", g.a, g.b)\n'
        "    g -> pair\n"
        "}\n"
    )
    config = _project(tmp_path).config
    config.externs = [
        _Extern(
            module="c",
            description="libc",
            declarations=[
                _Function.model_validate(
                    {
                        "name": "memset",
                        "type": "function",
                        "parameters": {"dest": "ptr", "value": "i32", "size": "u64"},
                        "return": "ptr",
                    }
                )
            ],
        )
    ]
    project = Project(config)
    lower_file(project, source)
    output = subprocess.run([project.build()], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output == "0 0\n"
