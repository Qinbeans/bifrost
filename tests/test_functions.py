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
            "a lambda cannot use 'n' of the function around it (yet); pass it as a parameter",
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
    hints = {row: label for (row, _), label in ((h.position, h.label) for h in document.inlay_hints())}
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
