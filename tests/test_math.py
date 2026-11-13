"""``std:math``: each function on ``f32`` and ``f64``, compiled to MLIR's math dialect and run.

The programs are built at -O0, where each operation is computed at run time
(LLVM's intrinsics, some calling libm), and at -O2, where LLVM folds what it
can; both give what Python's ``math`` does.
"""

import math
import re
import struct
import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

_EVERY = (
    "math.sqrt, math.floor, math.ceil, math.round, math.trunc, math.abs, math.sin, math.cos, math.tan, "
    "math.asin, math.acos, math.atan, math.atan2, math.tanh, math.exp, math.exp2, math.log, math.log2, "
    "math.log10, math.pow, math.fma, math.min, math.max"
)

# What each show_<type> prints, one value per line (as Python computes it, in _expected).
_SHOWN = [
    *("math.sqrt(x)", "math.floor(y)", "math.ceil(y)", "math.round(y)", "math.trunc(y)", "math.abs(y)"),
    *("math.sin(x)", "math.cos(x)", "math.tan(x)", "math.asin(x)", "math.acos(x)", "math.atan(x)"),
    *("math.atan2(y, x)", "math.tanh(x)", "math.exp(x)", "math.exp2(x)", "math.log(x)"),
    *("math.log2(x)", "math.log10(x)", "math.pow(x, y)", "math.fma(x, y, 1)"),
    *("math.min(x, y)", "math.max(x, y)", "math.min(math.sqrt(y), 2)", "math.pi * x"),
]


def _show(kind: str) -> str:
    lines = "".join(f'    io.printf("%.9g\\n", {shown})\n' for shown in _SHOWN)
    return f"let show_{kind} = [io.printf, {_EVERY}] (x: {kind}, y: {kind}) => null {{\n{lines}}}\n"


PROGRAM = (
    'let io = import("std:stdio")\nlet math = import("std:math")\n\n'
    + _show("f64")
    + "\n"
    + _show("f32")
    + r"""
let main = [io.printf, show_f64, show_f32, math.abs, math.min, math.max, math.sin] () => null {
    show_f64(0.7, -2.5)
    show_f32(0.7, -2.5)
    let n: i32 = -7
    io.printf("%d %d %d\n", math.abs(n), math.min(n, 3), math.max(n, 3))
    io.printf("%d %d\n", math.max(u8(200), u8(100)), math.min(u8(200), u8(100)))
    io.printf("%.9g\n", math.sin(math.pi / 2))
}
"""
)


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _expected(x: float, y: float) -> list[float]:
    """Compute what ``show_*`` prints, in Python (in f64)."""
    rounded = math.copysign(math.floor(abs(y) + 0.5), y)  # halves away from zero
    return [
        *(math.sqrt(x), math.floor(y), math.ceil(y), rounded, math.trunc(y), abs(y)),
        *(math.sin(x), math.cos(x), math.tan(x), math.asin(x), math.acos(x), math.atan(x)),
        *(math.atan2(y, x), math.tanh(x), math.exp(x), 2**x, math.log(x)),
        *(math.log2(x), math.log10(x), x**y, x * y + 1),
        *(min(x, y), max(x, y), 2.0, math.pi * x),  # min(NaN, 2) is 2: a NaN gives the other
    ]


def _project(tmp_path: Path, optimization: int = 0) -> Project:
    return Project(
        Config(
            path=tmp_path / "out",
            package={"name": "numbers", "version": "0", "description": ""},
            flags={"optimization": optimization, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _lower(tmp_path: Path, source: str, optimization: int = 0) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = _project(tmp_path, optimization)
    (tmp_path / "out").mkdir(exist_ok=True)
    lower_file(project, path)
    return project


@pytest.mark.parametrize("optimization", [0, 2])
def test_every_function_on_f64_and_f32(tmp_path: Path, optimization: int) -> None:
    executable = _lower(tmp_path, PROGRAM, optimization).build()
    lines = subprocess.run([executable], capture_output=True, text=True, check=True).stdout.splitlines()  # noqa: S603
    count = len(_SHOWN)
    wide, narrow = [float(line) for line in lines[:count]], [float(line) for line in lines[count : 2 * count]]
    assert wide == pytest.approx(_expected(0.7, -2.5), rel=1e-8)
    assert narrow == pytest.approx(_expected(_f32(0.7), -2.5), rel=1e-6)
    assert narrow[-1] != wide[-1]  # math.pi * x, computed in f32
    assert lines[2 * count :] == ["7 -7 3", "200 100", "1"]


def test_calls_become_math_operations(tmp_path: Path) -> None:
    source = (
        'let math = import("std:math")\n\n'
        "let hypot = [math.sqrt] (x: f64, y: f64) => f64 math.sqrt(x * x + y * y)\n"
        "let narrow = [math.sqrt, math.max] (x: f32) => f32 math.max(math.sqrt(x), 0)\n"
        "let main = [hypot, narrow] () => i32 i32(hypot(3.0, 4.0) + f64(narrow(9.0)))\n"
    )
    project = _lower(tmp_path, source, optimization=2)
    mlir, llvm = (path.read_text() for path in project.emit(mlir=True, llvm=True))
    assert "math.sqrt %" in mlir
    assert ": f32" in mlir.split("narrow", 1)[1]
    assert "arith.maxnumf" in mlir
    assert "bifrost_math" not in mlir  # no placeholder left, called or declared
    assert "@sqrt" not in llvm  # not a libm call: LLVM's intrinsic, folded into the constant
    assert "ret i32 8" in llvm
    assert subprocess.run([project.build()], check=False).returncode == 8  # noqa: S603


def test_math_needs_dependency_entries(tmp_path: Path) -> None:
    source = 'let math = import("std:math")\n\nlet root = (x: f64) => f64 math.sqrt(x)\n'
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source)
    assert (
        error.value.msg == "root calls 'math.sqrt' without depending on it; add it to the dependency list: [math.sqrt]"
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("(n: i64) => f64 math.sqrt(n)", "math.sqrt takes a float (f32 or f64), and n is i64; convert it: f64(n)"),
        (
            "(x: f32, y: f64) => f64 math.pow(y, x)",
            "math.pow takes numbers of one type, and y is f64 while x is f32; convert one: f64(x)",
        ),
        ("(n: u64) => u64 math.abs(n)", "math.abs takes a float or a signed integer, and n is u64, never below zero"),
        ("(s: str) => f64 math.sqrt(s)", "math.sqrt takes numbers, not a str (s)"),
        ("(x: f64) => f64 math.atan2(x)", "math.atan2 takes y, x: math.atan2(y, x)"),
        ("(n: i64) => i64 math.max(n, 2.5)", "2.5 is not a whole number, and math.max takes i64s here"),
    ],
)
def test_math_type_errors(tmp_path: Path, body: str, message: str) -> None:
    called = re.search(r"math\.\w+", body)
    assert called is not None
    source = f'let math = import("std:math")\n\nlet f = [{called[0]}] {body}\n'
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source)
    assert error.value.msg == message


def test_numbers_written_as_such_take_the_others_type(tmp_path: Path) -> None:
    source = (
        'let io = import("std:stdio")\nlet math = import("std:math")\n\nlet half = 0.5\n\n'
        "let main = [io.printf, math.sqrt, math.pow, math.max, math.cos] () => null {\n"
        "    let x: f32 = 3.0\n"
        "    let y = math.pow(x, 2)\n"  # f32, from x
        "    let s = math.sqrt(2)\n"  # f64: numbers alone are f64
        '    io.printf("%.9g %.9g %d %.9g\\n", y + half, s, math.max(2, -3), math.cos(math.pi))\n'
        "}\n"
    )
    executable = _lower(tmp_path, source).build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output == f"9.5 {math.sqrt(2):.9g} 2 -1\n"


def test_only_programs_using_math_link_libm(tmp_path: Path) -> None:
    assert not _lower(tmp_path, 'let io = import("std:stdio")\n').math.used


def test_the_editor_knows_math(tmp_path: Path) -> None:
    source = (
        'let math = import("std:math")\n\n'
        "let f = [math.sqrt] (x: f32) => null {\n"
        "    let r = math.sqrt(x)\n"
        "    math.\n"
        "}\n"
    )
    document = Document.open(tmp_path / "main.bif", source)
    completions = {completion.label: completion for completion in document.completions((4, 9))}
    assert {"sqrt", "atan2", "fma", "min", "pi", "e"} <= completions.keys()
    assert completions["sqrt"].detail == "math.sqrt(x: F) => F"
    assert completions["pi"].kind == "constant"
    hover = document.hover((3, 19)) or ""
    assert "math.sqrt = (x: F) => F" in hover
    assert "F is f32 or f64" in hover
    assert "let r: f32 = math.sqrt(x)" in (document.hover((3, 8)) or "")  # x's type
