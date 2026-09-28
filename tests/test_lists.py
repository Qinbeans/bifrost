"""Lists: ``#[1, 2, 3]`` values, ``i64[]`` types, indexing, ``len``, and ``forall``.

A list is freed once nothing uses it (in async functions too). ``#[a...b]`` is
the range from ``a`` up to ``b`` (not included): as a value, a list of those
numbers; in ``forall``, counting, with no list made.
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


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('let xs = #["a", "b"]', "arrays hold integers, floats, or bools"),
        ("let xs: str[] = #[]", "a list holds integers, floats or bools, not str"),
        ("let n = len(1, 2)", "len takes one list: len(xs)"),
        ("let xs = #[1, ...ys]", "spreading into a list is not supported yet"),
    ],
)
def test_errors(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _compile(tmp_path, "let main = [] () => null {\n    " + body + "\n}\n")
    assert message in error.value.msg


def test_lists_format() -> None:
    source = "let f = [] (xs: i64[]) => null {\n    forall x in #[0...len(xs)] {\n        let y = xs[x]\n    }\n}\n"
    assert format_source(source) == source
