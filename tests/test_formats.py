"""printf patterns are checked against their values, and integers are passed as their conversions read them."""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost import formats
from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

PROGRAM = r"""
let io = import("std:stdio")
let fmt = import("std:fmt")
let json = import("std:json")

let main = [io.printf, io.puts, fmt.format, json.encode] () => null {
    let big = 5000000000
    let small: i32 = -7
    let flag = true
    io.printf("%d %d %u %x %d\n", big, small, 3000000000, 255, flag)
    io.printf("%f %.1f %5.2f|%-4d|%*d|%c %% %s\n", 2, 2.25, 3.14159, 42, 5, 7, 65, "end")
    let text = fmt.format("%d items, %s", big, json.encode(#[1, 2]))
    io.puts(text)
}
"""


def _lower(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = Project(
        Config(
            path=tmp_path,
            package={"name": "formats", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O1, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def test_integers_print_whatever_their_size(tmp_path: Path) -> None:
    executable = _lower(tmp_path, PROGRAM).build()
    lines = subprocess.run([executable], capture_output=True, text=True, check=True).stdout.splitlines()  # noqa: S603
    assert lines == [
        "5000000000 -7 3000000000 ff 1",
        "2.000000 2.2  3.14|42  |    7|A % end",
        "5000000000 items, [1, 2]",
    ]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ('io.printf("%y\\n", 1)', "%y is not a printf conversion"),
        ('io.printf("%d %d\\n", 1)', "the pattern formats 2 values, but 1 follow it"),
        ('io.printf("%d\\n", 1, 2)', "the pattern formats 1 value, but 2 follow it"),
        ('io.printf("%s\\n", 3)', "%s prints strings, but 3 is an i64: use %d"),
        ('io.printf("%d\\n", "x")', '%d prints integers, but "x" is a str: use %s'),
        ('io.printf("%s\\n", #[1, 2])', "#[1, 2] is a list, which %s cannot print; print it with %v"),
        ('io.printf("%n\\n", 1)', "%n writes to memory"),
        ('let t = fmt.format("%s", 1.5)', "%s prints strings, but 1.5 is an f64: use %f"),
    ],
)
def test_mismatches_are_errors(tmp_path: Path, call: str, message: str) -> None:
    source = (
        'let io = import("std:stdio")\nlet fmt = import("std:fmt")\n'
        f"let main = [io.printf, fmt.format] () => null {{\n    {call}\n}}\n"
    )
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source)
    assert message in error.value.msg


def test_widen() -> None:
    assert formats.widen("%d %5.2f %-3lu %hd %x %c %%d %zu") == "%lld %5.2f %-3llu %hd %llx %c %%d %llu"
