import subprocess
from pathlib import Path

import pytest

from bifrost import std
from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

HELLO = r"""
let io = import("std:stdio")

let main = [io.printf, io.puts] () => null {
    io.puts("tab:\there")
    io.printf("%d + %d = %d\n", 20, 22, 20 + 22)
    io.printf("%.2f %s\x21\n", 3.14159, "done")
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path / "out",
            package={"name": "hello_std", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def test_stdio_is_bundled() -> None:
    assert "stdio" in std.available()
    printf = next(d for d in std.load("stdio").declarations if d.name == "printf")
    assert printf.variadic
    assert printf.parameters == {"format": "cstr"}


def test_printf_program_runs(tmp_path: Path) -> None:
    source = tmp_path / "hello.bif"
    source.write_text(HELLO)
    project = _project(tmp_path)
    (tmp_path / "out").mkdir()
    lower_file(project, source)
    executable = project.build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output == "tab:\there\n20 + 22 = 42\n3.14 done!\n"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            'let io = import("std:nope")\n',
            "unknown standard module 'std:nope' "
            "(available: std:fmt, std:json, std:mem, std:stdio, std:tasks, std:time)",
        ),
        (
            'let io = import("std:stdio")\nlet main = [io.puts] () => null {\n    io.puts("\\q")\n}\n',
            "unknown escape '\\q'",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    path = tmp_path / "bad.bif"
    path.write_text(source)
    with pytest.raises(BifrostError) as error:
        lower_file(_project(tmp_path), path)
    assert message in error.value.msg


def test_editor_knows_standard_modules(tmp_path: Path) -> None:
    document = Document.open(tmp_path / "hello.bif", HELLO)
    row = HELLO.splitlines().index('    io.printf("%d + %d = %d\\n", 20, 22, 20 + 22)')
    assert "stdio.printf = (format: str, ...) => i32" in (document.hover((row, 8)) or "")
    location = document.definition((row, 8))
    assert location is not None
    assert location.path == std.path("stdio")
    typing = Document.open(tmp_path / "x.bif", 'let io = import("std:stdio")\nlet f = () => null {\n    io.\n}\n')
    assert "printf" in {c.label for c in typing.completions((2, 7))}


def test_mem_is_a_builtin_standard_module(tmp_path: Path) -> None:
    assert std.available() == ["fmt", "json", "mem", "stdio", "tasks", "time"]
    typing = Document.open(tmp_path / "x.bif", 'let mem = import("std:mem")\nlet f = (c: mem.) => null {}\n')
    assert [c.label for c in typing.completions((1, len("let f = (c: mem.")))] == [
        "Unique",
        "Weak",
        "Shared",
        "Atomic",
        "UniqueGuard",
        "WeakGuard",
        "SharedGuard",
        "AtomicGuard",
    ]


def test_import_suggests_modules() -> None:
    # Inside examples/, so the modules of examples/config.yaml are offered too.
    source = 'let io = import("std:'
    document = Document.open(
        Path(__file__).parents[1] / "examples" / "raylib" / "src" / "hello_raylib" / "x.bif", source + "\n"
    )
    completions = document.completions((0, len(source)))
    # Standard modules, then exported Bifrost modules (from src/), then config.yaml's.
    assert [c.label for c in completions][:8] == [
        "std:fmt",
        "std:json",
        "std:mem",
        "std:stdio",
        "std:tasks",
        "std:time",
        "hello_raylib.helper:helper",
        "raylib",
    ]
    assert completions[0].detail == "formatting into owned strings: format"
    # Replaces what is typed inside the quotes, `std:`, so the colon does not split the name.
    assert completions[0].replace == ((0, len('let io = import("')), (0, len(source)))
    outside = [c.label for c in document.completions((0, 3))]  # on `let`, not in the string
    assert outside
    assert "std:mem" not in outside
