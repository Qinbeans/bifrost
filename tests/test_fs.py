"""``std:fs``: ``read_to_string``, ``write`` and ``exists``, compiled into the program and run."""

import shutil
import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.lowering import lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

PROGRAM = r"""
let io = import("std:stdio")
let fs = import("std:fs")
let mem = import("std:mem")

let checked = [fs.read_to_string] (path: str) => Result[mem.Unique[str], mem.Unique[str]] {
    let text = fs.read_to_string(path)?
    return ok(text)
}

let main = [io.printf, fs.read_to_string, fs.write, fs.exists, checked] () => null {
    let wrote = fs.write("note.txt", "héllo\nworld\n")
    match wrote { ok(_): io.printf("wrote\n"), err(e): io.printf("%s\n", e) }
    io.printf("%d %d %d\n", fs.exists("note.txt"), fs.exists("nope.txt"), fs.exists("."))
    let note = fs.read_to_string("note.txt")
    match note { ok(text): io.printf("[%s]\n", text), err(e): io.printf("%s\n", e) }
    let big = checked("big.txt")
    match big { ok(text): io.printf("%s\n", text), err(e): io.printf("%s\n", e) }
    let missing = fs.read_to_string("nope.txt")
    match missing { ok(_): io.printf("unexpected\n"), err(e): io.printf("%s\n", e) }
    let binary = checked("binary.dat")
    match binary { ok(_): io.printf("unexpected\n"), err(e): io.printf("%s\n", e) }
    let folder = fs.read_to_string(".")
    match folder { ok(_): io.printf("unexpected\n"), err(e): io.printf("%s\n", e) }
    let nowhere = fs.write("no/such/folder.txt", "x")
    match nowhere { ok(_): io.printf("unexpected\n"), err(e): io.printf("%s\n", e) }
}
"""


def _build(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = Project(
        Config(
            path=tmp_path / "out",
            package={"name": "files", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )
    (tmp_path / "out").mkdir()
    lower_file(project, path)
    return project.build()


def test_reading_and_writing_files(tmp_path: Path) -> None:
    executable = _build(tmp_path, PROGRAM)
    (tmp_path / "big.txt").write_text("x" * 100_000)  # past the first read's 4 KB
    (tmp_path / "binary.dat").write_bytes(b"a\xffb")
    output = subprocess.run([executable], capture_output=True, text=True, check=True, cwd=tmp_path).stdout  # noqa: S603
    assert output.splitlines() == [
        "wrote",
        "1 0 1",
        "[héllo",
        "world",
        "]",
        "x" * 100_000,
        "nope.txt: No such file or directory",
        "binary.dat: is not valid UTF-8",
        ".: Is a directory",
        "no/such/folder.txt: No such file or directory",
    ]
    assert (tmp_path / "note.txt").read_text() == "héllo\nworld\n"


@pytest.mark.skipif(shutil.which("valgrind") is None, reason="needs valgrind")
def test_files_free_what_they_read(tmp_path: Path) -> None:
    executable = _build(tmp_path, PROGRAM)
    (tmp_path / "big.txt").write_text("x" * 100_000)
    (tmp_path / "binary.dat").write_bytes(b"a\xffb")
    command = ["valgrind", "-q", "--leak-check=full", "--errors-for-leak-kinds=all", "--error-exitcode=9", executable]
    result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=tmp_path)  # noqa: S603
    assert result.returncode == 0, result.stderr


def test_only_programs_using_files_link_the_runtime(tmp_path: Path) -> None:
    _build(tmp_path, 'let io = import("std:stdio")\n\nlet main = [io.puts] () => null {\n    io.puts("hi")\n}\n')
    assert not (tmp_path / "out" / "bifrost_fs_runtime.o").exists()


def test_the_editor_knows_fs(tmp_path: Path) -> None:
    source = 'let fs = import("std:fs")\n\nlet main = [fs.exists] () => null {\n    fs.\n}\n'
    document = Document.open(tmp_path / "main.bif", source)
    labels = [completion.label for completion in document.completions((3, 7))]
    assert sorted(labels) == ["exists", "read_to_string", "write"]
