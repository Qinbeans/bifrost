"""Records (``#{name: value}``) and ``std:json``: values as JSON text.

The program prints through its own `puts` (declared with a ``json`` parameter), so it
uses no module that declares ``puts`` too (``std:stdio``, or ``std:fmt``, which loads it).
"""

import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.configs.schema import _Extern, _Function
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

PROGRAM = r"""
let json = import("std:json")
let c = import("c")

let User = struct {
    let id: i64,
    let admin: bool,
    let score: f64
}

let make_user = [] (id: i64) => User User(id: id, admin: id == 1, score: 99.5)

let main = [make_user, json.encode, c.show] () => null {
    let name = "user 7"
    let user = make_user(7)
    let small: i32 = 5
    let record = #{text: "a \"quote\"\n", name: name, sum: small + 1, ratio: 1 / 4, user: user, nested: #{ok: true}}
    let body = json.encode(record)
    c.show(body)
    let plain = json.encode(user)
    c.show(plain)
    c.show(#{direct: "to C"})
    c.show("[1, 2]")
}
"""


def _project(tmp_path: Path) -> Project:
    show = _Function.model_validate(
        {"name": "puts", "as": "show", "type": "function", "parameters": {"text": "json"}, "return": "i32"}
    )
    return Project(
        Config(
            path=tmp_path,
            package={"name": "records", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            externs=[_Extern(module="c", description="libc", declarations=[show])],
        )
    )


def test_records_encode_as_json(tmp_path: Path) -> None:
    source = tmp_path / "main.bif"
    source.write_text(PROGRAM)
    project = _project(tmp_path)
    unit = lower_file(project, source)
    with unit.errors():
        executable = project.build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output.splitlines() == [
        '{"text": "a \\"quote\\"\\n", "name": "user 7", "sum": 6, "ratio": 0.25, '
        '"user": {"id": 7, "admin": false, "score": 99.5}, "nested": {"ok": true}}',
        '{"id": 7, "admin": false, "score": 99.5}',
        '{"direct": "to C"}',  # a `json` parameter encodes a record passed to it
        "[1, 2]",  # and takes a str as JSON text already
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("    let t = json.encode(3)\n", "json.encode takes a record, an object or a list, not i64"),
        ("    let r = #{a: 1, a: 2}\n", "the record already has a field 'a'"),
        ("    let r = #{a: nothing()}\n", "cannot tell the type of field 'a'"),
        ("    let t = json.encode(1, 2)\n", "json.encode takes one value"),
    ],
)
def test_errors(tmp_path: Path, body: str, message: str) -> None:
    source = tmp_path / "bad.bif"
    source.write_text(
        'let json = import("std:json")\n'
        "let nothing = [] () => null {}\n"
        "let main = [json.encode, nothing] () => null {\n" + body + "}\n"
    )
    with pytest.raises(BifrostError) as error:
        _compile(_project(tmp_path), source)
    assert message in error.value.msg


def _compile(project: Project, source: Path) -> None:
    unit = lower_file(project, source)
    with unit.errors():
        _ = project.program.mlir


def test_a_trailing_record_hugs_its_call() -> None:
    source = (
        "let f = [http.json] (ctx: http.Context) => null {\n"
        '    http.json(ctx, 200, #{message: "hello", name: "a name long enough", protocol: "HTTP/2"})\n'
        "}\n"
    )
    assert format_source(source) == (
        "let f = [http.json] (ctx: http.Context) => null {\n"
        "    http.json(ctx, 200, #{\n"
        '        message: "hello",\n'
        '        name: "a name long enough",\n'
        '        protocol: "HTTP/2"\n'
        "    })\n"
        "}\n"
    )
