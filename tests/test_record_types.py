"""Record types: ``=> #{name: str, ms: i64}`` written out, or ``=> Record``, shaped like what is returned."""

import subprocess
from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

PROGRAM = """\
let stdio = import("std:stdio")
let time = import("std:time")
let tasks = import("std:tasks")

// Called before it is defined: it is lowered early, for the record it returns.
let main = [stdio.printf, tasks.gather, timed, pair, sign, describe] async () => null {
    let results = await tasks.gather(one: timed("one", 20), two: timed("two", 1))
    stdio.printf("%s %d %s\\n", results.one.name, results.one.ms >= 20, results.two.name)
    let p = pair(3)
    stdio.printf("%lld %lld %.1f\\n", p.low, p.high.value, p.high.half)
    let s = sign(-4)
    stdio.printf("%lld %d\\n", s.value, s.negative)
    describe(#{y: 2, x: 1})
}

let timed = [time.now, time.sleep] async (name: str, ms: i64) => Record {
    let started = time.now()
    await time.sleep(ms)
    return #{name: name, ms: time.now() - started}
}

let pair = [] (a: i64) => #{low: i64, high: #{value: i64, half: f64}} {
    return #{low: a, high: #{value: a * 2, half: 1.5}}
}

let sign = [] (n: i64) => Record {
    if n < 0 {
        return #{value: 0 - n, negative: true}
    }
    return #{negative: false, value: n}
}

let describe = [stdio.printf] (point: #{x: i64, y: i64}) => null {
    let copy: #{x: i64, y: i64} = point
    stdio.printf("%lld %lld\\n", copy.x, copy.y)
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "records_app", "version": "0", "description": "", "type": "async"},
            flags={"optimization": OptLevel.O1, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _lower(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "main.bif"
    path.write_text(source)
    project = _project(tmp_path)
    unit = lower_file(project, path)
    with unit.errors():
        _ = project.program.mlir
    return project


def test_functions_return_records(tmp_path: Path) -> None:
    executable = _lower(tmp_path, PROGRAM).build()
    lines = subprocess.run([executable], capture_output=True, text=True, check=True).stdout.splitlines()  # noqa: S603
    # The fields' order does not matter: `#{y: 2, x: 1}` is a `#{x: i64, y: i64}`, and so is either `sign` record.
    assert lines == ["one 1 two", "3 6 1.5", "4 1", "1 2"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "let f = [] (n: i64) => Record {\n    if n > 0 {\n        return #{value: n}\n    }\n"
            "    return #{value: n, negative: true}\n}",
            "f returns records of different shapes: #{value: i64} on line 3, and #{value: i64, negative: bool} here; "
            "consolidate them into one record with the same fields",
        ),
        ("let f = [] (n: i64) => Record {\n    return n\n}", "f returns a Record, but this returns i64"),
        (
            "let f = [f] (n: i64) => Record {\n    if n > 0 {\n        return f(n - 1)\n    }\n"
            "    return #{value: n}\n}",
            "the record f returns depends on a call of f itself; write its type: => #{...}",
        ),
        ("let f = [] (r: Record) => i64 {\n    return 1\n}", "Record is a function's result"),
        ("let f = [] () => #{a: i64, a: str} {\n    return #{a: 1}\n}", "the record type already has a field 'a'"),
        (
            'let f = [] () => #{a: i64} {\n    return #{a: "x"}\n}',
            "f returns #{a: i64}, but this is #{a: str}; give it the fields of #{a: i64}",
        ),
        (
            "let g = [] (p: #{x: i64, y: i64}) => i64 p.x\nlet f = [g] () => i64 g(#{x: 1})",
            "g takes #{x: i64, y: i64}, but this is #{x: i64}",
        ),
        (
            "let f = [] () => i64 {\n    let p: #{x: i64, y: i64} = #{x: 1}\n    return p.x\n}",
            "p is #{x: i64, y: i64}, but this is #{x: i64}",
        ),
    ],
)
def test_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source + "\n")
    assert message in error.value.msg
