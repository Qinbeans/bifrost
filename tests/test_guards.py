import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

COUNTER = r"""
let io = import("std:stdio")
let mem = import("std:mem")

let Context = struct {
    let width: i32,
    let counter: i32,
    static let new = [] () => Context Context(width: 800, counter: 0)
}

let tick = [io.printf] (ctx: mem.Weak[Context]) => null {
    let guard <- ctx
    guard.counter = guard.counter + 1
    io.printf("tick %d\n", guard.counter)
    guard -> ctx
}

let main = [tick, Context.new, io.printf] () => null {
    let ctx: mem.Unique[Context] = Context.new()
    tick(ctx)
    tick(ctx)
    let local <- ctx
    local.width = 1024
    io.printf("counter %d, width %d\n", local.counter, local.width)
    local -> ctx
}
"""


def _project(tmp_path: Path) -> Project:
    return Project(
        Config(
            path=tmp_path,
            package={"name": "guards", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _lower(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "test.bif"
    path.write_text(source)
    project = _project(tmp_path)
    lower_file(project, path)
    return project


def test_writes_through_a_guard_persist(tmp_path: Path) -> None:
    executable = _lower(tmp_path, COUNTER).build()
    output = subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output == "tick 1\ntick 2\ncounter 2, width 1024\n"


HEAD = 'let mem = import("std:mem")\nlet C = struct { let n: i32 }\n'
VALID = [
    "let f = (c: mem.Weak[C], x: i32) => null {\n    let g <- c\n    if x > 0 {\n        g -> c\n    } else {\n"
    "        g -> c\n    }\n}\n",
    "let f = (c: mem.Weak[C], x: i32) => null {\n    let g <- c\n    if x > 0 {\n"
    "        g -> c\n        return\n    }\n"
    "    g -> c\n}\n",
    "let f = (c: mem.Weak[C], x: i32) => null {\n    while x > 0 {\n        let g <- c\n        g.n = g.n + 1\n"
    "        g -> c\n    }\n}\n",
]


@pytest.mark.parametrize("source", VALID)
def test_valid_guards(tmp_path: Path, source: str) -> None:
    _lower(tmp_path, HEAD + source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "let f = (c: mem.Weak[C]) => null {\n    let g <- c\n    g.n = 1\n}\n",
            "g still holds c at the end of the scope it was locked in; release it with `g -> c`",
        ),
        (
            "let f = (c: mem.Weak[C], x: i32) => null {\n    let g <- c\n    if x > 0 {\n        g -> c\n    }\n}\n",
            "g is released only on some paths",
        ),
        (
            "let f = (c: mem.Weak[C], x: i32) => null {\n    let g <- c\n    if x > 0 {\n        return\n    }\n"
            "    g -> c\n}\n",
            "returns while g still holds c",
        ),
        (
            "let f = (c: mem.Weak[C], x: i32) => null {\n    let g <- c\n    while x > 0 {\n        g -> c\n    }\n}\n",
            "g is released inside a loop",
        ),
        (
            "let f = (c: mem.Weak[C]) => null {\n    let g <- c\n    match 1 { 1: { g -> c }, _: { g.n = 0 } }\n}\n",
            "in the `_` arm",
        ),
        (
            "let f = (c: mem.Weak[C]) => i32 {\n    let g <- c\n    let v = c\n    g -> c\n    return 0\n}\n",
            "c is locked by g",
        ),
        (
            "let f = (c: mem.Weak[C]) => i32 {\n    let g <- c\n    g -> c\n    return g.n\n}\n",
            "g was released on line",
        ),
        ("let f = (c: mem.Weak[C]) => null {\n    let g <- c\n    g -> c\n    g -> c\n}\n", "g was already released"),
        ("let f = (c: mem.Weak[C], d: mem.Weak[C]) => null {\n    let g <- c\n    g -> d\n}\n", "g holds c, not d"),
        ("let f = (c: mem.Weak[C]) => i32 c.n\n", "c is a mem.Weak; lock it"),
        (
            "let f = () => i32 {\n    let c: mem.Unique[C] = C(n: 1)\n    let d = c\n    return 0\n}\n",
            "c is a mem.Unique: lend it to a call or lock it",
        ),
        (
            "let g = (c: mem.Weak[C]) => null {}\nlet f = [g] () => null {\n    let n = 1\n    g(n)\n}\n",
            "n is an i64, but this parameter takes a mem.Weak[C]",
        ),
        (
            "let g = (c: mem.Weak[C]) => null {}\nlet f = [g] (c: C) => null {\n    g(c)\n}\n",
            "g is lent c (a mem.Weak, which it may change), but c is a parameter, which this function may not change",
        ),
        ("let f = () => null {\n    let c = C(n: 1)\n    c.n = 2\n}\n", "fields change only through a guard"),
        ("let f = (c: mem.Weak[C]) => null {\n    let g <- c\n    let h = g\n    g -> c\n}\n", "g is a guard"),
        ("let f = (c: mem.Weak[C]) => mem.Weak[C] c\n", "a function cannot return a mem.Weak"),
        ("let D = struct { let p: mem.Weak[C] }\n", "an object cannot store a mem.Weak"),
        ("let f = (c: mem.Unique[C]) => null {}\n", "passing ownership (a mem.Unique parameter)"),
        ("let f = () => null {\n    let c: mem.Weak[C] = C(n: 1)\n}\n", "a local cannot be a mem.Weak"),
        (
            "let f = (c: mem.Weak[C]) => null {\n    let g: mem.UniqueGuard[C] <- c\n    g -> c\n}\n",
            "c is a mem.Weak, so locking it gives a mem.WeakGuard, not a mem.UniqueGuard",
        ),
        (
            "let D = struct { let n: i32 }\nlet f = (c: mem.Weak[C]) => null {\n    let g: mem.WeakGuard[D] <- c\n"
            "    g -> c\n}\n",
            "c holds a C, not a D",
        ),
        ("let f = (c: mem.WeakGuard[C]) => null {}\n", "mem.WeakGuard is a guard's type; it only annotates a lock"),
    ],
)
def test_guard_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, HEAD + source)
    assert message in error.value.msg


def test_formats_guards() -> None:
    source = "let f = (ctx:mem.Weak[ C ]) => null {\n  let n:i32=1\n  let g<-ctx\n  g.n=g.n+n\n      g->ctx\n}\n"
    assert format_source(source) == (
        "let f = (ctx: mem.Weak[C]) => null {\n    let n: i32 = 1\n    let g <- ctx\n    g.n = g.n + n\n"
        "    g -> ctx\n}\n"
    )


def test_editor_knows_guards(tmp_path: Path) -> None:
    document = Document.open(tmp_path / "counter.bif", COUNTER)
    assert [d.message for d in document.diagnostics() if d.severity.name == "ERROR"] == []
    row = COUNTER.splitlines().index("    guard.counter = guard.counter + 1")
    assert document.hover((row, 20)) == "```bifrost\nlet guard: mem.WeakGuard[Context] <- ctx\n```"
    renamed = COUNTER.replace("guard", "myGuard")
    messages = [d.message for d in Document.open(tmp_path / "counter.bif", renamed).naming()]
    assert messages == ["guard 'myGuard' should be snake_case: 'my_guard'"]


def test_guard_types_are_hinted(tmp_path: Path) -> None:
    document = Document.open(tmp_path / "counter.bif", COUNTER)
    lines = COUNTER.splitlines()
    tick_lock, main_lock = lines.index("    let guard <- ctx"), lines.index("    let local <- ctx")
    locks = {tick_lock, main_lock}
    assert [(hint.position, hint.label) for hint in document.inlay_hints() if hint.position[0] in locks] == [
        ((tick_lock, len("    let guard")), ": mem.WeakGuard[Context]"),  # locks a `mem.Weak[Context]` parameter
        ((main_lock, len("    let local")), ": mem.UniqueGuard[Context]"),  # locks `let ctx: mem.Unique[Context] = ...`
    ]
    use = lines.index("    guard.counter = guard.counter + 1")
    assert document.hover((use, 20)) == "```bifrost\nlet guard: mem.WeakGuard[Context] <- ctx\n```"
    let_row = lines.index("    let ctx: mem.Unique[Context] = Context.new()")
    assert document.hover((let_row, 9)) == "```bifrost\nlet ctx: mem.Unique[Context] = Context.new()\n```"


def test_fields_resolve_through_guards_and_pointers(tmp_path: Path) -> None:
    document = Document.open(tmp_path / "counter.bif", COUNTER)
    lines = COUNTER.splitlines()
    use = lines.index("    guard.counter = guard.counter + 1")
    column = lines[use].index("counter")
    assert document.hover((use, column)) == "```bifrost\nlet counter: i32\n// a member of Context\n```"
    definition = document.definition((use, column))
    assert definition is not None
    assert lines[definition.range[0][0]].strip() == "let counter: i32,"
    typing = COUNTER.replace("    guard.counter = guard.counter + 1", "    guard.")
    completions = Document.open(tmp_path / "counter.bif", typing).completions((use, len("    guard.")))
    assert [c.label for c in completions] == ["width", "counter"]  # fields only, not the static `new`
