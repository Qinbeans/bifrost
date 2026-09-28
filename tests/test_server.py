from pathlib import Path

import pytest

from bifrost.server.analysis import Document, Severity, SymbolKind

# A folder of the raylib example project, so its config.yaml (raylib externs) applies.
EXAMPLES = Path(__file__).parents[1] / "examples" / "raylib" / "src"

SOURCE = """\
let raylib = import("raylib")
let limit = 10
let square = (a: i32) => i32 a * a
let main = [square, raylib.init_window] () => null {
    let size = square(limit)
    raylib.init_window(size, size, "t")
}
"""


@pytest.fixture
def document() -> Document:
    # Inside examples/, so the raylib externs of examples/config.yaml apply.
    return Document.open(EXAMPLES / "scratch.bif", SOURCE)


def _position(source: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    offset = -1
    for _ in range(occurrence + 1):
        offset = source.index(needle, offset + 1)
    line = source.count("\n", 0, offset)
    return line, offset - (source.rfind("\n", 0, offset) + 1)


def test_clean_document_has_no_diagnostics(document: Document) -> None:
    assert document.diagnostics() == []


def test_reports_syntax_errors() -> None:
    document = Document.open(EXAMPLES / "scratch.bif", "let f = (a: i32 => i32 a\n")
    assert [d.message for d in document.diagnostics()] == ["'(' is never closed; expected ')'"]


def test_reports_lowering_errors_at_their_token() -> None:
    source = "let g = () => i32 1\nlet f = () => i32 g()\n"
    (diagnostic,) = Document.open(EXAMPLES / "scratch.bif", source).diagnostics()
    assert "add it to the dependency list: [g]" in diagnostic.message
    assert diagnostic.range == ((1, 18), (1, 19))


def test_reports_type_errors() -> None:
    source = 'let f = (a: i32) => i32 {\n    let s = "x"\n    return a + s\n}\n'
    (diagnostic,) = Document.open(EXAMPLES / "scratch.bif", source).diagnostics()
    assert diagnostic.severity is Severity.ERROR
    assert diagnostic.range[0][0] == 1


def test_notes_a_missing_config(tmp_path: Path) -> None:
    (diagnostic,) = Document.open(tmp_path / "a.bif", "let a = 1\n").diagnostics()
    assert diagnostic.severity is Severity.INFORMATION


def test_symbols(document: Document) -> None:
    assert [(s.name, s.kind) for s in document.symbols()] == [
        ("raylib", SymbolKind.MODULE),
        ("limit", SymbolKind.CONSTANT),
        ("square", SymbolKind.FUNCTION),
        ("main", SymbolKind.FUNCTION),
    ]
    assert document.symbols()[2].detail == "(a: i32) => i32"


def test_definition_of_top_level_and_local_names(document: Document) -> None:
    square = document.definition(_position(SOURCE, "square(limit)"))
    assert square is not None
    assert square.range == ((2, 4), (2, 10))
    size = document.definition(_position(SOURCE, "size", 1))
    assert size is not None
    assert size.range == ((4, 8), (4, 12))


def test_definition_of_extern_is_in_config(document: Document) -> None:
    location = document.definition(_position(SOURCE, "init_window(size"))
    assert location is not None
    assert location.path.name == "config.yaml"
    row = location.range[0][0]
    assert "InitWindow" in location.path.read_text().splitlines()[row]


def test_hover(document: Document) -> None:
    assert "let square = (a: i32) => i32" in (document.hover(_position(SOURCE, "square(limit)")) or "")
    extern = document.hover(_position(SOURCE, "init_window(size"))
    assert extern is not None
    assert "raylib.init_window = (width: i32, height: i32, title: str) => null" in extern


def test_completes_module_members() -> None:
    source = 'let raylib = import("raylib")\nlet f = () => null {\n    raylib.\n}\n'
    document = Document.open(EXAMPLES / "scratch.bif", source)
    labels = {c.label for c in document.completions((2, 11))}
    assert {"init_window", "Color", "close_window"} <= labels
    assert "InitWindow" not in labels


def test_completes_names_in_scope(document: Document) -> None:
    labels = {c.label for c in document.completions(_position(SOURCE, "square(limit)"))}
    assert {"square", "limit", "size", "let", "i32"} <= labels


def test_outline_survives_syntax_errors() -> None:
    source = (
        'let rl = import("raylib")\nlet Context = struct {\n    width: i32,\n    title: str\n}\nlet f = () => i32 1\n'
    )
    document = Document.open(EXAMPLES / "scratch.bif", source)
    assert [s.name for s in document.symbols()] == ["rl", "Context", "f"]
    assert document.diagnostics()  # the struct fields need `let`, which is still reported


def test_every_feature_survives_every_half_typed_prefix() -> None:
    # As if typed one character at a time: no request may raise on any prefix.
    for end in range(len(SOURCE) + 1):
        document = Document.open(EXAMPLES / "scratch.bif", SOURCE[:end])
        document.diagnostics()
        document.symbols()
        lines = SOURCE[:end].split("\n")
        for position in [(0, 0), (len(lines) - 1, len(lines[-1]))]:
            document.hover(position)
            document.definition(position)
            document.completions(position)


def test_naming_conventions() -> None:
    source = (
        'let Rl = import("raylib")\n'
        "let point = struct { let X: i32, let y_pos: i32 }\n"
        "let drawText = (fontSize: i32) => null {\n    let myValue = fontSize\n}\n"
        "let Good = struct { let a: i32 }\nlet good_name = 1\n"
    )
    messages = [d.message for d in Document.open(EXAMPLES / "scratch.bif", source).naming()]
    assert messages == [
        "module 'Rl' should be snake_case: 'rl'",
        "object 'point' should be PascalCase: 'Point'",
        "field 'X' should be snake_case: 'x'",
        "function 'drawText' should be snake_case: 'draw_text'",
        "parameter 'fontSize' should be snake_case: 'font_size'",
        "variable 'myValue' should be snake_case: 'my_value'",
    ]


def test_naming_does_not_guess_on_half_typed_bindings() -> None:
    # `Context` is an object even though its struct body does not parse yet.
    source = "let Context = struct {\n    width: i32\n}\n"
    assert Document.open(EXAMPLES / "scratch.bif", source).naming() == []


def test_object_members() -> None:
    source = (
        "let Context = struct {\n    let width: i32,\n    static let new = [] () => Context Context(width: 1)\n}\n"
        "let main = [Context.new] () => null {\n    let ctx = Context.new()\n}\n"
    )
    document = Document.open(EXAMPLES / "scratch.bif", source)
    assert [d.message for d in document.diagnostics() if not d.unnecessary] == []  # `ctx` is unused, faded
    call = _position(source, "new()")
    assert document.hover(call) == "```bifrost\nstatic let new = () => Context\n// a member of Context\n```"
    definition = document.definition(call)
    assert definition is not None
    assert definition.range == ((2, 15), (2, 18))
    typing = Document.open(EXAMPLES / "scratch.bif", source.replace("Context.new()", "Context."))
    assert [c.label for c in typing.completions((5, len("    let ctx = Context.")))] == ["new"]
    bad = source.replace("static let new", "static let New")
    messages = [d.message for d in Document.open(EXAMPLES / "scratch.bif", bad).naming()]
    assert messages == ["static function 'New' should be snake_case: 'new'"]


def test_unused_names_are_faded() -> None:
    source = (
        "let f = (a: i32, b: i32, _c: i32) => i32 {\n    let x = a + 1\n    let unused = 2\n"
        "    let p: i32 = x\n    let g <- p\n    g -> p\n    return x\n}\n"
        "let h = (width: i32) => i32 f(width: 1)\n"
    )
    unused = Document.open(EXAMPLES / "scratch.bif", source).unused()
    assert [(d.range[0], d.message.split(" ")[0]) for d in unused] == [
        ((0, 17), "b"),  # a parameter never read
        ((2, 8), "unused"),
        ((4, 8), "g"),  # a guard that is locked and released but never read
        ((8, 9), "width"),  # `width: 1` names an argument; it does not read the parameter
    ]
    assert all(d.unnecessary for d in unused)


def test_locals_show_their_inferred_types() -> None:
    source = (
        'let io = import("std:stdio")\n'
        "let f = [io.printf] (w: i32) => null {\n"
        "    let a = 0\n"  # unconstrained: the default integer
        "    let k = 3\n"
        "    let m = k + w\n"  # makes `k` an i32
        "    let c = 1.5\n"
        '    let s = "hi"\n'
        "    let t: i32 = 2\n"  # written, so no hint
        '    io.printf("%d %d %f %s %d", a, m, c, s, t)\n'
        "}\n"
    )
    hints = Document.open(EXAMPLES / "scratch.bif", source).inlay_hints()
    assert [(row, label) for (row, _), label in ((h.position, h.label) for h in hints)] == [
        (2, ": i64"),
        (3, ": i32"),
        (4, ": i32"),
        (5, ": f64"),
        (6, ": str"),
    ]
