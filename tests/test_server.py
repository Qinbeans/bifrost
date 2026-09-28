import re
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
    hints = [h for h in Document.open(EXAMPLES / "scratch.bif", source).inlay_hints() if not h.parameter]
    assert [(row, label) for (row, _), label in ((h.position, h.label) for h in hints)] == [
        (2, ": i64"),
        (3, ": i32"),
        (4, ": i32"),
        (5, ": f64"),
        (6, ": str"),
    ]


RECORDS = """\
let io = import("std:stdio")
let time = import("std:time")
let tasks = import("std:tasks")

let User = struct { let id: i64, let score: f64 }

let fetch = [time.sleep] async (id: i64) => User {
    await time.sleep(1)
    return User(id: id, score: 1.5)
}

let show = [io.printf] (label: str) => null {
    let point = #{x: 1, y: 2.5, tag: #{name: label, ok: true}}
    io.printf("%lld %s", point.x, point.tag.name)
}

let main = [io.printf, tasks.gather, fetch] async () => null {
    let found = await tasks.gather(user: fetch(1), other: fetch(2))
    io.printf("%lld %f", found.user.id, found.other.score)
}
"""


@pytest.fixture
def records(tmp_path: Path) -> Document:
    config = (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    (tmp_path / "config.yaml").write_text(config)  # `type: async`, so main may be async
    return Document.open(tmp_path / "main.bif", RECORDS)


@pytest.mark.parametrize(
    ("needle", "occurrence", "expected"),
    [
        ("x: 1", 0, "x: i64\n// a field of #{x: i64, y: f64, tag: #{name: str, ok: bool}}"),
        ("ok: true", 0, "ok: bool\n// a field of #{name: str, ok: bool}"),
        ("name", 1, "name: str\n// a field of point.tag"),
        ("user.id", 0, "user: User\n// a field of found"),
        ("id, found", 0, "id: i64\n// a field of found.user"),
        ("score)", 0, "score: f64\n// a field of found.other"),
    ],
)
def test_hover_shows_record_fields(records: Document, needle: str, occurrence: int, expected: str) -> None:
    assert records.hover(_position(RECORDS, needle, occurrence)) == f"```bifrost\n{expected}\n```"


def test_records_show_their_types_in_async_main_too(records: Document) -> None:
    hints = [hint for hint in records.inlay_hints() if not hint.parameter]
    # A long record is shortened, and shown whole in the hint's tooltip.
    assert (hints[0].label, hints[0].tooltip) == (
        ": #{x: i64, y: f64, ...}",
        "#{x: i64, y: f64, tag: #{name: str, ok: bool}}",
    )
    assert [hint.label for hint in hints] == [
        ": #{x: i64, y: f64, ...}",
        ": i64",  # the fields of `point`
        ": f64",
        ": #{name: str, ok: bool}",
        ": str",
        ": bool",
        ": #{user: User, other: User}",
    ]


def test_record_fields_show_their_types_after_their_names(tmp_path: Path) -> None:
    source = (
        'let io = import("std:stdio")\n'
        'let json = import("std:json")\n'
        "let show = [io.printf, json.encode] (a: i64, w: i32, r: f64) => null {\n"
        "    let point = #{x: 1, y: 2.5}\n"
        '    let text = json.encode(#{s: a + 1, c: a < 3, n: -r, px: point.x, half: w / 2, name: "x", k: 2})\n'
        '    io.printf("%s", text)\n'
        "}\n"
    )
    (tmp_path / "config.yaml").write_text(
        (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    )
    hints = [h for h in Document.open(tmp_path / "main.bif", source).inlay_hints() if not h.parameter]
    line = source.splitlines()[4]
    start = line.index("#{")
    fields = [
        (re.split(r"#\{|, ", line[: hint.position[1]])[-1], hint.label)  # the field's name, then its type
        for hint in hints
        if hint.position[0] == 4 and hint.position[1] > start  # not `let text`'s
    ]
    assert fields == [
        ("s", ": i64"),
        ("c", ": bool"),
        ("n", ": f64"),
        ("px", ": i64"),
        ("half", ": i32"),
        ("name", ": str"),
        ("k", ": i64"),
    ]


def test_fields_read_through_a_guard_show_their_declared_types(tmp_path: Path) -> None:
    # The compiler holds a guard as a pointer, so the field's type comes from the object's declaration.
    source = (
        'let mem = import("std:mem")\n'
        'let json = import("std:json")\n'
        "let State = struct { let hits: i64 }\n"
        "let show = [json.encode] (app: mem.Shared[State]) => null {\n"
        "    let s <- app\n"
        "    let text = json.encode(#{hits: s.hits})\n"
        "    s -> app\n"
        "}\n"
    )
    (tmp_path / "config.yaml").write_text(
        (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    )
    document = Document.open(tmp_path / "main.bif", source)
    assert [
        (hint.position[0], hint.label)
        for hint in document.inlay_hints()
        if hint.position[0] == 5 and not hint.parameter
    ][-1] == (
        5,
        ": i64",
    )
    assert document.hover(_position(source, "hits: s")) == "```bifrost\nhits: i64\n// a field of #{hits: i64}\n```"


def test_arguments_show_the_parameters_they_fill(tmp_path: Path) -> None:
    source = (
        'let io = import("std:stdio")\n'
        "let area = (width: i64, height: i64) => i64 width * height\n"
        "let show = [io.printf, area] (height: i64) => null {\n"
        "    let scale = (factor: i64) => i64 factor * 2\n"
        '    io.printf("%lld %lld\\n", area(3, height), scale(4))\n'
        "    let named = area(width: 2, height: 5)\n"
        "}\n"
    )
    (tmp_path / "config.yaml").write_text(
        (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    )
    hints = [hint for hint in Document.open(tmp_path / "main.bif", source).inlay_hints() if hint.parameter]
    line = source.splitlines()[4]
    # `height` fills `height`, printf's values are variadic, and named arguments say their names already.
    assert [(hint.position[0], line[hint.position[1] :][:3], hint.label) for hint in hints] == [
        (4, '"%l', "format:"),
        (4, "3, ", "width:"),
        (4, "4))", "factor:"),
    ]


NAMED = """\
let time = import("std:time")
let tasks = import("std:tasks")
let width = 3
let Box = struct { let width: i64 }
let area = (width: i64, height: i64) => i64 width * height
let wait = [time.sleep] async (id: i64) => i64 {
    await time.sleep(1)
    return id
}
let main = [area, tasks.gather, wait] async () => null {
    let box = Box(width: width)
    let size = area(width: 2, height: 5)
    let found = await tasks.gather(width: wait(1))
}
"""


@pytest.fixture
def named(tmp_path: Path) -> Document:
    (tmp_path / "config.yaml").write_text(
        (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    )
    return Document.open(tmp_path / "main.bif", NAMED)


@pytest.mark.parametrize(
    ("needle", "expected"),
    [
        ("width: width", "let width: i64\n// a field of Box"),
        ("width: 2", "width: i64\n// a parameter of area"),
    ],
)
def test_a_named_argument_is_the_field_or_parameter_it_names(named: Document, needle: str, expected: str) -> None:
    # Not the top-level `width` of the same name.
    assert named.hover(_position(NAMED, needle)) == f"```bifrost\n{expected}\n```"
    declared = "let width: i64 }" if "field" in expected else "width: i64, height"
    location = named.definition(_position(NAMED, needle))
    assert location is not None
    assert location.range[0] == _position(NAMED, declared.removeprefix("let "))


def test_a_named_argument_of_a_builtin_is_not_the_name_it_shadows(named: Document) -> None:
    assert named.hover(_position(NAMED, "width: wait")) is None
    assert named.definition(_position(NAMED, "width: wait")) is None


@pytest.mark.parametrize("compiled_before", [False, True])
def test_completes_the_fields_of_records(records: Document, compiled_before: bool) -> None:
    from bifrost.server import analysis  # noqa: PLC0415 - to forget the types other tests compiled

    analysis._LAST_TYPES.clear()
    if compiled_before:
        records.diagnostics()
    # Half typed, the file does not compile: the fields come from before, or from the code as written.
    typing = RECORDS.replace('io.printf("%lld %f", found.user.id, found.other.score)', "found.")
    typed = RECORDS.replace('io.printf("%lld %s", point.x, point.tag.name)', "point.tag.")
    for source, expected in [
        (typing, [("user", "user: User"), ("other", "other: User")]),
        (typed, [("name", "name: str"), ("ok", "ok: bool")]),
    ]:
        document = Document.open(records.path, source)
        line = next(i for i, text in enumerate(source.splitlines()) if text.strip() in {"found.", "point.tag."})
        position = (line, len(source.splitlines()[line]))
        assert [(c.label, c.detail) for c in document.completions(position)] == expected


RECORD_RESULTS = """\
let stdio = import("std:stdio")

let sign = [] (n: i64) => Record {
    return #{value: n, negative: n < 0}
}

let main = [stdio.printf, sign] () => null {
    let s = sign(-4)
    stdio.printf("%lld", s.value)
}
"""


def test_record_results_show_their_shape(tmp_path: Path) -> None:
    from bifrost.server import analysis  # noqa: PLC0415 - to forget the types other tests compiled

    (tmp_path / "config.yaml").write_text(
        (Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text()
    )
    document = Document.open(tmp_path / "main.bif", RECORD_RESULTS)
    assert document.diagnostics() == []
    shape = "#{value: i64, negative: bool}"
    assert document.hover(_position(RECORD_RESULTS, "Record")) == (
        f"```bifrost\nRecord = {shape}\n// the result: records of one shape, from every `return`\n```"
    )
    assert document.hover(_position(RECORD_RESULTS, "sign = ")) == (
        f"```bifrost\nlet sign = (n: i64) => Record\n// Record is {shape}\n```"
    )
    assert f": {shape}" in [hint.label for hint in document.inlay_hints()]
    # Half typed, nothing compiles: the fields come from what `sign` returns, as written.
    analysis._LAST_TYPES.clear()
    typing = RECORD_RESULTS.replace('stdio.printf("%lld", s.value)', "s.")
    line = typing.splitlines().index("    s.")
    completions = Document.open(tmp_path / "main.bif", typing).completions((line, 6))
    assert [(c.label, c.detail) for c in completions] == [("value", "value: i64"), ("negative", "negative: bool")]


@pytest.mark.parametrize(
    ("whole", "short"),
    [
        ("#{task1: #{task1: str}, task2: str}", "#{task1: #{task1: str}, ...}"),
        ("#{a: #{b: i64, c: i64}}", "#{a: #{b: i64, ...}}"),
        ("#{value: i64, negative: bool}", "#{value: i64, negative: bool}"),  # short enough already
        ("mem.Unique[str]", "mem.Unique[str]"),
    ],
)
def test_long_records_are_shortened_in_hints(whole: str, short: str) -> None:
    from bifrost.server.analysis import _short_type  # noqa: PLC0415

    assert _short_type(whole, nested=False) == short
