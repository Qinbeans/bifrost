from pathlib import Path

import pytest

from bifrost.formatter import FormatError, format_source

MESSY = """\
// header
let raylib=import("raylib")
let Point = struct {let x:i32,let y :i32}


let add=(a:i32,b:i32)=>i32 a+b*2   // trailing
let classify = [this] (n: i32) => i32 {
  if n<0&&!(n==-5) {return -1} else if n==0 { return 0 }
  else {
      /* block comment */
      return classify(n-1)
  }
}
let pick=(n:i32)=>i32 match n {1:10,2:{return 20},_:0}
let main = [raylib.init_window, raylib.close_window, raylib.window_should_close, raylib.begin_drawing] () => null {
    raylib.init_window(800,600,"Hello")
    while !raylib.window_should_close() { raylib.begin_drawing() }
    raylib.close_window()
}
"""

FORMATTED = """\
// header
let raylib = import("raylib")
let Point = struct { let x: i32, let y: i32 }

let add = (a: i32, b: i32) => i32 a + b * 2  // trailing
let classify = [this] (n: i32) => i32 {
    if n < 0 && !(n == -5) {
        return -1
    } else if n == 0 {
        return 0
    } else {
        /* block comment */
        return classify(n - 1)
    }
}
let pick = (n: i32) => i32 match n {
    1: 10,
    2: {
        return 20
    },
    _: 0
}
let main = [
    raylib.init_window,
    raylib.close_window,
    raylib.window_should_close,
    raylib.begin_drawing
] () => null {
    raylib.init_window(800, 600, "Hello")
    while !raylib.window_should_close() {
        raylib.begin_drawing()
    }
    raylib.close_window()
}
"""


def test_formats_whitespace_and_layout() -> None:
    assert format_source(MESSY) == FORMATTED


def test_is_idempotent() -> None:
    assert format_source(FORMATTED) == FORMATTED


def test_short_lists_stay_inline() -> None:
    source = "let f = [g] () => i64 match n { 1: 10, _: 0 }\n"
    assert format_source(source) == source


def test_example_is_formatted() -> None:
    example = Path(__file__).parents[1] / "examples" / "raylib" / "src" / "hello_raylib" / "hello_raylib.bif"
    assert format_source(example.read_text()) == example.read_text()


def test_refuses_syntax_errors() -> None:
    with pytest.raises(FormatError, match="syntax errors"):
        format_source("let f = (a: i32 => i32 a\n")


def test_objects_and_long_calls() -> None:
    source = (
        "let Context = struct {\n    let width: i32,\n    let title: str,\n"
        "    static let new = [] () => Context{\n"
        '        return Context(width: 800, title: "A title that makes this call too long for one line")\n'
        "    }\n}\n"
    )
    assert format_source(source) == (
        "let Context = struct {\n    let width: i32,\n    let title: str,\n"
        "    static let new = [] () => Context {\n"
        "        return Context(\n"
        "            width: 800,\n"
        '            title: "A title that makes this call too long for one line"\n'
        "        )\n    }\n}\n"
    )


def test_a_long_line_inside_a_block_breaks_only_itself() -> None:
    source = (
        "module routes = {\n"
        "    /* A comment long enough that its line alone is wider than the formatter's limit of columns */\n"
        "    let health = [http.text] (ctx: http.Context) => null {\n"
        '        http.text(ctx, 200, "ok")\n'
        "    }\n"
        "}\n"
    )
    assert format_source(source) == source
