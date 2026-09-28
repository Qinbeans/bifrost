from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel
from mlir_python.lang import Function

from bifrost.configs import Config, ConfigBuilder
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project


def _project() -> Project:
    return Project(
        Config(
            package={"name": "test", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _lower(tmp_path: Path, source: str) -> Project:
    path = tmp_path / "test.bif"
    path.write_text(source)
    project = _project()
    lower_file(project, path)
    return project


def _function(project: Project, name: str) -> Function:
    return next(f for f in project.program.functions if f.name == name)


def test_functions_arithmetic_and_calls(tmp_path: Path) -> None:
    project = _lower(
        tmp_path,
        """
        // constants and forward calls
        let limit = 10
        let twice = [double] (a: i32) => i32 double(a)
        let double = (a: i32) => i32 a * 2 + limit - limit
        """,
    )
    assert _function(project, "twice")(21) == 42


def test_control_flow(tmp_path: Path) -> None:
    project = _lower(
        tmp_path,
        """
        let sum_to = (n: i32) => i32 {
            let total = 0
            let i = 0
            while i <= n {
                let total = total + i
                let i = i + 1
            }
            return total
        }
        let sign = (n: i32) => i32 {
            if n < 0 { return -1 } else if n == 0 { return 0 } else { return 1 }
        }
        let pick = (n: i32) => i32 {
            match n { 1: return 10, 2: { return 20 }, _: return 0 }
        }
        let pick_value = (n: i32) => i64 match n {
            1: 10, _: if n == 2 { 20 } else { 0 }
        }
        """,
    )
    assert _function(project, "sum_to")(4) == 10
    assert [_function(project, "sign")(n) for n in (-3, 0, 3)] == [-1, 0, 1]
    assert [_function(project, "pick")(n) for n in (1, 2, 3)] == [10, 20, 0]
    assert [_function(project, "pick_value")(n) for n in (1, 2, 3)] == [10, 20, 0]


def test_structs(tmp_path: Path) -> None:
    project = _lower(
        tmp_path,
        """
        let Point = struct { let x: i32, let y: i32 }
        let dot = (a: i32, b: i32) => i32 {
            let p = Point(a, b)
            return p.x * p.y
        }
        """,
    )
    assert _function(project, "dot")(6, 7) == 42


@pytest.mark.parametrize(
    ("source", "message", "line"),
    [
        ("let f = () => i32 missing(1)\n", "'missing' is not defined", 1),
        ("let a = 1\nlet a = 2\n", "'a' is already defined", 2),
        ('let m = import("nope")\n', "unknown module 'nope'", 1),
        (
            "let f = () => i32 {\n  let x = #[1, ...y]\n  return 0\n}\n",
            "spreading into a list is not supported yet",
            2,
        ),
    ],
)
def test_errors_point_at_bifrost_source(tmp_path: Path, source: str, message: str, line: int) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source)
    assert message in error.value.msg
    assert error.value.lineno == line


def test_compile_errors_are_mapped_to_bifrost_lines(tmp_path: Path) -> None:
    path = tmp_path / "test.bif"
    path.write_text('let f = (a: i32) => i32 {\n    let s = "x"\n    return a + s\n}\n')
    project = _project()
    unit = lower_file(project, path)
    with pytest.raises(BifrostError) as error, unit.errors():
        _function(project, "f")(1)
    assert (error.value.lineno, error.value.offset) == (2, 13)  # the string literal


def test_precedence_and_parentheses(tmp_path: Path) -> None:
    project = _lower(
        tmp_path,
        """
        let arithmetic = (x: i32) => i32 x + 2 * 3 - (x - 1) * 2
        let logic = (n: i32) => bool n < 0 && !(n == -5) || n > 10
        let power = (x: f64) => f64 -x ** 2
        """,
    )
    assert _function(project, "arithmetic")(4) == 4 + 6 - 6
    assert [_function(project, "logic")(n) for n in (-3, -5, 5, 11)] == [
        True,
        False,
        False,
        True,
    ]
    assert _function(project, "power")(3.0) == -9.0


def test_dependencies_allow_listed_calls_and_recursion(tmp_path: Path) -> None:
    project = _lower(
        tmp_path,
        """
        let Point = struct { let x: i32, let y: i32 }
        let square = (a: i32) => i32 a * a
        let factorial = [this, square] (n: i32) => i32 {
            let p = Point(n, n)
            if n <= 1 { return 1 }
            return n * factorial(n - 1) + square(0)
        }
        """,
    )
    assert _function(project, "factorial")(5) == 120


@pytest.mark.parametrize(
    ("source", "message", "line"),
    [
        (
            "let g = () => i32 1\nlet f = () => i32 g()\n",
            "f calls 'g' without depending on it; add it to the dependency list: [g]",
            2,
        ),
        (
            "let f = (n: i32) => i32 {\n  return f(n)\n}\n",
            "add it to the dependency list: [this]",
            2,
        ),
        (
            "let g = () => i32 1\nlet f = [g] () => i32 2\n",
            "'g' is a dependency of f but never called",
            2,
        ),
        (
            "let n = 3\nlet f = [n] () => i32 n\n",
            "'n' is not a function; list only functions",
            2,
        ),
        (
            "let g = () => i32 1\nlet f = [g, g] () => i32 g()\n",
            "'g' is already a dependency",
            2,
        ),
    ],
)
def test_dependency_list_is_enforced(tmp_path: Path, source: str, message: str, line: int) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source)
    assert message in error.value.msg
    assert error.value.lineno == line


def test_extern_calls_need_dependencies(tmp_path: Path) -> None:
    config = ConfigBuilder(Path(__file__).parents[1] / "examples" / "raylib" / "config.yaml").build()
    path = tmp_path / "test.bif"

    def lower(dependencies: str, body: str) -> None:
        path.write_text(f'let raylib = import("raylib")\nlet main = [{dependencies}] () => null {{\n{body}\n}}\n')
        lower_file(Project(config), path)

    lower("raylib.init_window", 'raylib.init_window(1, 2, "t")\nraylib.Color(0, 0, 0, 0)')
    with pytest.raises(BifrostError, match=r"\[raylib\.close_window\]"):
        lower("raylib.init_window", 'raylib.init_window(1, 2, "t")\nraylib.close_window()')
    with pytest.raises(BifrostError, match="'raylib' is not a function"):
        lower("raylib", 'raylib.init_window(1, 2, "t")')
    with pytest.raises(BifrostError, match=r"'raylib\.Color' is not a function"):
        lower("raylib.Color", "raylib.Color(0, 0, 0, 0)")


OBJECT = """
let Point = struct {
    let x: i32,
    let y: i32,
    static let new = [] (x: i32) => Point Point(x: x, y: x * 2),
    static let origin = () => Point Point(0, 0)
}
let sum = [Point.new] (a: i32) => i32 {
    let p = Point.new(a)
    return p.x + p.y
}
"""


def test_objects_have_static_functions_and_named_construction(tmp_path: Path) -> None:
    project = _lower(tmp_path, OBJECT)
    assert _function(project, "sum")(7) == 21
    assert {f.name for f in project.program.functions} >= {"Point_new", "Point_origin", "sum"}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            OBJECT.replace("[Point.new] (a: i32)", "(a: i32)"),
            "sum calls 'Point.new' without depending on it; add it to the dependency list: [Point.new]",
        ),
        (
            OBJECT.replace("static let new = []", "static let new = [super]"),
            "static function Point.new has no instance, so it cannot depend on super",
        ),
        (
            OBJECT.replace("static let new", "let new"),
            "a method (a function without `static`) is not supported yet",
        ),
        (OBJECT.replace("let y: i32", "static let y: i32"), "static data is not supported"),
        (OBJECT.replace("let y: i32", "let x: i32"), "Point already has a member named 'x'"),
        (OBJECT.replace("Point(x: x, y: x * 2)", "Point(x: x, x * 2)"), "a positional argument cannot follow"),
    ],
)
def test_object_errors(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, source)
    assert message in error.value.msg


def test_empty_dependency_list_means_no_dependencies(tmp_path: Path) -> None:
    project = _lower(tmp_path, "let one = [] () => i32 1\n")
    assert _function(project, "one")() == 1
    with pytest.raises(BifrostError, match="add it to the dependency list"):
        _lower(tmp_path, "let one = () => i32 1\nlet two = [] () => i32 one() + 1\n")


@pytest.mark.parametrize(
    "body",
    [
        "return 1",
        "if n > 0 { return 1 } else if n < 0 { return -1 } else { return 0 }",
        "match n { 1: return 10, _: { return 0 } }",
        "while true { return n }",
        "let m = n + 1\n    return m",
    ],
)
def test_bodies_that_return_on_every_path(tmp_path: Path, body: str) -> None:
    project = _lower(tmp_path, f"let f = (n: i64) => i64 {{\n    {body}\n}}\n")
    assert isinstance(_function(project, "f")(1), int)


@pytest.mark.parametrize(
    "body",
    [
        "let m = n + 1",
        "if n > 0 { return 1 }",
        "if n > 0 { return 1 } else { let m = 2 }",
        "match n { 1: return 10, 2: return 20 }",
        "while n > 0 { return n }",
        "forall i in #[0...3] { return i }",
    ],
)
def test_a_body_that_can_end_without_a_return_is_an_error(tmp_path: Path, body: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, f"let f = (n: i64) => i64 {{\n    {body}\n}}\n")
    assert error.value.msg == "f must return i64, but its body can end without a `return`; return a i64 at the end"
    assert error.value.lineno == 1  # at the result type
