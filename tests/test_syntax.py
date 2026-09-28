import pytest
import tree_sitter_bifrost
from tree_sitter import Language, Parser

from bifrost.syntax import syntax_errors

_PARSER = Parser(Language(tree_sitter_bifrost.language()))


@pytest.mark.parametrize(
    ("source", "message", "column"),
    [
        (
            "let f = [g, h,] () => null g()\n",
            "trailing commas are not allowed; remove this ',' or add a function to depend on after it",
            13,
        ),
        (
            "let f = (a: i32,) => i32 a\n",
            "trailing commas are not allowed; remove this ',' or add a parameter (a: i32) after it",
            15,
        ),
        ("let f = () => i32 g(1, 2,)\n", "remove this ',' or add an argument after it", 24),
        ("let a = #[1, 2,]\n", "remove this ',' or add a value after it", 14),
        ("let f = () => null { match x { 1: 2, } }\n", "add a match arm (1: value) after it", 35),
        ("let f = (a: i32 => i32 a\n", "'(' is never closed; expected ')'", 8),
        ("let a = 1 +\n", "expected a value after '+'", 10),
        ("let a =\n", "expected a value after '='", 6),
        ("let f = () => null { g(1 }\n", "'(' is never closed; expected ')'", 22),
        (
            "let P = struct {\n    x: i32\n}\n",
            "unexpected 'x' (expected 'let' or 'static' or '}')",
            None,
        ),
        ("let a = 1 1\n", "unexpected '1'", 10),
    ],
)
def test_describes_syntax_errors(source: str, message: str, column: int | None) -> None:
    problem = syntax_errors(_PARSER.parse(source.encode()).root_node)[0]
    assert message in problem.message
    if column is not None:
        assert problem.node.start_point == (0, column)


def test_valid_source_has_no_errors() -> None:
    assert syntax_errors(_PARSER.parse(b"let f = [g] (a: i32) => i32 g(a, 1)\n").root_node) == []
