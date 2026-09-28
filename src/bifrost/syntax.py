"""Describe Bifrost syntax errors in terms of the source, not the parser.

When tree-sitter recovers from an error it inserts a zero-width "missing" token
(whichever is cheapest: ``this`` in a dependency list, ``null`` for a value) or
wraps what it could not parse in an ``ERROR`` node. Neither says what went
wrong, so ``syntax_errors`` looks at the tokens around them instead: a comma
before a closing bracket is a trailing comma, an operator with nothing after it
is missing its operand, an opening bracket without its match (anywhere in the
file) is unclosed, and otherwise the first token that could not be parsed is
unexpected, with what the parser would have accepted there when that is short.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

import tree_sitter_bifrost
from tree_sitter import Language, Node

_LANGUAGE = Language(tree_sitter_bifrost.language())
_MAX_EXPECTED = 3  # list what the parser expected only when it is this short
_MAX_SHOWN = 20  # characters of an unexpected token to quote

# What each comma-separated list holds, for "add ... after it".
_ELEMENTS = {
    "dependency_list": "a function to depend on",
    "local_dependency_list": "a function to depend on",
    "parameter_list": "a parameter (a: i32)",
    "struct_assignment": "a field (let x: i32)",
    "match_expression": "a match arm (1: value)",
    "user_function_call": "an argument",
    "builtin_call": "an argument",
    "list": "a value",
    "tuple": "a value",
    "record": "a field (name: value)",
    "tuple_type": "a type",
}

_PAIRS = {"(": ")", "[": "]", "{": "}", "#[": "]", "#(": ")", "#{": "}", "<": ">"}
_CLOSERS = {")", "]", "}"}

# Tokens that recovery inserts to stand in for a whole name or value.
_NAMES = {"identifier", "simple_identifier"}
_PUNCTUATION = {")", "]", "}", ">", "(", "[", "{", "=>", ":", "=", ",", "."}


@dataclass(frozen=True)
class SyntaxProblem:
    """A syntax error: what is wrong, and the node to point at."""

    message: str
    node: Node


def _text(node: Node) -> str:
    return (node.text or b"").decode()


def _leaves(node: Node) -> Iterator[Node]:
    if node.child_count == 0:
        yield node
    for child in node.children:
        yield from _leaves(child)


def _adjacent_leaf(node: Node, *, forward: bool) -> Node | None:
    """Return the token just before (or after) ``node``, skipping comments."""
    current: Node | None = node
    while current is not None:
        sibling = current.next_sibling if forward else current.prev_sibling
        while sibling is not None:
            leaves = [leaf for leaf in _leaves(sibling) if leaf.type != "comment" and not leaf.is_missing]
            if leaves:
                return leaves[0] if forward else leaves[-1]
            sibling = sibling.next_sibling if forward else sibling.prev_sibling
        current = current.parent
    return None


def _trailing_comma(container: Node | None) -> str:
    element = _ELEMENTS.get(container.type) if container is not None else None
    fix = f"remove this ',' or add {element} after it" if element else "remove this ','"
    return f"trailing commas are not allowed; {fix}"


def _missing(node: Node) -> SyntaxProblem:
    previous = _adjacent_leaf(node, forward=False)
    if previous is not None and previous.type == ",":
        return SyntaxProblem(_trailing_comma(previous.parent), previous)
    after = f" after '{_text(previous)}'" if previous is not None else ""
    if node.type in _PUNCTUATION:
        return SyntaxProblem(f"expected '{node.type}'{after}", previous or node)
    # Recovery fills a missing value with a made-up name too; only a name that is
    # not an expression's (`let` ... `=`, a parameter) is really a name.
    in_expression = (
        node.parent is not None and node.parent.parent is not None and node.parent.parent.type == "expression"
    )
    what = "a name" if node.type in _NAMES and not in_expression else "a value"
    return SyntaxProblem(f"expected {what}{after}", previous or node)


def _unbalanced(root: Node) -> SyntaxProblem | None:
    """Find a bracket without its match, across the whole file."""
    stack: list[Node] = []
    for leaf in _leaves(root):
        if leaf.is_missing:
            continue
        if leaf.type in _PAIRS and leaf.type != "<":
            stack.append(leaf)
        elif leaf.type in _CLOSERS:
            if stack and _PAIRS[stack[-1].type] == leaf.type:
                stack.pop()
            elif any(_PAIRS[opener.type] == leaf.type for opener in stack):
                # `{ g(1 }`: the `}` has its `{`; the `(` between them was never closed.
                opener = stack[-1]
                return SyntaxProblem(f"'{opener.type}' is never closed; expected '{_PAIRS[opener.type]}'", opener)
            else:
                return SyntaxProblem(f"'{leaf.type}' has no matching opening bracket", leaf)
    if stack:
        opener = stack[-1]
        return SyntaxProblem(f"'{opener.type}' is never closed; expected '{_PAIRS[opener.type]}'", opener)
    return None


def _expected(previous: Node | None) -> str:
    """Return " (expected 'let' or '}')" when few tokens could follow ``previous``."""
    if previous is None:
        return ""
    symbols = _LANGUAGE.lookahead_iterator(previous.next_parse_state).symbols()
    tokens = sorted(
        {
            _LANGUAGE.node_kind_for_id(symbol)
            for symbol in symbols
            if _LANGUAGE.node_kind_is_visible(symbol) and not _LANGUAGE.node_kind_is_named(symbol)
        }
        - {"", "end"}
    )
    if not tokens or len(tokens) > _MAX_EXPECTED:
        return ""
    return " (expected " + " or ".join(f"'{token}'" for token in tokens) + ")"


def _first_nested_error(node: Node) -> Node | None:
    for child in node.children:
        if child.is_error:
            return _first_nested_error(child) or child
        found = _first_nested_error(child) if child.has_error else None
        if found is not None:
            return found
    return None


def _error(node: Node, source: bytes) -> SyntaxProblem:
    tokens = [leaf for leaf in _leaves(node) if leaf.type != "comment" and not leaf.is_missing]
    if not tokens:
        return SyntaxProblem("syntax error", node)
    first = tokens[0]
    following = _adjacent_leaf(node, forward=True)
    if first.type == "," and len(tokens) == 1 and following is not None and following.type in _CLOSERS:
        return SyntaxProblem(_trailing_comma(node.parent), first)
    # Recovery keeps the tokens it managed to read (``struct {``) in the ERROR
    # node; the first one it could not place is the nested ERROR, if any.
    target = _first_nested_error(node) or first
    word = re.match(rb"\w+|\S+", source[target.start_byte :])
    text = word.group().decode(errors="replace") if word else _text(target)
    shown = text if len(text) <= _MAX_SHOWN else text[: _MAX_SHOWN - 3] + "..."
    return SyntaxProblem(f"unexpected '{shown}'{_expected(_adjacent_leaf(target, forward=False))}", target)


def _problems(node: Node, source: bytes) -> list[SyntaxProblem]:
    if node.is_missing:
        return [_missing(node)]
    if node.is_error:
        return [_error(node, source)]
    if not node.has_error:
        return []
    return [problem for child in node.children for problem in _problems(child, source)]


def syntax_errors(root: Node) -> list[SyntaxProblem]:
    """Describe every syntax error under ``root``, outermost first.

    An unmatched bracket explains every error after it, so it is reported alone.
    """
    if not root.has_error:
        return []
    unbalanced = _unbalanced(root)
    if unbalanced is not None:
        return [unbalanced]
    return _problems(root, root.text or b"")


__all__ = ["SyntaxProblem", "syntax_errors"]
