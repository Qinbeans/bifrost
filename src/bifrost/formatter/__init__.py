"""Format Bifrost source from its tree-sitter CST.

The output uses four-space indentation, one item per line in blocks, and
spaces around binary operators. Dependency lists, structs and matches stay on
one line when they fit in ``WIDTH`` columns and break one entry per line when
they do not. Comments are kept; a construct holding a comment the formatter
does not place (inside a parameter list, say) is left exactly as written.

Formatting only changes whitespace: ``format_source`` checks that its output
has the same tokens as its input and raises ``FormatError`` otherwise.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import tree_sitter_bifrost
from tree_sitter import Language, Node, Parser

WIDTH = 88
INDENT = "    "

_LANGUAGE = Language(tree_sitter_bifrost.language())

# Nodes whose formatting places comments among their children.
_COMMENT_AWARE = {
    "source_file",
    "module",
    "block_expression",
    "struct_assignment",
    "match_expression",
    "dependency_list",
    "local_dependency_list",
}


class FormatError(ValueError):
    """The source cannot be formatted (it does not parse)."""


@dataclass
class _Entry:
    """One item of a sequence (a block's statement, a list's element)."""

    text: str
    is_comment: bool
    blank_before: bool
    trailing_comment: str | None = None


def _text(node: Node) -> str:
    return (node.text or b"").decode()


def _named(node: Node) -> list[Node]:
    return [child for child in node.named_children if child.type != "comment"]


def _leaves(node: Node) -> Iterator[tuple[str, str]]:
    if node.child_count == 0:
        text = _text(node)
        yield node.type, text.rstrip() if node.type == "comment" else text
    for child in node.children:
        yield from _leaves(child)


def parse(source: str) -> Node:
    """Parse ``source``, raising ``FormatError`` if it has syntax errors."""
    root = Parser(_LANGUAGE).parse(source.encode()).root_node
    if root.has_error:
        msg = "the source has syntax errors"
        raise FormatError(msg)
    return root


def format_source(source: str) -> str:
    """Return ``source`` formatted.

    Raises:
        FormatError: If ``source`` has syntax errors, or formatting would
            change more than whitespace.

    """
    root = parse(source)
    formatted = _Formatter().source_file(root)
    if list(_leaves(parse(formatted))) != list(_leaves(root)):
        msg = "formatting would change the program; please report this source"
        raise FormatError(msg)
    return formatted


class _Formatter:
    # Set while re-formatting a statement that did not fit: the next call
    # formatted (the outermost one) puts one argument per line.
    _break_next_call = False
    # Where the next record starts, when it is not at the start of its line: `f(a, #{`.
    _column: int | None = None

    # -- dispatch -----------------------------------------------------------------

    def format(self, node: Node, depth: int) -> str:
        """Format ``node``, whose first line starts at indentation ``depth``."""
        has_comment = any(child.type == "comment" for child in node.children)
        if has_comment and node.type not in _COMMENT_AWARE:
            return _text(node)
        handler: Callable[[Node, int], str] | None = getattr(self, f"_{node.type}", None)
        if handler is None:
            return _text(node)
        return handler(node, depth)

    def _expression(self, node: Node, depth: int) -> str:
        return self.format(_named(node)[0], depth)

    _condition = _getter_owner = _type_or_object = _statement = _expression
    _function_call = _expression

    # -- sequences ----------------------------------------------------------------

    def entries(self, children: list[Node], depth: int, item: Callable[[Node, int], str]) -> list[_Entry]:
        """Format ``children`` (items and comments) for one entry per line."""
        entries: list[_Entry] = []
        previous: Node | None = None
        for child in children:
            if child.type == "comment":
                same_line = (
                    previous is not None
                    and child.start_point.row == previous.end_point.row
                    and entries
                    and entries[-1].trailing_comment is None
                )
                if same_line:
                    entries[-1].trailing_comment = _text(child).rstrip()
                    previous = child
                    continue
            blank = previous is not None and child.start_point.row - previous.end_point.row > 1
            is_comment = child.type == "comment"
            text = _text(child).rstrip() if is_comment else item(child, depth)
            entries.append(_Entry(text, is_comment, blank))
            previous = child
        return entries

    @staticmethod
    def lines(entries: list[_Entry], depth: int, separator: str = "") -> str:
        """Render entries one per line, ``separator`` after all but the last item."""
        items = [i for i, entry in enumerate(entries) if not entry.is_comment]
        last = items[-1] if items else -1
        rendered = []
        for i, entry in enumerate(entries):
            if entry.blank_before and rendered:
                rendered.append("")
            text = entry.text
            if not entry.is_comment and i != last:
                text += separator
            if entry.trailing_comment:
                text += f"  {entry.trailing_comment}"
            rendered.append(INDENT * depth + text)
        return "\n".join(rendered)

    @staticmethod
    def inline(entries: list[_Entry]) -> str | None:
        """Entries on one line, or ``None`` when comments or newlines prevent it."""
        if any(e.is_comment or e.trailing_comment or "\n" in e.text for e in entries):
            return None
        return ", ".join(entry.text for entry in entries)

    def bracketed(
        self,
        node: Node,
        depth: int,
        brackets: tuple[str, str],
        prefix_width: int,
        spaced: bool,
    ) -> str:
        """Format a comma-separated, bracketed list inline or one entry per line."""
        children = list(node.named_children)
        entries = self.entries(children, depth + 1, self.format)
        opening, closing = brackets
        one_line = self.inline(entries)
        if one_line is not None:
            pad = " " if spaced and one_line else ""
            text = f"{opening}{pad}{one_line}{pad}{closing}"
            if prefix_width + len(text) <= WIDTH:
                return text
        body = self.lines(entries, depth + 1, ",")
        return f"{opening}\n{body}\n{INDENT * depth}{closing}"

    # -- top level ----------------------------------------------------------------

    def source_file(self, node: Node) -> str:
        entries = self.entries(node.named_children, 0, self.statement)
        text = self.lines(entries, 0)
        return text + "\n" if text else ""

    def _assignment(self, node: Node, depth: int) -> str:
        parts = _named(node)
        identifier, value = parts[0], parts[-1]
        type_node = node.child_by_field_name("type")
        annotation = f": {self.format(type_node, depth)}" if type_node is not None else ""
        head = f"let {_text(identifier)}{annotation} = "
        return head + self.value(value, depth, len(INDENT * depth + head))

    _local_assignment = _assignment

    def value(self, node: Node, depth: int, column: int) -> str:
        if node.type in {"function_definition", "local_function_definition"}:
            return self.function(node, depth, column)
        return self.format(node, depth)

    def _function_definition(self, node: Node, depth: int) -> str:
        return self.function(node, depth, len(INDENT * depth))

    _local_function_definition = _function_definition

    def function(self, node: Node, depth: int, column: int) -> str:
        """``[deps] async (params) => T body``, breaking the dependencies if too long."""
        parts = _named(node)
        dependencies = None
        if parts[0].type in {"dependency_list", "local_dependency_list"}:
            dependencies, parts = parts[0], parts[1:]
        parameters, return_type, body = parts
        is_async = "async " if node.child_by_field_name("async") is not None else ""
        signature = f"{is_async}{self.format(parameters, depth)} => {self.format(return_type, depth)}"
        opening = " {" if body.type == "block_expression" else ""
        head = ""
        if dependencies is not None:
            rest = len(signature) + len(opening) + 1
            head = self.dependencies(dependencies, depth, column, rest) + " "
        return f"{head}{signature} {self.format(body, depth)}"

    def dependencies(self, node: Node, depth: int, column: int, rest: int) -> str:
        entries = []
        for child in node.children:
            if child.is_named or child.type in {"this", "super"}:
                entries.append(child)
        formatted = self.entries(entries, depth + 1, lambda n, _: _text(n))
        one_line = self.inline(formatted)
        if one_line is not None and column + len(one_line) + 2 + rest <= WIDTH:
            return f"[{one_line}]"
        return f"[\n{self.lines(formatted, depth + 1, ',')}\n{INDENT * depth}]"

    _dependency_list = _local_dependency_list = lambda self, node, depth: self.dependencies(node, depth, 0, 0)

    def _parameter_list(self, node: Node, depth: int) -> str:
        return "(" + ", ".join(self.format(p, depth) for p in _named(node)) + ")"

    def _parameter(self, node: Node, depth: int) -> str:
        identifier, type_node = _named(node)
        return f"{_text(identifier)}: {self.format(type_node, depth)}"

    # -- structs ------------------------------------------------------------------

    def _struct_assignment(self, node: Node, depth: int) -> str:
        return "struct " + self.bracketed(node, depth, ("{", "}"), len(INDENT * depth) + 20, spaced=True)

    def _struct_field(self, node: Node, depth: int) -> str:
        parts = _named(node)
        modifier = "static " if node.child_by_field_name("modifier") is not None else ""
        text = f"{modifier}let {_text(parts[0])}"
        for part in parts[1:]:
            if part.type == "type_or_object":
                text += f": {self.format(part, depth)}"
            else:
                text += " = " + self.function(part, depth, len(text) + 3)
        return text

    def _module(self, node: Node, depth: int) -> str:
        name = _text(node.child_by_field_name("name"))
        body = [child for child in node.named_children if child.type in {"assignment", "comment"}]
        if not body:
            return f"module {name} = {{}}"
        entries = self.entries(body, depth + 1, self.statement)
        return f"module {name} = {{\n{self.lines(entries, depth + 1)}\n{INDENT * depth}}}"

    def _export_statement(self, node: Node, depth: int) -> str:
        del depth
        return "export(" + ", ".join(_text(name) for name in node.children_by_field_name("module")) + ")"

    # -- statements ---------------------------------------------------------------

    def _lock(self, node: Node, depth: int) -> str:
        source = self.format(node.child_by_field_name("source"), depth)
        type_node = node.child_by_field_name("type")
        annotation = f": {self.format(type_node, depth)}" if type_node is not None else ""
        return f"let {_text(node.child_by_field_name('guard'))}{annotation} <- {source}"

    def _release(self, node: Node, depth: int) -> str:
        source = self.format(node.child_by_field_name("source"), depth)
        return f"{_text(node.child_by_field_name('guard'))} -> {source}"

    def _field_assignment(self, node: Node, depth: int) -> str:
        target = self.format(node.child_by_field_name("target"), depth)
        return f"{target} = {self.format(node.child_by_field_name('value'), depth)}"

    def _guard_assignment(self, node: Node, depth: int) -> str:
        guard = _text(node.child_by_field_name("guard"))
        return f"{guard} = {self.format(node.child_by_field_name('value'), depth)}"

    def _block_expression(self, node: Node, depth: int) -> str:
        if not node.named_children:
            return "{}"
        entries = self.entries(node.named_children, depth + 1, self.statement)
        return "{\n" + self.lines(entries, depth + 1) + "\n" + INDENT * depth + "}"

    def _return_statement(self, node: Node, depth: int) -> str:
        values = _named(node)
        return "return " + self.format(values[0], depth) if values else "return"

    def _if(self, node: Node, depth: int) -> str:
        condition, body, *rest = _named(node)
        text = f"if {self.format(condition, depth)} {self.format(body, depth)}"
        if rest:
            text += f" else {self.format(rest[0], depth)}"
        return text

    def _while(self, node: Node, depth: int) -> str:
        condition, body = _named(node)
        return f"while {self.format(condition, depth)} {self.format(body, depth)}"

    def _forall(self, node: Node, depth: int) -> str:
        identifier, iterable, body = _named(node)
        iterated = self.format(iterable, depth)
        return f"forall {_text(identifier)} in {iterated} {self.format(body, depth)}"

    def _match_expression(self, node: Node, depth: int) -> str:
        subject, *_ = _named(node)
        arms = [c for c in node.named_children if c != subject]
        head = f"match {self.format(subject, depth)} "
        entries = self.entries(arms, depth + 1, self.format)
        blocks = any(_named(arm)[-1].type == "block_expression" for arm in arms if arm.type == "match_arm")
        one_line = None if blocks else self.inline(entries)
        if one_line is not None:
            text = f"{head}{{ {one_line} }}"
            if len(INDENT * depth) + len(text) <= WIDTH:
                return text
        body = self.lines(entries, depth + 1, ",")
        return f"{head}{{\n{body}\n{INDENT * depth}}}"

    def _match_arm(self, node: Node, depth: int) -> str:
        condition, body = _named(node)
        return f"{self.format(condition, depth)}: {self.format(body, depth)}"

    # -- expressions --------------------------------------------------------------

    def _binary_expression(self, node: Node, depth: int) -> str:
        left = self.format(node.child_by_field_name("left"), depth)
        operator = _text(node.child_by_field_name("operator"))
        right = self.format(node.child_by_field_name("right"), depth)
        return f"{left} {operator} {right}"

    def _unary_expression(self, node: Node, depth: int) -> str:
        operator = _text(node.child_by_field_name("operator"))
        return operator + self.format(node.child_by_field_name("argument"), depth)

    def _await_expression(self, node: Node, depth: int) -> str:
        return "await " + self.format(node.child_by_field_name("value"), depth)

    def _parenthesized_expression(self, node: Node, depth: int) -> str:
        return f"({self.format(_named(node)[0], depth)})"

    def statement(self, node: Node, depth: int) -> str:
        """Format a statement on one line, or with its outermost call broken if it is too long.

        Only the statement's first line counts: the lines of a block inside it are
        statements of their own, broken (or not) when they are formatted.
        """
        text = self.format(node, depth)
        if len(INDENT * depth + text.split("\n")[0]) <= WIDTH:
            return text
        self._break_next_call = True
        try:
            return self.format(node, depth)
        finally:
            self._break_next_call = False

    def _user_function_call(self, node: Node, depth: int) -> str:
        function = node.child_by_field_name("function")
        breaking, self._break_next_call = self._break_next_call, False
        children = [a for a in _named(node) if a != function]
        if breaking and children and _hugs(children[-1]):
            hugged = self.hug(_text(function), children, depth, _path_before(node))
            if hugged is not None:
                return hugged
        arguments = [self.format(a, depth + 1 if breaking else depth) for a in children]
        if breaking and arguments:
            inner = ",\n".join(INDENT * (depth + 1) + argument for argument in arguments)
            return f"{_text(function)}(\n{inner}\n{INDENT * depth})"
        return f"{_text(function)}({', '.join(arguments)})"

    def hug(self, function: str, children: list[Node], depth: int, path: str = "") -> str | None:
        """``f(a, b, #{`` ... ``})``: the leading arguments on the call's line, the last one's body below.

        ``None`` if the leading arguments do not fit on the line; ``path`` is what precedes the call
        on it (``http.``).
        """
        leading = [self.format(child, depth) for child in children[:-1]]
        if any("\n" in argument for argument in leading):
            return None
        head = f"{function}({''.join(argument + ', ' for argument in leading)}"
        self._column = len(INDENT * depth + path + head)
        last = self.format(children[-1], depth)
        self._column = None
        if len(INDENT * depth + path + head + last.split("\n")[0]) > WIDTH:
            return None
        return f"{head}{last})"

    def _named_argument(self, node: Node, depth: int) -> str:
        name = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        return f"{_text(name)}: {self.format(value, depth)}"

    _builtin_call = _user_function_call

    def _child_annotation(self, node: Node, depth: int) -> str:
        return ".".join(self.format(part, depth) for part in _named(node))

    def _get_expression(self, node: Node, depth: int) -> str:
        owner, index = _named(node)
        return f"{self.format(owner, depth)}[{self.format(index, depth)}]"

    def _literal(self, node: Node, depth: int) -> str:
        return self.format(_named(node)[0], depth)

    def _list(self, node: Node, depth: int) -> str:
        return self.bracketed(node, depth, ("#[", "]"), len(INDENT * depth), False)

    def _tuple(self, node: Node, depth: int) -> str:
        return self.bracketed(node, depth, ("#(", ")"), len(INDENT * depth), False)

    def _record(self, node: Node, depth: int) -> str:
        column, self._column = self._column, None
        return self.bracketed(node, depth, ("#{", "}"), column or len(INDENT * depth), False)

    def _record_field(self, node: Node, depth: int) -> str:
        name, value = node.child_by_field_name("name"), node.child_by_field_name("value")
        return f"{_text(name)}: {self.format(value, depth)}"

    def _spread_between(self, node: Node, depth: int) -> str:
        start, _, end = node.named_children
        return f"{self.format(start, depth)}...{self.format(end, depth)}"

    def _spread_action(self, node: Node, depth: int) -> str:
        return "..." + self.format(_named(node)[-1], depth)

    def _rest_of(self, node: Node, depth: int) -> str:
        return self.format(_named(node)[0], depth) + "..."

    # -- types --------------------------------------------------------------------

    def _type(self, node: Node, depth: int) -> str:
        inner = _named(node)
        return self.format(inner[0], depth) if inner else _text(node)

    def _generic_type(self, node: Node, depth: int) -> str:
        arguments = ", ".join(self.format(argument, depth) for argument in node.children_by_field_name("argument"))
        return f"{self.format(node.child_by_field_name('base'), depth)}[{arguments}]"

    def _list_type(self, node: Node, depth: int) -> str:
        return self.format(_named(node)[0], depth) + "[]"

    def _tuple_type(self, node: Node, depth: int) -> str:
        return "<" + ", ".join(self.format(t, depth) for t in _named(node)) + ">"

    def _record_type(self, node: Node, depth: int) -> str:
        """``#{name: str, ms: i64}``, on one line like other types."""
        return "#{" + ", ".join(self.format(field, depth) for field in _named(node)) + "}"

    def _record_type_field(self, node: Node, depth: int) -> str:
        name, kind = node.child_by_field_name("name"), node.child_by_field_name("type")
        return f"{_text(name)}: {self.format(kind, depth)}"

    def _function_type(self, node: Node, depth: int) -> str:
        parameters = ", ".join(self.format(p, depth) for p in node.children_by_field_name("parameter"))
        return_type = self.format(node.child_by_field_name("return_type"), depth)
        return f"({parameters}) => {return_type}"


def _path_before(call: Node) -> str:
    """Return the dotted path in front of a call, as formatted: ``http.`` for ``http.get(...)``."""
    wrapper = call.parent  # function_call
    chain = wrapper.parent if wrapper is not None else None
    if chain is None or chain.type != "child_annotation":
        return ""
    before = []
    for part in chain.named_children:
        if part == wrapper:
            break
        before.append("".join(_text(part).split()) + ".")
    return "".join(before)


def _hugs(argument: Node) -> bool:
    """Whether a last argument can open on the call's line and close with it: a record, list or block lambda."""
    while argument.type in {"expression", "literal"} and argument.named_children:
        argument = argument.named_children[0]
    if argument.type in {"record", "list"}:
        return True
    return argument.type == "local_function_definition" and argument.named_children[-1].type == "block_expression"


__all__ = ["WIDTH", "FormatError", "format_source", "parse"]
