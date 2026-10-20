"""The pipe operator: ``x |> f(a)`` is ``f(x, a)``, the value going in as the first argument.

The checks and the lowering read a pipe as the call it is. ``desugared`` gives
that call, made of the pipe's own nodes (its value, the function's name, the
arguments) held by a few nodes of its own, which act like tree-sitter's: the
passes reach it through their ``_children`` helpers, and treat it as any call.
Messages about it point at the pipe in the source.

The right side of ``|>``:

- a call: ``x |> f(a)`` is ``f(x, a)``, ``x |> m.f(a)`` is ``m.f(x, a)``;
- a function's name: ``x |> f`` is ``f(x)``, ``x |> m.f`` is ``m.f(x)``;
- a method of the value: ``xs |> .sorted()`` is ``xs.sorted()``;
- a lambda: ``x |> [] (y: i64) => i64 y + 1`` calls it with ``x`` (see ``lambda_call``).
"""

from typing import Any

from tree_sitter import Node

PIPE = "pipe_expression"
# A pipe into a lambda, ``x |> [] (y: i64) => i64 y + 1``: fields ``function`` (the
# lambda) and ``value`` (``x``); the lowering calls the lambda with the value.
LAMBDA_CALL = "pipe_lambda_call"

# The kinds of node a pipe's call is made of, numbered so that each has its own id.
_ROLES = {"function_call": 1, "user_function_call": 2, "builtin_call": 3, "child_annotation": 4, "identifier": 5}
_ROLES[LAMBDA_CALL] = 6


class Synthetic:
    """A node of the call a pipe is, standing in for tree-sitter's ``Node`` where the passes read one.

    It spans what it stands for in the source (``origin``), and has an id of its own, the
    same each time the pipe is read, so the passes agree on which call is which.
    """

    __slots__ = ("_fields", "id", "named_children", "origin", "parent", "pipe", "type")

    def __init__(
        self,
        kind: str,
        children: list[Any],
        origin: Node,
        parent: Any,  # noqa: ANN401 - a Node or a Synthetic
        fields: dict[str, Any] | None = None,
    ) -> None:
        self.type = kind
        self.named_children = children
        self.origin = origin
        self.parent = parent
        self._fields = fields or {}
        self.id = -(origin.id * 8 + _ROLES[kind])
        self.pipe: Node = origin  # the pipe it comes from (see ``desugared``)

    children = property(lambda self: self.named_children)
    text = property(lambda self: self.origin.text)
    start_point = property(lambda self: self.origin.start_point)
    end_point = property(lambda self: self.origin.end_point)
    start_byte = property(lambda self: self.origin.start_byte)
    end_byte = property(lambda self: self.origin.end_byte)
    range = property(lambda self: self.origin.range)
    has_error = property(lambda self: self.origin.has_error)
    child_count = property(lambda self: len(self.named_children))
    is_missing = False
    is_error = False
    is_named = True

    def child_by_field_name(self, name: str) -> Any:  # noqa: ANN401 - a Node or a Synthetic
        """Return the child in field ``name``: a call's ``function``, say."""
        return self._fields.get(name)

    def children_by_field_name(self, name: str) -> list[Any]:
        """Return the children in field ``name``."""
        found = self._fields.get(name)
        return [] if found is None else [found]

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` stands for the same part of the same pipe."""
        return isinstance(other, Synthetic) and other.id == self.id

    def __hash__(self) -> int:
        """Hash by id, as ``__eq__`` compares."""
        return hash(self.id)

    def __repr__(self) -> str:
        """Show what it stands for: ``<Synthetic function_call for 'x |> f'>``."""
        return f"<Synthetic {self.type} for {(self.origin.text or b'').decode()!r}>"


def _named(node: Any) -> list[Any]:  # noqa: ANN401
    return [child for child in node.named_children if child.type != "comment"]


def _unwrap(node: Any) -> Any:  # noqa: ANN401
    while node.type in {"expression", "parenthesized_expression"}:
        node = desugared(_named(node)[0]) if _named(node) else node
        if not _named(node):
            break
    return node


def desugared(node: Any) -> Any:  # noqa: ANN401 - a Node, or the Synthetic call a pipe is
    """Return the call a ``pipe_expression`` is, or ``node`` itself (also when the pipe's right side is none)."""
    if node.type != PIPE:
        return node
    call = _desugared(node)
    _mark(call, node)
    return call


def _mark(node: Any, pipe: Node) -> None:  # noqa: ANN401
    """Note in each node of the call a pipe is that it comes from ``pipe``."""
    if isinstance(node, Synthetic):
        node.pipe = pipe
        for child in node.named_children:
            _mark(child, pipe)


def _desugared(node: Node) -> Any:  # noqa: ANN401
    value = node.child_by_field_name("value")
    into = node.child_by_field_name("into")
    if value is None or into is None:
        return node
    parent = node.parent  # the `expression` holding the pipe
    if into.type == "pipe_method":
        return _method(node, value, _named(into)[0], parent)
    target = _unwrap(into)
    match target.type:
        case "local_function_definition":
            found = Synthetic(LAMBDA_CALL, [target, value], node, parent, {"function": target, "value": value})
        case "function_call":
            found = _call(target, value, parent)
        case "identifier":
            found = _named_call(node, target, value, parent)
        case "child_annotation":
            found = _into_member(node, target, value, parent)
        case _:
            found = None
    return found if found is not None else node  # none: the lowering says what the right side can be


def _call(function_call: Any, value: Node, parent: Any) -> Synthetic | None:  # noqa: ANN401
    """``f(a)``, with ``value`` first: ``f(value, a)``."""
    inner = _named(function_call)[0]
    function = inner.child_by_field_name("function")
    if function is None:
        return None
    outer = Synthetic("function_call", [], function_call, parent)
    arguments = [child for child in _named(inner) if child.type in {"expression", "named_argument"}]
    call = Synthetic(inner.type, [function, value, *arguments], inner, outer, {"function": function})
    outer.named_children = [call]
    return outer


def _named_call(origin: Node, function: Any, value: Node, parent: Any) -> Synthetic:  # noqa: ANN401
    """``f``: ``f(value)``."""
    outer = Synthetic("function_call", [], origin, parent)
    outer.named_children = [Synthetic("user_function_call", [function, value], origin, outer, {"function": function})]
    return outer


def _into_member(origin: Node, member: Node, value: Node, parent: Any) -> Any:  # noqa: ANN401
    """``m.f(a)`` or ``m.f``: ``m.f(value, a)``, ``m.f(value)``."""
    parts = _named(member)
    holder = Synthetic("child_annotation", [], origin, parent)
    last = parts[-1]
    if last.type == "function_call":
        call = _call(last, value, holder)
        if call is None:
            return origin
    elif last.type == "simple_identifier":
        name = Synthetic("identifier", [last], last, None)
        call = _named_call(last, name, value, holder)
        name.parent = call.named_children[0]
    else:
        return origin
    holder.named_children = [*parts[:-1], call]
    return holder


def _method(origin: Node, value: Node, function_call: Node, parent: Any) -> Any:  # noqa: ANN401
    """``.m(a)``: ``value.m(a)``, the value being what the method is called on."""
    holder = Synthetic("child_annotation", [], origin, parent)
    receiver = _unwrap(value)
    if receiver.type == "identifier":
        owner = [_named(receiver)[0]]  # a child_annotation starts with a simple_identifier
    elif receiver.type == "child_annotation":
        owner = _named(receiver)  # `a.b |> .m()` is `a.b.m()`
    elif receiver.type in {"function_call", "get_expression"}:
        owner = [receiver]
    else:
        owner = [value]  # any other value: `#[3, 1, 2] |> .sorted()`
    call = _call(function_call, value, holder)
    if call is None:
        return origin
    inner = call.named_children[0]
    inner.named_children = [child for child in inner.named_children if child is not value]  # the receiver
    holder.named_children = [*owner, call]
    return holder


def is_piped(node: Node) -> bool:
    """Whether ``node`` (or the expression holding it) is the value a pipe passes on: ``x`` in ``x |> f``."""
    current = node
    while current.parent is not None and current.parent.type in {"expression", "parenthesized_expression"}:
        current = current.parent
    parent = current.parent
    if parent is None or parent.type != PIPE:
        return False
    value = parent.child_by_field_name("value")
    return value is not None and value.id == current.id


def takes_piped(call: Node) -> bool:
    """Whether a ``user_function_call`` is a pipe's right side, its first argument the value piped in.

    ``add(4)`` in ``3 |> add(4)`` and ``m.add(4)`` in ``3 |> m.add(4)``, not a method: ``xs |> .take(2)``.
    """
    segment = call.parent  # the function_call
    holder = segment.parent if segment is not None else None
    if holder is not None and holder.type == "child_annotation" and _named(holder)[-1].id == segment.id:
        holder = holder.parent
    while holder is not None and holder.type == "parenthesized_expression":
        holder = holder.parent
    if holder is None or holder.type != "expression" or holder.parent is None or holder.parent.type != PIPE:
        return False
    into = holder.parent.child_by_field_name("into")
    return into is not None and into.id == holder.id


__all__ = ["LAMBDA_CALL", "PIPE", "Synthetic", "desugared", "is_piped", "takes_piped"]
