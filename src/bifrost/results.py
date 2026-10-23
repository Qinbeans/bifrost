"""``Result[T, E]``: a value, or the error that kept it from being made.

A function that can fail returns one: ``=> Result[User, str]``. It is built
with the prelude's ``ok(value)`` and ``err(error)`` (their types come from
where the result goes: the function's result, a ``let``'s written type, a
parameter's), read with ``match r { ok(user): ..., err(e): ... }``, and passed
on with ``?``: ``find(id)?`` is the value, or else returns the error from the
function at once (whose result must be a ``Result`` with the same error type).

A ``Result`` is an object of three fields, ``ok`` (whether it holds a value),
``value`` and ``error``, of which only one is set; the other is zero. One type
per pair of types, shared by every file, as records are. ``Result[null, E]``
has no ``value``: it says only whether something worked, and why not.

It owns what it holds when either side owns memory (a list, an owned string, a
closure): it frees only the side it holds (see ``owned.Drops``).
"""

import re

from mlir_python.lang import struct
from mlir_python.lang._types import ScalarType, StructType, scalar_type
from tree_sitter import Node

NAME = "Result"
# The prelude's constructors: `ok(value)` and `err(error)` (names a program may give other values).
OK, ERR = "ok", "err"

_PREFIX = "Result_"
_RESULTS: dict[tuple[int, int], tuple[ScalarType | None, ScalarType, type]] = {}
_DISPLAYS: dict[str, str] = {}


def _shown(kind: ScalarType | None) -> str:
    if kind is None:
        return "null"
    return "str" if kind.name == "cstr" else kind.name


def result_of(value: object, error: object) -> type:
    """Return the type ``Result[value, error]`` (``value`` may be ``None``/``type(None)``: nothing on success).

    One type per pair, kept alive here so that types compare by identity.
    """
    held = None if value in (None, type(None)) else scalar_type(value)
    failed = scalar_type(error)
    assert failed is not None
    key = (id(held), id(failed))
    if key not in _RESULTS:
        fields: dict[str, object] = {"ok": bool}
        if held is not None:
            fields["value"] = held.python if isinstance(held, StructType) else held
        fields["error"] = failed.python if isinstance(failed, StructType) else failed
        name = f"{_PREFIX}{len(_RESULTS)}"
        cls = struct(type(name, (), {"__annotations__": fields}))
        _RESULTS[key] = (held, failed, cls)
        _DISPLAYS[name] = f"{NAME}[{_shown(held)}, {_shown(failed)}]"
    return _RESULTS[key][2]


def is_result(kind: object) -> bool:
    """Whether ``kind`` is a ``Result[T, E]`` type."""
    found = scalar_type(kind)
    return isinstance(found, StructType) and found.name in _DISPLAYS


def parts(kind: object) -> tuple[ScalarType | None, ScalarType]:
    """Return what a ``Result`` type holds: its value's type (``None`` for ``Result[null, E]``) and its error's."""
    found = scalar_type(kind)
    assert isinstance(found, StructType)
    fields = dict(found.fields)
    value = fields.get("value")
    return (scalar_type(value) if value is not None else None), scalar_type(fields["error"])


def pattern(condition: Node) -> tuple[str, Node | None] | None:
    """For a match arm's condition, ``ok(v)`` or ``err(e)``: its side, and the identifier it binds.

    ``ok(_)`` (or ``ok()``, for a ``Result[null, E]``) binds nothing. ``None`` for other conditions.
    """
    node = condition
    while node.type in {"expression", "condition", "parenthesized_expression"} and node.named_children:
        node = node.named_children[0]
    call = node.named_children[0] if node.type == "function_call" and node.named_children else None
    if call is None or call.type != "user_function_call":
        return None
    function = call.child_by_field_name("function")
    side = (function.text or b"").decode() if function is not None else ""
    arguments = [child for child in call.named_children if child.type in {"expression", "named_argument"}]
    if side not in {OK, ERR} or len(arguments) > 1 or any(a.type == "named_argument" for a in arguments):
        return None
    if not arguments:
        return side, None
    inner = arguments[0]
    while inner.type in {"expression", "parenthesized_expression"} and inner.named_children:
        inner = inner.named_children[0]
    if inner.type == "default_var":
        return side, None
    return (side, inner) if inner.type == "identifier" else None


def exhaustive(conditions: list[Node]) -> bool:
    """Whether a match's arm conditions cover every Result: an ``ok(...)`` arm and an ``err(...)`` arm."""
    sides = {found[0] for found in map(pattern, conditions) if found is not None}
    return sides == {OK, ERR}


def display(text: str) -> str:
    """Write ``Result`` types in ``text`` as Bifrost does: ``Result_0`` is ``Result[i64, str]``."""
    previous = None
    while previous != text:  # one holding another: `Result[Result_1, str]`
        previous = text
        text = _NAMED.sub(lambda match: _DISPLAYS.get(match.group(), match.group()), text)
    return text


_NAMED = re.compile(rf"\b{_PREFIX}\d+\b")


__all__ = ["ERR", "NAME", "OK", "display", "exhaustive", "is_result", "parts", "pattern", "result_of"]
