"""Literal fields: ``let type: "employee"`` is a field that only ever holds that text.

The object stores it as a ``str`` (the literal's own text, never freed). The
compiler fills it in when the object is built, so it is left out of the call
(``Employee(name: "ada")``), and nothing may change it. JSON writes it like any
field; decoding checks the text holds it. A union tells its objects apart by
one (``let Actor = Employee | Service on "type"``: see ``unions``).
"""

from mlir_python.lang._types import scalar_type
from mlir_python.lang.types import StructType

# By the object's type: its literal fields, and the text each holds; kept alive so that types compare by identity.
_LITERALS: dict[int, tuple[type, dict[str, str]]] = {}


def fix(cls: type, fields: dict[str, str]) -> None:
    """Record that the object type ``cls`` has the literal ``fields`` (each field's name, and its text)."""
    if fields:
        _LITERALS[id(cls)] = (cls, dict(fields))


def literals(kind: object) -> dict[str, str]:
    """Return the literal fields of the object type ``kind`` (name to text), in the order it declares them."""
    found = scalar_type(kind)
    if not isinstance(found, StructType):
        return {}
    entry = _LITERALS.get(id(found.python))
    return dict(entry[1]) if entry is not None and entry[0] is found.python else {}


def given(kind: object) -> list[str]:
    """Return the fields of the object type ``kind`` that building one takes: all but its literal ones, in order."""
    found = scalar_type(kind)
    fixed = literals(found)
    return [name for name, _ in found.fields if name not in fixed] if isinstance(found, StructType) else []


__all__ = ["fix", "given", "literals"]
