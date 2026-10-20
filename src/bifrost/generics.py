"""Generics: functions and objects written once for any types, compiled once for each.

``let first[T] = [] (xs: T[]) => T xs[0]`` is a generic function: ``T`` stands
for a type. Each call compiles it for the types its arguments have (``first(#[1, 2])``
is ``first[i64]``), once per set of types, as Rust and C++ do; mlir-python only
ever sees the compiled functions, whose types are all known.

``let Box[T] = struct { let value: T }`` is a generic object: ``Box[i64]`` is
an object type of its own, made when it is written or when ``Box(value: 1)``
builds one (``T`` is then the type of what fills it).

A generic is bound where its name would be (a file's globals, or a module) as a
``Generic``: the definition, and what it has been compiled for. The file that
defines it compiles it, with the types its caller found, so names in its body
(its file's objects, imports and functions) mean what they mean there.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from tree_sitter import Node

FUNCTION = "function"
OBJECT = "object"


@dataclass
class Instance:
    """A generic compiled for some types: ``first[i64]``."""

    name: str  # how messages write it: `first[i64]`, `rules.pick[f64]`
    symbol: str  # its compiled name (a function's), or its object type's
    arguments: tuple[object, ...]  # the types its parameters stand for, as `SourceUnit._type` gives them
    result: Any = None  # a function's result type (a ScalarType), known before its body is compiled
    compiled: Any = None  # the compiled Function, or the object type; None while its body is compiled


@dataclass(eq=False)
class Generic:
    """A generic function or object: its definition, and what it has been compiled for."""

    kind: str  # FUNCTION or OBJECT
    unit: Any  # the SourceUnit defining it, which compiles it
    name: str  # its qualified Bifrost name: `first`, `rules.first`, `Box`
    symbol: str  # the base of its instances' compiled names
    node: Node  # the function_definition or struct_assignment
    parameters: list[str]  # `T`, `U`: the names of its type parameters
    module: str | None  # the module defining it, if any
    instances: dict[tuple[object, ...], Instance] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Write it as it is written: ``first[T]``."""
        return f"{self.name}[{', '.join(self.parameters)}]"


# Object types made from generic ones are named `Box__g0`; messages write `Box[i64]`.
_DISPLAYS: dict[str, str] = {}
# Each such object type -> the generic it was made from, and the types it was made for.
_MADE_FROM: dict[int, tuple[Generic, tuple[object, ...], type]] = {}
_MANGLED = re.compile(r"\b\w+__g\d+\b")


def made(cls: type, generic: Generic, arguments: tuple[object, ...], display: str) -> None:
    """Record that the object type ``cls`` is ``generic`` made for ``arguments``, written ``display``."""
    _DISPLAYS[cls.__name__] = display
    _MADE_FROM[id(cls)] = (generic, arguments, cls)


def made_from(cls: object) -> tuple[Generic, tuple[object, ...]] | None:
    """For an object type made from a generic one (``Box[i64]``), that generic and its type arguments."""
    found = _MADE_FROM.get(id(cls))
    return (found[0], found[1]) if found is not None and found[2] is cls else None


def display(text: str) -> str:
    """Write the object types made from generic ones in ``text`` as Bifrost does: ``Box__g0`` is ``Box[i64]``."""
    return _MANGLED.sub(lambda match: _DISPLAYS.get(match.group(), match.group()), text)


__all__ = ["FUNCTION", "OBJECT", "Generic", "Instance", "display", "made", "made_from"]
