"""Owned values: lists, and the records and objects that hold them.

A list (``T[]``) owns its items' memory, so it has one owner, which frees it
(see ``bifrost.ownership``); a record or object with a list among its fields
(at any depth) owns that list, so it is owned too. Everything else (numbers,
strings, records of those) is copied freely.

For each owned type this module generates, once per program, a function that
frees a value of it and everything it owns (``drop``), and one that makes a
deep copy of it (``clone``: what ``#[...xs]`` does when ``xs`` stays in use).
"""

import linecache
from types import CodeType, FunctionType
from typing import Any

from mlir_python.lang import Program
from mlir_python.lang._types import ScalarType, StructType, scalar_type

from bifrost.std import list_runtime

_LIST = "List["


# Keyed by the element type's identity, not its value: an object type re-made from edited
# source (the language server analyses again) compares equal to the old one, but holds
# another class, with other methods. Each entry keeps its element alive, so ids stay unique.
_LISTS: dict[int, tuple[ScalarType, ScalarType]] = {}


def list_of(element: object) -> ScalarType:
    """Return the type of a list of ``element`` values, written ``T[]``: a pointer to its first item.

    One object per element type, so that it can be named by identity.
    """
    kind = scalar_type(element)
    assert kind is not None
    if id(kind) not in _LISTS:
        _LISTS[id(kind)] = (kind, ScalarType(f"{_LIST}{kind.name}]", "ptr", 64, kind))
    return _LISTS[id(kind)][1]


# A string its holder owns and frees (from ``fmt.format``, say), where a list, record or
# object holds one: written ``mem.Unique[str]``. Other strings (literals, ones lent)
# are ``str``: the same pointer to text, which nothing here frees.
OWNED_STRING = ScalarType("mem.Unique[str]", "cstr", 64)


def is_list(kind: object) -> bool:
    return isinstance(kind, ScalarType) and kind.kind == "ptr" and kind.name.startswith(_LIST)


def owns(kind: object) -> bool:
    """Whether a value of ``kind`` owns memory: a list, or a record or object holding one."""
    kind = scalar_type(kind)
    if is_list(kind) or kind == OWNED_STRING:
        return True
    return isinstance(kind, StructType) and any(owns(field) for _, field in kind.fields)


class Drops:
    """The drop and clone functions of a program's owned types, generated as they are needed."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self._functions: dict[tuple[str, ScalarType], Any] = {}
        self._count = 0

    def drop(self, kind: ScalarType) -> Any:  # noqa: ANN401 - a compiled function
        """Return the function that frees a ``kind`` value and all it owns."""
        return self._get("drop", kind)

    def clone(self, kind: ScalarType) -> Any:  # noqa: ANN401 - a compiled function
        """Return the function that makes a deep copy of a ``kind`` value."""
        return self._get("clone", kind)

    def _get(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        if kind == OWNED_STRING:
            return list_runtime.bifrost_string_free if action == "drop" else list_runtime.bifrost_string_copy
        key = (action, kind)
        if key not in self._functions:
            make = self._list if is_list(kind) else self._struct
            self._functions[key] = make(action, kind)
        return self._functions[key]

    def _list(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        element = kind.element
        assert element is not None
        deep = owns(element)
        names: dict[str, object] = {
            "List": kind,
            "length": list_runtime.bifrost_list_length,
            "new": list_runtime.bifrost_list_new,
            "copy": list_runtime.bifrost_list_copy,
            "free": list_runtime.bifrost_list_free,
        }
        if action == "drop":
            lines = ["def {name}(items):"]
            if deep:
                names["inner"] = self.drop(element)
                lines += [
                    "    count = length(items)",
                    "    index = 0",
                    "    while index < count:",
                    "        inner(items[index])",
                    "        index = index + 1",
                ]
            lines.append("    free(items)")
            return self._function(lines, names, {"items": kind, "return": None})
        lines = [
            "def {name}(items):",
            "    count = length(items)",
            f"    copied = List(new(count, {element.size}))",
        ]
        if deep:
            names["inner"] = self.clone(element)
            lines += [
                "    index = 0",
                "    while index < count:",
                "        copied[index] = inner(items[index])",
                "        index = index + 1",
            ]
        else:
            lines.append(f"    copy(copied, items, count, {element.size})")
        lines.append("    return copied")
        return self._function(lines, names, {"items": kind, "return": kind})

    def _struct(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        assert isinstance(kind, StructType)
        assert kind.python is not None
        names: dict[str, object] = {"Value": kind.python}
        owned = [(field, self._get(action, scalar_type(held))) for field, held in kind.fields if owns(held)]
        for index, (_, function) in enumerate(owned):
            names[f"field_{index}"] = function
        if action == "drop":
            lines = ["def {name}(value):"]
            lines += [f"    field_{index}(value.{field})" for index, (field, _) in enumerate(owned)]
            return self._function(lines, names, {"value": kind, "return": None})
        cloned = {field: index for index, (field, _) in enumerate(owned)}
        arguments = ", ".join(
            f"{field}=field_{cloned[field]}(value.{field})" if field in cloned else f"{field}=value.{field}"
            for field, _ in kind.fields
        )
        lines = ["def {name}(value):", f"    return Value({arguments})"]
        return self._function(lines, names, {"value": kind, "return": kind})

    def _function(self, lines: list[str], names: dict[str, object], annotations: dict[str, object]) -> Any:  # noqa: ANN401
        """Compile generated source (``mlir_python.lang`` reads a function's source) into the program."""
        self._count += 1
        name = f"bifrost_owned_{self._count}"
        source = "\n".join(lines).format(name=name) + "\n"
        filename = f"<bifrost owned {name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        code = compile(source, filename, "exec")
        function_code = next(constant for constant in code.co_consts if isinstance(constant, CodeType))
        python: Any = FunctionType(function_code, names, name)
        python.__annotations__ = annotations
        return self.program.function(python)


__all__ = ["OWNED_STRING", "Drops", "is_list", "list_of", "owns"]
