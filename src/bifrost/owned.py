"""Owned values: lists, closures, and the records and objects that hold them.

A list (``T[]``) owns its items' memory, so it has one owner, which frees it
(see ``bifrost.ownership``); a closure (a function value) owns what it
captured; a record or object with either among its fields (at any depth) owns
that, so it is owned too. Everything else (numbers, strings, records of those)
is copied freely.

For each owned type this module generates, once per program, a function that
frees a value of it and everything it owns (``drop``), and one that makes a
deep copy of it (``clone``: what ``#[...xs]`` does when ``xs`` stays in use).
"""

import linecache
import re
from types import CodeType, FunctionType
from typing import Any

from mlir_python.lang import Fn, Program, Ptr, i32, ptr, stack, struct, u64
from mlir_python.lang._types import FnType, ScalarType, StructType, function_type, scalar_type

from bifrost.results import is_result
from bifrost.results import parts as result_parts
from bifrost.std import list_runtime
from bifrost.unions import TAG, slot, union

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


# A function value is a closure: the function, and what it captured, on the heap
# (``ptr(0)`` when it captured nothing). The function takes what it captured
# first: ``fn(env, x)``. What it captured starts with how to free and copy it,
# so a closure is freed or copied without knowing what it captured.
_CLOSURE = "closure_"
_CLOSURES: dict[FnType, type] = {}
_DISPLAYS: dict[str, str] = {}
CAPTURED = struct(type("captured", (), {"__annotations__": {"drop": Fn[[ptr], None], "clone": Fn[[ptr], ptr]}}))


def closure_of(signature: FnType) -> type:
    """Return the type of function values of ``signature`` (written ``(x: i64) => i64``): a closure.

    One type per signature, so that function values of one signature mix.
    """
    if signature not in _CLOSURES:
        name = f"{_CLOSURE}{len(_CLOSURES)}"
        function = function_type([ptr, *signature.params], signature.result)
        _CLOSURES[signature] = struct(type(name, (), {"__annotations__": {"fn": function, "env": ptr}}))
        parameters = ", ".join(parameter.name for parameter in signature.params)
        result = signature.result.name if signature.result is not None else "null"
        _DISPLAYS[name] = f"({parameters}) => {result}"
    return _CLOSURES[signature]


def is_closure(kind: object) -> bool:
    """Whether ``kind`` is a function value's type (see ``closure_of``)."""
    found = scalar_type(kind)
    return isinstance(found, StructType) and found.name.startswith(_CLOSURE)


def signature_of(kind: object) -> FnType:
    """Return the signature of a closure type: its function's, without what it captured."""
    found = scalar_type(kind)
    assert isinstance(found, StructType)
    function = dict(found.fields)["fn"]
    assert isinstance(function, FnType)
    return function_type(list(function.params[1:]), function.result)


def closure_names(text: str) -> str:
    """Write closure types in ``text`` as their signatures: ``closure_0`` is ``(i64) => i64``."""
    return re.sub(rf"\b{_CLOSURE}\d+\b", lambda match: _DISPLAYS.get(match.group(), match.group()), text)


# C structs that are handles C frees (`drop:` in config.yaml): id of the struct -> (struct, drop, clone).
_HANDLES: dict[int, tuple[type, Any, Any]] = {}


def own_handle(cls: type, drop: Any, clone: Any) -> None:  # noqa: ANN401 - compiled C functions
    """Make the C struct ``cls`` a handle Bifrost owns: freed with ``drop``, copied with ``clone`` (or never)."""
    _HANDLES[id(cls)] = (cls, drop, clone)


def handle_of(kind: object) -> tuple[Any, Any] | None:
    """For a C struct Bifrost owns, its drop and clone functions (clone ``None`` if it cannot be copied)."""
    found = scalar_type(kind)
    entry = _HANDLES.get(id(found.python)) if isinstance(found, StructType) else None
    return (entry[1], entry[2]) if entry is not None and entry[0] is found.python else None


def owns(kind: object) -> bool:
    """Whether a value of ``kind`` owns memory: a list, a closure, a C handle, or a record or object holding one."""
    kind = scalar_type(kind)
    if is_list(kind) or kind == OWNED_STRING or is_closure(kind) or handle_of(kind) is not None:
        return True
    return isinstance(kind, StructType) and any(owns(field) for _, field in kind.fields)


class CannotCopyError(ValueError):
    """A value holding a C handle that has no clone function."""


class Drops:
    """The drop and clone functions of a program's owned types, generated as they are needed."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self._functions: dict[tuple[str, ScalarType], Any] = {}
        self._environments: dict[tuple[tuple[str, ScalarType], ...], tuple[type, Any, Any]] = {}
        self._count = 0
        self._memset: Any = None  # C's memset, declared when a zero value is first made

    def drop(self, kind: ScalarType) -> Any:  # noqa: ANN401 - a compiled function
        """Return the function that frees a ``kind`` value and all it owns."""
        return self._get("drop", kind)

    def clone(self, kind: ScalarType) -> Any:  # noqa: ANN401 - a compiled function
        """Return the function that makes a deep copy of a ``kind`` value."""
        return self._get("clone", kind)

    def zero(self, kind: ScalarType) -> Any:  # noqa: ANN401 - a compiled function
        """Return the function that makes a ``kind`` value of all zeros: what a ``Result`` holds on its unset side."""
        return self._get("zero", kind)

    def _get(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        if kind == OWNED_STRING and action != "zero":
            return list_runtime.bifrost_string_free if action == "drop" else list_runtime.bifrost_string_copy
        key = (action, kind)
        if key not in self._functions:
            if action == "zero":
                make = self._zero
            elif handle_of(kind) is not None:
                make = self._handle
            elif is_result(kind):
                make = self._result
            elif union(kind) is not None:
                make = self._union
            else:
                make = self._list if is_list(kind) else self._closure if is_closure(kind) else self._struct
            self._functions[key] = make(action, kind)
        return self._functions[key]

    def _handle(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Free or copy a C handle: with the C functions config.yaml names (``drop:``, ``clone:``)."""
        found = handle_of(kind)
        assert found is not None
        function = found[0] if action == "drop" else found[1]
        if function is None:
            raise CannotCopyError(f"a {kind.name} cannot be copied (its declaration in config.yaml has no clone)")
        return function

    def _zero(self, _action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Zero a slot on the stack, and read it: zero numbers, null pointers, an empty closure."""
        if self._memset is None:

            def memset(pointer: ptr, byte: i32, size: u64) -> ptr: ...

            self._memset = self.program.extern(memset, name="memset")
        names: dict[str, object] = {"Value": kind.python if isinstance(kind, StructType) else kind}
        names |= {"stack": stack, "memset": self._memset, "u64": u64}
        lines = [
            "def {name}():",
            "    slot = stack(Value)",
            f"    memset(slot, 0, u64({kind.size}))",
            "    return slot[0]",
        ]
        return self.function(lines, names, {"return": kind})

    def _result(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Free or copy a ``Result``: only the side it holds (the other is zero)."""
        assert isinstance(kind, StructType)
        value, error = result_parts(kind)
        names: dict[str, object] = {"Value": kind.python}
        sides = [("value", value), ("error", error)]
        for field, held in sides:
            if held is not None and owns(held):
                names[f"{action}_{field}"] = self._get(action, held)
            if held is not None and action == "clone":
                names[f"zero_{field}"] = self.zero(held)
        if action == "drop":
            lines = ["def {name}(value):", "    if value.ok:"]
            lines.append("        drop_value(value.value)" if "drop_value" in names else "        pass")
            lines.append("    else:")
            lines.append("        drop_error(value.error)" if "drop_error" in names else "        pass")
            return self.function(lines, names, {"value": kind, "return": None})

        def side(field: str) -> str:
            return f"{field}=clone_{field}(value.{field})" if f"clone_{field}" in names else f"{field}=value.{field}"

        has_value = value is not None
        when_ok = ["ok=True", *([side("value")] if has_value else []), "error=zero_error()"]
        when_err = ["ok=False", *(["value=zero_value()"] if has_value else []), side("error")]
        lines = [
            "def {name}(value):",
            "    if value.ok:",
            f"        return Value({', '.join(when_ok)})",
            f"    return Value({', '.join(when_err)})",
        ]
        return self.function(lines, names, {"value": kind, "return": kind})

    def _union(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Free or copy a union: only the object its tag names (the others are zero)."""
        found = union(kind)
        assert found is not None
        assert isinstance(kind, StructType)
        names: dict[str, object] = {"Value": kind.python}
        for index, (_, held) in enumerate(found.variants):
            if owns(held):
                names[f"{action}_{index}"] = self._get(action, held)
            if action == "clone":
                names[f"zero_{index}"] = self.zero(held)
        count = len(found.variants)
        if action == "drop":
            return self.function(self._union_drop(count, names), names, {"value": kind, "return": None})
        return self.function(self._union_clone(count, names), names, {"value": kind, "return": kind})

    @staticmethod
    def _union_drop(count: int, names: dict[str, object]) -> list[str]:
        """Write the drop of a union of ``count`` objects: the one its tag names, if it owns anything."""
        lines = ["def {name}(value):"]
        for index in range(count):
            if f"drop_{index}" in names:
                lines += [f"    if value.{TAG} == {index}:", f"        drop_{index}(value.{slot(index)})"]
        return lines if len(lines) > 1 else [*lines, "    pass"]

    @staticmethod
    def _union_clone(count: int, names: dict[str, object]) -> list[str]:
        """Write the copy of a union of ``count`` objects: a copy of the one its tag names, the others zero."""
        lines = ["def {name}(value):"]
        for index in range(count):
            fields = [f"{TAG}=value.{TAG}"]
            for other in range(count):
                if other != index:
                    fields.append(f"{slot(other)}=zero_{other}()")
                elif f"clone_{index}" in names:
                    fields.append(f"{slot(index)}=clone_{index}(value.{slot(index)})")
                else:
                    fields.append(f"{slot(index)}=value.{slot(index)}")
            built = f"return Value({', '.join(fields)})"
            # The last object is what no tag before it named.
            lines += [f"    {built}"] if index == count - 1 else [f"    if value.{TAG} == {index}:", f"        {built}"]
        return lines

    def _closure(self, action: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Free or copy a closure: what it captured knows how (see ``environment``)."""
        names: dict[str, object] = {"Value": scalar_type(kind).python, "Captured": Ptr[CAPTURED], "ptr": ptr}
        if action == "drop":
            lines = [
                "def {name}(value):",
                "    if value.env != ptr(0):",
                "        header = Captured(value.env)",
                "        header[0].drop(value.env)",
            ]
            return self.function(lines, names, {"value": kind, "return": None})
        lines = [
            "def {name}(value):",
            "    env = value.env",
            "    if env != ptr(0):",
            "        header = Captured(env)",
            "        env = header[0].clone(env)",
            "    return Value(fn=value.fn, env=env)",
        ]
        return self.function(lines, names, {"value": kind, "return": kind})

    def environment(self, captures: list[tuple[str, ScalarType]]) -> tuple[type, Any, Any]:
        """Return the type of what a closure captures, and the functions that free and copy it.

        It holds ``captures``, after how to free and copy it (see ``CAPTURED``).
        """
        key = tuple(captures)
        if key not in self._environments:
            self._environments[key] = self._environment(captures)
        return self._environments[key]

    def _environment(self, captures: list[tuple[str, ScalarType]]) -> tuple[type, Any, Any]:
        fields: dict[str, object] = {"drop": Fn[[ptr], None], "clone": Fn[[ptr], ptr]}
        fields |= {name: kind.python if isinstance(kind, StructType) else kind for name, kind in captures}
        self._count += 1
        env = struct(type(f"captures_{self._count}", (), {"__annotations__": fields}))
        size = scalar_type(env).size
        owned = [(name, kind) for name, kind in captures if owns(kind)]
        names: dict[str, object] = {
            "Env": env,
            "At": Ptr[env],
            "new": list_runtime.bifrost_env_new,
            "free": list_runtime.bifrost_env_free,
        }
        for index, (_, kind) in enumerate(owned):
            names[f"drop_{index}"] = self.drop(kind)
            names[f"clone_{index}"] = self.clone(kind)
        drop = ["def {name}(env):", "    value = At(env)[0]"]
        drop += [f"    drop_{index}(value.{field})" for index, (field, _) in enumerate(owned)]
        drop.append("    free(env)")
        cloned = {field: index for index, (field, _) in enumerate(owned)}
        arguments = ", ".join(
            f"{field}=clone_{cloned[field]}(value.{field})" if field in cloned else f"{field}=value.{field}"
            for field in fields
        )
        clone = [
            "def {name}(env):",
            "    value = At(env)[0]",
            f"    copied = new({size})",
            f"    At(copied)[0] = Env({arguments})",
            "    return copied",
        ]
        dropping = self.function(drop, names, {"env": ptr, "return": None})
        cloning = self.function(clone, names, {"env": ptr, "return": ptr})
        return env, dropping, cloning

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
            names["is_null"] = list_runtime.bifrost_list_is_null
            lines = ["def {name}(items):", "    if is_null(items):", "        return"]
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
            return self.function(lines, names, {"items": kind, "return": None})
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
        return self.function(lines, names, {"items": kind, "return": kind})

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
            return self.function(lines, names, {"value": kind, "return": None})
        cloned = {field: index for index, (field, _) in enumerate(owned)}
        arguments = ", ".join(
            f"{field}=field_{cloned[field]}(value.{field})" if field in cloned else f"{field}=value.{field}"
            for field, _ in kind.fields
        )
        lines = ["def {name}(value):", f"    return Value({arguments})"]
        return self.function(lines, names, {"value": kind, "return": kind})

    def function(self, lines: list[str], names: dict[str, object], annotations: dict[str, object]) -> Any:  # noqa: ANN401
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


__all__ = [
    "CAPTURED",
    "OWNED_STRING",
    "CannotCopyError",
    "Drops",
    "closure_names",
    "closure_of",
    "handle_of",
    "is_closure",
    "is_list",
    "list_of",
    "own_handle",
    "owns",
    "signature_of",
]
