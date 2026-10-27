"""What ``json.decode`` and ``form.decode`` compile to: one decoder per type, written by the compiler.

``let user: Result[User, mem.Unique[str]] = json.decode(text)`` calls a function
generated for ``User``: it reads the text with the decoding runtime
(``std/decode_runtime.c``), field by field as ``User`` declares them, and gives
``ok(user)``, or ``err(message)`` (``missing field `name```, ``expected an
integer at 12, near '"x"}'``). What it made before it failed is freed.

A decoder makes what it reads: strings are new ones, so a type decoded holds
``mem.Unique[str]``, never ``str`` (which would be lent from somewhere). JSON
decodes numbers, bools, owned strings, lists, records and objects, nested;
unknown fields are skipped. An object's literal field (``let type: "employee"``)
must hold its text, and a union's object is the one whose literal field (the
union's ``on``) holds the text given, wherever it is in the JSON object. A form (``a=1&b=x+y``: a query, a form body, a
path's parameters) decodes a record or object of numbers, bools and owned
strings, or one such value alone (a path with one parameter).
"""

import linecache
from collections.abc import Callable
from types import CodeType, FunctionType
from typing import Any

from mlir_python.lang import Program, cstr, f64, i64, ptr
from mlir_python.lang._types import ScalarType, StructType, scalar_type

from bifrost.literals import literals
from bifrost.owned import OWNED_STRING, Drops, is_closure, is_list, owns
from bifrost.results import is_result, result_of
from bifrost.std import list_runtime
from bifrost.unions import Union, slot, union


class DecodeError(Exception):
    """A type that cannot be decoded; the message says why."""


# The runtime's functions: name -> (parameters, result), as Python annotations.
_RUNTIME: dict[str, tuple[list[object], object]] = {
    "json": ([cstr], ptr),
    "form": ([cstr], ptr),
    "free": ([ptr], None),
    "failed": ([ptr], bool),
    "error": ([ptr], cstr),
    "missing": ([ptr, cstr], None),
    "duplicate": ([ptr, cstr], None),
    "key_is": ([ptr, cstr], bool),
    "json_end": ([ptr], None),
    "json_string": ([ptr], cstr),
    "json_int": ([ptr], i64),
    "json_float": ([ptr], f64),
    "json_bool": ([ptr], bool),
    "json_object": ([ptr], bool),
    "json_key": ([ptr], bool),
    "json_skip": ([ptr], None),
    "json_array": ([ptr], bool),
    "json_more": ([ptr], bool),
    "json_peek": ([ptr, cstr], None),
    "peeked_is": ([ptr, cstr], bool),
    "unknown": ([ptr, cstr], None),
    "json_literal": ([ptr, cstr], None),
    "form_field": ([ptr, cstr], bool),
    "form_single": ([ptr], None),
    "form_int": ([ptr], i64),
    "form_float": ([ptr], f64),
    "form_bool": ([ptr], bool),
    "form_string": ([ptr], cstr),
}

JSON, FORM = "json", "form"


class Decoders:
    """The decoders of a program's types, generated as they are needed (as ``Drops`` generates drops)."""

    def __init__(self, program: Program, drops: Drops) -> None:
        self.program = program
        self.drops = drops
        self.used = False  # whether the program decodes anything: then the runtime is linked in
        self._runtime: dict[str, Any] = {}
        self._readers: dict[tuple[str, ScalarType], Any] = {}
        self._decoders: dict[tuple[str, ScalarType], Any] = {}
        self._count = 0

    def decoder(self, how: str, kind: ScalarType) -> Any:  # noqa: ANN401 - a compiled function
        """Return the function decoding text (``how``: JSON or a form) into a ``Result[kind, mem.Unique[str]]``.

        Raises:
            DecodeError: If ``kind`` cannot be decoded that way.

        """
        self.check(how, kind)
        key = (how, kind)
        if key not in self._decoders:
            self.used = True
            result = result_of(kind, OWNED_STRING)
            names: dict[str, object] = {
                "Result": result,
                "start": self.runtime(how),
                "read": self.reader(how, kind),
                "end": self.runtime("json_end"),
                "failed": self.runtime("failed"),
                "message": self.runtime("error"),
                "finish": self.runtime("free"),
                "zero_value": self.drops.zero(kind),
                "zero_error": self.drops.zero(OWNED_STRING),
            }
            lines = ["def {name}(text):", "    p = start(text)", "    value = read(p)"]
            if how == JSON:
                lines.append("    end(p)")
            lines += ["    if failed(p):", "        error = message(p)"]
            if owns(kind):
                names["drop"] = self.drops.drop(kind)
                lines.append("        drop(value)")
            lines += [
                "        finish(p)",
                "        return Result(ok=False, value=zero_value(), error=error)",
                "    finish(p)",
                "    return Result(ok=True, value=value, error=zero_error())",
            ]
            self._decoders[key] = self.function(lines, names, {"text": cstr, "return": result})
        return self._decoders[key]

    def check(self, how: str, kind: ScalarType, where: str = "") -> None:
        """Reject a type that cannot be decoded (``how``), saying why.

        Raises:
            DecodeError: If it cannot.

        """
        shown = where or kind.name
        if kind == cstr:
            msg = f"{shown} is a str, which would be lent from the text; decoding makes strings: mem.Unique[str]"
            raise DecodeError(msg)
        if kind == OWNED_STRING or kind.kind in {"int", "uint", "float", "bool"}:
            return
        if is_closure(kind) or is_result(kind):
            raise DecodeError(f"{shown} has no {how} form")
        if union(kind) is not None:
            self._check_union(how, kind, shown)
            return
        if is_list(kind):
            if how == FORM:
                raise DecodeError(f"a form holds no lists ({shown})")
            assert kind.element is not None
            self.check(how, kind.element, f"an item of {shown}")
            return
        if isinstance(kind, StructType):
            self._check_fields(how, kind)
            return
        raise DecodeError(f"{shown} has no {how} form")

    def _check_fields(self, how: str, kind: StructType) -> None:
        """Reject a record or object with a field that cannot be decoded (its literal ones are checked, not decoded).

        Raises:
            DecodeError: If one cannot.

        """
        fixed = literals(kind)
        for field, held in kind.fields:
            if field in fixed:
                continue
            inner = scalar_type(held)
            assert inner is not None
            if how == FORM and (isinstance(inner, StructType) or is_list(inner)):
                raise DecodeError(f"a form holds numbers, bools and strings, not {field}'s")
            self.check(how, inner, f"field {field}")

    def _check_union(self, how: str, kind: ScalarType, shown: str) -> None:
        """Reject a union that cannot be decoded: one in a form, or one with an object that cannot be.

        Raises:
            DecodeError: If it cannot.

        """
        found = union(kind)
        assert found is not None
        if how == FORM:
            raise DecodeError(f"a form holds no unions ({shown})")
        for written, held in found.variants:
            self.check(how, held, written)

    def reader(self, how: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Return the function reading one ``kind`` value from a reader (the runtime's), as ``how`` writes it."""
        key = (how, kind)
        if key not in self._readers:
            found = union(kind)
            if how == FORM and isinstance(kind, StructType):
                self._readers[key] = self._form_fields(kind)
            elif how == FORM:
                self._readers[key] = self._form_single(kind)
            elif found is not None:
                self._readers[key] = self._json_union(kind, found)
            elif isinstance(kind, StructType):
                self._readers[key] = self._json_object(kind)
            elif is_list(kind):
                self._readers[key] = self._json_list(kind)
            else:
                self._readers[key] = self._scalar(JSON, kind)
        return self._readers[key]

    def _scalar(self, how: str, kind: ScalarType) -> Any:  # noqa: ANN401
        """Read a number, bool or string: the runtime's reading, as ``kind``."""
        part = (
            "string"
            if kind.kind == "cstr"
            else "bool"
            if kind.kind == "bool"
            else "float"
            if kind.kind == "float"
            else "int"
        )
        names: dict[str, object] = {"get": self.runtime(f"{how}_{part}"), "Kind": getattr(kind, "python", None) or kind}
        body = "    return get(p)" if part in {"string", "bool"} else "    return Kind(get(p))"
        return self.function(["def {name}(p):", body], names, {"p": ptr, "return": kind})

    def _form_single(self, kind: ScalarType) -> Any:  # noqa: ANN401
        names: dict[str, object] = {"single": self.runtime("form_single"), "get": self._scalar(FORM, kind)}
        lines = ["def {name}(p):", "    single(p)", "    return get(p)"]
        return self.function(lines, names, {"p": ptr, "return": kind})

    def _form_fields(self, kind: StructType) -> Any:  # noqa: ANN401
        """Read a record or object from a form: each field from the pair of its name (the last one given)."""
        names: dict[str, object] = {
            "Value": kind.python,
            "field": self.runtime("form_field"),
            "missing": self.runtime("missing"),
        }
        lines = ["def {name}(p):"]
        fixed = literals(kind)  # its literal fields hold their text, given or not
        for index, (field, held) in enumerate(kind.fields):
            if field in fixed:
                lines.append(f"    f_{index} = {fixed[field]!r}")
                continue
            inner = scalar_type(held)
            assert inner is not None
            names[f"get_{index}"] = self._scalar(FORM, inner)
            names[f"zero_{index}"] = self.drops.zero(inner)
            lines += [
                f"    f_{index} = zero_{index}()",
                f'    if field(p, "{field}"):',
                f"        f_{index} = get_{index}(p)",
                "    else:",
                f'        missing(p, "{field}")',
            ]
        arguments = ", ".join(f"{field}=f_{index}" for index, (field, _) in enumerate(kind.fields))
        lines.append(f"    return Value({arguments})")
        return self.function(lines, names, {"p": ptr, "return": kind})

    def _json_object(self, kind: StructType) -> Any:  # noqa: ANN401
        """Read a record or object from a JSON object: its fields by name, in any order (others skipped)."""
        names: dict[str, object] = {
            "Value": kind.python,
            "object": self.runtime("json_object"),
            "key": self.runtime("json_key"),
            "key_is": self.runtime("key_is"),
            "skip": self.runtime("json_skip"),
            "missing": self.runtime("missing"),
            "duplicate": self.runtime("duplicate"),
        }
        lines = ["def {name}(p):"]
        fixed = literals(kind)  # its literal fields: read to check they hold their text
        if fixed:
            names["literal"] = self.runtime("json_literal")
        for index, (field, held) in enumerate(kind.fields):
            if field in fixed:
                lines += [f"    f_{index} = {fixed[field]!r}", f"    seen_{index} = False"]
                continue
            inner = scalar_type(held)
            assert inner is not None
            names[f"get_{index}"] = self.reader(JSON, inner)
            names[f"zero_{index}"] = self.drops.zero(inner)
            lines += [f"    f_{index} = zero_{index}()", f"    seen_{index} = False"]
        lines += ["    if object(p):", "        while key(p):"]
        for index, (field, held) in enumerate(kind.fields):
            branch = "if" if index == 0 else "elif"
            lines += [f'            {branch} key_is(p, "{field}"):']
            lines += [f"                if seen_{index}:", f'                    duplicate(p, "{field}")']
            if field in fixed:
                lines += [f"                literal(p, {fixed[field]!r})", f"                seen_{index} = True"]
                continue
            if owns(held):  # a duplicate's first value is freed, not lost
                names[f"drop_{index}"] = self.drops.drop(scalar_type(held))
                lines.append(f"                drop_{index}(f_{index})")
            lines += [f"                f_{index} = get_{index}(p)", f"                seen_{index} = True"]
        lines += ["            else:", "                skip(p)"]
        for index, (field, _) in enumerate(kind.fields):
            lines += [f"    if not seen_{index}:", f'        missing(p, "{field}")']
        arguments = ", ".join(f"{field}=f_{index}" for index, (field, _) in enumerate(kind.fields))
        lines.append(f"    return Value({arguments})")
        return self.function(lines, names, {"p": ptr, "return": kind})

    def _json_union(self, kind: ScalarType, found: Union) -> Any:  # noqa: ANN401
        """Read a union from a JSON object: the object of it whose literal field (its ``on``) holds the text given."""
        names: dict[str, object] = {
            "Value": getattr(kind, "python", kind),
            "peek": self.runtime("json_peek"),
            "peeked_is": self.runtime("peeked_is"),
            "unknown": self.runtime("unknown"),
            "skip": self.runtime("json_skip"),
        }
        lines = ["def {name}(p):", f'    peek(p, "{found.discriminator}")']
        for index, (_, held) in enumerate(found.variants):
            names[f"get_{index}"] = self.reader(JSON, held)
            names[f"zero_{index}"] = self.drops.zero(held)

        def build(index: int | None) -> str:
            slots = ", ".join(
                f"{slot(other)}=" + (f"get_{other}(p)" if other == index else f"zero_{other}()")
                for other in range(len(found.variants))
            )
            return f"Value(tag={index or 0}, {slots})"

        for index, value in enumerate(found.values):
            lines += [f"    if peeked_is(p, {value!r}):", f"        return {build(index)}"]
        lines += [f'    unknown(p, "{found.discriminator}")', "    skip(p)", f"    return {build(None)}"]
        return self.function(lines, names, {"p": ptr, "return": kind})

    def _json_list(self, kind: ScalarType) -> Any:  # noqa: ANN401
        """Read a list from a JSON array, item by item (growing in place)."""
        element = kind.element
        assert element is not None
        names: dict[str, object] = {
            "List": kind,
            "array": self.runtime("json_array"),
            "more": self.runtime("json_more"),
            "get": self.reader(JSON, element),
            "new": list_runtime.bifrost_list_new,
            "grow": list_runtime.bifrost_list_grow,
            "length": list_runtime.bifrost_list_length,
        }
        lines = [
            "def {name}(p):",
            f"    items = List(new(0, {element.size}))",
            "    if array(p):",
            "        while more(p):",
            "            item = get(p)",
            f"            items = List(grow(items, 1, {element.size}))",
            "            items[length(items) - 1] = item",
            "    return items",
        ]
        return self.function(lines, names, {"p": ptr, "return": kind})

    def runtime(self, name: str) -> Any:  # noqa: ANN401
        """Declare (once) the runtime's ``bifrost_decode_<name>``."""
        if name not in self._runtime:
            parameters, result = _RUNTIME[name]
            arguments = [f"a{index}" for index in range(len(parameters))]
            stub: Callable[..., Any] = self._stub(f"bifrost_decode_{name}", arguments)
            stub.__annotations__ = dict(zip(arguments, parameters, strict=True)) | {"return": result}
            self._runtime[name] = self.program.extern(stub, name=f"bifrost_decode_{name}")
        return self._runtime[name]

    @staticmethod
    def _stub(name: str, arguments: list[str]) -> Callable[..., Any]:
        source = f"def {name}({', '.join(arguments)}):\n    ...\n"
        filename = f"<bifrost decode {name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        code = compile(source, filename, "exec")
        function_code = next(constant for constant in code.co_consts if isinstance(constant, CodeType))
        return FunctionType(function_code, {}, name)

    def function(self, lines: list[str], names: dict[str, object], annotations: dict[str, object]) -> Any:  # noqa: ANN401
        """Compile generated source into the program (see ``Drops.function``)."""
        self._count += 1
        name = f"bifrost_decode_{self._count}"
        source = "\n".join(lines).format(name=name) + "\n"
        filename = f"<bifrost decoding {name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        code = compile(source, filename, "exec")
        function_code = next(constant for constant in code.co_consts if isinstance(constant, CodeType))
        python: Any = FunctionType(function_code, names, name)
        python.__annotations__ = annotations
        return self.program.function(python)


__all__ = ["FORM", "JSON", "DecodeError", "Decoders"]
