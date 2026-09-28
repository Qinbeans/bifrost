"""``std:json``: values as JSON text.

Imported like any standard module (``let json = import("std:json")``), but
built into the compiler.

- ``json.encode(value)``: the JSON text of a record (``#{id: 7, name: "x"}``)
  or an object (a ``struct``), nested ones included, as a ``mem.Unique[str]``.
  The compiler writes the conversion from the value's field names and types:
  strings are escaped, integers and floats written as numbers, bools as
  ``true``/``false``.

A C function can take JSON too: a parameter declared ``json`` in
``config.yaml`` (``body: json``) receives a record or object encoded at the
call, and freed after it; a ``str`` passes through as JSON text already.
"""

from bifrost.std.fmt import Builtin

ENCODE = Builtin(
    "encode",
    "(value: record or object) => mem.Unique[str]",
    "the JSON text of a record or object, as an owned string",
    module="json",
)

FUNCTIONS = {builtin.name: builtin for builtin in (ENCODE,)}

__all__ = ["ENCODE", "FUNCTIONS"]
