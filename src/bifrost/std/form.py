"""``std:form``: values as forms, ``a=1&b=x+y`` (application/x-www-form-urlencoded).

Imported like any standard module (``let form = import("std:form")``), but
built into the compiler.

- ``form.decode(text)``: the record or object a form holds, as a ``Result[T,
  mem.Unique[str]]`` (``T`` is the type written where it goes): each field from
  the pair of its name, numbers, bools and owned strings (``mem.Unique[str]``),
  percent-decoded. A single number or string decodes from a form of one pair.
  It is what a query string and a submitted HTML form are, and how an HTTP
  server hands over a path's parameters.
"""

from bifrost.std.fmt import Builtin

DECODE = Builtin(
    "decode",
    "(text: str) => Result[T, mem.Unique[str]]",
    "the record a form (a=1&b=x) holds, of the type written where it goes, or what is wrong with it",
    module="form",
)

FUNCTIONS = {builtin.name: builtin for builtin in (DECODE,)}

__all__ = ["DECODE", "FUNCTIONS"]
