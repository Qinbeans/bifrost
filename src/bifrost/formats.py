"""Checking printf patterns against the values they format, at compile time.

C's ``printf`` family (and ``fmt.format``, built on ``snprintf``) reads its
values by the pattern's conversions: a value of another type is read as garbage
(``%d`` for an ``i64`` reads half of it; ``%s`` for a number reads memory at
that address). The pattern is usually a literal, so the compiler checks it:
one value per conversion (and per ``*`` width or precision), each of a type
the conversion reads.

``%v`` prints any value, as its type calls for: a number as a number, a bool
as ``true`` or ``false``, a string as it is, and a list or record as its JSON
text (``[1, 2]``, ``{"name": "ada"}``). An object prints what its
``to_string`` returns.

Numbers are passed as their conversion reads them: every integer conversion
is made 64-bit (``%d`` becomes ``%lld``, and the value an ``i64``), so ``%d``
prints any integer, whatever its size; ``%f`` prints an integer as a float.
Only ``%hd`` and ``%hhd`` (and ``%c``, and a ``*`` width) keep C's ``int``.
"""

import re
from dataclasses import dataclass

from mlir_python.lang._types import ScalarType, StructType

from bifrost.owned import is_list

# The C functions that format, and which of their parameters is the pattern.
PATTERNS = {"printf": 0, "fprintf": 1, "dprintf": 1, "sprintf": 1, "snprintf": 2}

_CONVERSION = re.compile(
    r"%(?P<flags>[-+ #0]*)(?P<width>\*|\d+)?(?:\.(?P<precision>\*|\d*))?"
    r"(?P<length>hh|h|ll|l|j|z|t|L)?(?P<letter>.?)"
)
_WIDE = {"ll", "l", "j", "z", "t"}  # 64-bit integers


@dataclass(frozen=True)
class Read:
    """What one value of a pattern is read as: ``int``, ``float``, ``string``, ``pointer``."""

    what: str
    conversion: str  # as written, e.g. ``%d``, or ``*`` for a width
    wide: bool = False  # read as 64 bits (``%d`` is, once the pattern is widened)
    unsigned: bool = False  # ``%u``, ``%x``, ``%o``


class FormatError(Exception):
    """A pattern that does not fit its values; ``index`` is the value at fault, or ``None`` for the pattern."""

    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.index = index


def reads(pattern: str) -> list[Read]:
    """Return what each value ``pattern`` formats is read as, in order.

    Raises:
        FormatError: For a conversion C does not have (``%v``), or one Bifrost does not allow (``%n``).

    """
    found: list[Read] = []
    for match in _CONVERSION.finditer(pattern):
        if match.group(0) == "%%":
            continue
        found += [Read("int", "*") for star in (match["width"], match["precision"]) if star == "*"]
        found.append(_read(match))
    return found


def _read(match: re.Match[str]) -> Read:
    """Return what one conversion (``%5.2f``) reads its value as."""
    letter, length, written = match["letter"], match["length"] or "", match.group(0)
    kinds = {"f": "float", "s": "string", "p": "pointer"}
    if letter and letter in "diouxX":
        return Read("int", written, wide=length in _WIDE or not length, unsigned=letter not in "di")
    if letter == "c" and not length:
        return Read("int", written)
    if letter and letter in "eEfFgGaA" and length != "L":
        return Read("float", written)
    if letter in {"s", "p"} and not length:
        return Read(kinds[letter], written)
    if letter == "v" and not length:
        return Read("value", written)
    if letter == "n":
        raise FormatError(f"{written} writes to memory, which Bifrost does not allow in a pattern")
    shown = written if letter else f"{written} at the end"
    msg = (
        f"{shown} is not a printf conversion: use %v (any value), %d (integers), %f (floats), %s (strings), "
        "%p (addresses), or %%"
    )
    raise FormatError(msg)


def widen(pattern: str) -> str:
    """Make every integer conversion 64-bit (``%d`` -> ``%lld``, ``%zu`` -> ``%llu``); keep the others."""

    def wider(match: re.Match[str]) -> str:
        length = match["length"] or ""
        if match["letter"] not in "diouxX" or not match["letter"] or length in {"h", "hh", "L"}:
            return match.group(0)
        precision = f".{match['precision']}" if match["precision"] is not None else ""
        return f"%{match['flags']}{match['width'] or ''}{precision}ll{match['letter']}"

    return _CONVERSION.sub(wider, pattern)


def fill(pattern: str, conversions: list[str]) -> str:
    """Replace each ``%v`` with the conversion its value needs (``lld``, ``g``, ``s``), keeping its flags and width."""
    remaining = iter(conversions)

    def filled(match: re.Match[str]) -> str:
        if match["letter"] != "v" or match["length"]:
            return match.group(0)
        precision = f".{match['precision']}" if match["precision"] is not None else ""
        return f"%{match['flags']}{match['width'] or ''}{precision}{next(remaining)}"

    return _CONVERSION.sub(filled, pattern)


def check(pattern: str, values: list[tuple[str, ScalarType | None, bool]]) -> list[Read]:
    """Check ``values`` (each as written, its type if known, and whether it is a literal) against ``pattern``.

    Return what each value is read as, for the lowering to pass literals as read.

    Raises:
        FormatError: At the first value that does not fit, or when there are too many or too few.

    """
    wanted = reads(pattern)
    if len(values) != len(wanted):
        count = f"{len(wanted)} value{'s' if len(wanted) != 1 else ''}"
        raise FormatError(f"the pattern formats {count}, but {len(values)} follow it")
    for index, (read, (text, kind, literal)) in enumerate(zip(wanted, values, strict=True)):
        problem = _misfit(read, text, kind, literal=literal)
        if problem:
            raise FormatError(problem, index)
    return wanted


def _misfit(read: Read, text: str, kind: ScalarType | None, *, literal: bool) -> str:
    """Explain why a value of ``kind`` does not fit ``read``; ``""`` if it does (or its type is not known)."""
    del literal
    if kind is None or read.what == "value":
        return ""  # anything prints with %v (an object, through its to_string)
    shown = kind.name.replace("cstr", "str")
    if is_list(kind) or isinstance(kind, StructType):
        what = "a list" if is_list(kind) else "a record or object"
        if read.what == "pointer" and is_list(kind):
            return ""
        return f"{text} is {what}, which {read.conversion} cannot print; print it with %v"
    fits = {
        "int": kind.kind in {"int", "uint", "bool"},
        "float": kind.kind in {"float", "int", "uint"},
        "string": kind.kind == "cstr",
        "pointer": kind.kind in {"ptr", "cstr"},
    }[read.what]
    if fits:
        return ""
    names = {"int": "integers", "float": "numbers", "string": "strings", "pointer": "addresses"}
    by_kind = {"int": "%d", "uint": "%u", "bool": "%d", "float": "%f", "cstr": "%s"}.get(kind.kind, "%p")
    article = _article(shown)
    return f"{read.conversion} prints {names[read.what]}, but {text} is {article} {shown}: use {by_kind}"


def _article(name: str) -> str:
    """Return "a" or "an" for a type's name, as it is said: an i64, an f64, a u64, a str."""
    spelled = re.match(r"[a-z]\d", name) is not None  # said letter by letter: "eff sixty-four"
    return "an" if name[:1] in ("aefhilmnorsx" if spelled else "aeiou") else "a"


__all__ = ["PATTERNS", "FormatError", "Read", "check", "fill", "reads", "widen"]
