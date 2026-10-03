"""A list's own functions, called on it like methods: ``xs.contains(3)``, ``xs.map(f)``, ``g.sort()``.

They are built into the compiler, like ``len``: no import, no dependency
entry. Each is generated once per element type (and, for ``map``, per
function type), from source, as the drop and clone functions are (see
``bifrost.owned``).

Reading a list (any list, a view too):

- ``contains(x)``, ``index_of(x)`` (-1 when absent): numbers, bools and strings;
- ``map(f)``, ``filter(f)``: a new list, of what ``f`` gives, or of the items
  ``f`` keeps (copies);
- ``sorted()``, ``reversed()``: a new list, sorted (numbers and strings,
  ascending) or reversed.

Changing it in place, through a guard (``let g <- xs``), like item writes:
``sort()`` and ``reverse()``. Sorting is a heapsort: O(n log n), in place.
"""

from typing import Any

from mlir_python.lang._types import ScalarType, StructType, scalar_type

from bifrost.formats import article
from bifrost.owned import Drops, is_closure, list_of, owns, signature_of
from bifrost.std import list_runtime

READING = {"contains", "index_of", "map", "filter", "sorted", "reversed"}
CHANGING = {"sort", "reverse"}  # in place, through a guard
METHODS = READING | CHANGING

_STRINGS = ("cstr",)


# How each reads, for the editor: its signature (`T` is the item type) and what it does.
_DESCRIBED = {
    "contains": ("(x: T) => bool", "whether x is one of its items"),
    "index_of": ("(x: T) => i64", "the index of x's first appearance, or -1"),
    "map": ("(f: (x: T) => U) => U[]", "a new list of what f gives for each item"),
    "filter": ("(f: (x: T) => bool) => T[]", "a new list of copies of the items f keeps"),
    "sorted": ("() => T[]", "a new list of copies of its items, smallest first"),
    "reversed": ("() => T[]", "a new list of copies of its items, last first"),
    "sort": ("() => null", "sorts it in place, smallest first (through a guard)"),
    "reverse": ("() => null", "reverses it in place (through a guard)"),
}


def describe(name: str, element: str) -> tuple[str, str]:
    """Return a list method's signature, for items of type ``element`` (as written), and what it does."""
    signature, summary = _DESCRIBED[name]
    return signature.replace("T", element), summary


class ListMethodError(Exception):
    """A list method used wrongly: ``message`` says how."""


def comparable(element: ScalarType) -> bool:
    """Whether items of ``element`` can be compared: numbers, bools and strings."""
    return element.kind in {"int", "uint", "float", "bool", *_STRINGS}


def orderable(element: ScalarType) -> bool:
    """Whether items of ``element`` can be sorted: numbers and strings."""
    return element.kind in {"int", "uint", "float", *_STRINGS}


def result(name: str, kind: ScalarType, function: object = None) -> ScalarType | None:
    """Return what calling the method ``name`` on a list of type ``kind`` gives (``function``: what ``map`` takes)."""
    match name:
        case "contains":
            return scalar_type(bool)
        case "index_of":
            return scalar_type(int)
        case "filter" | "sorted" | "reversed":
            return kind
        case "map":
            if not is_closure(function):
                return None
            given = signature_of(function).result
            return list_of(given) if given is not None else None
    return None


class ListMethods:
    """The list methods of a program, generated as they are needed."""

    def __init__(self, drops: Drops) -> None:
        self.drops = drops
        self._functions: dict[tuple[object, ...], Any] = {}

    def get(self, name: str, kind: ScalarType, function: ScalarType | None = None) -> Any:  # noqa: ANN401
        """Return the compiled method ``name`` of lists of type ``kind`` (``function``: what ``map`` takes)."""
        key = (name, kind, function)
        if key not in self._functions:
            self._functions[key] = getattr(self, f"_{name}")(kind, function)
        return self._functions[key]

    # -- what the generated source uses ---------------------------------------------

    def _names(self, kind: ScalarType) -> dict[str, object]:
        element = kind.element
        assert element is not None
        names: dict[str, object] = {
            "List": kind,
            "length": list_runtime.bifrost_list_length,
            "new": list_runtime.bifrost_list_new,
            "set_length": list_runtime.bifrost_list_set_length,
            "compare": list_runtime.bifrost_string_compare,
        }
        if owns(element):
            names["clone"] = self.drops.clone(element)
        return names

    @staticmethod
    def _equal(element: ScalarType, a: str, b: str) -> str:
        return f"compare({a}, {b}) == 0" if element.kind in _STRINGS else f"{a} == {b}"

    @staticmethod
    def _less(element: ScalarType, a: str, b: str) -> str:
        return f"compare({a}, {b}) < 0" if element.kind in _STRINGS else f"{a} < {b}"

    @staticmethod
    def _copy(element: ScalarType, value: str) -> str:
        """Write a copy of an item: deep, for one that owns (the new list owns its copy)."""
        return f"clone({value})" if owns(element) else value

    @staticmethod
    def _annotation(kind: object) -> object:
        found = scalar_type(kind)
        return found.python if isinstance(found, StructType) else kind

    # -- the methods ----------------------------------------------------------------------

    def _contains(self, kind: ScalarType, _: object) -> Any:  # noqa: ANN401
        element = kind.element
        assert element is not None
        lines = [
            "def {name}(items, value):",
            "    count = length(items)",
            "    index = 0",
            "    found = False",
            "    while index < count and not found:",
            f"        found = {self._equal(element, 'items[index]', 'value')}",
            "        index = index + 1",
            "    return found",
        ]
        value = self._annotation(element)
        return self.drops.function(lines, self._names(kind), {"items": kind, "value": value, "return": bool})

    def _index_of(self, kind: ScalarType, _: object) -> Any:  # noqa: ANN401
        element = kind.element
        assert element is not None
        lines = [
            "def {name}(items, value):",
            "    count = length(items)",
            "    index = 0",
            "    found = -1",
            "    while index < count and found < 0:",
            f"        if {self._equal(element, 'items[index]', 'value')}:",
            "            found = index",
            "        index = index + 1",
            "    return found",
        ]
        value = self._annotation(element)
        return self.drops.function(lines, self._names(kind), {"items": kind, "value": value, "return": int})

    def _sift(self, kind: ScalarType) -> Any:  # noqa: ANN401
        """Move ``items[root]`` down a heap of ``count`` items until both children are smaller."""
        element = kind.element
        assert element is not None
        lines = [
            "def {name}(items, root, count):",
            "    going = True",
            "    while going and root * 2 + 1 < count:",
            "        child = root * 2 + 1",
            f"        if child + 1 < count and {self._less(element, 'items[child]', 'items[child + 1]')}:",
            "            child = child + 1",
            f"        if {self._less(element, 'items[root]', 'items[child]')}:",
            "            held = items[root]",
            "            items[root] = items[child]",
            "            items[child] = held",
            "            root = child",
            "        else:",
            "            going = False",
        ]
        return self.drops.function(lines, self._names(kind), {"items": kind, "root": int, "count": int, "return": None})

    def _sort(self, kind: ScalarType, _: object) -> Any:  # noqa: ANN401
        names = self._names(kind) | {"sift": self._sift(kind)}
        lines = [
            "def {name}(items):",
            "    count = length(items)",
            "    start = count // 2 - 1",
            "    while start >= 0:",
            "        sift(items, start, count)",
            "        start = start - 1",
            "    end = count - 1",
            "    while end > 0:",
            "        held = items[0]",
            "        items[0] = items[end]",
            "        items[end] = held",
            "        sift(items, 0, end)",
            "        end = end - 1",
        ]
        return self.drops.function(lines, names, {"items": kind, "return": None})

    def _reverse(self, kind: ScalarType, _: object) -> Any:  # noqa: ANN401
        lines = [
            "def {name}(items):",
            "    low = 0",
            "    high = length(items) - 1",
            "    while low < high:",
            "        held = items[low]",
            "        items[low] = items[high]",
            "        items[high] = held",
            "        low = low + 1",
            "        high = high - 1",
        ]
        return self.drops.function(lines, self._names(kind), {"items": kind, "return": None})

    def _sorted(self, kind: ScalarType, _: object) -> Any:  # noqa: ANN401
        names = self._names(kind) | {"copy": self.drops.clone(kind), "sort": self.get("sort", kind)}
        lines = ["def {name}(items):", "    copied = copy(items)", "    sort(copied)", "    return copied"]
        return self.drops.function(lines, names, {"items": kind, "return": kind})

    def _reversed(self, kind: ScalarType, _: object) -> Any:  # noqa: ANN401
        element = kind.element
        assert element is not None
        lines = [
            "def {name}(items):",
            "    count = length(items)",
            f"    copied = List(new(count, {element.size}))",
            "    index = 0",
            "    while index < count:",
            f"        copied[index] = {self._copy(element, 'items[count - 1 - index]')}",
            "        index = index + 1",
            "    return copied",
        ]
        return self.drops.function(lines, self._names(kind), {"items": kind, "return": kind})

    def _map(self, kind: ScalarType, function: object) -> Any:  # noqa: ANN401
        mapped = result("map", kind, function)
        assert mapped is not None
        assert mapped.element is not None
        names = self._names(kind) | {"Mapped": mapped}
        lines = [
            "def {name}(items, f):",
            "    count = length(items)",
            f"    mapped = Mapped(new(count, {mapped.element.size}))",
            "    index = 0",
            "    while index < count:",
            "        mapped[index] = f.fn(f.env, items[index])",
            "        index = index + 1",
            "    return mapped",
        ]
        annotations = {"items": kind, "f": self._annotation(function), "return": mapped}
        return self.drops.function(lines, names, annotations)

    def _filter(self, kind: ScalarType, function: object) -> Any:  # noqa: ANN401
        element = kind.element
        assert element is not None
        lines = [
            "def {name}(items, f):",
            "    count = length(items)",
            f"    kept = List(new(count, {element.size}))",
            "    found = 0",
            "    index = 0",
            "    while index < count:",
            "        if f.fn(f.env, items[index]):",
            f"            kept[found] = {self._copy(element, 'items[index]')}",
            "            found = found + 1",
            "        index = index + 1",
            "    set_length(kept, found)",
            "    return kept",
        ]
        annotations = {"items": kind, "f": self._annotation(function), "return": kind}
        return self.drops.function(lines, self._names(kind), annotations)


def check(name: str, kind: ScalarType, arguments: list[ScalarType | None]) -> None:
    """Check a call of the list method ``name`` on a ``kind`` list with arguments of these types.

    Raises:
        ListMethodError: Saying what is wrong, and how to call it.

    """
    element = kind.element
    assert element is not None
    shown = {"contains": "xs.contains(x)", "index_of": "xs.index_of(x)", "map": "xs.map(f)", "filter": "xs.filter(f)"}
    takes_one = name in shown
    if takes_one != (len(arguments) == 1) or len(arguments) > 1:
        usage = shown.get(name, f"xs.{name}()")
        raise ListMethodError(f"{name} takes {'one argument' if takes_one else 'no arguments'}: {usage}")
    if name in {"contains", "index_of"} and not comparable(element):
        raise ListMethodError(f"{name} compares numbers, bools and strings, not {element.name}")
    if name in {"sort", "sorted"} and not orderable(element):
        raise ListMethodError(f"{name} orders numbers and strings, not {element.name}")
    if name in {"map", "filter"}:
        function = arguments[0]
        if not is_closure(function):
            raise ListMethodError(f"{name} takes a function: xs.{name}([] (x: T) => ... )")
        signature = signature_of(function)
        params = list(signature.params)
        takes = params[0] if len(params) == 1 else None
        if takes is None or not (takes == element or (takes.kind == "cstr" and element.kind == "cstr")):
            raise ListMethodError(f"{name}'s function takes one item, {article(element.name)} {element.name}")
        if name == "filter" and signature.result != scalar_type(bool):
            raise ListMethodError("filter's function says whether to keep an item: it returns bool")
        if name == "map" and signature.result is None:
            raise ListMethodError("map's function gives each new item: it returns a value")


__all__ = [
    "CHANGING",
    "METHODS",
    "READING",
    "ListMethodError",
    "ListMethods",
    "check",
    "comparable",
    "describe",
    "result",
]
