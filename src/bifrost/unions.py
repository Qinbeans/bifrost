"""Tagged unions: ``let Actor = Employee | Service on "type"`` is a value that is one of those objects.

Each object of the union has a literal field named after ``on`` (see ``literals``),
``let type: "employee"``, holding a different text: it tells them apart as JSON
(``json.decode`` reads it to know which object it is reading), and is always
right, since nothing can change it.

A union is made by putting one of its objects where the union goes (a ``let``'s
written type, a parameter, a field, a returned value); the compiler wraps it.
It is read with ``match actor { Employee(e): ..., Service(s): ... }``: each arm
names an object of the union, and binds the one it holds (``Employee(_)`` binds
nothing). Arms naming every object of the union cover it; ``_`` covers the rest.

A union is an object of a ``tag`` (which object it holds, by its place in the
union) and one field per object, of which only the tagged one is set; the others
are zero. It owns what it holds when any of its objects owns memory, and frees
only the one it holds (see ``owned.Drops``).
"""

from dataclasses import dataclass

from mlir_python.lang import i32, struct
from mlir_python.lang._types import scalar_type
from mlir_python.lang.types import ScalarType, StructType
from tree_sitter import Node

from bifrost.literals import literals

TAG = "tag"


@dataclass(frozen=True)
class Union:
    """A union type's objects: each one's name as written (``Employee``, ``models.Employee``) and its type."""

    name: str
    variants: tuple[tuple[str, StructType], ...]
    discriminator: str  # the literal field telling the objects apart
    values: tuple[str, ...]  # the text it holds in each object, in their order

    def index(self, written: str) -> int | None:
        """Return the place of the object written ``written`` in the union, if it is one of its objects."""
        return next((index for index, (name, _) in enumerate(self.variants) if name == written), None)

    def of(self, kind: object) -> int | None:
        """Return the place of the object type ``kind`` in the union, if it is one of its objects."""
        found = scalar_type(kind)
        return next((index for index, (_, held) in enumerate(self.variants) if held == found), None)

    def shown(self) -> str:
        """Write the union as it is declared: ``Employee | Service``."""
        return " | ".join(name for name, _ in self.variants)

    def shared(self, field: str) -> ScalarType | None:
        """Return the type of the field ``field`` when each object has one of that type: the union reads it."""
        found = {scalar_type(held.field(field)[1]) if held.field(field) else None for _, held in self.variants}
        only = found.pop() if len(found) == 1 else None
        return only if isinstance(only, ScalarType) else None

    def declared(self) -> str:
        """Write the union with what tells its objects apart: ``Employee | Service on "type"``."""
        return f'{self.shown()} on "{self.discriminator}"'


# By the union's type, kept alive here so that types compare by identity.
_UNIONS: dict[int, tuple[type, Union]] = {}


def slot(index: int) -> str:
    """Name the field that holds a union's object at ``index`` (a name no Bifrost field can have)."""
    return f"__variant_{index}"


def union_of(name: str, variants: list[tuple[str, StructType]], discriminator: str) -> type:
    """Make the union ``name`` of ``variants`` (each object's written name and type): a new type, every time.

    ``discriminator`` is the literal field each object has, with a text of its own.

    Raises:
        ValueError: An object appears twice, the union has fewer than two, or the
            discriminator is not a literal field of each, holding a different text.

    """
    if len(variants) < 2:  # noqa: PLR2004
        raise ValueError(f'a union is of two objects or more: let {name} = A | B on "{discriminator}"')
    seen: dict[int, str] = {}
    values: dict[str, str] = {}  # each text, and the object holding it
    for written, kind in variants:
        if id(kind) in seen:
            raise ValueError(f"{written} is in {name} twice")
        seen[id(kind)] = written
        value = _discriminant(name, written, kind, discriminator)
        if value in values:
            msg = f'{values[value]} and {written} both have {discriminator} "{value}"; '
            msg += f"each object of {name} needs its own"
            raise ValueError(msg)
        values[value] = written
    fields: dict[str, object] = {TAG: i32}
    for index, (_, kind) in enumerate(variants):
        fields[slot(index)] = kind.python
    cls = struct(type(name, (), {"__annotations__": fields}))
    _UNIONS[id(cls)] = (cls, Union(name, tuple(variants), discriminator, tuple(values)))
    return cls


def _discriminant(name: str, written: str, kind: StructType, discriminator: str) -> str:
    """Return the text the object ``written`` holds in its literal field ``discriminator``.

    Raises:
        ValueError: It has no such field, or one that is not literal.

    """
    fixed = literals(kind)
    if discriminator in fixed:
        return fixed[discriminator]
    example = f'let {discriminator}: "{written.rsplit(".", maxsplit=1)[-1].lower()}"'
    if any(field == discriminator for field, _ in kind.fields):
        msg = (
            f"{name} tells its objects apart by {discriminator}, which {written} can set to any text; fix it: {example}"
        )
        raise ValueError(msg)
    raise ValueError(f"{name} tells its objects apart by {discriminator}, which {written} does not have: {example}")


def union(kind: object) -> Union | None:
    """Return what the union type ``kind`` holds; ``None`` when ``kind`` is not a union."""
    found = scalar_type(kind)
    if not isinstance(found, StructType):
        return None
    entry = _UNIONS.get(id(found.python))
    return entry[1] if entry is not None and entry[0] is found.python else None


def is_union(kind: object) -> bool:
    """Whether ``kind`` is a union type."""
    return union(kind) is not None


def is_variant(kind: object) -> bool:
    """Whether the object type ``kind`` is one of a union's objects."""
    found = scalar_type(kind)
    return any(held == found for _, entry in _UNIONS.values() for _, held in entry.variants)


def covers(kinds: set[int]) -> bool:
    """Whether some union holds exactly the object types ``kinds`` (by ``id``): arms naming them all cover it.

    For a match on a value whose type is not yet known; one whose type is known asks its union.
    """
    return any({id(held) for _, held in entry.variants} == kinds for _, entry in _UNIONS.values())


def variant_type(kind: ScalarType, index: int) -> StructType:
    """Return the type of the object at ``index`` of the union type ``kind``."""
    found = union(kind)
    assert found is not None
    return found.variants[index][1]


def _call(condition: Node) -> tuple[str, Node] | None:
    """For ``Employee(e)`` or ``models.Employee(e)``: the dotted name called, and the ``user_function_call``."""
    node = condition
    while node.type in {"expression", "condition", "parenthesized_expression"} and node.named_children:
        node = node.named_children[0]
    path: list[str] = []
    if node.type == "child_annotation":  # `models.Employee(e)`
        *names, node = node.named_children
        if not names or any(part.type != "simple_identifier" for part in names):
            return None
        path = [(part.text or b"").decode() for part in names]
    call = node.named_children[0] if node.type == "function_call" and node.named_children else None
    function = call.child_by_field_name("function") if call is not None and call.type == "user_function_call" else None
    if call is None or function is None:
        return None
    return ".".join([*path, (function.text or b"").decode()]), call


def pattern(condition: Node) -> tuple[str, Node | None] | None:
    """For a match arm's condition, ``Employee(e)``: the object it names (as written), and the identifier it binds.

    ``Employee(_)`` and ``Employee()`` bind nothing. ``None`` for a condition of
    another shape; whether it names an object of the union is the caller's to check.
    """
    found = _call(condition)
    if found is None:
        return None
    written, call = found
    arguments = [child for child in call.named_children if child.type in {"expression", "named_argument"}]
    if len(arguments) > 1 or any(argument.type == "named_argument" for argument in arguments):
        return None
    inner = arguments[0] if arguments else None
    while inner is not None and inner.type in {"expression", "parenthesized_expression"} and inner.named_children:
        inner = inner.named_children[0]
    if inner is None or inner.type == "default_var":
        return written, None
    return (written, inner) if inner.type == "identifier" else None


__all__ = ["TAG", "Union", "covers", "is_union", "is_variant", "pattern", "slot", "union", "union_of", "variant_type"]
