"""Objects built on others: ``let Employee = struct { let employee_id: str } & Person``.

An Employee has each of Person's fields (after its own, in the order Person
declares them) and Person's methods, which see the Person part of it. And it is
a Person wherever one goes: given to a ``p: Person`` parameter it is lent as one
(its Person fields, read in place); kept as one (``let p: Person = e``, a
returned value, an object's field, a list's item) it moves into one, and its own
fields are freed. Its own functions are its own: Person's ``greet`` sees a
Person, and an Employee's ``greet`` is not called through a Person.
"""

from mlir_python.lang._types import scalar_type
from mlir_python.lang.types import StructType

# By the object's type: the objects it is built on, kept alive so that types compare by identity.
_BASES: dict[int, tuple[type, tuple[StructType, ...]]] = {}


def build_on(cls: type, bases: list[StructType]) -> None:
    """Record that the object type ``cls`` is built on ``bases``, in the order written."""
    if bases:
        _BASES[id(cls)] = (cls, tuple(bases))


def bases(kind: object) -> tuple[StructType, ...]:
    """Return the objects ``kind`` is built on directly; none for any other type."""
    found = scalar_type(kind)
    if not isinstance(found, StructType):
        return ()
    entry = _BASES.get(id(found.python))
    return entry[1] if entry is not None and entry[0] is found.python else ()


def derives(kind: object, base: object) -> bool:
    """Whether the object type ``kind`` is built on ``base``, directly or through another."""
    wanted = scalar_type(base)
    return any(found == wanted or derives(found, wanted) for found in bases(kind))


def is_base(kind: object) -> bool:
    """Whether some object type is built on ``kind`` (so a value of that type may go where a ``kind`` does)."""
    wanted = scalar_type(kind)
    return isinstance(wanted, StructType) and any(
        held == wanted or derives(held, wanted) for _, found in _BASES.values() for held in found
    )


__all__ = ["bases", "build_on", "derives", "is_base"]
