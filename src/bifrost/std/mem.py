"""``std:mem``: containers that say who owns a value.

Imported like any standard module (``let mem = import("std:mem")``), but built
into the compiler: its containers are types, checked at compile time, not C
declarations.

- ``mem.Unique[T]``: the one owner of a value, held in a local
  (``let ctx: mem.Unique[Context] = Context.new()``). It is reached only
  through a guard (``let guard <- ctx``) and lent to calls, never copied.
- ``mem.Weak[T]``: a reference lent to a call, as a parameter
  (``(ctx: mem.Weak[Context])``). It is reached only through a guard, and
  cannot be returned or stored, so it never outlives its owner.
- ``mem.Shared[T]``: a value with several owners, in one thread
  (``let counter: mem.Shared[Counter] = Counter(hits: 0)``). ``let other =
  counter`` makes another owner; the value is freed when its last owner's
  scope ends. A function may return one, and take one as a parameter (which
  borrows it for the call).
- ``mem.Atomic[T]``: a ``mem.Shared`` that several threads may own: its owners
  are counted atomically, and its guard locks a mutex.

Locking one gives the guard named after it: ``let guard <- ctx`` on a
``mem.Weak[Context]`` is a ``mem.WeakGuard[Context]``. A lock may say so
(``let guard: mem.WeakGuard[Context] <- ctx``); the compiler checks it matches.

``mem.Unique`` and ``mem.Weak`` cost nothing at run time: a ``mem.Weak``
compiles to a plain pointer, and a ``mem.Unique`` local to the value itself.
``mem.Shared`` and ``mem.Atomic`` are pointers to the value on the heap, after
a count of its owners (see ``mem_runtime``). Their guards are checked at run
time too: locking a ``mem.Atomic`` waits for its mutex, and locking a
``mem.Shared`` that another owner holds stops the program. Lending one to a
``mem.Weak`` parameter locks it for the call.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Container:
    """One of ``mem``'s containers, as ``mem.Unique`` resolves in a type."""

    name: str
    summary: str
    supported: bool
    guards: str | None = None  # for a guard type, the container it locks ("Weak" for WeakGuard)

    def __repr__(self) -> str:
        """Show it as Bifrost code names it: ``mem.Unique``."""
        return f"mem.{self.name}"


UNIQUE = Container("Unique", "the one owner of a value; lent to calls, reached through a guard", supported=True)
WEAK = Container("Weak", "a reference lent to a call; cannot be returned or stored", supported=True)
SHARED = Container("Shared", "one value with several owners, freed with the last; for one thread", supported=True)
ATOMIC = Container("Atomic", "a mem.Shared for several threads: counted atomically, locked by a mutex", supported=True)

UNIQUE_GUARD = Container("UniqueGuard", "a lock on a mem.Unique: exclusive access, compile-time only", True, "Unique")
WEAK_GUARD = Container("WeakGuard", "a lock on a mem.Weak: access for the call, compile-time only", True, "Weak")
SHARED_GUARD = Container("SharedGuard", "a lock on a mem.Shared: one at a time, checked at run time", True, "Shared")
ATOMIC_GUARD = Container("AtomicGuard", "a lock on a mem.Atomic: holds its mutex until released", True, "Atomic")
CELLS = (SHARED, ATOMIC)  # the containers that are counted cells on the heap

CONTAINERS = {
    container.name: container
    for container in (UNIQUE, WEAK, SHARED, ATOMIC, UNIQUE_GUARD, WEAK_GUARD, SHARED_GUARD, ATOMIC_GUARD)
}


def guard_of(container: Container) -> Container:
    """Return the guard a lock on ``container`` gives: ``mem.Weak`` -> ``mem.WeakGuard``."""
    return CONTAINERS[f"{container.name}Guard"]


__all__ = [
    "ATOMIC",
    "ATOMIC_GUARD",
    "CELLS",
    "CONTAINERS",
    "SHARED",
    "SHARED_GUARD",
    "UNIQUE",
    "UNIQUE_GUARD",
    "WEAK",
    "WEAK_GUARD",
    "Container",
    "guard_of",
]
