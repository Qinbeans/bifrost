"""The ``bifrost`` dialect: Bifrost's own MLIR operations, for passes over the linked program.

Lowering writes plain MLIR (calls into the runtimes), and that is what the
program keeps. The dialect exists between: ``bifrost.passes`` raises the calls
it knows into these operations, rewrites them where that is safe, and lowers
them back before codegen, which never sees one. It is defined at run time,
through IRDL (``mlir_python.irdl``), so its operations have no traits and no
memory effects: a pass reasons about them itself.

The operations so far are those of ``mem.Shared`` and ``mem.Atomic`` cells
(``bifrost.std.mem_runtime``), one per runtime function. Each says which kind
of cell it works on, ``kind = "shared"`` or ``"atomic"``::

    %c = bifrost.new %size {kind = "shared"}          bifrost_shared_new(size)
    bifrost.retain %c {kind = "shared"}               bifrost_shared_retain(c)
    bifrost.release %c {kind = "shared"}              bifrost_shared_release(c)
    %left = bifrost.unref %c {kind = "shared"}        bifrost_shared_unref(c)
    bifrost.destroy %c {kind = "shared"}              bifrost_shared_destroy(c)
    bifrost.lock %c, %message {kind = "shared"}       bifrost_shared_lock(c, message)
    bifrost.lock %c {kind = "atomic"}                 bifrost_atomic_lock(c)
    bifrost.unlock %c {kind = "shared"}               bifrost_shared_unlock(c)

A cell is the ``!llvm.ptr`` to its value; only a ``mem.Shared`` lock takes a
message (what to print when the cell is already locked).
"""

from typing import ClassVar

import mlir_python as ir
from mlir_python import irdl
from mlir_python.dialects import llvm

KINDS = ("shared", "atomic")  # the cells: mem.Shared and mem.Atomic

bifrost = irdl.Dialect("bifrost")


def _pointer() -> ir.Type:
    return llvm.PointerType()


def _kind(name: str) -> irdl.Is:
    return irdl.Is(lambda: ir.StringAttr(name))


class CellOp(irdl.Op):
    """An operation on a ``mem.Shared`` or ``mem.Atomic`` cell: the runtime's ``bifrost_<kind>_<ACTION>``."""

    ACTION: ClassVar[str] = ""
    RESULT: ClassVar[str | None] = None  # the name of its result, if the runtime function returns one
    kind = irdl.Attr(irdl.AnyOf(*(_kind(kind) for kind in KINDS)))


@bifrost.operation
class New(CellOp):
    """A new cell for a ``size``-byte value, with one owner."""

    ACTION = "new"
    RESULT = "cell"
    size = irdl.Operand(irdl.Is(lambda: ir.IntegerType(64)))
    cell = irdl.Result(irdl.Is(_pointer))


@bifrost.operation
class Retain(CellOp):
    """Count another owner of ``cell``."""

    ACTION = "retain"
    cell = irdl.Operand(irdl.Is(_pointer))


@bifrost.operation
class Release(CellOp):
    """Drop an owner of ``cell``; the last frees it."""

    ACTION = "release"
    cell = irdl.Operand(irdl.Is(_pointer))


@bifrost.operation
class Unref(CellOp):
    """Drop an owner of ``cell`` without freeing it; ``left`` is how many are left."""

    ACTION = "unref"
    RESULT = "left"
    cell = irdl.Operand(irdl.Is(_pointer))
    left = irdl.Result(irdl.Is(lambda: ir.IntegerType(64)))


@bifrost.operation
class Destroy(CellOp):
    """Free ``cell`` (after ``unref`` left no owner)."""

    ACTION = "destroy"
    cell = irdl.Operand(irdl.Is(_pointer))


@bifrost.operation
class Lock(CellOp):
    """Take ``cell``'s guard: wait for a ``mem.Atomic``'s mutex; stop with ``message`` if a ``mem.Shared`` is held."""

    ACTION = "lock"
    cell = irdl.Operand(irdl.Is(_pointer))
    message = irdl.OptionalOperand(irdl.Is(_pointer))


@bifrost.operation
class Unlock(CellOp):
    """Give ``cell``'s guard back."""

    ACTION = "unlock"
    cell = irdl.Operand(irdl.Is(_pointer))


CELL_OPS: tuple[type[CellOp], ...] = (New, Retain, Release, Unref, Destroy, Lock, Unlock)
