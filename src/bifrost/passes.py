"""Bifrost's passes over the linked program, in the ``bifrost`` dialect (``bifrost.dialect``).

Every project's program runs ``PIPELINE`` (see ``Project``): mlir-python runs a
module's passes on the program it links, every imported module's functions
included, in structured control flow, before buffer deallocation and codegen.

The cell passes, in order:

- ``RaiseCells`` rewrites the calls lowering writes into the ``mem.Shared`` and
  ``mem.Atomic`` runtime (``bifrost_shared_retain(c)``, ...) as ``bifrost``
  operations (``bifrost.retain %c {kind = "shared"}``, ...).
- ``OptimizeCells`` removes pairs that cancel out, where that is provably
  invisible (see ``OptimizeCells``).
- ``LowerCells`` rewrites what is left back into the same calls, and checks no
  ``bifrost`` operation is left for codegen.

And, on each function, ``ReassociateSums`` lets a float ``xs.sum()`` add its
items in any order, so its loop vectorizes.

First of all, ``LowerMath`` makes each call of a ``std:math`` placeholder
(``bifrost.math_sqrt_f64(x)``) the operation it stands for (``math.sqrt %x``),
which then counts as inert between cell operations.
"""

import dataclasses
import re

import mlir_python as ir
from mlir_python import passes
from mlir_python.dialects import arith, func, math

from bifrost.dialect import CELL_OPS, KINDS, CellOp, Lock, Release, Retain, Unlock
from bifrost.list_methods import SUMMED
from bifrost.std import math as std_math

_ACTIONS = {op.ACTION: op for op in CELL_OPS}
_RUNTIME_CALL = re.compile(rf"bifrost_({'|'.join(KINDS)})_({'|'.join(_ACTIONS)})")

# Operations that do nothing a cell's owners or guard could see: no call, no synchronization, no
# control flow. Loads and stores are listed separately: they must not be atomic or volatile.
_INERT_DIALECTS = ("arith.", "math.")
_INERT = {
    "llvm.mlir.constant",
    "llvm.mlir.addressof",
    "llvm.mlir.undef",
    "llvm.mlir.poison",
    "llvm.mlir.zero",
    "llvm.getelementptr",
    "llvm.extractvalue",
    "llvm.insertvalue",
    "llvm.alloca",
    "llvm.bitcast",
    "llvm.ptrtoint",
    "llvm.inttoptr",
    "llvm.zext",
    "llvm.sext",
    "llvm.trunc",
    "memref.load",
    "memref.store",
    "memref.alloca",
}
_PLAIN_MEMORY = {"llvm.load", "llvm.store"}  # inert unless atomic (ordering != 0) or volatile


def _kind(op: CellOp) -> str:
    kind = op.kind
    assert isinstance(kind, ir.StringAttr)
    return kind.value


@dataclasses.dataclass(frozen=True, kw_only=True)
class RaiseCells(passes.PythonPass):
    """Rewrites calls to the mem.Shared and mem.Atomic runtime as bifrost cell operations."""

    def run(self, op: ir.Operation) -> None:
        """Raise every runtime call nested in ``op``."""
        calls: list[ir.Operation] = []
        op.walk(lambda inner: calls.append(inner) if _runtime_call(inner) is not None else None)
        for call in calls:
            found = _runtime_call(call)
            assert found is not None
            kind, cell_op = found
            results = {cell_op.RESULT: call.results[0].type} if cell_op.RESULT is not None else {}
            operands = list(call.operands)
            if cell_op is Lock and len(operands) == 1:
                operands.append(None)  # a mem.Atomic lock has no message
            with ir.InsertionPoint.before(call), call.location:
                raised = cell_op.create(*operands, kind=ir.StringAttr(kind), **results)
            for old, new in zip(call.results, raised.operation.results, strict=True):
                old.replace_all_uses_with(new)
            call.erase()


def _runtime_call(op: ir.Operation) -> tuple[str, type[CellOp]] | None:
    """``("shared", Retain)`` for a call to ``bifrost_shared_retain``, ...; ``None`` for anything else."""
    if op.name != "func.call":
        return None
    match = _RUNTIME_CALL.fullmatch(op.attributes["callee"].value)
    return (match[1], _ACTIONS[match[2]]) if match is not None else None


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptimizeCells(passes.PythonPass):
    """Removes cell retains and releases, and unlocks and locks, that cancel out.

    Two pairs, each on the same cell value, of the same kind, in the same block:

    - ``bifrost.retain %c`` then ``bifrost.release %c``: the owner that retained
      ``%c`` already holds it, so it outlives the pair, and the count is the same
      after. Removed only if nothing between could see the count or drop
      another owner (any call, any other cell operation but a retain, anything
      that synchronizes or branches): then no owner can be the last in between,
      in this thread or (for a ``mem.Atomic``) another one that waits for it.
    - ``bifrost.unlock %c`` then ``bifrost.lock %c``: the guard is held across
      what is between instead of given back and taken again. A ``mem.Shared`` is
      used by one thread, and its lock only stops the program if the cell is
      held, which it cannot be right after its unlock; so the lock and its
      message never fire. For a ``mem.Atomic``, holding the mutex over the gap
      only rules out one interleaving (another thread taking the mutex in it),
      which the program cannot rely on; as nothing between waits, calls, or
      synchronizes, no deadlock or new wait is added. Removed only if nothing
      but inert operations is between. A loop's lock comes before its unlock
      in its body's block, so a polling loop still gives the mutex back at the
      end of each iteration, and another thread can take it.

    Operations between are only "inert" ones (``_INERT``, and plain loads and
    stores); the pass repeats until nothing more is removed, so pairs around
    removed pairs go too.
    """

    def run(self, op: ir.Operation) -> None:
        """Remove the pairs in every block nested in ``op``."""
        blocks: list[ir.Block] = []

        def visit(inner: ir.Operation) -> None:
            blocks.extend(block for region in inner.regions for block in region.blocks)

        op.walk(visit)
        for block in blocks:
            while pairs := _cancelling(block):
                for first, second in pairs:
                    first.operation.erase()
                    second.operation.erase()


def _cancelling(block: ir.Block) -> list[tuple[CellOp, CellOp]]:
    """Find the retain/release and unlock/lock pairs in ``block`` that cancel out (see ``OptimizeCells``)."""
    pairs: list[tuple[CellOp, CellOp]] = []
    retained: dict[tuple[ir.Value, str], list[Retain]] = {}  # cell -> its retains since the last barrier
    unlocked: dict[tuple[ir.Value, str], Unlock] = {}  # cell -> its unlock, if nothing but inert ops since

    for operation in block.operations:
        name = operation.name
        if name == Retain.NAME:
            retain = Retain(operation)
            retained.setdefault((retain.cell, _kind(retain)), []).append(retain)
            unlocked.clear()
        elif name == Release.NAME:
            release = Release(operation)
            open_retains = retained.get((release.cell, _kind(release)))
            if open_retains:
                pairs.append((open_retains.pop(), release))
            else:
                retained.clear()  # it may free its cell, or another it aliases
            unlocked.clear()
        elif name == Unlock.NAME:
            unlock = Unlock(operation)
            retained.clear()
            unlocked = {(unlock.cell, _kind(unlock)): unlock}
        elif name == Lock.NAME:
            lock = Lock(operation)
            unlock = unlocked.get((lock.cell, _kind(lock)))
            if unlock is not None:
                pairs.append((unlock, lock))
            retained.clear()
            unlocked.clear()
        elif not _inert(operation):
            retained.clear()
            unlocked.clear()
    return pairs


def _inert(op: ir.Operation) -> bool:
    """Whether ``op`` does nothing a cell could see (see ``_INERT``)."""
    if op.regions:
        return False
    name = op.name
    if name in _PLAIN_MEMORY:
        attributes = op.attributes
        ordering = attributes.get("ordering")
        return "volatile_" not in attributes and (ordering is None or str(ordering) == "0 : i64")
    return name in _INERT or name.startswith(_INERT_DIALECTS)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LowerCells(passes.PythonPass):
    """Rewrites bifrost cell operations back into calls to the mem.Shared and mem.Atomic runtime."""

    def run(self, op: ir.Operation) -> None:
        """Lower every cell operation nested in ``op``.

        Raises:
            RuntimeError: If a ``bifrost`` operation is left.

        """
        for cell_op in CELL_OPS:
            for found in cell_op.all(op):
                operation = found.operation
                operands = list(operation.operands)
                callee = f"bifrost_{_kind(found)}_{cell_op.ACTION}"
                with ir.InsertionPoint.before(operation), operation.location:
                    call = func.CallOp(callee, operands, [result.type for result in operation.results])
                found.replace_with(list(call.results))
        left: list[str] = []
        op.walk(lambda inner: left.append(inner.name) if inner.name.startswith("bifrost.") else None)
        if left:
            msg = f"bifrost operations left after lowering: {', '.join(sorted(set(left)))}"
            raise RuntimeError(msg)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReassociateSums(passes.PythonPass):
    """Marks the additions of a generated float ``sum`` as reassociable (``fastmath<reassoc>``).

    Adding left to right, each addition waits for the one before, so LLVM keeps
    the loop scalar; free to reorder them, it adds several items at once, in
    vector lanes, and the lanes at the end. The total can then differ from a
    left-to-right one in the last bits. A ``sum`` function (named ``SUMMED``
    and a count, by ``list_methods``) adds only its items to its total, so each
    of its additions is marked; other functions keep their order.
    """

    ANCHOR = func.FuncOp

    def run(self, op: ir.Operation) -> None:
        """Mark the float additions of ``op``, if it is a sum function."""
        assert isinstance(op, func.FuncOp)
        if not op.sym_name.startswith(f"{SUMMED}_"):  # never one of the program's own (see SUMMED)
            return
        added: list[arith.AddFOp] = []
        op.walk(lambda inner: added.append(inner) if isinstance(inner, arith.AddFOp) else None)
        for addition in added:
            addition.fastmath |= arith.FastMathFlags.REASSOC


# Each std:math function's operation: (on floats, on signed integers, on unsigned integers).
_MATH_OPS: dict[str, tuple[type[ir.Operation], type[ir.Operation] | None, type[ir.Operation] | None]] = {
    "sqrt": (math.SqrtOp, None, None),
    "floor": (math.FloorOp, None, None),
    "ceil": (math.CeilOp, None, None),
    "round": (math.RoundOp, None, None),
    "trunc": (math.TruncOp, None, None),
    "abs": (math.AbsFOp, math.AbsIOp, None),
    "sin": (math.SinOp, None, None),
    "cos": (math.CosOp, None, None),
    "tan": (math.TanOp, None, None),
    "asin": (math.AsinOp, None, None),
    "acos": (math.AcosOp, None, None),
    "atan": (math.AtanOp, None, None),
    "atan2": (math.Atan2Op, None, None),
    "tanh": (math.TanhOp, None, None),
    "exp": (math.ExpOp, None, None),
    "exp2": (math.Exp2Op, None, None),
    "log": (math.LogOp, None, None),
    "log2": (math.Log2Op, None, None),
    "log10": (math.Log10Op, None, None),
    "pow": (math.PowFOp, None, None),
    "fma": (math.FmaOp, None, None),
    "min": (arith.MinNumFOp, arith.MinSIOp, arith.MinUIOp),  # minnum: a NaN gives the other, as C's fmin
    "max": (arith.MaxNumFOp, arith.MaxSIOp, arith.MaxUIOp),
}
_MATH_CALL = re.compile(rf"{re.escape(std_math.PREFIX)}({'|'.join(_MATH_OPS)})_([fiu])(8|16|32|64)")


@dataclasses.dataclass(frozen=True, kw_only=True)
class LowerMath(passes.PythonPass):
    """Rewrites calls of ``std:math``'s placeholders as math (and arith) operations, and drops their declarations.

    ``bifrost.math_sqrt_f32(x)`` is ``math.sqrt %x : f32``; ``bifrost.math_min_u64(a, b)``
    is ``arith.minui``. LLVM gets its intrinsics (``llvm.sqrt.f32``) instead of a call it
    cannot see into.
    """

    def run(self, op: ir.Operation) -> None:
        """Lower every placeholder call nested in ``op``; erase the declarations left unused.

        Raises:
            RuntimeError: If a placeholder is still called.

        """
        calls: list[ir.Operation] = []
        declarations: list[ir.Operation] = []

        def visit(inner: ir.Operation) -> None:
            if _calls_placeholder(inner):
                calls.append(inner)
            elif inner.name == "llvm.func" and inner.attributes["sym_name"].value.startswith(std_math.PREFIX):
                declarations.append(inner)

        op.walk(visit)
        for call in calls:
            match = _MATH_CALL.fullmatch(call.attributes["callee"].value)
            if match is None:
                continue  # reported below
            on_float, on_signed, on_unsigned = _MATH_OPS[match[1]]
            lowered = {"f": on_float, "i": on_signed, "u": on_unsigned}[match[2]]
            assert lowered is not None, f"{match[0]} has no operation (lowering checks what each function takes)"
            with ir.InsertionPoint.before(call), call.location:
                operation = lowered(*call.operands)
            call.results[0].replace_all_uses_with(operation.result)
            call.erase()
        left: list[str] = []
        op.walk(lambda inner: left.append(inner.attributes["callee"].value) if _calls_placeholder(inner) else None)
        if left:
            msg = f"std:math placeholders still called after lowering: {', '.join(sorted(set(left)))}"
            raise RuntimeError(msg)
        for declaration in declarations:
            declaration.erase()


def _calls_placeholder(op: ir.Operation) -> bool:
    """Whether ``op`` calls a ``std:math`` placeholder (declared as a C function, so with ``llvm.call``)."""
    return (
        op.name == "llvm.call"
        and "callee" in op.attributes
        and op.attributes["callee"].value.startswith(std_math.PREFIX)
    )


PIPELINE: tuple[passes.PipelineElement, ...] = (
    LowerMath(),
    RaiseCells(),
    OptimizeCells(),
    LowerCells(),
    ir.Nested(func.FuncOp, [ReassociateSums()]),
)
"""What every project's program runs on the program it links."""
