"""Bifrost's passes over the linked program (``bifrost.passes``), on MLIR written by hand.

The programs call the ``mem.Shared`` and ``mem.Atomic`` runtime as lowering
does. Each runs through ``RaiseCells``, ``OptimizeCells`` and ``LowerCells``;
the tests count the calls left. Whole programs are in ``test_cells``.
``ReassociateSums`` runs on functions shaped like the generated ``sum``.
"""

import itertools

import mlir_python as ir
import pytest
from mlir_python.dialects import arith

from bifrost.dialect import Lock, Retain
from bifrost.list_methods import SUMMED
from bifrost.passes import PIPELINE, LowerCells, OptimizeCells, RaiseCells

RUNTIME = """
func.func private @bifrost_shared_new(i64) -> !llvm.ptr
func.func private @bifrost_shared_retain(!llvm.ptr)
func.func private @bifrost_shared_release(!llvm.ptr)
func.func private @bifrost_shared_unref(!llvm.ptr) -> i64
func.func private @bifrost_shared_destroy(!llvm.ptr)
func.func private @bifrost_shared_lock(!llvm.ptr, !llvm.ptr)
func.func private @bifrost_shared_unlock(!llvm.ptr)
func.func private @bifrost_atomic_retain(!llvm.ptr)
func.func private @bifrost_atomic_release(!llvm.ptr)
func.func private @bifrost_atomic_lock(!llvm.ptr)
func.func private @bifrost_atomic_unlock(!llvm.ptr)
func.func private @use(!llvm.ptr)
llvm.mlir.global internal constant @message("locked\\00") {addr_space = 0 : i32}
"""


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx, ir.Location.unknown():
        yield ctx


def _run(body: str, pipeline: tuple[object, ...] = PIPELINE) -> str:
    """Run ``pipeline`` on ``f(%a, %b)``, whose body is ``body``; return the module's text."""
    module = ir.Module.parse(f"{RUNTIME}\nfunc.func @f(%a: !llvm.ptr, %b: !llvm.ptr) {{\n{body}\n  return\n}}")
    ir.PassManager(ir.Module, list(pipeline)).run(module)
    module.verify()
    return str(module)


def _calls(text: str, action: str) -> int:
    return text.count(f"call @bifrost_shared_{action}(") + text.count(f"call @bifrost_atomic_{action}(")


_MESSAGES = itertools.count()  # each lock's message gets a name of its own


def _retain(cell: str, kind: str = "shared") -> str:
    return f"  func.call @bifrost_{kind}_retain({cell}) : (!llvm.ptr) -> ()"


def _release(cell: str, kind: str = "shared") -> str:
    return f"  func.call @bifrost_{kind}_release({cell}) : (!llvm.ptr) -> ()"


def _unlock(cell: str, kind: str = "shared") -> str:
    return f"  func.call @bifrost_{kind}_unlock({cell}) : (!llvm.ptr) -> ()"


def _lock(cell: str, kind: str = "shared") -> str:
    if kind == "atomic":
        return f"  func.call @bifrost_atomic_lock({cell}) : (!llvm.ptr) -> ()"
    message = f"%m{next(_MESSAGES)}"
    address = f"  {message} = llvm.mlir.addressof @message : !llvm.ptr\n"
    return address + f"  func.call @bifrost_shared_lock({cell}, {message}) : (!llvm.ptr, !llvm.ptr) -> ()"


def _use(cell: str) -> str:
    return f"  func.call @use({cell}) : (!llvm.ptr) -> ()"


INERT = "  %one = arith.constant 1 : i64\n  %v = llvm.load %a : !llvm.ptr -> i64\n  %w = arith.addi %v, %one : i64"


def test_calls_become_bifrost_operations_and_back() -> None:
    body = "\n".join(
        [
            "  %size = arith.constant 8 : i64",
            "  %c = func.call @bifrost_shared_new(%size) : (i64) -> !llvm.ptr",
            _retain("%c"),
            _use("%c"),
            _lock("%c"),
            _unlock("%c"),
            _lock("%b", "atomic"),
            _unlock("%b", "atomic"),
            "  %left = func.call @bifrost_shared_unref(%c) : (!llvm.ptr) -> i64",
            "  func.call @bifrost_shared_destroy(%c) : (!llvm.ptr) -> ()",
        ]
    )
    raised = _run(body, (RaiseCells(),))
    assert "call @bifrost_" not in raised
    for op in ("new", "retain", "lock", "unlock", "unref", "destroy"):
        assert f'"bifrost.{op}"' in raised
    assert 'kind = "atomic"' in raised
    assert 'kind = "shared"' in raised
    lowered = _run(body, (RaiseCells(), LowerCells()))
    assert '"bifrost.' not in lowered
    assert lowered == _run(body, ())  # the same calls, in the same places


def test_a_retain_then_a_release_of_the_same_cell_cancel_out() -> None:
    text = _run("\n".join([_retain("%a"), INERT, _release("%a")]))
    assert _calls(text, "retain") == _calls(text, "release") == 0
    text = _run("\n".join([_retain("%a", "atomic"), INERT, _release("%a", "atomic")]))
    assert _calls(text, "retain") == _calls(text, "release") == 0


def test_nested_pairs_cancel_out() -> None:
    text = _run("\n".join([_retain("%a"), _retain("%b"), _release("%b"), _release("%a")]))
    assert _calls(text, "retain") == _calls(text, "release") == 0


@pytest.mark.parametrize(
    ("between", "why"),
    [
        (_use("%a"), "a call could drop another owner, or read the count"),
        (_release("%b"), "%b may be the same cell: it could free it"),
        ("  %left = func.call @bifrost_shared_unref(%b) : (!llvm.ptr) -> i64", "an unref reads the count"),
        (_unlock("%b", "atomic"), "an unlock lets another thread go on, and drop its owner"),
        (_lock("%b"), "a lock may wait"),
        ("  %v = llvm.load %a atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64", "an atomic load synchronizes"),
        (
            "  %true = arith.constant true\n  scf.if %true {\n" + _use("%a") + "\n  }",
            "a branch, with a call in it",
        ),
    ],
)
def test_a_retain_and_a_release_with_anything_visible_between_are_kept(between: str, why: str) -> None:
    text = _run("\n".join([_retain("%a"), between, _release("%a")]))
    assert _calls(text, "retain") == 1, why
    assert _calls(text, "release") >= 1, why


def test_only_the_same_cell_and_order_cancel_out() -> None:
    text = _run("\n".join([_retain("%a"), _release("%b")]))  # %b may alias %a, but that is not provable
    assert _calls(text, "retain") == _calls(text, "release") == 1
    text = _run("\n".join([_release("%a"), _retain("%a")]))  # the release may free %a
    assert _calls(text, "retain") == _calls(text, "release") == 1


def test_a_retain_and_release_in_other_blocks_are_kept() -> None:
    body = "\n".join(
        [_retain("%a"), "  %true = arith.constant true", "  scf.if %true {", _release("%a"), "  }", _release("%a")]
    )
    text = _run(body)
    assert _calls(text, "retain") == 1
    assert _calls(text, "release") == 2


@pytest.mark.parametrize("kind", ["shared", "atomic"])
def test_an_unlock_then_a_lock_of_the_same_cell_merge(kind: str) -> None:
    body = "\n".join(
        [_lock("%a", kind), _use("%a"), _unlock("%a", kind), INERT, _lock("%a", kind), _unlock("%a", kind)]
    )
    text = _run(body)
    assert _calls(text, "lock") == _calls(text, "unlock") == 1
    f = text[text.index("func.func @f") :]
    assert f.index(f"{kind}_lock(%arg0") < f.index("call @use") < f.index(f"{kind}_unlock(%arg0")


@pytest.mark.parametrize(
    "between",
    [
        _use("%a"),  # it may lock %a itself (through another owner): merged, that would stop the program
        _retain("%a"),
        _lock("%b"),
        _unlock("%b"),
    ],
)
def test_an_unlock_and_a_lock_with_anything_but_inert_operations_between_are_kept(between: str) -> None:
    text = _run("\n".join([_lock("%a"), _unlock("%a"), between, _lock("%a"), _unlock("%a")]))
    assert _calls(text, "lock") >= 2


def test_a_lock_then_an_unlock_are_kept() -> None:
    # An empty guard still stops the program if a mem.Shared is held, and synchronizes a mem.Atomic.
    text = _run("\n".join([_lock("%a"), _unlock("%a"), _lock("%b", "atomic"), _unlock("%b", "atomic")]))
    assert _calls(text, "lock") == _calls(text, "unlock") == 2


def test_merging_locks_lets_retains_cancel_out() -> None:
    text = _run("\n".join([_lock("%b"), _retain("%a"), _unlock("%b"), _lock("%b"), _release("%a"), _unlock("%b")]))
    assert _calls(text, "retain") == _calls(text, "release") == 0
    assert _calls(text, "lock") == _calls(text, "unlock") == 1


def test_operations_are_verified() -> None:
    module = ir.Module.parse(f"{RUNTIME}\nfunc.func @f(%a: !llvm.ptr) {{\n  return\n}}")
    function = next(iter(module.body.operations[-1].regions[0].blocks))
    (cell,) = function.arguments
    with ir.InsertionPoint.at_block_begin(function):
        Retain.create(cell, kind=ir.StringAttr("shared"))
        Lock.create(cell, None, kind=ir.StringAttr("atomic"))
    module.verify()
    with ir.InsertionPoint.at_block_begin(function):
        Retain.create(cell, kind=ir.StringAttr("unique"))
    with pytest.raises(ir.MLIRError, match=r"bifrost\.retain"):
        module.verify()


def test_optimize_alone_leaves_calls_as_they_are() -> None:
    body = "\n".join([_retain("%a"), _release("%a")])
    assert _run(body, (OptimizeCells(),)) == _run(body, ())


def _adding(name: str, kind: str) -> str:
    """Write a function ``name`` adding up ``n`` items of type ``kind`` at ``%items``, as the generated sum does."""
    add = "arith.addf" if kind.startswith("f") else "arith.addi"
    zero = "0.0" if kind.startswith("f") else "0"
    return f"""
func.func @{name}(%items: !llvm.ptr, %n: i64) -> {kind} {{
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %zero = arith.constant {zero} : {kind}
  %total = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> ({kind}) : i64 {{
    %at = llvm.getelementptr %items[%i] : (!llvm.ptr, i64) -> !llvm.ptr, {kind}
    %item = llvm.load %at : !llvm.ptr -> {kind}
    %next = {add} %acc, %item : {kind}
    scf.yield %next : {kind}
  }}
  return %total : {kind}
}}"""


def test_a_float_sum_adds_in_any_order() -> None:
    # Only the generated sum functions' additions are marked: a function of the program's own adds
    # left to right, as written; an integer sum needs nothing (its order changes nothing).
    sums = [_adding(f"{SUMMED}_1", "f32"), _adding(f"{SUMMED}_2", "f64"), _adding(f"{SUMMED}_3", "i64")]
    module = ir.Module.parse("\n".join([*sums, _adding("total", "f32")]))
    ir.PassManager(ir.Module, list(PIPELINE)).run(module)
    module.verify()
    marked = {
        function.sym_name: [op.fastmath for op in _ops(function, arith.AddFOp)] for function in module.body.operations
    }
    reassoc, none = arith.FastMathFlags.REASSOC, arith.FastMathFlags.NONE
    assert marked == {f"{SUMMED}_1": [reassoc], f"{SUMMED}_2": [reassoc], f"{SUMMED}_3": [], "total": [none]}
    assert "fastmath<reassoc>" in str(module)


def _ops(op: ir.Operation, kind: type[ir.Operation]) -> list[ir.Operation]:
    found: list[ir.Operation] = []
    op.walk(lambda inner: found.append(inner) if isinstance(inner, kind) else None)
    return found
