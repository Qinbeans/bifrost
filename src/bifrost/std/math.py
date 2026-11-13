"""``std:math``: square roots, rounding, trigonometry, exponents, and the numbers pi and e.

Imported like any standard module (``let math = import("std:math")``), but
built into the compiler: a call becomes an operation of MLIR's ``math``
dialect (``math.sqrt``, ``math.floor``, ...), or ``arith``'s for ``min`` and
``max``, so LLVM sees its intrinsics (``llvm.sqrt.f64``), which it folds,
vectorizes, or turns into one instruction, rather than an opaque call into
libm.

Each function works on ``f32`` and ``f64`` alike, in the type of its
arguments: ``math.sqrt(x)`` with an ``f32`` ``x`` is an ``f32``. Arguments
are of one type; a number written as one (``math.pow(x, 2)``), a constant
(``math.pi``), or arithmetic of them (``2 * math.pi``), takes the type the
others give, or ``f64``. An integer is an error that says to convert it
(``math.sqrt(f64(n))``), except for ``abs`` (signed integers too), ``min``
and ``max`` (any number type).

Lowering calls a placeholder per function and type, ``bifrost.math_<name>_<type>``
(``bifrost.math_sqrt_f32``), declared as a C function but never linked: the
``LowerMath`` pass (``bifrost.passes``) rewrites each call into its operation,
and drops the declarations. No function of a program can be so named: a
Bifrost name has no dot, and neither has the symbol it compiles to.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any

from mlir_python.lang import Function, Program
from mlir_python.lang.types import ScalarType

from bifrost.std.fmt import Builtin

FLOATS = "floats"  # f32 and f64
SIGNED = "signed"  # floats and signed integers
NUMBERS = "numbers"  # every number type
PREFIX = "bifrost.math_"  # the placeholders' names: bifrost.math_sqrt_f64


@dataclass(frozen=True, repr=False)  # shown as Builtin is: math.sqrt
class MathFunction(Builtin):
    """A function of ``std:math``: what it takes (``FLOATS``, ``SIGNED``, ``NUMBERS``) and its parameters' names."""

    module: str = "math"
    parameters: tuple[str, ...] = ("x",)
    takes: str = FLOATS


def _floats(name: str, summary: str, parameters: tuple[str, ...] = ("x",)) -> MathFunction:
    signature = f"({', '.join(f'{parameter}: F' for parameter in parameters)}) => F"
    return MathFunction(name, signature, f"{summary}; F is f32 or f64", parameters=parameters)


FUNCTIONS: dict[str, MathFunction] = {
    function.name: function
    for function in (
        _floats("sqrt", "the square root of x"),
        _floats("floor", "x rounded down to a whole number"),
        _floats("ceil", "x rounded up to a whole number"),
        _floats("round", "x rounded to the nearest whole number, halves away from zero"),
        _floats("trunc", "x rounded toward zero to a whole number"),
        MathFunction("abs", "(x: N) => N", "how far x is from zero; N is f32, f64 or a signed integer", takes=SIGNED),
        _floats("sin", "the sine of x radians"),
        _floats("cos", "the cosine of x radians"),
        _floats("tan", "the tangent of x radians"),
        _floats("asin", "the angle, in radians, whose sine is x"),
        _floats("acos", "the angle, in radians, whose cosine is x"),
        _floats("atan", "the angle, in radians, whose tangent is x"),
        _floats("atan2", "the angle, in radians, from the x axis to the point (x, y)", ("y", "x")),
        _floats("tanh", "the hyperbolic tangent of x"),
        _floats("exp", "e to the power x"),
        _floats("exp2", "2 to the power x"),
        _floats("log", "the natural logarithm of x"),
        _floats("log2", "the base-2 logarithm of x"),
        _floats("log10", "the base-10 logarithm of x"),
        _floats("pow", "x to the power y", ("x", "y")),
        _floats("fma", "a * b + c, rounded once", ("a", "b", "c")),
        MathFunction(
            "min",
            "(a: N, b: N) => N",
            "the smaller of a and b (a float NaN gives the other); N is any number type",
            parameters=("a", "b"),
            takes=NUMBERS,
        ),
        MathFunction(
            "max",
            "(a: N, b: N) => N",
            "the larger of a and b (a float NaN gives the other); N is any number type",
            parameters=("a", "b"),
            takes=NUMBERS,
        ),
    )
}


@dataclass(frozen=True)
class Constant:
    """A number of ``std:math``, as ``math.pi`` resolves in the editor; code reads its value, as a top-level ``let``."""

    name: str
    value: float
    summary: str

    def __repr__(self) -> str:
        """Show it as Bifrost code names it: ``math.pi``."""
        return f"math.{self.name}"


CONSTANTS: dict[str, Constant] = {
    constant.name: constant
    for constant in (
        Constant("pi", math.pi, "the ratio of a circle's circumference to its diameter, 3.14159..."),
        Constant("e", math.e, "the base of the natural logarithm, 2.71828..."),
    )
}
VALUES = {name: constant.value for name, constant in CONSTANTS.items()}  # `math.pi` in code: 3.14159...


def placeholder_name(function: MathFunction, kind: ScalarType) -> str:
    """Return the name of ``function``'s placeholder for ``kind``: ``bifrost.math_sqrt_f64``."""
    return f"{PREFIX}{function.name}_{kind.name}"


class Placeholders:
    """The placeholders calls of ``std:math`` lower to, declared once per program."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self._declared: dict[str, Function[..., Any]] = {}

    @property
    def used(self) -> bool:
        """Whether a ``std:math`` function is called: then the program links libm, where LLVM may call it."""
        return bool(self._declared)

    def get(self, function: MathFunction, kind: ScalarType, python: object) -> Function[..., Any]:
        """Return ``function``'s placeholder for arguments of ``kind`` (whose Python annotation is ``python``)."""
        name = placeholder_name(function, kind)
        if name not in self._declared:
            stub = _stub(name, len(function.parameters))
            arguments = stub.__code__.co_varnames[: stub.__code__.co_argcount]
            stub.__annotations__ = dict.fromkeys(arguments, python) | {"return": python}
            self._declared[name] = self.program.extern(stub, name=name)
        return self._declared[name]


def _stub(name: str, arity: int) -> Callable[..., None]:
    """Return a stub named ``name`` taking ``arity`` arguments, to declare a placeholder with."""
    original = {1: _one, 2: _two, 3: _three}[arity]
    return FunctionType(original.__code__, original.__globals__, name)


def _one(x): ...  # noqa: ANN001, ANN202


def _two(a, b): ...  # noqa: ANN001, ANN202


def _three(a, b, c): ...  # noqa: ANN001, ANN202


__all__ = [
    "CONSTANTS",
    "FLOATS",
    "FUNCTIONS",
    "NUMBERS",
    "PREFIX",
    "SIGNED",
    "VALUES",
    "Constant",
    "MathFunction",
    "Placeholders",
]
