"""``std:fmt``: formatting text into owned strings.

Imported like any standard module (``let fmt = import("std:fmt")``), but built
into the compiler.

- ``fmt.format(pattern, ...)``: formats like C's ``printf`` (``%s``, ``%d``,
  ``%.2f``, ...) into a string allocated to fit, and returns it as a
  ``mem.Unique[str]``: the caller owns it, and it is freed when its owner's
  scope ends, unless ownership moves on (returned, or passed to a
  ``mem.Unique[str]`` parameter).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Builtin:
    """A function the compiler expands itself, as ``fmt.format`` resolves in code."""

    name: str
    signature: str
    summary: str
    module: str = "fmt"  # the standard module it belongs to: std:fmt, std:json

    def __repr__(self) -> str:
        """Show it as Bifrost code names it: ``fmt.format``."""
        return f"{self.module}.{self.name}"


FORMAT = Builtin(
    "format",
    "(pattern: str, ...) => mem.Unique[str]",
    "formats like printf (%s, %d, %.2f) into an owned string, freed when its owner's scope ends",
)

FUNCTIONS = {builtin.name: builtin for builtin in (FORMAT,)}

__all__ = ["FORMAT", "FUNCTIONS", "Builtin"]
