"""``std:fs``: reading and writing whole files, as Rust's ``std::fs`` does.

Imported like any standard module (``let fs = import("std:fs")``), but built
into the compiler: its functions are written for each program that imports
it, and call ``std/fs_runtime.c``, which the program then links.

- ``fs.read_to_string(path)``: the file's text, as a ``Result[mem.Unique[str],
  mem.Unique[str]]``: ``ok(text)``, or ``err("index.html: No such file or
  directory")``. Text is UTF-8 without NUL bytes; other files are an error.
- ``fs.write(path, text)``: write ``text`` to the file, replacing what it held
  (or creating it), as a ``Result[null, mem.Unique[str]]``.
- ``fs.exists(path)``: whether ``path`` names a file or folder.

Reading blocks: in a server, read files once at start (into its state) rather
than in a handler, which would hold up every other request meanwhile.
"""

import linecache
from collections.abc import Callable
from types import CodeType, FunctionType
from typing import Any

from mlir_python.lang import Program, cstr, ptr

from bifrost.owned import OWNED_STRING, Drops
from bifrost.results import result_of
from bifrost.std.fmt import Builtin

READ_TO_STRING = Builtin(
    "read_to_string",
    "(path: str) => Result[mem.Unique[str], mem.Unique[str]]",
    "the file's text, or why it cannot be read (`index.html: No such file or directory`)",
    module="fs",
)

WRITE = Builtin(
    "write",
    "(path: str, text: str) => Result[null, mem.Unique[str]]",
    "write text to the file, replacing what it held (or creating it), or say why it cannot",
    module="fs",
)

EXISTS = Builtin("exists", "(path: str) => bool", "whether path names a file or folder", module="fs")

FUNCTIONS = {builtin.name: builtin for builtin in (READ_TO_STRING, WRITE, EXISTS)}

# The runtime's functions: name -> (parameters, result), as Python annotations.
_RUNTIME: dict[str, tuple[list[object], object]] = {
    "read": ([cstr], ptr),
    "failed": ([ptr], bool),
    "text": ([ptr], cstr),
    "error": ([ptr], cstr),
    "done": ([ptr], None),
    "write": ([cstr, cstr], cstr),
    "is_null": ([cstr], bool),
    "exists": ([cstr], bool),
}


class Files:
    """``std:fs``'s functions, compiled into a program the first time a file imports it."""

    def __init__(self, program: Program, drops: Drops, owned_signatures: dict[str, tuple[bool, set[int]]]) -> None:
        self.program = program
        self.drops = drops
        self.owned_signatures = owned_signatures
        self.used = False  # whether a file imports std:fs: then the runtime is linked in
        self._functions: dict[str, Any] = {}

    def functions(self) -> dict[str, Any]:
        """Return the module's functions by name (compiling them the first time)."""
        if not self._functions:
            self.used = True
            self._functions = {
                "read_to_string": self._read_to_string(),
                "write": self._write(),
                "exists": self._runtime("exists"),
            }
        return self._functions

    def _read_to_string(self) -> Any:  # noqa: ANN401 - a compiled function
        result = result_of(OWNED_STRING, OWNED_STRING)
        names: dict[str, object] = {
            "Result": result,
            "read": self._runtime("read"),
            "failed": self._runtime("failed"),
            "take_text": self._runtime("text"),
            "take_error": self._runtime("error"),
            "done": self._runtime("done"),
            "zero": self.drops.zero(OWNED_STRING),
        }
        lines = [
            "def {name}(path):",
            "    p = read(path)",
            "    if failed(p):",
            "        error = take_error(p)",
            "        done(p)",
            "        return Result(ok=False, value=zero(), error=error)",
            "    text = take_text(p)",
            "    done(p)",
            "    return Result(ok=True, value=text, error=zero())",
        ]
        return self._function("read_to_string", lines, names, {"path": cstr, "return": result})

    def _write(self) -> Any:  # noqa: ANN401 - a compiled function
        result = result_of(None, OWNED_STRING)
        names: dict[str, object] = {
            "Result": result,
            "write": self._runtime("write"),
            "is_null": self._runtime("is_null"),
            "zero": self.drops.zero(OWNED_STRING),
        }
        lines = [
            "def {name}(path, text):",
            "    error = write(path, text)",
            "    if is_null(error):",
            "        return Result(ok=True, error=zero())",
            "    return Result(ok=False, error=error)",
        ]
        return self._function("write", lines, names, {"path": cstr, "text": cstr, "return": result})

    def _runtime(self, name: str) -> Any:  # noqa: ANN401 - a C function
        """Declare the runtime's ``bifrost_fs_<name>``."""
        parameters, result = _RUNTIME[name]
        arguments = [f"a{index}" for index in range(len(parameters))]
        stub: Callable[..., Any] = _compiled(
            f"bifrost_fs_{name}", [f"def {{name}}({', '.join(arguments)}):", "    ..."]
        )
        stub.__annotations__ = dict(zip(arguments, parameters, strict=True)) | {"return": result}
        return self.program.extern(stub, name=f"bifrost_fs_{name}")

    def _function(  # noqa: ANN202 - a compiled function
        self, name: str, lines: list[str], names: dict[str, object], annotations: dict[str, object]
    ):
        """Compile one of the module's functions, whose result its caller owns."""
        symbol = f"bifrost_std_fs_{name}"
        python: Any = _compiled(symbol, lines, names)
        python.__annotations__ = annotations
        self.owned_signatures[symbol] = (True, set())
        return self.program.function(python)


def _compiled(name: str, lines: list[str], names: dict[str, object] | None = None) -> Callable[..., Any]:
    """Return the Python function generated source defines (named ``name``), for the program to compile."""
    source = "\n".join(lines).format(name=name) + "\n"
    filename = f"<bifrost std:fs {name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    code = compile(source, filename, "exec")
    function_code = next(constant for constant in code.co_consts if isinstance(constant, CodeType))
    return FunctionType(function_code, names or {}, name)


__all__ = ["EXISTS", "FUNCTIONS", "READ_TO_STRING", "WRITE", "Files"]
