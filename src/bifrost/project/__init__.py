import ast
import inspect
import linecache
import re
import subprocess
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any

from mlir_python.lang import (
    Fn,
    Function,
    Program,
    Token,
    cstr,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    ptr,
    struct,
    u8,
    u16,
    u32,
    u64,
)

from bifrost import std
from bifrost.configs import Config
from bifrost.configs.schema import Declaration, _Extern, _Function
from bifrost.naming import to_snake_case
from bifrost.owned import Drops

# The C functions behind owned values: name -> (parameters, result).
# `(Request, i32) => None` in an extern declaration: a function type (one level; no nesting).
_FUNCTION_TYPE = re.compile(r"\(([^()]*)\)\s*=>\s*([^()]+)")

_RUNTIME = {
    "malloc": ({"size": "u64"}, "cstr"),
    "realloc": ({"pointer": "cstr", "size": "u64"}, "cstr"),
    "free": ({"pointer": "cstr"}, "None"),
}


@dataclass(frozen=True)
class Context:
    """Configured extern functions and types available to project handlers."""

    extern_table: dict[str, type | Callable[..., Any]]
    type_map: dict[str, type]


class _ContextLookupLowerer(ast.NodeTransformer):
    def __init__(self, context_name: str, project: "Project") -> None:
        self.context_name = context_name
        self.project = project
        self.globals: dict[str, Any] = {}

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if not (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == self.context_name
            and node.value.attr in {"extern_table", "type_map"}
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            return self.generic_visit(node)

        table = getattr(self.project.context, node.value.attr)
        key = node.slice.value
        if key not in table:
            msg = f"Unknown {node.value.attr} entry {key!r}"
            raise ValueError(msg)

        name = f"__bifrost_{node.value.attr}_{len(self.globals)}"
        self.globals[name] = table[key]
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)


class Project:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.program = Program()
        self.extern_table: dict[str, type | Callable[..., Any]] = {}
        self.type_map: dict[str, type] = {
            "i8": i8,
            "i16": i16,
            "i32": i32,
            "i64": i64,
            "u8": u8,
            "u16": u16,
            "u32": u32,
            "u64": u64,
            "f32": f32,
            "f64": f64,
            "cstr": cstr,
            "ptr": ptr,
            "token": Token,  # something that completes later: a function returning one pauses
            "bool": bool,
            "None": type(None),
        }
        self.context = Context(self.extern_table, self.type_map)
        self._std: dict[str, _Extern] = {}  # standard modules loaded by `import("std:...")`
        # Bifrost files loaded by `import("file:module")`, by resolved path, and the chain being loaded.
        self.units: dict[Path, Any] = {}
        self.loading: list[Path] = []
        # Compiled symbol -> (whether it returns an owned string, which parameters take one).
        self.owned_signatures: dict[str, tuple[bool, set[int]]] = {}
        # Compiled symbol -> what each parameter takes: a lent type (mem.Weak), a cell (mem.Shared/Atomic), or None.
        self.pointer_params: dict[str, list[object | None]] = {}
        # Compiled symbol -> the (container, type) of the mem.Shared or mem.Atomic it returns, if it returns one.
        self.cell_results: dict[str, tuple[Any, object] | None] = {}
        # C symbol -> the indexes of its `json` parameters (a record or object passed to one is encoded).
        self.json_parameters: dict[str, set[int]] = {}
        # Compiled symbols of the Bifrost functions that pause (async functions).
        self.pausing: set[str] = set()
        # Record fields ((name, type), ...) -> their record type, shared by every file.
        self.records: dict[tuple[tuple[str, object], ...], type] = {}
        # An object's type -> its methods (functions without `static`) -> (Bifrost name, compiled symbol).
        self.methods: dict[type, dict[str, tuple[str, str]]] = {}
        # Compiled symbols of the methods that change their object (they lock `super`): it is lent to them.
        self.changing: set[str] = set()
        # The functions freeing and copying lists, and the records and objects holding them.
        self.drops = Drops(self.program)
        # C symbol -> a function of a runtime Bifrost links in, declared when first called.
        self._runtime_externs: dict[str, Function[..., Any]] = {}
        self._init_declarations()

    def _resolve_type(self, module: str, type_name: str) -> type:
        """Resolve scalar and module-local struct types from configuration.

        ``(Request) => None`` is a function type: a C function pointer, such as a callback.
        """
        if type_name in self.type_map:
            return self.type_map[type_name]

        function = _FUNCTION_TYPE.fullmatch(type_name.strip())
        if function is not None:
            parameters = [part.strip() for part in function.group(1).split(",") if part.strip()]
            result = self._resolve_type(module, function.group(2).strip())
            return Fn[
                [self._resolve_type(module, parameter) for parameter in parameters],
                None if result is type(None) else result,
            ]

        declaration = self.extern_table.get(f"{module}_{type_name}")
        if isinstance(declaration, type):
            return declaration

        msg = f"Unknown type {type_name!r} in extern module {module!r}"
        raise ValueError(msg)

    def _create_struct(self, module: str, declaration: Declaration) -> type:
        return struct(
            type(
                declaration.name,
                (),
                {
                    "__annotations__": {
                        to_snake_case(field_name): self._resolve_type(module, field_type)
                        for field_name, field_type in declaration.fields.items()
                    }
                },
            )
        )

    def _create_extern(self, module: str, declaration: Declaration) -> Callable[..., Any]:
        parameter_names = ", ".join([*declaration.parameters, *(["*args"] if declaration.variadic else [])])
        source = f"def extern_function({parameter_names}):\n    ...\n"
        filename = f"<bifrost extern {module}.{declaration.name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        module_code = compile(source, filename, "exec")
        function_code = next(code for code in module_code.co_consts if isinstance(code, CodeType))
        extern_function = FunctionType(function_code, {})

        extern_function.__name__ = declaration.name
        extern_function.__annotations__ = {
            parameter_name: cstr if parameter_type == "json" else self._resolve_type(module, parameter_type)
            for parameter_name, parameter_type in declaration.parameters.items()
        }
        json_indexes = {index for index, kind in enumerate(declaration.parameters.values()) if kind == "json"}
        if json_indexes:
            self.json_parameters[declaration.name] = json_indexes
        extern_function.__annotations__["return"] = self._resolve_type(module, declaration.return_type)
        return self.program.extern(extern_function, name=declaration.name)

    def _init_declarations(self) -> None:
        for extern in self.config.externs:
            self._declare_module(extern)

    def runtime_extern(self, declaration: Callable[..., Any]) -> Function[..., Any]:
        """Declare (once) a C function of a runtime Bifrost links in, like the async runtime's."""
        name = declaration.__name__
        if name not in self._runtime_externs:
            self._runtime_externs[name] = self.program.extern(declaration, name=name)
        return self._runtime_externs[name]

    def load_std(self, name: str) -> _Extern:
        """Declare the bundled standard module ``std:name`` (once), and return it.

        Raises:
            KeyError: If there is no such standard module.
            ValueError: If one of its symbols is already declared by the project.

        """
        if name not in self._std:
            extern = std.load(name)
            self._declare_module(extern)
            self._std[name] = extern
        return self._std[name]

    def runtime(self, name: str) -> Callable[..., Any]:
        """Return a C function the compiler itself calls, declaring it once.

        ``malloc``, ``realloc`` and ``free`` manage the memory of owned values
        (``mem.Unique[str]``); Bifrost code does not call them directly. ``snprintf``
        is ``std:stdio``'s, so importing that module as well does not declare it twice.
        """
        if name == "snprintf":
            self.load_std("stdio")
            return self.extern_table["std:stdio_snprintf"]
        key = f"__bifrost_{name}"
        if key not in self.extern_table:
            parameters, result = _RUNTIME[name]
            declaration = _Function.model_validate(
                {"name": name, "type": "function", "parameters": parameters, "return": result}
            )
            self.extern_table[key] = self._create_extern("__bifrost", declaration)
        return self.extern_table[key]

    def _declare_module(self, extern: _Extern) -> None:
        module = extern.module
        for declaration in extern.declarations:
            match declaration.type:
                case "function":
                    extern_func = self._create_extern(module, declaration)
                    self.extern_table[f"{module}_{declaration.name}"] = extern_func
                    setattr(self, to_snake_case(declaration.name), extern_func)
                case "struct":
                    extern_struct = self._create_struct(module, declaration)
                    self.type_map[f"{module}_{declaration.name}"] = extern_struct
                    setattr(self, f"{module}_{declaration.name}", extern_struct)

    def main(self, handler: Callable[[Context], None]) -> Callable[..., Any]:
        """Register a context-based entry point with lowered registry accesses."""
        lines, _ = inspect.getsourcelines(handler)
        module = ast.parse(textwrap.dedent("".join(lines)))
        definition = module.body[0]
        if not isinstance(definition, ast.FunctionDef) or len(definition.args.args) != 1:
            msg = "@project.main handlers must accept exactly one Context parameter"
            raise TypeError(msg)

        context_name = definition.args.args[0].arg
        lowerer = _ContextLookupLowerer(context_name, self)
        definition = lowerer.visit(definition)
        definition.decorator_list = []
        definition.args.args = []
        definition.args.posonlyargs = []
        definition.args.defaults = []
        ast.fix_missing_locations(definition)

        if any(isinstance(node, ast.Name) and node.id == context_name for node in ast.walk(definition)):
            msg = "Context may only be used for literal extern_table or type_map lookups"
            raise ValueError(msg)

        source = ast.unparse(definition) + "\n"
        filename = f"<bifrost main {handler.__name__}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        module_code = compile(source, filename, "exec")
        function_code = next(code for code in module_code.co_consts if isinstance(code, CodeType))
        lowered_handler = FunctionType(function_code, lowerer.globals, handler.__name__)
        return self.program.main(lowered_handler)

    def build(self) -> Path:
        """Build the project executable (with Bifrost's async runtime, if anything pauses)."""
        return self.program.build_executable(
            self.config.path / self.config.package.name,
            opt_level=self.config.flags.optimization,
            libraries=self.config.libraries,
            linker=self.config.flags.linker,
            async_runtime=self._async_runtime() if self.pausing or self._returns_tokens() else None,
        )

    def _returns_tokens(self) -> bool:
        """Whether a C function returns a ``token``: then its caller, or C code, runs coroutines."""
        return any(
            getattr(getattr(extern, "python", None), "__annotations__", {}).get("return") is Token
            for extern in self.extern_table.values()
        )

    def _async_runtime(self) -> Path:
        """Compile Bifrost's async runtime (``std/async_runtime.c``) into the build folder.

        Raises:
            RuntimeError: If the compiler fails.

        """
        source = Path(std.__file__).parent / "async_runtime.c"
        output = self.config.path / "bifrost_async.o"
        command = [self.config.flags.linker, "-c", "-O2", "-fPIC", str(source), "-o", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode != 0:
            msg = f"compiling the async runtime failed: {' '.join(command)}\n{result.stderr}"
            raise RuntimeError(msg)
        return output
