"""Lower a Bifrost tree-sitter CST onto ``mlir_python.lang`` bindings.

Every top-level ``let`` becomes a binding in one shared namespace:

- ``let m = import("raylib")``: a namespace of the configured externs of that
  module (``m.InitWindow``, ``m.Color``).
- ``let P = struct { let x: i32, ... }``: an ``mlir_python.lang.struct``.
- ``let n = 42``: a compile-time constant.
- ``let f = (a: i32) => i32 ...``: a Python ``ast.FunctionDef`` registered with
  ``program.function`` (or ``program.main`` for ``main``), so the
  ``mlir_python.lang`` compiler type-checks and lowers the body.

A function's dependency list names every function it calls, and nothing else:
``let main = [draw, raylib.InitWindow, this] () => ...``. Calling an unlisted
function, listing one that is never called, or listing something that is not a
function (a module, struct, or constant) is an error; building a struct value
(``raylib.Color(...)``) is not a call. Without a list, a function calls nothing.

Functions are values: naming one without calling it (``http.get(app, "/",
index)``) passes a pointer to it, typed ``(ctx: http.Context) => null``, and
counts as depending on it. Calling a function value (a parameter, a local)
needs no entry: the value's type already says what it is. A lambda, ``[deps]
(x: i32) => i32 x * 2`` written where a value goes, is lifted to a function of
its own; it has its own dependency list and cannot use the enclosing
function's locals yet.

Constructs without a binding yet raise ``BifrostError`` at their location in
the ``.bif`` file, and so do compile errors in the generated functions.
"""

import ast
import keyword
import linecache
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType, GenericAlias, SimpleNamespace
from typing import Any

import tree_sitter_bifrost
from mlir_python.lang import (
    Array,
    CompileError,
    Fn,
    Function,
    Ptr,
    Token,
    cstr,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    ptr,
    stack,
    struct,
    u8,
    u16,
    u32,
    u64,
)
from mlir_python.lang._types import scalar_type
from mlir_python.lang.types import FnType, ScalarType, StructType
from mlir_python.lang.types import array as mlir_array
from tree_sitter import Language, Node, Parser

from bifrost import guards, ownership, std
from bifrost.configs.schema import _Extern
from bifrost.naming import extern_name
from bifrost.project import Project
from bifrost.std import fmt, json, json_runtime, mem, mem_runtime, tasks
from bifrost.syntax import syntax_errors

_LANGUAGE = Language(tree_sitter_bifrost.language())

_PRIMITIVES: dict[str, object] = {
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
    "bool": bool,
    "str": cstr,
    "null": type(None),
}

# A C function's `ptr` parameter, to which an owned local is lent (see `_opaque_params`).
_OPAQUE = object()
# A C function's `json` parameter: a record or object passed to it is encoded (see `_json_argument`).
_JSON = object()
# A C function's callback parameter that returns a `token` (a task C starts and waits for; see `_task_argument`).
_TASK = object()


@dataclass(frozen=True)
class _Cell:
    """A ``mem.Shared`` or ``mem.Atomic`` parameter: it borrows a cell of the same kind (see ``_pointees``)."""

    container: mem.Container
    held: object  # the type in the cell


def _json_quote(text: str) -> str:
    """Quote a field name as JSON: ``"name"``."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", "'": "'", '"': '"'}
_HEX_ESCAPE = 3  # `x` and two hex digits

_BINARY_OPERATORS: dict[str, type[ast.operator]] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
    "%": ast.Mod,
    "**": ast.Pow,
    "<<": ast.LShift,
    ">>": ast.RShift,
    "&": ast.BitAnd,
    "|": ast.BitOr,
    "^": ast.BitXor,
}

_COMPARISONS: dict[str, type[ast.cmpop]] = {
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
    "==": ast.Eq,
    "!=": ast.NotEq,
}

_BOOLEAN_OPERATORS: dict[str, type[ast.boolop]] = {"&&": ast.And, "||": ast.Or}

_UNARY_OPERATORS: dict[str, type[ast.unaryop]] = {"-": ast.USub, "!": ast.Not}


_FN_TYPE = re.compile(r"Fn\[\[([^\[\]]*)\], ([^\[\],]+)\]")


def display_types(text: str) -> str:
    """Write the compiler's type names as Bifrost does: ``Fn[[i32], None]`` is ``(i32) => null``."""
    previous = None
    while previous != text:
        previous = text
        text = _FN_TYPE.sub(lambda m: f"({m.group(1)}) => {'null' if m.group(2) == 'None' else m.group(2)}", text)
    text = re.sub(r"\bArray\[(\w+)\]", r"\1[]", text)
    return re.sub(r"\bcstr\b", "str", text) if "=>" in text else text


class BifrostError(SyntaxError):
    """A Bifrost source error, located in the ``.bif`` file."""

    def __init__(self, message: str, path: Path, source: bytes, point: tuple[int, int]) -> None:
        row, column = point
        lines = source.decode().splitlines()
        text = lines[row] if row < len(lines) else ""
        super().__init__(message, (str(path), row + 1, column + 1, text))

    def __str__(self) -> str:
        """Format as ``file:line:column: message``, the source line, and a caret."""
        caret = " " * ((self.offset or 1) - 1) + "^"
        location = f"{self.filename}:{self.lineno}:{self.offset}"
        return f"{location}: {self.msg}\n    {self.text}\n    {caret}"


def _children(node: Node) -> list[Node]:
    """Named children, without comments (an extra that can appear anywhere)."""
    return [child for child in node.named_children if child.type != "comment"]


def _unwrap(node: Node) -> Node:
    """Return the node a wrapper holds.

    The wrappers are ``expression``, ``condition``, ``getter_owner`` and
    ``parenthesized_expression``.
    """
    while node.type in {
        "expression",
        "condition",
        "getter_owner",
        "parenthesized_expression",
    }:
        node = _children(node)[0]
    return node


def _text(node: Node) -> str:
    return (node.text or b"").decode()


@dataclass(frozen=True)
class _PendingFunction:
    """A function to lower: how Bifrost names it, its compiled symbol, and its object."""

    name: str  # `draw`, or `Context.new` for an object's static function
    symbol: str  # `draw`, `Context_new`
    node: Node
    owner: object = None  # the object (or module) a static function or member belongs to
    module: str | None = None  # the module the function is a member of, if any


def _statement_call(item: Node) -> Node | None:
    """Return the call a statement makes as a whole (`draw(ctx)`, `let x = f(ctx)`), which may lend locals."""
    value = item
    if item.type in {"local_assignment", "return_statement"} and _children(item):
        value = _children(item)[-1]
    if value.type != "expression":
        return None
    inner = _unwrap(value)
    if inner.type == "child_annotation":
        inner = _children(inner)[-1]
    if inner.type != "function_call":
        return None
    return _children(inner)[0]


def _type_label(kind: object) -> str:
    """Name a type in messages: ``AppState``, ``i64``."""
    found = scalar_type(kind)
    return getattr(kind, "__name__", None) or (found.name if found is not None else repr(kind))


def _returns(node: Node) -> bool:
    """Whether a statement (or block) returns on every path, so that nothing runs after it.

    A ``return``; an ``if`` whose branches, ``else`` included, all return; a
    ``match`` with a ``_`` arm whose arms all return; a block with such a
    statement; or ``while true`` (only ``return`` leaves it).
    """
    node = _unwrap(node)
    match node.type:
        case "return_statement":
            return True
        case "block_expression":
            return any(_returns(statement) for statement in _children(node))
        case "if":
            branches = _children(node)[1:]  # after the condition: `then`, and `else` if there is one
            return len(branches) == 2 and all(_returns(branch) for branch in branches)  # noqa: PLR2004
        case "match_expression":
            arms = [arm for arm in _children(node) if arm.type == "match_arm"]
            default = any(_text(_children(arm)[0]) == "_" for arm in arms)
            return default and all(_returns(_children(arm)[-1]) for arm in arms)
        case "while":
            condition = _unwrap(_children(node)[0])
            return _text(condition) == "true"
    return False


def _is_statement(node: Node) -> bool:
    """Whether the expression ``node`` is in (``await f()``, ``(f())``) is a statement of its own, its value unused."""
    around = {"function_call", "child_annotation", "await_expression", "expression", "parenthesized_expression"}
    current = node.parent
    while current is not None and current.type in around:
        current = current.parent
    return current is not None and current.type == "block_expression"


def _is_async(function: Node) -> bool:
    """Whether a function definition is written ``async``."""
    return function.child_by_field_name("async") is not None


def _display(path: Path, root: Path) -> str:
    """Name a file in messages: ``root/shop/utils.bif`` is ``shop/utils.bif``."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _file_key(path: Path, root: Path) -> str:
    """Name a file for symbols and messages: ``root/utils/text.bif`` is ``utils_text``."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(path.name)
    return "_".join(relative.with_suffix("").parts)


# `=> Record`: the function returns a record, shaped like the records it returns.
RECORD = "Record"
_INFERRED = object()  # its result while it is being lowered


def _shown(kind: ScalarType) -> str:
    """Write a type as Bifrost does: ``#{ms: i64, name: str}`` for a record, ``str`` for a ``cstr``."""
    if isinstance(kind, StructType) and kind.python.__name__ == "record":
        return "#{" + ", ".join(f"{name}: {_shown(field)}" for name, field in kind.fields) + "}"
    return "str" if kind == cstr else display_types(kind.name)


class _FunctionScope:
    """The names local to one Bifrost function, and what it may call.

    Locals follow Python scoping (the whole body); the dependency list names
    the functions the body may call.
    """

    def __init__(self, function: str, parameters: set[str], symbol: str) -> None:
        self.function = function
        self.symbol = symbol  # its compiled name
        self.is_async = False  # written `async`: an `async def`, which may `await` calls that pause
        self.awaited: set[int] = set()  # ids of the call nodes written `await f(x)`
        self.names = set(parameters)
        self.temporaries = 0
        # Dependency ("raylib.InitWindow", or the function's own name for
        # `this`) -> its entry in the list.
        self.dependencies: dict[str, Node] = {}
        self.called: set[str] = set()
        self.pointers: set[str] = set()  # `mem.Weak` parameters and cells (pointers)
        self.cells: dict[str, Any] = {}  # `mem.Shared`/`mem.Atomic` parameters and owners -> their container
        self.cell_parameters: set[str] = set()  # the cells that are parameters (borrowed, never released)
        self.shared_locals: set[str] = set()  # `let x: mem.Unique[T] = ...` (owned values, lent to calls)
        self.held: dict[str, tuple[Any, object]] = {}  # name -> (its mem container, the type it holds)
        self.types: dict[str, ScalarType | None] = {}  # local -> its type, when the compiler can tell
        self.plan = ownership.Plan()  # where owned values are freed
        self.guards: dict[str, str] = {}  # guard -> the name it locks, as of the statement being lowered
        # A local lent to a `ptr[]` parameter is copied to the stack and back
        # around the statement that lends it.
        self.lend_call: Node | None = None  # the one call in this statement that may lend
        self.before: list[ast.stmt] = []
        self.after: list[ast.stmt] = []
        self.result: object = None  # its result type, as written (`_INFERRED` for `Record`, until lowered)
        # For a function written `=> Record`: each returned value, with its type.
        self.results: list[tuple[Node, ScalarType | None]] | None = None

    def temporary(self) -> str:
        self.temporaries += 1
        return f"__bifrost_match_{self.temporaries}"


class SourceUnit:
    """One ``.bif`` file lowered into a ``Project``'s program."""

    def __init__(
        self,
        project: Project,
        path: Path,
        source: bytes | None = None,
        *,
        root: Path | None = None,
        is_root: bool = True,
    ) -> None:
        self.project = project
        self.path = path
        self.root = root or path.parent  # where `import("a.b:module")` looks for a/b.bif
        self.is_root = is_root  # the file being built, whose `main` is the entry point
        # Symbols of functions in other files are prefixed, so files cannot clash.
        self.prefix = "" if is_root else _file_key(path, self.root) + "_"
        self.exports: set[str] = set()
        self._module: str | None = None  # the module whose member is being lowered
        self._members: dict[str, set[str]] = {}  # module -> its members' names
        self.source = path.read_bytes() if source is None else source
        self.tree = Parser(_LANGUAGE).parse(self.source)
        self.globals: dict[str, Any] = {}
        # Generated-source filename -> {(line, column): .bif point}.
        self._positions: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
        self._scope: _FunctionScope | None = None
        self._outer: list[_FunctionScope] = []  # the functions around the lambda being lowered
        self._functions: set[str] = set()  # top-level functions, bound after lowering

    # -- errors -----------------------------------------------------------------

    def error(self, node: Node, message: str) -> BifrostError:
        """Build a ``BifrostError`` located at ``node``."""
        return BifrostError(message, self.path, self.source, node.start_point)

    def _unsupported(self, node: Node, what: str | None = None) -> BifrostError:
        return self.error(node, f"{what or node.type.replace('_', ' ')} is not supported yet")

    @contextmanager
    def errors(self) -> Iterator[None]:
        """Report compile errors in generated functions at their ``.bif`` location."""
        try:
            yield
        except CompileError as error:
            owner = next(
                (unit for unit in [self, *self.project.units.values()] if error.filename in unit._positions), None
            )
            if owner is not self and owner is not None:
                with owner.errors():
                    raise
            positions = self._positions.get(error.filename or "")
            if positions is None:
                raise
            line, column = error.lineno or 1, (error.offset or 1) - 1
            point = positions.get((line, column)) or next(
                (point for (row, _), point in sorted(positions.items()) if row == line),
                positions.get((1, 0), (0, 0)),
            )
            raise BifrostError(display_types(error.msg), self.path, self.source, point) from error

    # -- top level ----------------------------------------------------------------

    def lower(self) -> None:
        """Bind every top-level ``let`` and register the functions."""
        root = self.tree.root_node
        problems = syntax_errors(root)
        if problems:
            raise self.error(problems[0].node, problems[0].message)
        self.project.loading.append(self.path.resolve())
        try:
            functions = self._bind(root)
            # Every name is known before any body is lowered, so functions can call
            # each other in any order.
            self._functions = {function.name for function in functions}
            self._pointer_params = {}
            self._function_nodes = {function.name: function.node for function in functions}
            self._pending = {function.name: function for function in functions}
            self._records = self.project.records  # records with the same fields have the same type, in every file
            self._inferred: dict[str, type] = {}  # `=> Record` function -> the record it returns
            self._inferring: set[str] = set()  # `=> Record` functions being lowered
            self._lowered: dict[str, Callable[..., Any]] = {}  # functions lowered early, for their record
            self._owned_signatures: dict[str, tuple[bool, set[int]]] = {}
            self._cell_results: dict[str, tuple[mem.Container, object] | None] = {}
            for function in functions:
                self._module = function.module
                self._signature(function.name, function.node, function.symbol)
            self._pausing: set[str] = set()  # functions of this file that pause
            self._find_pausing(functions)
            for function in functions:
                self._module = function.module
                python = self._lowered.pop(function.name, None) or self._function(
                    function.name, function.node, function.symbol
                )
                self._register(function, python)
            self._module = None
        finally:
            self.project.loading.pop()

    def _bind(self, root: Node) -> list[_PendingFunction]:
        """Bind the top-level values and objects; return the functions to lower."""
        functions: list[_PendingFunction] = []
        exports: list[Node] = []
        for item in _children(root):
            if item.type == "function_definition":
                raise self.error(item, "a function needs a name: let name = (...) => ...")
            if item.type == "export_statement":
                exports.extend(item.children_by_field_name("module"))
            elif item.type == "module":
                functions += self._bind_module(item, functions)
            else:
                functions += self._bind_let(item, functions, None, self.globals)
        for export in exports:
            name = _text(export)
            if name not in self._members:
                raise self.error(export, f"'{name}' is not a module in this file; export(...) names modules")
            self.exports.add(name)
        return functions

    def _bind_module(self, node: Node, functions: list[_PendingFunction]) -> list[_PendingFunction]:
        """Bind ``module name = { ... }`` as a namespace of its members."""
        identifier = node.child_by_field_name("name")
        name = self._identifier(identifier)
        if name in self.globals or any(name == function.name for function in functions):
            raise self.error(identifier, f"'{name}' is already defined")
        namespace = SimpleNamespace()
        self.globals[name] = namespace
        members = node.children_by_field_name("member")
        self._members[name] = {self._identifier(_children(member)[0]) for member in members}
        self._module = name
        found: list[_PendingFunction] = []
        for member in members:
            found += self._bind_let(member, [*functions, *found], name, vars(namespace))
        self._module = None
        return found

    def _bind_let(
        self, assignment: Node, functions: list[_PendingFunction], module: str | None, scope: dict[str, Any]
    ) -> list[_PendingFunction]:
        """Bind one ``let`` in ``scope`` (the file's globals, or a module's namespace)."""
        identifier, value = _children(assignment)
        name = self._identifier(identifier)
        qualified = f"{module}.{name}" if module else name
        if name in scope or any(qualified == function.name for function in functions):
            raise self.error(identifier, f"'{name}' is already defined")
        symbol = self.prefix + qualified.replace(".", "_")
        owner = self.globals[module] if module else None
        match value.type:
            case "function_definition":
                return [_PendingFunction(qualified, symbol, value, owner, module)]
            case "struct_assignment":
                scope[name], statics = self._struct(name, value)
                return [
                    _PendingFunction(f"{qualified}.{member}", f"{symbol}_{member}", definition, scope[name], module)
                    for member, definition in statics
                ]
            case "expression":
                scope[name] = self._constant(_unwrap(value))
                return []
            case _:
                raise self._unsupported(value, "a top-level block")

    def _qualify(self, name: str) -> str:
        """Inside a module, a member's bare name means the member: ``greet`` is ``helper.greet``."""
        if self._module is not None and name in self._members.get(self._module, set()):
            return f"{self._module}.{name}"
        return name

    def _register(self, function: _PendingFunction, python: Callable[..., Any]) -> None:
        """Add a lowered function to the program, and bind it where Bifrost finds it."""
        try:
            is_main = function.name == "main" and self.is_root
            if is_main and function.name in self._pausing:
                python = self._async_main(function, python)
            register = self.project.program.main if is_main else self.project.program.function
            compiled = register(python)
        except ValueError as error:
            raise self.error(function.node, str(error)) from error
        if function.owner is None:
            self.globals[function.name] = compiled
        else:
            # `Context.new()` (or `helper.greet()`) finds it as an attribute of its object (or module).
            setattr(function.owner, function.name.rpartition(".")[2], compiled)

    def _async_main(self, function: _PendingFunction, body: Callable[..., Any]) -> Callable[..., Any]:
        """Return the entry point of an async ``main``: a plain function that runs it to the end."""
        self.globals["bifrost_async_main"] = self.project.program.function(body)
        result = body.__annotations__["return"]
        run = self._at(
            ast.Call(self._at(ast.Name("bifrost_async_main", ast.Load()), function.node), [], []), function.node
        )
        statement = ast.Return(run) if result is not type(None) else ast.Expr(run)
        definition = self._at(
            ast.FunctionDef(
                name="main",
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[self._at(statement, function.node)],
                decorator_list=[],
                type_params=[],
            ),
            function.node,
        )
        return self._compile(definition, {"return": result})

    def _constant(self, node: Node) -> object:
        if node.type == "function_call" and _children(node)[0].type == "builtin_call":
            call = _children(node)[0]
            builtin = _text(call.child_by_field_name("function"))
            if builtin == "import":
                return self._import_module(call)
            raise self._unsupported(call, f"top-level {builtin}()")
        negate = node.type == "unary_expression" and _text(node.child_by_field_name("operator")) == "-"
        literal = _unwrap(node.child_by_field_name("argument")) if negate else node
        if literal.type == "literal":
            value = self._literal(literal)
            if isinstance(value, ast.Constant) and isinstance(value.value, (bool, int, float)):
                if negate and not isinstance(value.value, bool):
                    return -value.value
                if not negate:
                    return value.value
        raise self.error(node, "a top-level value must be a number, a boolean, or import(...)")

    def _import_module(self, call: Node) -> SimpleNamespace:
        arguments = [_unwrap(a) for a in _children(call) if a.type == "expression"]
        if len(arguments) != 1 or arguments[0].type != "literal" or _children(arguments[0])[0].type != "string":
            raise self.error(call, 'import takes one module name: import("raylib")')
        module = _text(arguments[0])[1:-1]
        if module == f"{std.PREFIX}mem":
            return SimpleNamespace(**mem.CONTAINERS)
        if module == f"{std.PREFIX}fmt":
            return SimpleNamespace(**fmt.FUNCTIONS)
        if module == f"{std.PREFIX}json":
            return SimpleNamespace(**json.FUNCTIONS)
        if module == f"{std.PREFIX}tasks":
            return SimpleNamespace(**tasks.FUNCTIONS)
        if ":" in module and not module.startswith(std.PREFIX):
            return self._import_file_module(module, arguments[0])
        extern = self._find_module(module, arguments[0])
        # Externs keep their C symbols but take Bifrost names: InitWindow is
        # init_window, rlVertexBuffer is RlVertexBuffer.
        bindings: dict[str, object] = {}
        names: dict[str, str] = {}
        for declaration in extern.declarations:
            key = f"{module}_{declaration.name}"
            name = extern_name(declaration)
            if name in names:
                msg = f"{module}'s {names[name]} and {declaration.name} would both be named {name!r}"
                raise self.error(arguments[0], msg)
            names[name] = declaration.name
            bindings[name] = self.project.extern_table.get(key) or self.project.type_map[key]
        return SimpleNamespace(**bindings)

    def _import_file_module(self, spec: str, node: Node) -> SimpleNamespace:
        """``import("a.b:module")``: the exported ``module`` of ``a/b.bif``, from the project root.

        A leading ``.`` looks next to this file instead, and each further ``.`` one folder up,
        as Python's relative imports do (``.helper:helper``, ``..shared:text``).
        """
        file_part, _, module = spec.partition(":")
        path = self._resolve_file(file_part, node)
        resolved = path.resolve()
        if resolved in self.project.loading:
            chain = [*self.project.loading[self.project.loading.index(resolved) :], resolved]
            cycle = " -> ".join(_display(p, self.root) for p in chain)
            raise self.error(node, f"import cycle: {cycle}")
        unit = self.project.units.get(resolved)
        if unit is None:
            unit = SourceUnit(self.project, path, root=self.root, is_root=False)
            try:
                unit.lower()
            except BifrostError as error:
                where = f"{_display(path, self.root)}:{error.lineno}"
                raise self.error(node, f"in {where}: {error.msg}") from error
            self.project.units[resolved] = unit
        if module not in unit.exports:
            exported = ", ".join(sorted(unit.exports)) or "none"
            raise self.error(node, f"{path.name} does not export module '{module}' (exports: {exported})")
        return unit.globals[module]

    def _resolve_file(self, file_part: str, node: Node) -> Path:
        dots = len(file_part) - len(file_part.lstrip("."))
        base = self.root if dots == 0 else self.path.parent
        for _ in range(max(dots - 1, 0)):
            base = base.parent
        names = file_part[dots:].split(".")
        if not all(names) or names[0] == "std":
            raise self.error(node, f"'{file_part}' is not a file name: use dots between folders, like utils.text")
        path = base.joinpath(*names[:-1], names[-1] + ".bif")
        if not path.is_file():
            raise self.error(node, f"cannot find {path.name} for '{file_part}' (looked for {path})")
        return path

    def _find_module(self, module: str, node: Node) -> _Extern:
        """Find ``module``: a standard one bundled with the compiler (``std:stdio``) or a configured one."""
        if module.startswith(std.PREFIX):
            try:
                return self.project.load_std(module.removeprefix(std.PREFIX))
            except KeyError:
                known = ", ".join(f"{std.PREFIX}{name}" for name in std.available())
                raise self.error(node, f"unknown standard module {module!r} (available: {known})") from None
            except ValueError as error:
                raise self.error(node, f"cannot load {module}: {error}") from None
        extern = next((e for e in self.project.config.externs if e.module == module), None)
        if extern is None:
            known = ", ".join(repr(e.module) for e in self.project.config.externs) or "none"
            raise self.error(node, f"unknown module {module!r} (configured: {known}; standard: std:...)")
        return extern

    def _struct(self, name: str, node: Node) -> tuple[type, list[tuple[str, Node]]]:
        """Build an object's data type, and collect its ``static`` functions.

        ``let x: T`` members are stored fields, laid out as a C struct.
        ``static let f = (...) => ...`` members are functions called on the type
        (``Context.new()``); they take no space in the object.
        """
        annotations: dict[str, object] = {}
        statics: list[tuple[str, Node]] = []
        for member in _children(node):
            parts = _children(member)
            member_name = self._identifier(parts[0])
            is_static = member.child_by_field_name("modifier") is not None
            if member_name in annotations or any(member_name == other for other, _ in statics):
                raise self.error(parts[0], f"{name} already has a member named '{member_name}'")
            function = next((part for part in parts if part.type == "local_function_definition"), None)
            if function is not None and is_static:
                statics.append((member_name, function))
            elif function is not None:
                raise self._unsupported(member, "a method (a function without `static`)")
            elif is_static:
                raise self.error(member, f"static data is not supported; '{member_name}' must be a field or a function")
            else:
                annotations[member_name] = self._field_type(parts[1])
        if not annotations:
            raise self.error(node, f"struct {name} declares no fields")
        try:
            return struct(type(name, (), {"__annotations__": annotations})), statics
        except TypeError as error:
            raise self.error(node, str(error)) from error

    def _field_type(self, type_node: Node) -> object:
        """Return the type of an object's field; a field cannot hold a ``mem`` container (yet)."""
        found = self._container(type_node)
        if found is None:
            return self._type(type_node)
        container = found[0]
        if container is mem.WEAK:
            raise self.error(type_node, "an object cannot store a mem.Weak: it would outlive the call that lends it")
        if container is mem.UNIQUE:
            raise self._unsupported(type_node, "an object owning another (a mem.Unique field)")
        if container in mem.CELLS:
            raise self._unsupported(type_node, f"an object holding a {container!r} (a {container!r} field)")
        return self._type(type_node)

    # -- types --------------------------------------------------------------------

    def _type(self, node: Node) -> object:
        node = _children(node)[0] if node.type == "type_or_object" else node
        match node.type:
            case "type":
                return self._written_type(node)
            case "generic_type":
                return self._generic_type(node)
            case "record_type":
                return self._record_shape(node)
            case "identifier" if _text(node) == RECORD:
                msg = f"{RECORD} is a function's result, the record it returns: `() => {RECORD} #{{x: 1}}`"
                raise self.error(node, msg)
            case "identifier" | "child_annotation":
                found = self._static(node)
                if not (isinstance(found, type) and hasattr(found, "__lang_struct__")):
                    raise self.error(node, f"'{_text(node)}' is not a type")
                return found
            case _:
                raise self._unsupported(node)

    def _record_shape(self, node: Node) -> type:
        """``#{name: str, ms: i64}``: the type of records with these fields (in this order)."""
        fields: list[tuple[str, object]] = []
        for field in _children(node):
            name_node = field.child_by_field_name("name")
            name = self._identifier(name_node)
            if any(name == other for other, _ in fields):
                raise self.error(name_node, f"the record type already has a field '{name}'")
            kind = self._type(field.child_by_field_name("type"))
            if kind is type(None):
                raise self.error(field, f"field '{name}' needs a type with values, not null")
            held = scalar_type(kind)
            fields.append((name, held.python if isinstance(held, StructType) else held))
        return self._record_type(tuple(fields))

    def _written_type(self, node: Node) -> object:
        """Resolve a ``type`` node: a keyword (``i32``, ``str``, ``null``), a tuple, a function or a list type."""
        inner = _children(node)
        if not inner:
            keyword_text = _text(node)
            if keyword_text not in _PRIMITIVES:
                raise self._unsupported(node, f"type {keyword_text}")
            return _PRIMITIVES[keyword_text]
        match inner[0].type:
            case "null":
                return type(None)
            case "tuple_type":
                # tuple[i32, i32], built at runtime rather than written as a type.
                return GenericAlias(tuple, tuple(self._type(part) for part in _children(inner[0])))
            case "function_type":
                return self._function_type(inner[0])
            case "list_type":
                element = self._type(_children(inner[0])[0])
                try:
                    return Array[element]
                except TypeError:
                    name = getattr(element, "__name__", _text(_children(inner[0])[0]))
                    raise self.error(inner[0], f"a list holds integers, floats or bools, not {name} (yet)") from None
        raise self._unsupported(inner[0])

    def _function_type(self, node: Node) -> object:
        """``(ctx: http.Context) => null`` is ``Fn[[http.Context], None]``: a pointer to such a function."""
        parameters: list[object] = []
        for parameter in node.children_by_field_name("parameter"):
            type_node = _children(parameter)[1] if parameter.type == "parameter" else parameter
            if self._container(type_node) is not None:
                raise self._unsupported(type_node, "a mem container in a function type")
            parameters.append(self._type(type_node))
        return_node = node.child_by_field_name("return_type")
        if self._container(return_node) is not None:
            raise self._unsupported(return_node, "a mem container in a function type")
        result = self._type(return_node)
        try:
            return Fn[parameters, None if result is type(None) else result]
        except TypeError as error:
            raise self.error(node, str(error)) from error

    def _generic_type(self, node: Node) -> object:
        """``mem.Weak[T]`` is ``Ptr[T]``; what each container allows is checked here and by ``guards``."""
        base_node = node.child_by_field_name("base")
        base = self._static(base_node)
        if not isinstance(base, mem.Container):
            raise self._unsupported(node, f"generic type {_text(base_node)}[...]")
        if base.guards is not None:
            msg = f"{base!r} is a guard's type; it only annotates a lock: `let guard: {base!r}[...] <- value`"
            raise self.error(node, msg)
        if not base.supported:
            raise self._unsupported(node, f"{base!r}")
        arguments = node.children_by_field_name("argument")
        if len(arguments) != 1:
            raise self.error(node, f"{base!r} takes one type: {base!r}[Context]")
        if base is mem.UNIQUE:
            return self._type(arguments[0])  # the owner holds the value itself (a str: its pointer)
        return Ptr[self._type(arguments[0])]

    def _container(self, type_node: Node | None) -> tuple[mem.Container, Node] | None:
        """For a ``mem.X[T]`` type node, return the container and ``T``'s node; ``None`` for other types."""
        node = type_node
        while node is not None and node.type == "type_or_object" and node.named_children:
            node = node.named_children[0]
        if node is None or node.type != "generic_type":
            return None
        try:
            base = self._static(node.child_by_field_name("base"))
        except BifrostError:
            return None
        arguments = node.children_by_field_name("argument")
        return (base, arguments[0]) if isinstance(base, mem.Container) and len(arguments) == 1 else None

    def _is(self, container: mem.Container, type_node: Node | None) -> bool:
        found = self._container(type_node)
        return found is not None and found[0] is container

    def _signature(self, name: str, node: Node, symbol: str) -> None:
        """Record what calls of a function need to know: what its parameters take, and what it returns."""
        self._pointer_params[name] = self.project.pointer_params[symbol] = self._pointees(node)
        self._owned_signatures[name] = self.project.owned_signatures[symbol] = self._owned_signature(node)
        parts = _children(node)
        return_node = parts[parts.index(next(part for part in parts if part.type == "parameter_list")) + 1]
        cell = self._cell(return_node)
        found = self._container(return_node)
        result = (cell, self._type(found[1])) if cell is not None and found is not None else None
        self._cell_results[name] = self.project.cell_results[symbol] = result

    def _find_pausing(self, functions: list[_PendingFunction]) -> None:
        """Record the functions written ``async``: they may pause, and their callers ``await`` them."""
        for function in functions:
            if _is_async(function.node):
                self._pausing.add(function.name)
                self.project.pausing.add(function.symbol)

    def _pauses(self, callee: object) -> bool:
        """Whether calling ``callee`` (as ``_callee`` returns it) can pause."""
        if callee is tasks.GATHER:
            return True
        if isinstance(callee, str):
            return callee in self._pausing
        if isinstance(callee, Function):
            if callee.kind == "extern":
                return getattr(callee.python, "__annotations__", {}).get("return") is Token
            return callee.name in self.project.pausing
        return False

    def _cell(self, type_node: Node | None) -> mem.Container | None:
        """For a ``mem.Shared[T]`` or ``mem.Atomic[T]`` type node, its container; ``None`` for other types."""
        found = self._container(type_node)
        return found[0] if found is not None and found[0] in mem.CELLS else None

    def _cell_result(self, value: Node) -> tuple[mem.Container, object] | None:
        """For a call returning a ``mem.Shared`` or ``mem.Atomic``, its container and held type."""
        callee = self._callee(_unwrap(value))
        if isinstance(callee, str):
            return self._cell_results.get(callee)
        if isinstance(callee, Function):
            return self.project.cell_results.get(callee.name)
        return None

    def _owned_signature(self, function: Node) -> tuple[bool, set[int]]:
        """Whether ``function`` returns an owned value (a string or a cell), and which parameters take a string."""
        parts = _children(function)
        parameter_list = next(part for part in parts if part.type == "parameter_list")
        return_node = parts[parts.index(parameter_list) + 1]
        moved = {
            index
            for index, parameter in enumerate(_children(parameter_list))
            if self._owned_type(_children(parameter)[1])
        }
        return self._owned_type(return_node) or self._cell(return_node) is not None, moved

    def _pointees(self, function: Node) -> list[object | None]:
        """Return what each parameter of ``function`` takes.

        That is the lent type of a ``mem.Weak``, a ``_Cell`` for a
        ``mem.Shared`` or ``mem.Atomic``, and ``None`` for the others.
        """
        parts = _children(function)
        parameter_list = next(part for part in parts if part.type == "parameter_list")
        pointees: list[object | None] = []
        for parameter in _children(parameter_list):
            type_node = _children(parameter)[1]
            found = self._container(type_node)
            if found is not None and found[0] is mem.WEAK:
                pointees.append(self._type(found[1]))
            elif found is not None and found[0] in mem.CELLS:
                pointees.append(_Cell(found[0], self._type(found[1])))
            else:
                pointees.append(None)
        return pointees

    def _static(self, node: Node) -> object:
        """Resolve a top-level name or dotted name (``raylib.Color``)."""
        parts = [node] if node.type == "identifier" else _children(node)
        if any(part.type == "function_call" for part in parts):
            raise self.error(node, f"'{_text(node)}' is not a name")
        name = self._qualify(_text(parts[0]))
        if name in self._functions and len(parts) == 1:
            return None
        module, _, member = name.partition(".")
        if module not in self.globals:
            raise self.error(parts[0], f"'{name}' is not defined")
        found = self.globals[module]
        if member:
            found = getattr(found, member)
        for part in parts[1:]:
            if not hasattr(found, _text(part)):
                raise self.error(part, f"'{_text(node)}' is not defined")
            found = getattr(found, _text(part))
        return found

    # -- functions ----------------------------------------------------------------

    def _function(self, name: str, node: Node, symbol: str | None = None) -> Callable[..., Any]:
        """Lower a function; ``name`` is how Bifrost calls it, ``symbol`` its compiled name."""
        parts = _children(node)
        dependency_list = parts[0] if parts[0].type in {"dependency_list", "local_dependency_list"} else None
        parameter_list, return_node, body = parts[1:] if dependency_list else parts
        annotations: dict[str, object] = {}
        arguments = []
        for parameter in _children(parameter_list):
            identifier, type_node = _children(parameter)
            parameter_name = self._identifier(identifier)
            annotations[parameter_name] = self._type(type_node)
            arguments.append(self._at(ast.arg(parameter_name), identifier))
        inferring = _text(return_node) == RECORD
        annotations["return"] = _INFERRED if inferring else self._type(return_node)
        self._check_returns(name, return_node, body, annotations["return"])
        if inferring:
            self._inferring.add(name)
        held = self._held(parameter_list, return_node, body)
        self._check_guards(body, held)
        owned_parameters = {
            self._identifier(_children(parameter)[0])
            for parameter in _children(parameter_list)
            if self._owned_type(_children(parameter)[1])
        }
        cells = {name: container for name, (container, _) in held.items() if container in mem.CELLS}
        cell_parameters = set(cells) & set(annotations)
        try:
            plan = ownership.check(body, owned_parameters, self._ownership_oracle(), cell_parameters)
        except ownership.OwnershipError as error:
            raise self.error(error.node, error.message) from None

        self._scope = scope = _FunctionScope(name, set(annotations) - {"return"}, symbol or name)
        scope.result = annotations["return"]
        scope.results = [] if inferring else None
        scope.is_async = is_async = name in self._pausing
        if is_async and name == "main" and self.is_root:
            self._check_async_main(node)
            symbol = scope.symbol = "bifrost_async_main"  # run by a plain `main` (see `_register`)
        scope.types = {parameter: scalar_type(kind) for parameter, kind in annotations.items() if parameter != "return"}
        scope.held, scope.plan, scope.cells, scope.cell_parameters = held, plan, cells, cell_parameters
        scope.pointers = {name for name, (container, _) in held.items() if container is mem.WEAK} | set(cells)
        scope.shared_locals = {name for name, (container, _) in held.items() if container is mem.UNIQUE}
        if dependency_list is not None:
            self._dependencies(dependency_list)
        self._check_local_names(body)
        self._scope.names |= (
            {self._identifier(_children(let)[0]) for let in self._descendants(body, "local_assignment")}
            | {self._identifier(lock.child_by_field_name("guard")) for lock in self._descendants(body, "lock")}
            | {self._identifier(_children(loop)[0]) for loop in self._descendants(body, "forall")}
        )
        statements = self._body(body, returns=annotations["return"] is not type(None))
        for dependency, entry in self._scope.dependencies.items():
            if dependency not in self._scope.called:
                raise self.error(entry, f"'{dependency}' is a dependency of {name} but never called")
        if inferring:
            annotations["return"] = self._inferred_record(name, self._scope.results or [])
            self._inferring.discard(name)
        self._scope = None

        definition_type = ast.AsyncFunctionDef if is_async else ast.FunctionDef
        definition = self._at(
            definition_type(
                name=symbol or name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=arguments,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=statements,
                decorator_list=[],
                type_params=[],
            ),
            node,
        )
        return self._compile(definition, annotations)

    def _check_local_names(self, body: Node) -> None:
        """Reject a ``let`` named like a module of this file: calls through the module would stop working."""
        for let in self._descendants(body, "local_assignment"):
            identifier = _children(let)[0]
            if isinstance(self.globals.get(_text(identifier)), SimpleNamespace):
                msg = f"'{_text(identifier)}' is a module of this file (an import); give this value another name"
                raise self.error(identifier, msg)

    def _check_guards(self, body: Node, held: dict[str, tuple[mem.Container, object]]) -> None:
        """Check the guards in ``body`` (see ``guards``), knowing which calls pause and which values are whole."""
        kinds = {name: repr(container) for name, (container, _) in held.items()}
        values = frozenset(name for name, (_, kind) in held.items() if not isinstance(scalar_type(kind), StructType))
        try:
            guards.check(body, kinds, pauses=lambda call: self._pauses(self._callee(call)), values=values)
        except guards.GuardError as error:
            raise self.error(error.node, error.message) from None

    def _held(self, parameter_list: Node, return_node: Node, body: Node) -> dict[str, tuple[mem.Container, object]]:
        """Check where each ``mem`` container may appear; return each held name's container and type.

        ``mem.Weak`` is for parameters (lent to the call), ``mem.Unique`` for
        locals (the owner); neither may be returned.
        """
        if self._is(mem.WEAK, return_node):
            msg = "a function cannot return a mem.Weak: it cannot outlive the call that lends it"
            raise self.error(return_node, msg)
        if self._is(mem.UNIQUE, return_node) and not self._owned_type(return_node):
            raise self._unsupported(return_node, "returning ownership of anything but a str (a mem.Unique result)")
        held: dict[str, tuple[mem.Container, object]] = {}
        for parameter in _children(parameter_list):
            identifier, type_node = _children(parameter)
            found = self._container(type_node)
            if self._owned_type(type_node):
                continue  # an owned string: checked by `ownership`
            if found is not None and found[0] is mem.UNIQUE:
                msg = "passing ownership (a mem.Unique parameter); borrow it with mem.Weak"
                raise self._unsupported(type_node, msg)
            if found is not None:
                held[self._identifier(identifier)] = (found[0], self._type(found[1]))
        self._held_locals(body, held)
        return held

    def _held_locals(self, body: Node, held: dict[str, tuple[mem.Container, object]]) -> None:
        """Add the locals of ``body`` that hold a ``mem`` container to ``held``."""
        for let in self._descendants(body, "local_assignment"):
            found = self._container(let.child_by_field_name("type"))
            if found is not None and found[0] is mem.WEAK:
                msg = "a local cannot be a mem.Weak, which is only lent to a call; own the value with mem.Unique"
                raise self.error(let.child_by_field_name("type"), msg)
            if found is not None and not self._owned_type(let.child_by_field_name("type")):
                held[self._identifier(_children(let)[0])] = (found[0], self._type(found[1]))
            elif let.child_by_field_name("type") is None and (cell := self._inferred_cell(let, held)) is not None:
                held[self._identifier(_children(let)[0])] = cell

    def _inferred_cell(self, let: Node, held: dict[str, tuple[mem.Container, object]]) -> tuple[Any, object] | None:
        """``let other = counter`` and ``let counter = make_counter()`` own a cell too: return its kind."""
        value = _unwrap(_children(let)[-1])
        source = held.get(_text(value)) if value.type == "identifier" else None
        return source if source is not None and source[0] in mem.CELLS else self._cell_result(value)

    def _owned_type(self, type_node: Node | None) -> bool:
        """Whether a type node is ``mem.Unique[str]``: an owned string, freed by its owner."""
        found = self._container(type_node)
        return found is not None and found[0] is mem.UNIQUE and self._type(found[1]) == cstr

    def _callee(self, node: Node) -> object:
        """Return what a call expression calls: a function's name here, a Function elsewhere, or a builtin."""
        if node.type == "function_call":
            inner = _children(node)[0]
            function = inner.child_by_field_name("function") if inner.type == "user_function_call" else None
            return self._qualify(_text(function)) if function is not None else None
        if node.type != "child_annotation":
            return None
        *path, last = _children(node)
        if last.type != "function_call" or not path or any(part.type != "simple_identifier" for part in path):
            return None
        function = _children(last)[0].child_by_field_name("function")
        if function is None:
            return None
        names = [*self._qualify(_text(path[0])).split("."), *(_text(part) for part in path[1:]), _text(function)]
        dotted = ".".join(names)
        if dotted in self._functions:
            return dotted
        target: object = self.globals.get(names[0])
        for name in names[1:]:
            target = getattr(target, name, None)
        return target

    def _ownership_oracle(self) -> ownership.Oracle:
        def signature(node: Node) -> tuple[bool, set[int]]:
            callee = self._callee(node)
            if callee in (fmt.FORMAT, json.ENCODE):
                return True, set()
            if isinstance(callee, str):
                return self._owned_signatures.get(callee, (False, set()))
            if isinstance(callee, Function):
                return self.project.owned_signatures.get(callee.name, (False, set()))
            return False, set()

        return ownership.Oracle(
            owned_call=lambda node: signature(node)[0],
            moved_arguments=lambda node: signature(node)[1],
            owned_type=self._owned_type,
            cell_call=lambda node: self._cell_result(node) is not None,
            cell_type=lambda node: self._cell(node) is not None,
        )

    def _dependencies(self, node: Node) -> None:
        """Record the functions a dependency list lets the current function call.

        Entries are top-level functions, externs (``raylib.InitWindow``), and
        ``this`` (the function itself).
        """
        assert self._scope is not None
        for entry in node.children:
            if (not entry.is_named and entry.type not in {"this", "super"}) or entry.type == "comment":
                continue
            if entry.type == "super":
                function = self._scope.function
                raise self.error(entry, f"static function {function} has no instance, so it cannot depend on super")
            if entry.type == "this":
                dependency = self._scope.function
            else:
                parts = _text(entry).split(".")
                dependency = ".".join([self._qualify(parts[0]), *parts[1:]])
                if not self._is_function(entry):
                    raise self.error(entry, f"'{dependency}' is not a function; list only functions")
            if dependency in self._scope.dependencies:
                raise self.error(entry, f"'{dependency}' is already a dependency")
            self._scope.dependencies[dependency] = entry

    def _is_function(self, node: Node) -> bool:
        """Whether a (dotted) name is a top-level function, an object's static function, or an extern."""
        parts = _text(node).split(".")
        if ".".join([self._qualify(parts[0]), *parts[1:]]) in self._functions:
            return True
        found = self._static(node)
        return isinstance(found, (Function, fmt.Builtin))

    def _depend(self, dependency: str, node: Node, verb: str = "calls") -> None:
        """Check a call to ``dependency`` (or, with ``verb="uses"``, a use as a value) against the dependency list."""
        assert self._scope is not None
        if dependency not in self._scope.dependencies:
            listed = "this" if dependency == self._scope.function else dependency
            raise self.error(
                node,
                f"{self._scope.function} {verb} '{dependency}' without depending on "
                f"it; add it to the dependency list: [{listed}]",
            )
        self._scope.called.add(dependency)

    def _compile(self, definition: ast.FunctionDef, annotations: dict[str, object]) -> Callable[..., Any]:
        """Give the generated function real source, which ``mlir_python.lang`` reads."""
        ast.fix_missing_locations(definition)
        source = ast.unparse(definition) + "\n"
        filename = f"<bifrost {self.path}:{definition.name}>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        # The reparsed tree has the same shape, so its nodes line up with ours.
        positions: dict[tuple[int, int], tuple[int, int]] = {}
        for ours, theirs in zip(ast.walk(definition), ast.walk(ast.parse(source).body[0]), strict=False):
            point = getattr(ours, "bifrost_point", None)
            if point is not None and hasattr(theirs, "lineno"):
                positions.setdefault((theirs.lineno, theirs.col_offset), point)
        self._positions[filename] = positions

        module_code = compile(source, filename, "exec")
        function_code = next(code for code in module_code.co_consts if isinstance(code, CodeType))
        python = FunctionType(function_code, self.globals, definition.name)
        python.__annotations__ = annotations
        return python

    # -- statements ---------------------------------------------------------------

    def _block(self, node: Node) -> list[ast.stmt]:
        statements: list[ast.stmt] = []
        for item in _children(node):
            assert self._scope is not None
            self._scope.lend_call = _statement_call(item)
            lowered = self._block_item(item)
            statements.extend([*self._scope.before, *lowered])
            if item.type != "return_statement":
                statements.extend(self._scope.after)
            self._scope.before, self._scope.after, self._scope.lend_call = [], [], None
        statements += [self._free(name, node) for name in self._scope.plan.at_end.get(node.id, [])]
        return statements or [self._at(ast.Pass(), node)]

    def _free(self, name: str, node: Node) -> ast.stmt:
        """``free(name)``: the end of an owned string's life (for a cell, the end of one owner's)."""
        assert self._scope is not None
        if name in self._scope.cells:
            return self._cell_statement("release", name, node)
        self.globals["__bifrost_free"] = self.project.runtime("free")
        call = ast.Call(self._at(ast.Name("__bifrost_free", ast.Load()), node), [ast.Name(name, ast.Load())], [])
        return self._at(ast.Expr(self._at(call, node)), node)

    def _block_item(self, item: Node) -> list[ast.stmt]:
        """Lower one statement of a block (its lends are added around it by ``_block``)."""
        assert self._scope is not None
        statements: list[ast.stmt] = []
        match item.type:
            case "lock":
                guard = self._identifier(item.child_by_field_name("guard"))
                source = _text(_unwrap(item.child_by_field_name("source")))
                self._scope.guards[guard] = source
                self._check_guard_type(item)
                if source in self._scope.cells:
                    statements.append(self._lock(source, item, f"{source} is already locked by another guard"))
            case "release":
                source = _text(_unwrap(item.child_by_field_name("source")))
                if source in self._scope.cells:  # other guards are checked by `guards`; nothing runs
                    statements.append(self._cell_statement("unlock", source, item))
            case "field_assignment":
                target = self._field_target(item.child_by_field_name("target"))
                value = self._expression(item.child_by_field_name("value"))
                statements.append(self._at(ast.Assign([target], value), item))
            case "guard_assignment":  # `guard = v`: write the value it holds (checked by `guards`)
                guarded = self._guarded(item.child_by_field_name("guard"), ast.Store())
                assert guarded is not None
                value = self._expression(item.child_by_field_name("value"))
                statements.append(self._at(ast.Assign([guarded], value), item))
            case "local_assignment":
                statements.append(self._local_assignment(item))
            case "return_statement":
                statements.extend(self._return_statement(item))
            case _:
                statements.extend(self._statement(item))
        return statements

    def _local_assignment(self, item: Node) -> ast.stmt:
        """``let x = v``, or ``let x: T = v`` (a ``mem.Unique`` local is a plain value, lent to calls)."""
        assert self._scope is not None
        parts = _children(item)
        identifier, value = parts[0], parts[-1]
        if value.type != "expression":
            raise self._unsupported(value, "a nested function or block value")
        target = self._at(ast.Name(self._identifier(identifier), ast.Store()), identifier)
        type_node = item.child_by_field_name("type")
        if _text(identifier) in self._scope.cells:
            return self._cell_assignment(item, target, value)
        known = self._static_type(value)
        lowered = self._expression(value)
        if type_node is not None:
            found = self._container(type_node)
            known = scalar_type(self._type(found[1] if found is not None else type_node))
            self._check_record(
                value, known.python if isinstance(known, StructType) else None, f"{_text(identifier)} is"
            )
        self._scope.types[_text(identifier)] = known
        if type_node is None or self._is(mem.UNIQUE, type_node):
            if type_node is not None:
                self._type(type_node)  # resolve it, to report unknown types here
            return self._at(ast.Assign([target], lowered), item)
        self._scope.temporaries += 1
        annotation = f"__bifrost_type_{self._scope.temporaries}"
        self.globals[annotation] = self._type(type_node)
        annotation_name = self._at(ast.Name(annotation, ast.Load()), type_node)
        return self._at(ast.AnnAssign(target, annotation_name, lowered, simple=1), item)

    def _cell_assignment(self, item: Node, target: ast.Name, value: Node) -> ast.stmt:
        """``let c: mem.Shared[T] = v`` puts ``v`` in a new cell; ``let other = c`` is another owner of ``c``'s."""
        assert self._scope is not None
        name = target.id
        container, held = self._scope.held[name]
        self._scope.types[name] = scalar_type(held)
        inner = _unwrap(value)
        source = _text(inner) if inner.type == "identifier" else None
        found = self._cell_result(inner)
        if source in self._scope.cells or found is not None:
            other, other_held = self._scope.held[source] if source in self._scope.cells else found
            what = source or f"{_text(inner).split('(')[0]}(...)"
            if other is not container:
                raise self.error(inner, f"{what} is a {other!r}, not a {container!r}")
            if other_held != held:
                raise self.error(inner, f"{what} holds a {_type_label(other_held)}, not a {_type_label(held)}")
            if source is not None:
                self._scope.after.append(self._cell_statement("retain", name, item))
            return self._at(ast.Assign([target], self._expression(value)), item)
        # A new cell: allocate it, then move the value in.
        pointer = self._global(Ptr[held], "type")
        size = self._at(ast.Constant(scalar_type(held).size), item)
        raw = self._cell_call("new", container, [size], item)
        allocate = self._at(ast.Call(self._at(pointer, item), [raw], []), item)
        owner = self._at(ast.Name(name, ast.Load()), item)
        element = self._at(ast.Subscript(owner, ast.Constant(0), ast.Store()), item)
        self._scope.after.append(self._at(ast.Assign([element], self._expression(value)), item))
        annotation = self._at(self._global(Ptr[held], "type"), item)
        return self._at(ast.AnnAssign(target, annotation, allocate, simple=1), item)

    def _cell_call(self, action: str, container: mem.Container, arguments: list[ast.expr], node: Node) -> ast.Call:
        """Call the runtime for a cell: ``bifrost_shared_retain(c)``, ``bifrost_atomic_lock(c)``, ..."""
        kind = "shared" if container is mem.SHARED else "atomic"
        function = self._at(self._global(getattr(mem_runtime, f"bifrost_{kind}_{action}"), "mem"), node)
        return self._at(ast.Call(function, arguments, []), node)

    def _cell_statement(self, action: str, name: str, node: Node) -> ast.stmt:
        """``retain``, ``release``, or ``unlock`` the cell ``name`` holds, as a statement."""
        assert self._scope is not None
        pointer = self._at(ast.Name(name, ast.Load()), node)
        return self._at(ast.Expr(self._cell_call(action, self._scope.cells[name], [pointer], node)), node)

    def _lock(self, name: str, node: Node, conflict: str) -> ast.stmt:
        """Lock the cell ``name``: wait for a mem.Atomic's mutex; stop the program if a mem.Shared is held."""
        assert self._scope is not None
        container = self._scope.cells[name]
        pointer = self._at(ast.Name(name, ast.Load()), node)
        arguments: list[ast.expr] = [pointer]
        if container is mem.SHARED:
            where = f"{_display(self.path, self.root)}:{node.start_point[0] + 1}"
            message = f"{where}: {conflict}; a mem.Shared allows one at a time (a mem.Atomic waits for it)\n"
            arguments.append(self._at(ast.Constant(message), node))
        return self._at(ast.Expr(self._cell_call("lock", container, arguments, node)), node)

    def _retain_returned(self, value: Node | None) -> list[ast.stmt]:
        """Count the caller as another owner when a cell parameter is returned."""
        assert self._scope is not None
        inner = _unwrap(value) if value is not None else None
        if inner is None or inner.type != "identifier" or _text(inner) not in self._scope.cell_parameters:
            return []
        return [self._cell_statement("retain", _text(inner), inner)]

    def _check_guard_type(self, lock: Node) -> None:
        """Check a lock's written type (``let g: mem.WeakGuard[Context] <- ctx``) against what it locks."""
        assert self._scope is not None
        annotation = lock.child_by_field_name("type")
        if annotation is None:
            return
        source = _text(_unwrap(lock.child_by_field_name("source")))
        found = self._container(annotation)
        if found is None or found[0].guards is None:
            raise self.error(annotation, "a guard's type is a mem guard, like mem.WeakGuard[Context]")
        # A plain local is owned by its function alone, like a mem.Unique.
        container, held_type = self._scope.held.get(source, (mem.UNIQUE, None))
        expected = mem.guard_of(container)
        if found[0] is not expected:
            msg = f"{source} is a {container!r}, so locking it gives a {expected!r}, not a {found[0]!r}"
            raise self.error(annotation, msg)
        if held_type is not None and self._type(found[1]) != held_type:
            name = getattr(held_type, "__name__", repr(held_type))
            raise self.error(found[1], f"{source} holds a {name}, not a {_text(found[1])}")

    def _return_statement(self, node: Node) -> list[ast.stmt]:
        """``return value``, freeing the owners still alive first (after computing the value)."""
        assert self._scope is not None
        values = _children(node)
        if values:
            self._returned(values[0])
        retains = self._retain_returned(values[0] if values else None)
        value = self._expression(values[0]) if values else None
        # What runs after the returned call (unlocking a cell lent to it), then the frees.
        after, self._scope.after = self._scope.after, []
        frees = after + [self._free(name, node) for name in self._scope.plan.before_return.get(node.id, [])]
        if not frees:
            return [*retains, self._at(ast.Return(value), node)]
        if value is None:
            return [*retains, *frees, self._at(ast.Return(None), node)]
        self._scope.temporaries += 1
        result = f"__bifrost_result_{self._scope.temporaries}"
        store = self._at(ast.Assign([self._at(ast.Name(result, ast.Store()), node)], value), node)
        return [*retains, store, *frees, self._at(ast.Return(self._at(ast.Name(result, ast.Load()), node)), node)]

    def _arm_body(self, node: Node) -> list[ast.stmt]:
        match node.type:
            case "block_expression":
                return self._block(node)
            case "return_statement":
                before, self._scope.before = self._scope.before, []
                lowered = self._return_statement(node)
                statements = [*self._scope.before, *lowered]
                self._scope.before = before
                return statements
            case _:
                return self._statement(node)

    def _statement(self, node: Node) -> list[ast.stmt]:
        inner = _unwrap(node)
        match inner.type:
            case "if":
                return [self._if_statement(inner)]
            case "while":
                condition, body = _children(inner)
                return [
                    self._at(
                        ast.While(self._expression(condition), self._block(body), []),
                        inner,
                    )
                ]
            case "match_expression":
                return self._match_statement(inner)
            case "forall":
                return [self._forall(inner)]
            case _:
                return [self._at(ast.Expr(self._expression(inner)), inner)]

    def _forall(self, node: Node) -> ast.For:
        """``forall x in xs { ... }``, or ``forall i in #[0...10] { ... }`` (counting, with no list made)."""
        assert self._scope is not None
        identifier, iterated, body = _children(node)
        name = self._identifier(identifier)
        literal = _unwrap(iterated)
        inner = _children(literal)[0] if literal.type == "literal" else literal
        span = self._span(inner) if inner.type == "list" else None
        if span is not None:
            bounds = [self._expression(bound) for bound in span]
            iterable: ast.expr = self._at(ast.Call(self._at(ast.Name("range", ast.Load()), node), bounds, []), node)
            self._scope.types[name] = self._static_type(span[0]) or i64
        else:
            iterable = self._expression(iterated)
            kind = self._static_type(iterated)
            self._scope.types[name] = kind.element if kind is not None else None
        target = self._at(ast.Name(name, ast.Store()), identifier)
        return self._at(ast.For(target, iterable, self._block(body), []), node)

    def _if_statement(self, node: Node) -> ast.If:
        condition, body, *rest = _children(node)
        orelse: list[ast.stmt] = []
        if rest:
            orelse = [self._if_statement(rest[0])] if rest[0].type == "if" else self._block(rest[0])
        return self._at(ast.If(self._expression(condition), self._block(body), orelse), node)

    def _match_statement(self, node: Node) -> list[ast.stmt]:
        """``match x { 1: a, _: b }`` is ``if x == 1 { a } else { b }``."""
        scrutinee, *arms = _children(node)
        subject = self._expression(scrutinee)
        prelude: list[ast.stmt] = []
        if not isinstance(subject, ast.Name):
            assert self._scope is not None
            temporary = self._scope.temporary()
            store = self._at(ast.Name(temporary, ast.Store()), scrutinee)
            prelude.append(self._at(ast.Assign([store], subject), scrutinee))
            subject = self._at(ast.Name(temporary, ast.Load()), scrutinee)

        branches, default = self._match_arms(arms, subject, self._arm_body)
        chain: list[ast.stmt] = default or []
        for test, body, arm in reversed(branches):
            chain = [self._at(ast.If(test, body, chain), arm)]
        return prelude + chain

    def _match_arms[T](
        self, arms: list[Node], subject: ast.expr, lower: Callable[[Node], T]
    ) -> tuple[list[tuple[ast.expr, T, Node]], T | None]:
        branches: list[tuple[ast.expr, T, Node]] = []
        default: T | None = None
        for arm in arms:
            condition, body = _children(arm)
            if default is not None:
                raise self.error(arm, "this arm is unreachable: '_' already matched")
            if _unwrap(condition).type == "default_var":
                default = lower(body)
                continue
            test = self._at(
                ast.Compare(subject, [ast.Eq()], [self._expression(condition)]),
                condition,
            )
            branches.append((test, lower(body), arm))
        return branches, default

    # -- expressions --------------------------------------------------------------

    def _expression(self, node: Node) -> ast.expr:
        node = _unwrap(node)
        lower = _EXPRESSIONS.get(node.type)
        if lower is None:
            raise self._unsupported(node)
        return lower(self, node)

    def _unary(self, node: Node) -> ast.expr:
        operator = _UNARY_OPERATORS[_text(node.child_by_field_name("operator"))]
        operand = self._expression(node.child_by_field_name("argument"))
        return self._at(ast.UnaryOp(operator(), operand), node)

    def _function_call(self, node: Node) -> ast.expr:
        return self._call(_children(node)[0], None)

    def _get_expression(self, node: Node) -> ast.expr:
        owner, index = _children(node)
        if index.type == "rest_of":
            raise self._unsupported(index, "slicing")
        value = ast.Subscript(self._expression(owner), self._expression(index), ast.Load())
        return self._at(value, node)

    def _default_var(self, node: Node) -> ast.expr:
        raise self.error(node, "'_' is only a match arm's default")

    def _statement_only(self, node: Node) -> ast.expr:
        raise self.error(node, f"{node.type} is a statement, not a value")

    def _binary(self, node: Node) -> ast.expr:
        operator = _text(node.child_by_field_name("operator"))
        left = self._expression(node.child_by_field_name("left"))
        right = self._expression(node.child_by_field_name("right"))
        if operator in _COMPARISONS:
            return self._at(ast.Compare(left, [_COMPARISONS[operator]()], [right]), node)
        if operator in _BOOLEAN_OPERATORS:
            return self._at(ast.BoolOp(_BOOLEAN_OPERATORS[operator](), [left, right]), node)
        return self._at(ast.BinOp(left, _BINARY_OPERATORS[operator](), right), node)

    def _literal(self, node: Node) -> ast.expr:
        inner = _children(node)[0]
        match inner.type:
            case "number":
                number = _children(inner)[0]
                text = _text(number)
                value: object = (
                    float(text) if number.type == "float" else int(text, 10 if number.type == "integer" else 0)
                )
            case "string":
                value = self._unescape(inner)
            case "boolean":
                value = _text(inner) == "true"
            case "tuple":
                return self._at(
                    ast.Tuple([self._expression(e) for e in _children(inner)], ast.Load()),
                    node,
                )
            case "record":
                return self._record(inner)
            case "list":
                return self._list(inner)
            case _:
                raise self._unsupported(inner, f"a {inner.type} value")
        return self._at(ast.Constant(value), node)

    def _list(self, node: Node) -> ast.expr:
        """``#[1, 2, 3]``, or ``#[0...10]`` (0 to 9): a list, freed when nothing uses it any more."""
        items = _children(node)
        span = self._span(node)
        if span is None:
            if any(item.type != "expression" for item in items):
                raise self._unsupported(next(i for i in items if i.type != "expression"), "spreading into a list")
            return self._at(ast.List([self._expression(item) for item in items], ast.Load()), node)
        # A range as a list: fill a new one with start, start + 1, ... (end - 1).
        assert self._scope is not None
        start, end = span
        self._scope.temporaries += 1
        prefix = f"__bifrost_range_{self._scope.temporaries}"
        kind = self._static_type(start) or i64

        def name(identifier: str, context: ast.expr_context | None = None) -> ast.Name:
            return self._at(ast.Name(identifier, context or ast.Load()), node)

        count = self._at(ast.BinOp(name(f"{prefix}_end"), ast.Sub(), name(f"{prefix}_start")), node)
        allocate = ast.Call(name("__bifrost_array"), [self._global(kind, "type"), name(f"{prefix}_count")], [])
        self.globals["__bifrost_array"] = mlir_array
        element = ast.Subscript(name(prefix), name(f"{prefix}_index"), ast.Store())
        value = ast.BinOp(name(f"{prefix}_start"), ast.Add(), name(f"{prefix}_index"))
        fill = ast.For(
            name(f"{prefix}_index", ast.Store()),
            ast.Call(name("range"), [name(f"{prefix}_count")], []),
            [self._at(ast.Assign([self._at(element, node)], self._at(value, node)), node)],
            [],
        )
        self._scope.before += [
            self._at(ast.Assign([name(f"{prefix}_start", ast.Store())], self._expression(start)), node),
            self._at(ast.Assign([name(f"{prefix}_end", ast.Store())], self._expression(end)), node),
            self._at(ast.Assign([name(f"{prefix}_count", ast.Store())], count), node),
            self._at(ast.Assign([name(prefix, ast.Store())], self._at(allocate, node)), node),
            self._at(fill, node),
        ]
        return name(prefix)

    @staticmethod
    def _span(list_node: Node) -> tuple[Node, Node] | None:
        """For ``#[a...b]``, return ``a`` and ``b`` (a range, ``b`` excluded); ``None`` for other lists."""
        items = _children(list_node)
        if len(items) != 1 or items[0].type != "spread_between":
            return None
        start, end = (child for child in _children(items[0]) if child.type != "ellipsis")
        return start, end

    def _unescape(self, string: Node) -> str:
        r"""Decode a string literal's escapes: ``\n``, ``\t``, ``\r``, ``\0``, ``\\``, quotes, ``\xHH``."""
        text = _text(string)[1:-1]

        def decode(match: re.Match[str]) -> str:
            escape = match.group(1)
            if escape in _ESCAPES:
                return _ESCAPES[escape]
            if escape.startswith("x") and len(escape) == _HEX_ESCAPE:
                return chr(int(escape[1:], 16))
            row, column = string.start_point
            point = (row, column + 1 + match.start())
            msg = f"unknown escape '\\{escape}' (use \\n, \\t, \\r, \\0, \\\\, \\', \\\" or \\xHH)"
            raise BifrostError(msg, self.path, self.source, point)

        return re.sub(r"\\(x[0-9a-fA-F]{2}|.)", decode, text, flags=re.DOTALL)

    def _load(self, node: Node) -> ast.expr:
        name = self._identifier(node)
        assert self._scope is not None
        qualified = self._qualify(name)
        if qualified != name and name not in self._scope.names:
            module = self._at(ast.Name(self._module, ast.Load()), node)
            return self._at(ast.Attribute(module, name, ast.Load()), node)
        if not ({name} & (self._scope.names | self.globals.keys() | self._functions)):
            if any(name in outer.names for outer in self._outer):
                msg = f"a lambda cannot use '{name}' of the function around it (yet); pass it as a parameter"
                raise self.error(node, msg)
            raise self.error(node, f"'{name}' is not defined")
        return self._at(ast.Name(name, ast.Load()), node)

    def _value(self, node: Node) -> ast.expr:
        """Lower a name used as a value; a function's name is the function itself, a dependency."""
        assert self._scope is not None
        name = self._identifier(node)
        qualified = self._qualify(name)
        if qualified in self._functions and name not in self._scope.names:
            self._depend(qualified, node, "uses")
        guarded = self._guarded(node)  # a guard on a whole value reads it (checked by `guards`)
        return guarded if guarded is not None else self._load(node)

    def _lambda(self, node: Node) -> ast.expr:
        """Lift a function written where a value goes to a function of its own."""
        outer = self._scope
        assert outer is not None
        row, column = node.start_point
        name = f"the lambda on line {row + 1}"
        symbol = f"{outer.symbol}_lambda_{row + 1}_{column}"
        self._signature(name, node, symbol)
        if _is_async(node):
            self._pausing.add(name)
            self.project.pausing.add(symbol)
        self._outer.append(outer)
        try:
            python = self._function(name, node, symbol)
        finally:
            self._outer.pop()
            self._scope = outer
        self.globals[symbol] = self.project.program.function(python)
        return self._at(ast.Name(symbol, ast.Load()), node)

    def _child_annotation(self, node: Node) -> ast.expr:
        """``a.b.f(x).c``: attribute access and calls, left to right."""
        value: ast.expr | None = None
        path: list[str] = []  # the dotted name so far, e.g. ["raylib"]
        for part in _children(node):
            if part.type == "simple_identifier":
                if value is None:
                    value = self._guarded(part) or self._load(part)
                else:
                    value = self._at(ast.Attribute(value, _text(part), ast.Load()), part)
                path.append(_text(part))
            else:
                value = self._call(_children(part)[0], value, path)
                path = []  # a call's result is a value, not a name
        assert value is not None
        if path and len(path) == len(_children(node)):
            self._depend_on_value(path, node)
        return value

    def _depend_on_value(self, path: list[str], node: Node) -> None:
        """``helper.greet`` or ``http.text`` used as a value: a function, so a dependency."""
        assert self._scope is not None
        if path[0] in self._scope.names or path[0] in self._scope.guards:
            return  # a field of a local
        names = [*self._qualify(path[0]).split("."), *path[1:]]
        dotted = ".".join(names)
        if dotted in self._functions:
            self._depend(dotted, node, "uses")
            return
        target: object = self.globals.get(names[0])
        for attribute in names[1:]:
            target = getattr(target, attribute, None)
        if isinstance(target, fmt.Builtin):
            raise self.error(node, f"{dotted} is built into the compiler, so it cannot be a function value")
        if isinstance(target, Function):
            self._depend(".".join(path), node, "uses")

    def _guarded(self, part: Node, context: ast.expr_context | None = None) -> ast.expr | None:
        """Return what a guard reaches: ``ctx[0]`` for a locked pointer, the value for a locked local."""
        assert self._scope is not None
        source = self._scope.guards.get(_text(part))
        if source is None:
            return None
        if source in self._scope.pointers:
            pointer = self._at(ast.Name(source, ast.Load()), part)
            return self._at(ast.Subscript(pointer, ast.Constant(0), context or ast.Load()), part)
        return self._at(ast.Name(source, context or ast.Load()), part)

    def _field_target(self, target: Node) -> ast.expr:
        """``guard.a.b`` as an assignment target."""
        root, *fields = _children(target)
        value = self._guarded(root)
        assert value is not None  # `guards` allows field assignment only through a guard
        for index, field_node in enumerate(fields):
            context = ast.Store() if index == len(fields) - 1 else ast.Load()
            value = self._at(ast.Attribute(value, _text(field_node), context), field_node)
        return value

    def _call(self, node: Node, owner: ast.expr | None, path: list[str] | None = None) -> ast.expr:
        if node.type == "builtin_call":
            return self._language_call(node)
        function = node.child_by_field_name("function")
        name = self._identifier(function)
        assert self._scope is not None
        if path and path[0] not in self._scope.names:
            path = [*self._qualify(path[0]).split("."), *path[1:]]
        callee_name = ".".join([*path, name]) if owner is not None and path else self._qualify(name)
        pointees = self._pointer_params.get(callee_name, [])
        pauses = callee_name in self._pausing and name not in self._scope.names
        target: object = None
        if owner is None:
            callee: ast.expr = self._load(function)
            if callee_name in self._functions and name not in self._scope.names:
                self._depend(callee_name, function)
        else:
            callee = self._at(ast.Attribute(owner, name, ast.Load()), function)
            target = self._member_target(path or [], function)
            if isinstance(target, fmt.Builtin):
                return self._builtin_call(target, node)
            if isinstance(target, Function) and target.kind != "extern":
                pointees = self.project.pointer_params.get(target.name, pointees)  # a function of another file
            if isinstance(target, Function) and target.kind == "extern":
                pointees = self._extern_pointees(target)
            pauses = pauses or self._pauses(target)
        dotted = ".".join([*(path or []), name])
        self._check_record_arguments(node, target if owner is not None else callee_name, dotted)
        call = self._at(ast.Call(callee, *self._arguments(node, pointees)), node)
        if not pauses and node.id in self._scope.awaited:
            raise self.error(node, f"{dotted}(...) does not pause, so there is nothing to await")
        return self._awaited(call, node, dotted) if pauses else call

    def _language_call(self, node: Node) -> ast.expr:
        """Lower a call of a function of the language itself: ``len(xs)``."""
        builtin = _text(node.child_by_field_name("function"))
        arguments = [argument for argument in _children(node) if argument.type == "expression"]
        if builtin != "len":
            raise self._unsupported(node, f"{builtin}()")
        if len(arguments) != 1:
            raise self.error(node, "len takes one list: len(xs)")
        length = ast.Call(self._at(ast.Name("len", ast.Load()), node), [self._expression(arguments[0])], [])
        return self._at(length, node)

    def _builtin_call(self, builtin: fmt.Builtin, node: Node) -> ast.expr:
        """Expand a call of a function built into the compiler: ``fmt.format``, ``json.encode``, ``tasks.*``."""
        expansions = {
            fmt.FORMAT: self._format,
            json.ENCODE: self._encode_call,
            tasks.GATHER: self._gather,
            tasks.IGNORE: self._ignore,
        }
        return expansions[builtin](node)

    def _extern_pointees(self, extern: Function) -> list[object | None]:
        """Return what a C function's parameters take: ``void *``, ``json``, a task callback, or plain values."""
        pointees = self._opaque_params(extern)
        marked = [(index, _JSON) for index in self.project.json_parameters.get(extern.name, set())]
        for index, marker in [*marked, *((index, _TASK) for index in self._task_params(extern))]:
            pointees = [*pointees, *[None] * (index + 1 - len(pointees))]
            pointees[index] = marker
        return pointees

    def _awaited(self, call: ast.Call, node: Node, name: str) -> ast.expr:
        """Wait for a call that pauses, written ``await f(x)`` in an async function."""
        assert self._scope is not None
        if node.id not in self._scope.awaited:
            msg = (
                f"{name}(...) is async and is not awaited. Did you forget `await {name}(...)`? "
                f"If not waiting is deliberate, start it with `tasks.ignore({name}(...))`"
            )
            raise self.error(node, msg)
        return self._at(ast.Await(call), node)

    def _await(self, node: Node) -> ast.expr:
        """``await f(x)``: the call must pause, and the function must be ``async``."""
        assert self._scope is not None
        value = _unwrap(node.child_by_field_name("value"))
        call = _children(value)[-1] if value.type == "child_annotation" else value
        if call.type != "function_call":
            raise self.error(value, "await a call to a function that pauses: `await http.sleep(100)`")
        inner = _children(call)[0]
        function = inner.child_by_field_name("function")
        name = _text(value).split("(")[0] if function is not None else _text(inner)
        if not self._pauses(self._callee(value)):
            raise self.error(node, f"{name}(...) does not pause, so there is nothing to await; call it without `await`")
        if not self._scope.is_async:
            where = self._scope.function
            hint = " (with `type: async` in config.yaml's package)" if where == "main" and self.is_root else ""
            raise self.error(node, f"`await` is only allowed in an async function; mark {where} `async`{hint}")
        self._scope.awaited.add(inner.id)
        return self._expression(value)

    def _body(self, body: Node, *, returns: bool) -> list[ast.stmt]:
        """Lower a function's body: a block, or one expression (its result, if it ``returns``)."""
        assert self._scope is not None
        if body.type == "block_expression":
            return self._block(body)
        if returns:
            self._returned(body)
        value = self._expression(body)
        result = self._at(ast.Return(value) if returns else ast.Expr(value), body)
        return [*self._scope.before, *self._retain_returned(body), result]

    def _returned(self, value: Node) -> None:
        """Note what a function returns: the shape of its ``Record``, or checked against its written record type."""
        assert self._scope is not None
        if self._scope.results is not None:
            self._scope.results.append((value, self._static_type(value)))
            return
        self._check_record(value, self._scope.result, f"{self._scope.function} returns")

    def _check_record(self, value: Node, expected: object, what: str) -> None:
        """Where a record type is expected (``what`` it is for), reject a record with other fields, naming both."""
        if not (isinstance(expected, type) and expected.__name__ == "record"):
            return
        kind = self._static_type(value)
        if isinstance(kind, StructType) and kind.python.__name__ == "record" and kind.python is not expected:
            wanted = _shown(scalar_type(expected))
            raise self.error(value, f"{what} {wanted}, but this is {_shown(kind)}; give it the fields of {wanted}")

    def _check_record_arguments(self, call: Node, callee: object, name: str) -> None:
        """Check the records passed to a Bifrost function against its parameters' record types."""
        if not (isinstance(callee, str) and callee in self._function_nodes) and not (
            isinstance(callee, Function) and callee.kind != "extern"
        ):
            return
        parameters = self._call_signature(callee)[1]
        arguments = [argument for argument in _children(call) if argument.type == "expression"]
        for argument, expected in zip(arguments, parameters, strict=False):
            self._check_record(argument, expected, f"{name} takes")

    def _inferred_record(self, name: str, results: list[tuple[Node, ScalarType | None]]) -> type:
        """Return the record type a function written ``=> Record`` returns: one shape, from all its returns."""
        first: tuple[Node, StructType] | None = None
        for value, kind in results:
            if kind is None:
                msg = f"cannot tell the record {name} returns here; name its fields' values first: let x: i64 = ..."
                raise self.error(value, msg)
            if not (isinstance(kind, StructType) and kind.python.__name__ == "record"):
                raise self.error(value, f"{name} returns a {RECORD}, but this returns {_shown(kind)}")
            if first is None:
                first = (value, kind)
            elif kind.python is not first[1].python:
                msg = (
                    f"{name} returns records of different shapes: {_shown(first[1])} on line "
                    f"{first[0].start_point[0] + 1}, and {_shown(kind)} here; consolidate them into one record "
                    f"with the same fields (or write the result type: => #{{...}})"
                )
                raise self.error(value, msg)
        assert first is not None  # `_check_returns`: it returns on every path
        self._inferred[name] = first[1].python
        return first[1].python

    def _record_result(self, name: str, return_node: Node) -> type:
        """Return the record the ``=> Record`` function ``name`` returns, lowering it first if need be."""
        if name in self._inferred:
            return self._inferred[name]
        if name in self._inferring:
            msg = f"the record {name} returns depends on a call of {name} itself; write its type: => #{{...}}"
            raise self.error(return_node, msg)
        pending = self._pending[name]
        scope, module = self._scope, self._module
        self._module = pending.module
        try:
            self._lowered[name] = self._function(pending.name, pending.node, pending.symbol)
        finally:
            self._scope, self._module = scope, module
        return self._inferred[name]

    def _check_returns(self, name: str, return_node: Node, body: Node, result: object) -> None:
        """Require a function with a result to return it on every path, before anything else is checked in it."""
        if result is type(None) or body.type != "block_expression" or _returns(body):
            return
        written = " ".join(_text(return_node).split())
        msg = f"{name} must return {written}, but its body can end without a `return`; return a {written} at the end"
        raise self.error(return_node, msg)

    def _check_async_main(self, node: Node) -> None:
        """Allow an async ``main`` only when config.yaml says so (``package.type: async``)."""
        if self.project.config.package.type == "async":
            return
        keyword = node.child_by_field_name("async") or node
        msg = "main is async, but config.yaml's package.type is sync; set `type: async` to run main on the event loop"
        raise self.error(keyword, msg)

    def _member_target(self, path: list[str], function: Node) -> object:
        """Check a call of ``path.function`` against the dependency list; return what it calls, when known."""
        assert self._scope is not None
        name = _text(function)
        dotted = ".".join([*path, name]) if path else None
        if dotted in self._functions and path[0] not in self._scope.names:
            self._depend(dotted, function)  # an object's static function, e.g. Context.new
            return None
        if not path or path[0] in self._scope.names or path[0] not in self.globals:
            return None
        target = self.globals[path[0]]
        for attribute in [*path[1:], name]:
            target = getattr(target, attribute, None)
        if isinstance(target, (Function, fmt.Builtin)):
            self._depend(".".join([*path, name]), function)
        return target

    @staticmethod
    def _task_params(extern: Function) -> list[int]:
        """Return the indexes of a C function's parameters that take a callback returning a ``token``."""
        annotations = getattr(extern.python, "__annotations__", {})
        kinds = [kind for name, kind in annotations.items() if name != "return"]
        return [index for index, kind in enumerate(kinds) if isinstance(kind, FnType) and kind.result == Token]

    def _task_argument(self, argument: Node) -> ast.expr:
        """Pass a function where C starts a task: as is if it pauses, else wrapped in an async function."""
        value = self._expression(argument)
        found = self._function_reference(_unwrap(argument), value)
        if found is None:
            return value  # a function value held in a local: compiled code checks its type
        pausing, annotations = found
        if pausing:
            return value  # its function value starts it, returning its token
        assert self._scope is not None
        row, column = argument.start_point
        symbol = f"{self._scope.symbol}_task_{row + 1}_{column}"
        parameters = [name for name in annotations if name != "return"]

        def name(identifier: str) -> ast.Name:
            return self._at(ast.Name(identifier, ast.Load()), argument)

        call = self._at(ast.Call(value, [name(parameter) for parameter in parameters], []), argument)
        definition = self._at(
            ast.AsyncFunctionDef(
                name=symbol,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[self._at(ast.arg(parameter), argument) for parameter in parameters],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[self._at(ast.Expr(call), argument)],
                decorator_list=[],
                type_params=[],
            ),
            argument,
        )
        wrapper = {parameter: annotations[parameter] for parameter in parameters} | {"return": type(None)}
        self.globals[symbol] = self.project.program.function(self._compile(definition, wrapper))
        return name(symbol)

    def _function_reference(self, node: Node, value: ast.expr) -> tuple[bool, dict[str, object]] | None:
        """For a function named or written as ``node``: whether it pauses, and its parameter types."""
        assert self._scope is not None
        if node.type == "local_function_definition":  # a lambda, lowered to `value`
            assert isinstance(value, ast.Name)
            compiled = self.globals[value.id]
            return compiled.name in self.project.pausing, dict(compiled.python.__annotations__)
        if node.type not in {"identifier", "child_annotation"} or _text(node).split(".")[0] in self._scope.names:
            return None
        parts = _text(node).split(".")
        qualified = ".".join([self._qualify(parts[0]), *parts[1:]])
        if qualified in self._function_nodes:
            return qualified in self._pausing, self._annotations(self._function_nodes[qualified])
        target: object = self.globals.get(parts[0])
        for attribute in parts[1:]:
            target = getattr(target, attribute, None)
        if isinstance(target, Function) and target.kind != "extern":
            return target.name in self.project.pausing, dict(target.python.__annotations__)
        return None

    def _annotations(self, function: Node) -> dict[str, object]:
        """Return a function's parameter types, as its compiled parameters take them."""
        parts = _children(function)
        parameter_list = next(part for part in parts if part.type == "parameter_list")
        return {
            self._identifier(_children(parameter)[0]): self._type(_children(parameter)[1])
            for parameter in _children(parameter_list)
        }

    @staticmethod
    def _opaque_params(extern: Function) -> list[object | None]:
        """Mark a C function's ``ptr`` (``void *``) parameters: an owned local passed to one is lent."""
        annotations = getattr(extern.python, "__annotations__", {})
        return [_OPAQUE if kind is ptr else None for name, kind in annotations.items() if name != "return"]

    # -- running async calls together ----------------------------------------------

    def _gather_calls(self, call: Node) -> list[tuple[str | None, Node, ScalarType | None, list[object]]]:
        """Check the arguments of ``tasks.gather``; return each one's name, call, result type and parameter types."""
        entries: list[tuple[str | None, Node, ScalarType | None, list[object]]] = []
        for argument in _children(call):
            if argument.type not in {"expression", "named_argument"}:
                continue
            named = argument.type == "named_argument"
            name = self._identifier(argument.child_by_field_name("name")) if named else None
            value = argument.child_by_field_name("value") if named else argument
            inner = _unwrap(value)
            callee = self._callee(inner)
            what = _text(inner).split("(")[0]
            if inner.type not in {"function_call", "child_annotation"} or not self._pauses(callee):
                msg = f"tasks.gather runs calls that pause, like fetch_user(id); {what} is not one"
                raise self.error(inner, msg)
            if any(name is not None and name == other for other, *_ in entries):
                raise self.error(argument, f"tasks.gather already has a result named '{name}'")
            result, parameters, pointees = self._call_signature(callee)
            if any(p is not None and not isinstance(p, _Cell) for p in pointees):
                msg = f"{what} borrows a mem.Weak, which tasks.gather cannot lend to a call running alongside others"
                raise self.error(inner, msg)
            if name is None and result is not None:
                raise self.error(argument, f"{what}(...) returns a value; name it: tasks.gather(result: {what}(...))")
            if name is not None and result is None:
                raise self.error(argument, f"{what}(...) returns nothing; pass it without a name")
            entries.append((name, value, result, parameters))
        if not entries:
            raise self.error(call, "tasks.gather takes the calls to run: tasks.gather(user: fetch_user(id))")
        return entries

    def _call_signature(self, callee: object) -> tuple[ScalarType | None, list[object], list[object | None]]:
        """Return what calling ``callee`` gives, its parameter types, and what each parameter takes."""
        if isinstance(callee, str):
            node = self._function_nodes[callee]
            return self._returns(callee), list(self._annotations(node).values()), self._pointer_params[callee]
        assert isinstance(callee, Function)
        annotations = dict(getattr(callee.python, "__annotations__", {}))
        result = annotations.pop("return", None)
        pointees = self.project.pointer_params.get(callee.name, []) if callee.kind != "extern" else []
        returned = None if result in (None, type(None), Token) else scalar_type(result)
        return returned, list(annotations.values()), pointees

    def _gather(self, call: Node) -> ast.expr:
        """``await tasks.gather(a: f(x), b: g(y))``: start each call, then wait for all; a record of the results.

        Each call runs in a small async function of its own, which stores its
        result in a slot of the caller's; the caller starts them all (through
        their function values, which return their tokens), then awaits each token.
        """
        assert self._scope is not None
        if call.id not in self._scope.awaited:
            raise self.error(call, "tasks.gather(...) pauses; wait for it with `await tasks.gather(...)`")
        self._scope.temporaries += 1
        number = self._scope.temporaries
        self.globals["__bifrost_stack"] = stack

        def name(identifier: str, context: ast.expr_context | None = None) -> ast.Name:
            return self._at(ast.Name(identifier, context or ast.Load()), call)

        def assign(target: str, value: ast.expr) -> ast.stmt:
            return self._at(ast.Assign([name(target, ast.Store())], value), call)

        tokens: list[str] = []
        fields: list[tuple[str, object]] = []
        results: list[ast.keyword] = []
        for index, (field, value, result, parameters) in enumerate(self._gather_calls(call)):
            prefix = f"__bifrost_gather_{number}_{index}"
            symbol, target = self._task_function(value, "gather", result, parameters)
            passed = list(target.args)
            if result is not None:
                allocate = ast.Call(name("__bifrost_stack"), [self._global(result, "type")], [])
                self._scope.before.append(assign(f"{prefix}_slot", self._at(allocate, call)))
                passed = [name(f"{prefix}_slot"), *passed]
                fields.append((field, result.python if isinstance(result, StructType) else result))
                read = ast.Subscript(name(f"{prefix}_slot"), ast.Constant(0), ast.Load())
                results.append(self._at(ast.keyword(arg=field, value=self._at(read, call)), call))
            # Its function value starts it and returns its token.
            self._scope.before.append(assign(f"{prefix}_start", name(symbol)))
            self._scope.before.append(assign(prefix, self._at(ast.Call(name(f"{prefix}_start"), passed, []), call)))
            tokens.append(prefix)
        self._scope.before += [self._at(ast.Expr(self._at(ast.Await(name(token)), call)), call) for token in tokens]
        if not fields:
            if not _is_statement(call):
                msg = (
                    "tasks.gather(...) gives nothing here, since none of its calls returns a value; "
                    "write `await tasks.gather(...)` on its own, or name the calls that return values"
                )
                raise self.error(call, msg)
            return self._at(ast.Constant(0), call)  # a statement: its value is never read
        record = self._record_type(tuple(fields))
        return self._at(ast.Call(self._global(record, "record"), [], results), call)

    def _task_function(
        self, value: Node, kind: str, result: ScalarType | None, parameters: list[object]
    ) -> tuple[str, ast.Call]:
        """Define a small async function that awaits the call ``value`` and stores its result, if any.

        It takes the call's arguments (after a slot for the result, a
        ``Ptr[result]``); its function value starts it and returns its token.
        Return its name, and the call as lowered (the callee and its arguments, checked).
        """
        assert self._scope is not None
        inner = _unwrap(value)
        function_call = _children(inner)[-1] if inner.type == "child_annotation" else inner
        self._scope.awaited.add(_children(function_call)[0].id)
        lowered = self._expression(value)
        assert isinstance(lowered, ast.Await)
        assert isinstance(lowered.value, ast.Call)
        target = lowered.value

        def name(identifier: str, context: ast.expr_context | None = None) -> ast.Name:
            return self._at(ast.Name(identifier, context or ast.Load()), value)

        arguments = [f"__bifrost_argument_{i}" for i in range(len(target.args))]
        annotations: dict[str, object] = dict(zip(arguments, parameters, strict=True))
        waited: ast.expr = ast.Await(ast.Call(target.func, [name(a) for a in arguments], []))
        if result is None:
            body: ast.stmt = ast.Expr(waited)
        else:
            annotations = {"__bifrost_slot": Ptr[result], **annotations}
            slot = ast.Subscript(name("__bifrost_slot"), ast.Constant(0), ast.Store())
            body = ast.Assign([slot], waited)
        row, column = value.start_point
        symbol = f"{self._scope.symbol}_{kind}_{row + 1}_{column}"
        definition = self._at(
            ast.AsyncFunctionDef(
                name=symbol,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[self._at(ast.arg(a), value) for a in annotations],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[self._at(body, value)],
                decorator_list=[],
                type_params=[],
            ),
            value,
        )
        ast.fix_missing_locations(definition)
        self.globals[symbol] = self.project.program.function(
            self._compile(definition, {**annotations, "return": type(None)})
        )
        return symbol, target

    def _ignored_call(self, call: Node) -> tuple[Node, list[object]]:
        """Check the argument of ``tasks.ignore``; return the call and its parameter types."""
        arguments = [argument for argument in _children(call) if argument.type in {"expression", "named_argument"}]
        if len(arguments) != 1 or arguments[0].type != "expression":
            raise self.error(call, 'tasks.ignore takes one call to start: tasks.ignore(log("done"))')
        value = arguments[0]
        inner = _unwrap(value)
        callee = self._callee(inner)
        what = _text(inner).split("(")[0]
        if inner.type not in {"function_call", "child_annotation"} or not self._pauses(callee):
            raise self.error(inner, f"tasks.ignore starts a call that pauses, like log(text); {what} is not one")
        result, parameters, pointees = self._call_signature(callee)
        if result is not None:
            msg = f"{what}(...) returns a value, which tasks.ignore would drop; wait for it with `await {what}(...)`"
            raise self.error(inner, msg)
        if any(pointee is not None for pointee in pointees):
            msg = f"{what} borrows a value, which could be gone before a call nobody waits for is done"
            raise self.error(inner, msg)
        function_call = _children(inner)[-1] if inner.type == "child_annotation" else inner
        passed = [argument for argument in _children(_children(function_call)[0]) if argument.type == "expression"]
        for argument, kind in zip(passed, parameters, strict=False):
            if kind is cstr:
                literal = _unwrap(argument)
                if literal.type == "literal" and _children(literal)[0].type == "string":
                    continue
            elif scalar_type(kind).kind in {"int", "uint", "float", "bool"}:
                continue
            msg = (
                f"tasks.ignore passes {what} numbers, booleans and string literals: "
                f"{_text(argument)} could be gone before a call nobody waits for is done"
            )
            raise self.error(argument, msg)
        return value, parameters

    def _ignore(self, call: Node) -> ast.expr:
        """``tasks.ignore(log("done"))``: start a call and let go of its token, not waiting for it."""
        assert self._scope is not None
        value, parameters = self._ignored_call(call)
        symbol, target = self._task_function(value, "ignore", None, parameters)
        self._scope.temporaries += 1
        prefix = f"__bifrost_ignore_{self._scope.temporaries}"
        self.globals["__bifrost_forget"] = self.project.runtime_extern(tasks.forget)

        def name(identifier: str, context: ast.expr_context | None = None) -> ast.Name:
            return self._at(ast.Name(identifier, context or ast.Load()), call)

        # Its function value starts it and returns its token, which nothing waits on.
        self._scope.before += [
            self._at(ast.Assign([name(f"{prefix}_start", ast.Store())], name(symbol)), call),
            self._at(ast.Assign([name(prefix, ast.Store())], ast.Call(name(f"{prefix}_start"), target.args, [])), call),
        ]
        return self._at(ast.Call(name("__bifrost_forget"), [name(prefix)], []), call)

    def _await_type(self, node: Node) -> ScalarType | None:
        """Return the type of ``await x``: a gather's record, or what the awaited call returns."""
        value = _unwrap(node.child_by_field_name("value"))
        if self._callee(value) is not tasks.GATHER:
            return self._static_type(value)
        call = _children(_children(value)[-1])[0]
        fields = [
            (name, result.python if isinstance(result, StructType) else result)
            for name, _, result, _ in self._gather_calls(call)
            if name is not None and result is not None
        ]
        return scalar_type(self._record_type(tuple(fields))) if fields else None

    # -- records and JSON ----------------------------------------------------------

    def _record(self, node: Node) -> ast.expr:
        """``#{id: 7, name: n}``: a value of an unnamed object type, its fields typed by their values."""
        assert self._scope is not None
        fields: list[tuple[str, object]] = []
        values: list[ast.keyword] = []
        for field in _children(node):
            name_node, value = field.child_by_field_name("name"), field.child_by_field_name("value")
            name = self._identifier(name_node)
            if any(name == other for other, _ in fields):
                raise self.error(name_node, f"the record already has a field '{name}'")
            kind = self._static_type(value)
            if kind is None:
                msg = f"cannot tell the type of field '{name}'; name the value first: let {name}: i64 = ..."
                raise self.error(value, msg)
            fields.append((name, kind.python if isinstance(kind, StructType) else kind))
            values.append(self._at(ast.keyword(arg=name, value=self._expression(value)), field))
        if not fields:
            raise self.error(node, "a record needs a field: #{name: value}")
        record = self._record_type(tuple(fields))
        return self._at(ast.Call(self._global(record, "record"), [], values), node)

    def _record_type(self, fields: tuple[tuple[str, object], ...]) -> type:
        """Return the object type of records with these fields (one type per set of fields, so they mix).

        The fields' order does not matter: ``#{y: 2, x: 1}`` is a ``#{x: i64, y: i64}``.
        Its fields are laid out (and written as JSON) in the order first met.
        """
        key = tuple(sorted(fields, key=lambda field: field[0]))
        if key not in self._records:
            self._records[key] = struct(type("record", (), {"__annotations__": dict(fields)}))
        return self._records[key]

    def _global(self, value: object, hint: str) -> ast.Name:
        """Return a name the generated code can use for ``value`` (a type, a runtime function)."""
        name = f"__bifrost_{hint}_{id(value)}"
        self.globals[name] = value
        return ast.Name(name, ast.Load())

    def _encode_call(self, call: Node) -> ast.expr:
        """``json.encode(value)``: its JSON text, as an owned string."""
        positional, named = self._arguments(call)
        del positional
        arguments = [a for a in _children(call) if a.type == "expression"]
        if named or len(arguments) != 1:
            raise self.error(call, "json.encode takes one value: json.encode(#{id: 7})")
        return self._encode(arguments[0])

    def _json_argument(self, argument: Node) -> ast.expr:
        """Encode a record or object passed to a C ``json`` parameter, and free it after the call."""
        assert self._scope is not None
        kind = self._static_type(argument)
        if not isinstance(kind, StructType):
            return self._expression(argument)  # a str: JSON text already
        text = self._encode(argument)
        self._scope.after.append(self._free(text.id, argument))
        return text

    def _encode(self, value: Node) -> ast.Name:
        """Append statements that write ``value``'s JSON text; return the name holding the owned text."""
        assert self._scope is not None
        kind = self._static_type(value)
        if not isinstance(kind, StructType):
            name = kind.name if kind is not None else "an unknown type"
            raise self.error(value, f"json.encode takes a record or an object, not {name}")
        self._scope.temporaries += 1
        number = self._scope.temporaries
        writer = _JsonWriter(self, value, f"__bifrost_json_{number}")
        writer.write(self._expression(value), kind)
        writer.flush()
        text = self._at(ast.Name(f"__bifrost_json_text_{number}", ast.Store()), value)
        self.globals["__bifrost_stack"] = stack
        allocate = ast.Call(
            ast.Name("__bifrost_stack", ast.Load()), [self._global(json_runtime.JsonBuffer, "type")], []
        )
        self._scope.before += [
            self._at(ast.Assign([writer.name(ast.Store())], self._at(allocate, value)), value),
            writer.call("begin"),
            *writer.statements,
            self._at(ast.Assign([text], self._at(writer.invoke("end"), value)), value),
        ]
        return self._at(ast.Name(text.id, ast.Load()), value)

    # -- static types ------------------------------------------------------------------

    def _static_type(self, node: Node) -> ScalarType | None:
        """Return an expression's type when the compiler can tell before compiling it, else ``None``.

        It can for literals, names, fields, calls and arithmetic.
        """
        node = _unwrap(node)
        find = _STATIC_TYPES.get(node.type)
        return find(self, node) if find is not None else None

    def _call_type(self, node: Node) -> ScalarType | None:
        inner = _children(node)[0]
        if inner.type == "builtin_call" and _text(inner.child_by_field_name("function")) == "len":
            return i64
        function = inner.child_by_field_name("function") if inner.type == "user_function_call" else None
        return self._returns(self._qualify(_text(function))) if function is not None else None

    def _unary_type(self, node: Node) -> ScalarType | None:
        if _text(node.child_by_field_name("operator")) == "!":
            return scalar_type(bool)
        return self._static_type(node.child_by_field_name("argument"))

    def _literal_type(self, literal: Node) -> ScalarType | None:
        match literal.type:
            case "number":
                return f64 if _children(literal)[0].type == "float" else i64
            case "string":
                return cstr
            case "boolean":
                return scalar_type(bool)
            case "list":
                return self._list_type(literal)
            case "record":
                return self._record_literal_type(literal)
            case _:
                return None

    def _record_literal_type(self, literal: Node) -> ScalarType | None:
        """``#{id: 7}``: an object type with a field per value, if every value's type is known."""
        fields = []
        for field in _children(literal):
            kind = self._static_type(field.child_by_field_name("value"))
            if kind is None:
                return None
            name = _text(field.child_by_field_name("name"))
            fields.append((name, kind.python if isinstance(kind, StructType) else kind))
        return scalar_type(self._record_type(tuple(fields))) if fields else None

    def _list_type(self, literal: Node) -> ScalarType | None:
        """``#[1, 2]`` is ``i64[]``; ``#[0...n]`` is a list of ``0``'s type."""
        span = self._span(literal)
        first = span[0] if span is not None else next(iter(_children(literal)), None)
        kind = self._static_type(first) if first is not None else None
        if kind is None and span is not None:
            kind = i64
        return scalar_type(Array[kind]) if kind is not None and kind.kind in {"int", "uint", "float", "bool"} else None

    def _name_type(self, name: str) -> ScalarType | None:
        assert self._scope is not None
        name = self._scope.guards.get(name, name)  # a guard reaches what it locks
        if name in self._scope.held:
            return scalar_type(self._scope.held[name][1])
        return self._scope.types.get(name)

    def _path_type(self, node: Node) -> ScalarType | None:
        """``user.name``, ``s.hits``, ``http.param(ctx, "id")``, ``Context.new().width``."""
        assert self._scope is not None
        parts = _children(node)
        kind: ScalarType | None = None
        path: list[str] = []
        for index, part in enumerate(parts):
            if part.type == "simple_identifier":
                if index == 0 and _text(part) in (self._scope.names | set(self._scope.guards)):
                    kind = self._name_type(_text(part))
                elif kind is not None:
                    found = kind.field(_text(part)) if isinstance(kind, StructType) else None
                    kind = found[1] if found else None
                else:
                    path.append(_text(part))
            else:
                function = _children(part)[0].child_by_field_name("function")
                if function is None or kind is not None:
                    return None  # a call on a value: methods do not exist yet
                kind = self._returns(".".join([*path, _text(function)]))
                path = []
                if kind is None:
                    return None
        return kind

    def _returns(self, dotted: str) -> ScalarType | None:
        """Return the result type of calling ``dotted``: a function of this file, an extern, or a constructor."""
        first, _, rest = dotted.partition(".")
        qualified = f"{self._qualify(first)}.{rest}" if rest else self._qualify(first)
        node = self._function_nodes.get(qualified)
        if node is not None:
            parts = [part for part in _children(node) if part.type not in {"dependency_list", "local_dependency_list"}]
            if _text(parts[1]) == RECORD:
                return scalar_type(self._record_result(qualified, parts[1]))
            found = self._container(parts[1])
            return scalar_type(self._type(found[1] if found is not None else parts[1]))
        names = qualified.split(".")
        target: object = self.globals.get(names[0])
        for attribute in names[1:]:
            target = getattr(target, attribute, None)
        if target is fmt.FORMAT or target is json.ENCODE:
            return cstr
        if isinstance(target, type) and hasattr(target, "__lang_struct__"):
            return scalar_type(target)  # constructing an object
        if isinstance(target, Function):
            return scalar_type(getattr(target.python, "__annotations__", {}).get("return"))
        return None

    def _binary_type(self, node: Node) -> ScalarType | None:
        operator = _text(node.child_by_field_name("operator"))
        if operator in _COMPARISONS or operator in _BOOLEAN_OPERATORS:
            return scalar_type(bool)
        left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
        left_kind, right_kind = self._static_type(left), self._static_type(right)
        if operator == "/" and not (
            (left_kind and left_kind.kind == "float") or (right_kind and right_kind.kind == "float")
        ):
            return f64
        # A literal takes the type of the other side: `hits + 1` is `hits`'s type.
        if _unwrap(left).type == "literal" and right_kind is not None:
            return right_kind
        return left_kind or right_kind

    def _format(self, call: Node) -> ast.expr:
        """Expand ``fmt.format(pattern, ...)`` into a string allocated to fit.

        Formats into 64 bytes, then, if the text needs more, grows the buffer to
        its exact size and formats again. The arguments are evaluated once.
        """
        assert self._scope is not None
        positional, named = self._arguments(call)
        if named or not positional:
            raise self.error(call, 'fmt.format takes a pattern and its values: fmt.format("%s: %d", name, count)')
        self._scope.temporaries += 1
        number = self._scope.temporaries
        text, size = f"__bifrost_text_{number}", f"__bifrost_size_{number}"
        for helper in ("malloc", "realloc", "snprintf"):
            self.globals[f"__bifrost_{helper}"] = self.project.runtime(helper)
        self.globals["__bifrost_u64"] = u64

        def name(identifier: str, context: ast.expr_context | None = None) -> ast.Name:
            return self._at(ast.Name(identifier, context or ast.Load()), call)

        def invoke(function: str, *arguments: ast.expr) -> ast.Call:
            return self._at(ast.Call(name(function), list(arguments), []), call)

        def assign(target: str, value: ast.expr) -> ast.stmt:
            return self._at(ast.Assign([name(target, ast.Store())], value), call)

        values: list[ast.expr] = []
        for index, value in enumerate(positional):
            if isinstance(value, (ast.Name, ast.Constant)):
                values.append(value)
            else:
                temporary = f"__bifrost_value_{number}_{index}"
                self._scope.before.append(assign(temporary, value))
                values.append(name(temporary))
        initial = self._at(ast.Constant(64), call)
        needed = self._at(ast.BinOp(name(size), ast.Add(), self._at(ast.Constant(1), call)), call)
        grow = self._at(
            ast.If(
                self._at(ast.Compare(name(size), [ast.GtE()], [self._at(ast.Constant(64), call)]), call),
                [
                    assign(text, invoke("__bifrost_realloc", name(text), invoke("__bifrost_u64", needed))),
                    self._at(
                        ast.Expr(invoke("__bifrost_snprintf", name(text), invoke("__bifrost_u64", needed), *values)),
                        call,
                    ),
                ],
                [],
            ),
            call,
        )
        self._scope.before += [
            assign(text, invoke("__bifrost_malloc", invoke("__bifrost_u64", initial))),
            assign(size, invoke("__bifrost_snprintf", name(text), invoke("__bifrost_u64", initial), *values)),
            grow,
        ]
        return name(text)

    def _arguments(
        self, call: Node, pointees: list[object | None] | None = None
    ) -> tuple[list[ast.expr], list[ast.keyword]]:
        """Lower a call's arguments: positional ones, then named ones (``width: 800``)."""
        positional: list[ast.expr] = []
        named: list[ast.keyword] = []
        for argument in _children(call):
            if argument.type == "expression":
                if named:
                    raise self.error(argument, "a positional argument cannot follow a named one")
                index = len(positional)
                pointee = pointees[index] if pointees and index < len(pointees) else None
                opaque = pointee is _OPAQUE
                if opaque:
                    pointee = self._held_type(argument)
                if pointee is _JSON:
                    positional.append(self._json_argument(argument))
                    continue
                if pointee is _TASK:
                    positional.append(self._task_argument(argument))
                    continue
                if isinstance(pointee, _Cell):
                    positional.append(self._cell_argument(argument, pointee))
                    continue
                positional.append(
                    self._lend(argument, pointee, call, opaque=opaque)
                    if pointee is not None
                    else self._argument(argument)
                )
            elif argument.type == "named_argument":
                name = self._identifier(argument.child_by_field_name("name"))
                value = self._expression(argument.child_by_field_name("value"))
                named.append(self._at(ast.keyword(arg=name, value=value), argument))
        return positional, named

    def _held_type(self, argument: Node) -> object | None:
        """Return the type an owned local, lent parameter or guard passed as ``argument`` holds, if it is one."""
        assert self._scope is not None
        inner = _unwrap(argument)
        if inner.type != "identifier":
            return None
        name = self._scope.guards.get(_text(inner), _text(inner))
        if name not in self._scope.pointers and name not in self._scope.shared_locals:
            return None
        return self._scope.held[name][1]

    def _argument(self, argument: Node) -> ast.expr:
        """Lower a plain argument; a guard can only be lent to a ``mem.Weak`` parameter."""
        assert self._scope is not None
        inner = _unwrap(argument)
        source = self._scope.guards.get(_text(inner)) if inner.type == "identifier" else None
        if source is not None and isinstance(scalar_type(self._scope.held.get(source, (None, None))[1]), StructType):
            raise self.error(inner, f"{_text(inner)} is a guard; it can only be lent to a mem.Weak parameter")
        return self._expression(argument)  # a guard on a whole value passes that value

    def _cell_argument(self, argument: Node, cell: _Cell) -> ast.expr:
        """Pass a ``mem.Shared`` or ``mem.Atomic`` parameter a cell of the same kind (the call borrows it)."""
        assert self._scope is not None
        inner = _unwrap(argument)
        name = _text(inner) if inner.type == "identifier" else None
        if name is None or name not in self._scope.cells:
            what = f"{name} is not one" if name is not None else "pass one by name"
            raise self.error(inner, f"this parameter takes a {cell.container!r}[{_type_label(cell.held)}]; {what}")
        container, held = self._scope.held[name]
        if container is not cell.container:
            raise self.error(inner, f"{name} is a {container!r}, but this parameter takes a {cell.container!r}")
        if held != cell.held:
            raise self.error(inner, f"{name} holds a {_type_label(held)}, not a {_type_label(cell.held)}")
        return self._at(ast.Name(name, ast.Load()), inner)

    def _lend(self, argument: Node, pointee: object, call: Node, *, opaque: bool = False) -> ast.expr:
        """Pass a ``mem.Weak`` argument: a lent parameter or guard as is, an owned local by lending it.

        A cell is locked for the call, unless the parameter is C's ``void *``
        (``opaque``), which takes the cell itself: the C code passes it on to
        functions that lock it, like a web server's handlers.
        """
        assert self._scope is not None
        inner = _unwrap(argument)
        if inner.type != "identifier":
            raise self.error(inner, "a mem.Weak parameter takes a name: an owned value, a lent one, or a guard")
        name = self._scope.guards.get(_text(inner), _text(inner))
        if name in self._scope.cells and name == _text(inner) and not opaque:
            if call != self._scope.lend_call:
                raise self.error(inner, f"lend {name} in a call that is a statement of its own, like `draw({name})`")
            self._scope.before.append(self._lock(name, inner, f"{name} is locked, so it cannot be lent"))
            self._scope.after.append(self._cell_statement("unlock", name, inner))
            return self._at(ast.Name(name, ast.Load()), inner)
        if name in self._scope.pointers:
            return self._at(ast.Name(name, ast.Load()), inner)
        if name not in self._scope.shared_locals:
            msg = f"{name} is not owned; declare it `let {name}: mem.Unique[...] = ...` to lend it"
            raise self.error(inner, msg)
        if call != self._scope.lend_call:
            raise self.error(inner, f"lend {name} in a call that is a statement of its own, like `draw({name})`")
        # Copy the local to the stack, pass its address, and copy it back after the call.
        self._scope.temporaries += 1
        slot = f"__bifrost_lend_{self._scope.temporaries}"
        kind = f"__bifrost_type_{self._scope.temporaries}"
        self.globals[kind] = pointee
        self.globals["__bifrost_stack"] = stack

        def name_node(identifier: str, context: ast.expr_context) -> ast.Name:
            return self._at(ast.Name(identifier, context), inner)

        def element(context: ast.expr_context) -> ast.Subscript:
            return self._at(ast.Subscript(name_node(slot, ast.Load()), ast.Constant(0), context), inner)

        allocate = ast.Call(name_node("__bifrost_stack", ast.Load()), [name_node(kind, ast.Load())], [])
        self._scope.before += [
            self._at(ast.Assign([name_node(slot, ast.Store())], self._at(allocate, inner)), inner),
            self._at(ast.Assign([element(ast.Store())], name_node(name, ast.Load())), inner),
        ]
        self._scope.after.append(self._at(ast.Assign([name_node(name, ast.Store())], element(ast.Load())), inner))
        return name_node(slot, ast.Load())

    def _if_expression(self, node: Node) -> ast.expr:
        """``if c { a } else { b }`` as a value, when each branch is one expression."""
        condition, body, *rest = _children(node)
        if not rest:
            raise self.error(node, "an if used as a value needs an else")
        orelse = self._if_expression(rest[0]) if rest[0].type == "if" else self._single(rest[0])
        return self._at(ast.IfExp(self._expression(condition), self._single(body), orelse), node)

    def _match_expression(self, node: Node) -> ast.expr:
        scrutinee, *arms = _children(node)
        subject = self._expression(scrutinee)
        if not isinstance(subject, (ast.Name, ast.Constant)):
            raise self.error(scrutinee, "a match used as a value needs a name to match on")
        branches, default = self._match_arms(arms, subject, self._single)
        if default is None:
            raise self.error(node, "a match used as a value needs a '_' arm")
        value = default
        for test, body, arm in reversed(branches):
            value = self._at(ast.IfExp(test, body, value), arm)
        return value

    def _single(self, node: Node) -> ast.expr:
        """Lower the one expression a value-producing branch holds."""
        items = _children(node) if node.type == "block_expression" else [node]
        if len(items) != 1 or items[0].type != "expression":
            raise self.error(node, "a branch used as a value must be a single expression")
        return self._expression(items[0])

    # -- helpers ------------------------------------------------------------------

    def _identifier(self, node: Node) -> str:
        name = _text(node)
        if keyword.iskeyword(name):
            raise self.error(node, f"'{name}' is reserved")
        return name

    def _at[N: ast.AST](self, python: N, node: Node) -> N:
        python.bifrost_point = node.start_point  # type: ignore[attr-defined]
        return python

    @staticmethod
    def _descendants(node: Node, kind: str) -> Iterator[Node]:
        """``kind`` nodes of one function's body (a lambda in it is a function of its own)."""
        for child in node.named_children:
            if child.type == kind:
                yield child
            if child.type != "local_function_definition":
                yield from SourceUnit._descendants(child, kind)


class _JsonWriter:
    """The statements that append a value's JSON text to a buffer, for ``json.encode``."""

    def __init__(self, unit: SourceUnit, node: Node, buffer: str) -> None:
        self.unit = unit
        self.node = node
        self.buffer = buffer
        self.statements: list[ast.stmt] = []
        self.pending: list[str] = []  # constant text not yet appended, merged into one call

    def name(self, context: ast.expr_context | None = None) -> ast.Name:
        return self.unit._at(ast.Name(self.buffer, context or ast.Load()), self.node)

    def invoke(self, function: str, *arguments: ast.expr) -> ast.Call:
        runtime = getattr(json_runtime, f"bifrost_json_{function}")
        callee = self.unit._global(runtime, "json")
        return self.unit._at(ast.Call(callee, [self.name(), *arguments], []), self.node)

    def call(self, function: str, *arguments: ast.expr) -> ast.stmt:
        return self.unit._at(ast.Expr(self.invoke(function, *arguments)), self.node)

    def flush(self) -> None:
        if self.pending:
            self.statements.append(self.call("raw", ast.Constant("".join(self.pending))))
            self.pending.clear()

    def write(self, expression: ast.expr, kind: ScalarType) -> None:
        """Append ``expression``, of type ``kind``: an object field by field, or a scalar."""
        if isinstance(kind, StructType):
            self.write_object(expression, kind)
            return
        self.flush()
        unit = self.unit
        if kind.kind == "cstr":
            self.statements.append(self.call("string", expression))
        elif kind.kind == "bool":
            self.statements.append(self.call("bool", expression))
        elif kind.is_integer or kind.kind == "float":
            convert = i64 if kind.is_integer else f64
            value = unit._at(ast.Call(unit._global(convert, "type"), [expression], []), self.node)
            self.statements.append(self.call("int" if kind.is_integer else "float", value))
        else:
            raise unit.error(self.node, f"a {kind.name} has no JSON form")

    def write_object(self, expression: ast.expr, kind: StructType) -> None:
        unit = self.unit
        assert unit._scope is not None
        unit._scope.temporaries += 1
        holder = f"__bifrost_json_value_{unit._scope.temporaries}"
        store = unit._at(ast.Name(holder, ast.Store()), self.node)
        self.statements.append(unit._at(ast.Assign([store], expression), self.node))
        self.pending.append("{")
        for index, (field, field_kind) in enumerate(kind.fields):
            self.pending.append(("" if index == 0 else ", ") + _json_quote(field) + ": ")
            read = ast.Attribute(ast.Name(holder, ast.Load()), field, ast.Load())
            self.write(unit._at(read, self.node), field_kind)
        self.pending.append("}")


# How the compiler finds each expression node's type before compiling it (see ``SourceUnit._static_type``).
_STATIC_TYPES: dict[str, Callable[[SourceUnit, Node], ScalarType | None]] = {
    "literal": lambda unit, node: unit._literal_type(_children(node)[0]),
    "identifier": lambda unit, node: unit._name_type(_text(node)),
    "child_annotation": SourceUnit._path_type,
    "function_call": SourceUnit._call_type,
    "binary_expression": SourceUnit._binary_type,
    "unary_expression": SourceUnit._unary_type,
    "await_expression": SourceUnit._await_type,
    "get_expression": lambda unit, node: getattr(unit._static_type(_children(node)[0]), "element", None),
}

# How each expression node lowers (see ``SourceUnit._expression``).
_EXPRESSIONS: dict[str, Callable[[SourceUnit, Node], ast.expr]] = {
    "binary_expression": SourceUnit._binary,
    "unary_expression": SourceUnit._unary,
    "await_expression": SourceUnit._await,
    "literal": SourceUnit._literal,
    "identifier": SourceUnit._value,
    "local_function_definition": SourceUnit._lambda,
    "child_annotation": SourceUnit._child_annotation,
    "function_call": SourceUnit._function_call,
    "get_expression": SourceUnit._get_expression,
    "if": SourceUnit._if_expression,
    "match_expression": SourceUnit._match_expression,
    "default_var": SourceUnit._default_var,
    "while": SourceUnit._statement_only,
    "forall": SourceUnit._statement_only,
}


def lower_file(project: Project, path: Path, source: bytes | None = None, root: Path | None = None) -> SourceUnit:
    """Parse ``path`` (or ``source``, its unsaved text) and register its functions with ``project``'s program.

    ``root`` is where ``import("a.b:module")`` finds ``a/b.bif``: the project's folder (the one with
    config.yaml), or ``path``'s folder by default.
    """
    unit = SourceUnit(project, path, source, root=root)
    unit.lower()
    return unit


__all__ = ["BifrostError", "SourceUnit", "display_types", "lower_file"]
