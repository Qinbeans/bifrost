"""What the language server knows about one Bifrost document.

Everything here works on source text and returns plain data, so it is tested
without a client. Positions are ``(line, character)`` in UTF-16 code units, as
the Language Server Protocol counts them.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

import tree_sitter_bifrost
from mlir_python.codegen import OptLevel
from mlir_python.lang import CompileError
from tree_sitter import Language, Node, Parser

from bifrost import std
from bifrost.configs import Config, ConfigBuilder, source_root
from bifrost.configs.schema import Declaration, _Extern
from bifrost.lowering import RECORD, BifrostError, display_types, lower_file
from bifrost.naming import extern_name, is_pascal_case, is_snake_case, to_pascal_case, to_snake_case
from bifrost.owned import is_list
from bifrost.project import Project
from bifrost.std import fmt, json, mem, tasks
from bifrost.syntax import syntax_errors

_LANGUAGE = Language(tree_sitter_bifrost.language())
_MODULE_MEMBER = 2  # parts in ``module.Member``

KEYWORDS = ["let", "struct", "if", "else", "while", "forall", "in", "match", "return", "this", "true", "false", "null"]
TYPES = [
    *(f"{kind}{bits}" for kind in "iu" for bits in (8, 16, 32, 64)),
    "f32",
    "f64",
    "bool",
    "str",
]

type Position = tuple[int, int]
type Range = tuple[Position, Position]


class Severity(Enum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


class SymbolKind(Enum):
    FUNCTION = "function"
    STRUCT = "struct"
    MODULE = "module"
    CONSTANT = "constant"


@dataclass(frozen=True)
class Diagnostic:
    range: Range
    message: str
    severity: Severity = Severity.ERROR
    unnecessary: bool = False  # shown faded, e.g. an unused variable


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: SymbolKind
    detail: str
    range: Range
    selection: Range


@dataclass(frozen=True)
class Location:
    path: Path
    range: Range


@dataclass(frozen=True)
class Completion:
    label: str
    kind: str  # "function", "struct", "module", "constant", "variable", "keyword", "type"
    detail: str = ""
    replace: Range | None = None  # the text the completion replaces, when not just the word at the cursor
    documentation: str = ""  # shown beside the list, e.g. an extern's doc


@dataclass(frozen=True)
class InlayHint:
    position: Position
    label: str  # e.g. ": Guard[Context]", or "status:" before an argument
    parameter: bool = False  # names the parameter an argument fills, rather than a type
    tooltip: str = ""  # the whole type, where the label shortens it


@contextmanager
def _capturing_types(sink: dict[str, dict[str, Any]]) -> Iterator[None]:
    """Record each compiled function's inferred local types in ``sink``, keyed by symbol.

    mlir_python keeps them only while a function compiles, so this wraps its
    compiler to copy them out.
    """
    from mlir_python.lang._compiler import FunctionCompiler  # noqa: PLC0415 - no public hook yet

    original = FunctionCompiler.compile

    def compile_and_record(compiler: FunctionCompiler) -> None:
        original(compiler)
        found = dict(compiler.variable_types)
        if len(compiler.signature.results) == 1:
            found[_RESULT] = compiler.signature.results[0]  # what it returns: the shape of a `Record`
        sink[compiler.function.name] = found

    FunctionCompiler.compile = compile_and_record  # type: ignore[method-assign]
    try:
        yield
    finally:
        FunctionCompiler.compile = original  # type: ignore[method-assign]


# The key of a function's result among its compiled local types (no local is named `return`).
_RESULT = "return"

# Each file's local types from its last compile, by function symbol: while a line
# is half typed (`results.`) nothing compiles, and completion still needs them.
_LAST_TYPES: dict[Path, dict[str, dict[str, Any]]] = {}


def _remember_types(path: Path, inferred: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return ``inferred`` over the functions' types from before, and keep that for next time."""
    merged = {**_LAST_TYPES.get(path, {}), **inferred}
    _LAST_TYPES[path] = merged
    return merged


def find_config(path: Path) -> Path | None:
    """Return the nearest ``config.yaml`` in ``path``'s directory or above."""
    for directory in [path.parent, *path.parent.parents]:
        candidate = directory / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def default_config() -> Config:
    """Return a configuration with no externs, for files outside a project."""
    return Config(
        package={"name": "untitled", "version": "0.0.0", "description": ""},
        flags={"optimization": OptLevel.O0, "linker": "clang"},
        libraries=[],
        externs=[],
    )


def _text(node: Node) -> str:
    return (node.text or b"").decode()


def _named(node: Node) -> list[Node]:
    """Named children, without comments or what error recovery made up.

    The document is analysed while it is being typed, so the tree may hold
    ``ERROR`` nodes and zero-width "missing" ones; nothing here assumes a shape.
    """
    return [c for c in node.named_children if c.type not in {"comment", "ERROR"} and not c.is_missing]


def _unwrap(node: Node) -> Node:
    while node.type in {"expression", "parenthesized_expression"} and _named(node):
        node = _named(node)[0]
    return node


def _descendants(node: Node, *kinds: str, own: bool = False) -> list[Node]:
    """``kinds`` nodes under ``node``; ``own``: only a function's own, not those of a lambda in it."""
    found = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in kinds:
            found.append(current)
        if not (own and current is not node and current.type == "local_function_definition"):
            stack.extend(reversed(current.named_children))
    return found


# Where a name is declared or only named, rather than read: `let x`, `x: i32`,
# `let g <- ...`, `g -> x`, `f(width: ...)`. (Calling `f(...)` reads `f`: it
# may be a function value.)
_NOT_USES = {
    "local_assignment": 0,
    "parameter": 0,
    "lock": "guard",
    "release": "*",
    "named_argument": "name",
    "record_field": "name",
}


def _uses(function: Node) -> list[Node]:
    """Return the names ``function``'s body reads: plain names and the roots of dotted names."""
    found = []
    for node in _descendants(function, "identifier", "child_annotation", own=True):
        if node.type == "child_annotation":
            root = _named(node)[0]
            if root.type == "simple_identifier":
                found.append(root)
            continue
        parent = node.parent
        rule = _NOT_USES.get(parent.type) if parent is not None else None
        if rule == "*":
            continue
        if isinstance(rule, int) and _named(parent)[rule] == node:
            continue
        if isinstance(rule, str) and parent.child_by_field_name(rule) == node:
            continue
        found.append(node)
    return found


_SKIPPED_FOLDERS = {"build", "node_modules", "__pycache__"}


def _is_file_module(spec: str) -> bool:
    """``utils.text:greeting`` names a module in a Bifrost file; ``std:stdio`` and ``raylib`` do not."""
    return ":" in spec and not spec.startswith(std.PREFIX)


def _module_description(module: Node) -> str | None:
    """Return a module's description: the block comment at the top of its body."""
    for child in module.children:
        if child.type == "comment" and _text(child).startswith("/*"):
            return " ".join(_text(child).removeprefix("/*").removesuffix("*/").strip(" *\n").split())
        if child.type == "assignment":
            return None
    return None


def _name_of(node: Node) -> Node | None:
    """Return the identifier a ``let``, parameter or assignment binds, if it has one."""
    parts = _named(node)
    return parts[0] if parts and parts[0].type == "identifier" else None


@dataclass
class Document:
    """One ``.bif`` file's text, parsed, with the project configuration it uses."""

    path: Path
    text: str
    config_path: Path | None = None
    config: Config | None = None
    _lines: list[bytes] = field(init=False, repr=False)
    # The `=> Record` functions whose returns are being read (a record holding a call of itself stops there).
    _resolving: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Split the source into lines, for position conversions."""
        self._lines = self.source.split(b"\n")
        self._resolving = set()

    @classmethod
    def open(cls, path: Path, text: str) -> "Document":
        """Parse ``text`` as ``path``, loading the nearest ``config.yaml``."""
        config_path = find_config(path)
        return cls(path, text, config_path)

    @cached_property
    def source(self) -> bytes:
        """The text as UTF-8, which tree-sitter positions count in."""
        return self.text.encode()

    @cached_property
    def root(self) -> Node:
        """The parsed syntax tree."""
        return Parser(_LANGUAGE).parse(self.source).root_node

    # -- positions ------------------------------------------------------------------

    def position(self, point: tuple[int, int]) -> Position:
        """Convert a tree-sitter ``(row, byte column)`` to an LSP position."""
        row, column = point
        line = self._lines[row] if row < len(self._lines) else b""
        return row, len(line[:column].decode(errors="replace").encode("utf-16-le")) // 2

    def point(self, position: Position) -> tuple[int, int]:
        """Convert an LSP position to a tree-sitter ``(row, byte column)``."""
        row, character = position
        line = self._lines[row] if row < len(self._lines) else b""
        text = line.decode(errors="replace").encode("utf-16-le")[: character * 2]
        return row, len(text.decode("utf-16-le", errors="replace").encode())

    def range(self, node: Node) -> Range:
        """Return ``node``'s span as an LSP range."""
        return self.position(node.start_point), self.position(node.end_point)

    def node_at(self, position: Position) -> Node | None:
        """Return the smallest named node at ``position``."""
        point = self.point(position)
        return self.root.named_descendant_for_point_range(point, point)

    # -- configuration ----------------------------------------------------------------

    def load_config(self) -> Config:
        """Load (once) the project configuration, or a default one outside a project."""
        if self.config is None:
            self.config = ConfigBuilder(self.config_path).build() if self.config_path else default_config()
        return self.config

    # -- diagnostics ------------------------------------------------------------------

    def diagnostics(self) -> list[Diagnostic]:
        """Report syntax errors, or else lowering and type errors; and naming warnings."""
        return [*self._program_errors(), *self.naming(), *self.unused()]

    def naming(self) -> list[Diagnostic]:
        """Warn about names that break the conventions: snake_case, and PascalCase objects."""
        found = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.named_children))
            if node.type not in {"assignment", "local_assignment", "lock", "parameter", "struct_field", "module"}:
                continue
            identifier = _name_of(node)
            if identifier is None:
                continue
            name, what = _text(identifier), _kind_of_binding(node)
            if what is None:
                continue
            if what == "object" and not is_pascal_case(name):
                message = f"object '{name}' should be PascalCase: '{to_pascal_case(name)}'"
            elif what != "object" and not is_snake_case(name):
                message = f"{what} '{name}' should be snake_case: '{to_snake_case(name)}'"
            else:
                continue
            found.append(Diagnostic(self.range(identifier), message, Severity.WARNING))
        return found

    def unused(self) -> list[Diagnostic]:
        """Fade parameters, locals and guards that are never used (names starting with `_` are exempt)."""
        found = []
        for function in _descendants(self.root, "function_definition", "local_function_definition"):
            declared = [
                identifier
                for binding in _descendants(function, "parameter", "local_assignment", "lock", own=True)
                if (identifier := _name_of(binding)) is not None and not _text(identifier).startswith("_")
            ]
            used = {_text(name) for name in _uses(function)}
            for identifier in declared:
                name = _text(identifier)
                if name not in used:
                    message = f"{name} is never used; remove it, or name it _{name} to keep it"
                    found.append(Diagnostic(self.range(identifier), message, Severity.HINT, unnecessary=True))
        return found

    def _program_errors(self) -> list[Diagnostic]:
        return self._compiled[0]

    @cached_property
    def _compiled(self) -> tuple[list[Diagnostic], dict[str, dict[str, Any]]]:
        """Compile the document once: its errors, and each function's inferred local types (by symbol)."""
        inferred: dict[str, dict[str, Any]] = {}
        syntax = self._syntax_errors()
        if syntax:
            return syntax, _remember_types(self.path, inferred)  # half typed: the types it last compiled with
        found = []
        if self.config_path is None:
            found.append(
                Diagnostic(
                    ((0, 0), (0, 0)),
                    "no config.yaml in this directory or above it, so only standard modules (std:...) can be imported",
                    Severity.INFORMATION,
                )
            )
        try:
            config = self.load_config()
        except Exception as error:  # noqa: BLE001 - any invalid config is reported, not raised
            return [*found, Diagnostic(((0, 0), (0, 0)), f"cannot load {self.config_path}: {error}")], inferred
        try:
            project = Project(config)
            unit = lower_file(project, self.path, self.source, root=self.project_root)
            with unit.errors(), _capturing_types(inferred):
                _ = project.program.mlir  # type-check every function
        except BifrostError as error:
            found.append(self._error_diagnostic(error))
        except CompileError as error:
            found.append(Diagnostic(((0, 0), (0, 0)), error.msg))
        return found, _remember_types(self.path, inferred)

    def _syntax_errors(self) -> list[Diagnostic]:
        return [Diagnostic(self.range(p.node), p.message) for p in syntax_errors(self.root)]

    def _error_diagnostic(self, error: BifrostError) -> Diagnostic:
        """Span the token a ``BifrostError`` points at."""
        start = ((error.lineno or 1) - 1, (error.offset or 1) - 1)
        node = self.root.descendant_for_point_range(self._byte_point(start), self._byte_point(start))
        end = node.end_point if node is not None and node.start_point == self._byte_point(start) else None
        start_position = self.position(self._byte_point(start))
        return Diagnostic((start_position, self.position(end) if end else start_position), error.msg)

    def _byte_point(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a ``(row, character)`` point from a ``BifrostError`` to bytes."""
        row, column = point
        line = self._lines[row] if row < len(self._lines) else b""
        return row, len(line.decode(errors="replace")[:column].encode())

    # -- symbols ----------------------------------------------------------------------

    def bindings(self) -> dict[str, tuple[Node, Node]]:
        """Return each top-level ``let``'s name -> (assignment, value), and those of modules' members."""
        found = {}
        items = _named(self.root)
        for module in (item for item in items if item.type == "module"):
            items += module.children_by_field_name("member")
        for assignment in items:
            if assignment.type != "assignment":
                continue
            parts = _named(assignment)
            identifier = _name_of(assignment)
            if identifier is None or len(parts) < 2:  # noqa: PLR2004 - a name and a value
                continue
            found.setdefault(_text(identifier), (assignment, parts[-1]))
        return found

    def symbols(self) -> list[Symbol]:
        """Outline the top-level bindings: functions, structs, modules, constants."""
        symbols = []
        for name, (assignment, value) in self.bindings().items():
            kind, detail = self._describe(value)
            identifier = _name_of(assignment)
            if identifier is None:
                continue
            symbols.append(Symbol(name, kind, detail, self.range(assignment), self.range(identifier)))
        for module in (item for item in _named(self.root) if item.type == "module"):
            identifier = module.child_by_field_name("name")
            if identifier is not None:
                detail = _module_description(module) or "module"
                symbols.append(
                    Symbol(_text(identifier), SymbolKind.MODULE, detail, self.range(module), self.range(identifier))
                )
        return symbols

    def _describe(self, value: Node) -> tuple[SymbolKind, str]:
        if value.type == "function_definition":
            return SymbolKind.FUNCTION, self._signature(value)
        if value.type == "struct_assignment":
            return SymbolKind.STRUCT, "struct"
        if _text(value).startswith("import("):
            return SymbolKind.MODULE, _text(value)
        return SymbolKind.CONSTANT, _text(value)

    @staticmethod
    def _signature(function: Node) -> str:
        """Return ``(a: i32) => i32``, or ``async (a: i32) => i32``: a function's parameters and result."""
        parts = [p for p in _named(function) if p.type not in {"dependency_list", "local_dependency_list"}]
        if len(parts) < 2 or parts[0].type != "parameter_list":  # noqa: PLR2004 - parameters and result
            return "(...)"
        parameters, result = parts[0], parts[1]
        is_async = "async " if function.child_by_field_name("async") is not None else ""
        return f"{is_async}{' '.join(_text(parameters).split())} => {_text(result)}"

    # -- definitions and hover ----------------------------------------------------------

    def _enclosing_function(self, node: Node) -> Node | None:
        while node is not None and node.type not in {"function_definition", "local_function_definition"}:
            node = node.parent
        return node

    def _local(self, name: str, node: Node) -> Node | None:
        """Return the parameter or ``let`` that binds ``name`` where ``node`` is."""
        function = self._enclosing_function(node)
        if function is None:
            return None
        for parameter in function.named_children:
            if parameter.type == "parameter_list":
                for candidate in _named(parameter):
                    identifier = _name_of(candidate)
                    if identifier is not None and _text(identifier) == name:
                        return candidate
        for current in _descendants(function, "local_assignment", "lock", own=True):
            identifier = _name_of(current)
            if identifier is not None and _text(identifier) == name:
                return current
        return None

    def _describe_local(self, local: Node) -> str:
        """``let guard: Guard[Context] <- ctx``: a local's binding, with its type when it is known."""
        if local.type == "parameter":
            return " ".join(_text(local).split())
        identifier = _name_of(local)
        first_line = _text(local).split("\n")[0]
        kind = self._type_of_binding(local)
        if identifier is None or kind is None or local.child_by_field_name("type") is not None:
            return first_line  # unknown, or already written: `let ctx: mem.Shared[Context] = ...`
        name = _text(identifier)
        rest = first_line[first_line.index(name) + len(name) :]  # ` <- ctx`, ` = Context.new()`
        return f"let {name}: {kind}{rest}"

    def type_of(self, name: str, node: Node) -> str | None:
        """Return the type of local ``name`` where ``node`` is, when known: ``Context``, ``Guard[Context]``.

        In a method, ``super`` is the object it is called on.
        """
        if name == "super":
            return self._receiver(node)
        local = self._local(name, node)
        return self._type_of_binding(local) if local is not None else None

    def _receiver(self, node: Node) -> str | None:
        """Return the object a method is a member of, for ``super`` in it: ``Team``."""
        function = self._enclosing_function(node)
        member = function.parent if function is not None else None
        if member is None or member.type != "struct_field" or member.child_by_field_name("modifier") is not None:
            return None
        owner = member.parent.parent if member.parent is not None else None  # struct_assignment, then its `let`
        name = _name_of(owner) if owner is not None and owner.type == "assignment" else None
        return _text(name) if name is not None else None

    def _type_of_binding(self, local: Node) -> str | None:
        parts = _named(local)
        if local.type == "parameter" and len(parts) > 1:
            return " ".join(_text(parts[1]).split())
        if local.type == "lock":
            source = local.child_by_field_name("source")
            locked = self.type_of(_text(_unwrap(source)), local) if source is not None else None
            declared = local.child_by_field_name("type")
            if declared is not None:
                return " ".join(_text(declared).split())
            return self._guard_type(locked) if locked else None
        if local.type == "local_assignment" and len(parts) > 1:
            declared = local.child_by_field_name("type")
            if declared is not None:
                return " ".join(_text(declared).split())
            value = _unwrap(parts[-1])
            written = self._written_type(value, local) if value.type == "await_expression" else None
            return self._type_of_value(value, local) or self._inferred_type(local) or written
        return None

    def _guard_type(self, locked: str) -> str:
        """Name the guard a lock gives: ``mem.Weak[Context]`` -> ``mem.WeakGuard[Context]``.

        A plain local is owned by its function alone, so its guard is a ``UniqueGuard``.
        """
        held = re.fullmatch(r"(\w+)\.(Unique|Weak|Shared|Atomic)\[(.*)\]", locked)
        if held is not None and self._module_of(held.group(1)) == f"{std.PREFIX}mem":
            return f"{held.group(1)}.{held.group(2)}Guard[{held.group(3).strip()}]"
        alias = next((name for name in self.bindings() if self._module_of(name) == f"{std.PREFIX}mem"), None)
        return f"{alias}.UniqueGuard[{locked}]" if alias else f"UniqueGuard[{locked}]"

    def _shared_inner(self, type_text: str) -> str:
        """``mem.Weak[Context]``, ``mem.WeakGuard[Context]`` -> ``Context``; other types are unchanged."""
        shared = re.fullmatch(r"(\w+)\.(?:Unique|Weak|Shared|Atomic)(?:Guard)?\[(.*)\]", type_text)
        if shared is not None and self._module_of(shared.group(1)) == f"{std.PREFIX}mem":
            return shared.group(2).strip()
        return type_text

    def _inferred_type(self, local: Node) -> str | None:
        """Return the type the compiler inferred for a ``let``.

        A literal takes its type from how it is used: ``let i = 0`` is an ``i64``,
        or an ``i32`` if it is added to one.
        """
        identifier = _name_of(local)
        symbol = self._symbol_of(local)
        if identifier is None or symbol is None:
            return None
        kind = self._compiled[1].get(symbol, {}).get(_text(identifier))
        if kind is None or (kind.kind == "ptr" and not is_list(kind)):
            return None
        return _type_name(kind)

    def _compiled_type(self, local: Node) -> Any:  # noqa: ANN401 - a compiler type
        """Return the compiler's type of a local or parameter (a record's has its fields), if it compiled."""
        identifier = _name_of(local)
        symbol = self._symbol_of(local)
        if identifier is None or symbol is None:
            return None
        return self._compiled[1].get(symbol, {}).get(_text(identifier))

    def _symbol_of(self, node: Node) -> str | None:
        """Return the compiled symbol of the function ``node`` is in: ``draw``, or ``Context_new``."""
        current: Node | None = node
        while current is not None and current.type not in {"function_definition", "local_function_definition"}:
            current = current.parent
        if current is not None and current.parent is not None and current.parent.type == "expression":
            # A lambda: named after the function around it and where it starts.
            outer = self._symbol_of(current.parent)
            row, column = current.start_point
            return f"{outer}_lambda_{row + 1}_{column}" if outer is not None else None
        binding = current.parent if current is not None else None
        name = _name_of(binding) if binding is not None else None
        if binding is None or name is None:
            return None
        if binding.type == "assignment" and binding.parent is not None and binding.parent.type == "module":
            module = binding.parent.child_by_field_name("name")
            return f"{_text(module)}_{_text(name)}" if module is not None else None
        if binding.type == "assignment":
            if _text(name) == "main" and current is not None and current.child_by_field_name("async") is not None:
                return "bifrost_async_main"  # an async main is compiled under this name, and run by a plain main
            return _text(name)
        owner = binding.parent.parent if binding.parent is not None else None  # struct_assignment -> assignment
        owner_name = _name_of(owner) if owner is not None and owner.type == "assignment" else None
        return f"{_text(owner_name)}_{_text(name)}" if owner_name is not None else None

    def _type_of_value(self, value: Node, where: Node) -> str | None:
        """Return the type of an expression, for the simple cases: calls, construction, names."""
        if value.type == "local_function_definition":
            return self._signature(value)  # a lambda
        if value.type == "identifier":
            return self._type_of_name(_text(value), where)
        if value.type == "function_call":
            call = _named(value)[0]
            function = call.child_by_field_name("function")
            return self._type_of_call([], _text(function)) if function is not None else None
        if value.type == "child_annotation":
            return self._type_of_chain_call(value, where)
        return None

    def _type_of_chain_call(self, value: Node, where: Node) -> str | None:
        """Return what ``a.b.f(x)`` gives: a module's or object's function, or a method of a local."""
        parts = _named(value)
        last = parts[-1]
        if last.type != "function_call" or not all(p.type == "simple_identifier" for p in parts[:-1]):
            return None
        function = _named(last)[0].child_by_field_name("function")
        if function is None:
            return None
        owner = [_text(p) for p in parts[:-1]]
        method = self._method_node(owner, _text(function), where)
        return self._result_of(method) if method is not None else self._type_of_call(owner, _text(function))

    def _method_node(self, owner: list[str], name: str, where: Node) -> Node | None:
        """For ``value.name(...)`` on a local (or ``super``), return the method's definition."""
        if len(owner) != 1 or (owner[0] != "super" and self._local(owner[0], where) is None):
            return None
        found = self._struct_of(owner[0], where)
        if found is None:
            return None
        for member in _named(found[1]):
            if member.type == "struct_field" and _declares(member, name):
                return next((p for p in _named(member) if p.type == "local_function_definition"), None)
        return None

    def _type_of_name(self, name: str, where: Node) -> str | None:
        """Return the type of a local, or of a function named as a value: ``(x: i32) => i32``."""
        binding = self.bindings().get(name)
        if self._local(name, where) is None and binding is not None and binding[1].type == "function_definition":
            return self._signature(binding[1])
        return self.type_of(name, where)

    def _type_of_call(self, owner: list[str], name: str) -> str | None:
        """Return the result type of calling ``owner.name(...)`` (``owner`` empty for ``name(...)``)."""
        if not owner:
            return self._type_of_top_call(name)
        if len(owner) != 1:
            return None
        binding = self.bindings().get(owner[0])
        if binding is not None and binding[1].type == "struct_assignment":
            return self._type_of_static_call(binding[1], name)
        module = self._module_of(owner[0])
        builtins = _BUILTIN_FUNCTIONS.get(module or "", {})
        if name in builtins:
            return builtins[name].signature.rpartition(" => ")[2]
        declaration = self._declaration(module, name) if module else None
        if declaration is None:
            return None
        return f"{owner[0]}.{name}" if declaration.type == "struct" else _bifrost_type(declaration.return_type)

    def _type_of_top_call(self, name: str) -> str | None:
        """Return the result type of ``Name(...)`` (a constructor) or ``function(...)``."""
        binding = self.bindings().get(name)
        if binding is None:
            return None
        if binding[1].type == "struct_assignment":
            return name
        if binding[1].type == "function_definition":
            return self._result_of(binding[1])
        return None

    def _type_of_static_call(self, struct: Node, name: str) -> str | None:
        """Return the result type of ``Object.name(...)``."""
        for member in struct.named_children:
            identifier = _name_of(member) if member.type == "struct_field" else None
            if identifier is not None and _text(identifier) == name:
                function = next((p for p in _named(member) if p.type == "local_function_definition"), None)
                return self._result_of(function) if function is not None else None
        return None

    def _result_of(self, function: Node) -> str | None:
        """Return a function's result type as written, or, for ``Record``, the record it returns."""
        written = self._signature(function).rpartition(" => ")[2] or None
        if written != RECORD:
            return written
        shown = self._record_text(self._returned_entries(function))
        return self._record_returned(function) or shown or written

    def _list_text(self, value: Node, where: Node) -> str | None:
        """Write a list literal's type from its first item as written: ``#[1, 2]`` is ``i64[]``."""
        items = [item for item in _named(value) if item.type in {"expression", "spread_action", "spread_between"}]
        if not items:
            return None
        first = items[0]
        if first.type == "spread_between":  # `#[0...n]`: numbers from the first
            return f"{self._written_type(_named(first)[0], where) or 'i64'}[]"
        if first.type == "spread_action":  # `#[...xs, x]`: what xs is
            return self._written_type(_named(first)[-1], where)
        kinds = [self._written_type(item, where) for item in items if item.type == "expression"]
        element = "f64" if "f64" in kinds and set(kinds) <= {"i64", "f64"} else kinds[0]
        return f"{element}[]" if element else None

    def _record_text(self, entries: list[tuple[str, Node]] | None) -> str | None:
        """Write a record's type from its fields as written: ``#{name: str, ms: i64}``.

        Each value is typed where it is written (``n`` is the parameter of the function returning it).
        """
        if not entries:
            return None
        return "#{" + ", ".join(f"{name}: {self._written_type(v, v) or '?'}" for name, v in entries) + "}"

    def _record_returned(self, function: Node) -> str | None:
        """Return the record a function written ``=> Record`` returns, once it has compiled: ``#{ms: i64}``."""
        symbol = self._symbol_of(function)
        kind = self._compiled[1].get(symbol, {}).get(_RESULT) if symbol is not None else None
        return _type_name(kind) if kind is not None else None

    def inlay_hints(self) -> list[InlayHint]:
        """Show the type of each guard and untyped ``let``: ``let guard: mem.WeakGuard[Context] <- ctx``.

        And of each record field, after its name: ``#{name: str: http.param(ctx, "name")}``.
        """
        hints = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.named_children)
            if node.type == "record_field":
                hints.extend(self._record_field_hint(node))
                continue
            if node.type == "named_argument":
                hints.extend(self._named_argument_hint(node))
                continue
            if node.type == "user_function_call":
                hints.extend(self._argument_hints(node))
                continue
            if node.type not in {"lock", "local_assignment"} or node.child_by_field_name("type") is not None:
                continue
            identifier = _name_of(node)
            kind = self._type_of_binding(node)
            if identifier is not None and kind is not None:
                hints.append(InlayHint(self.position(identifier.end_point), f": {kind}"))
        return sorted((_shortened(hint) for hint in hints), key=lambda hint: hint.position)

    def _argument_hints(self, call: Node) -> list[InlayHint]:
        """Name the parameter (or field) each positional argument fills, as clangd does: ``Member(name: "ada")``.

        Not for an argument that already says it (``ctx`` for ``ctx``, ``s.hits``
        for ``hits``), a named one, or one of a variadic function's extra arguments.
        """
        called = self._called(call)
        names = self._parameter_names(*called, call) if called is not None else None
        hints = []
        arguments = [argument for argument in _named(call) if argument.type == "expression"]
        for argument, name in zip(arguments, names or [], strict=False):
            written = _text(_unwrap(argument)).rsplit(".", 1)[-1]
            if written != name:
                hints.append(InlayHint(self.position(argument.start_point), f"{name}:", parameter=True))
        return hints

    def _called(self, call: Node) -> tuple[list[str], str] | None:
        """For a ``user_function_call``, return what it calls: ``(["http"], "text")`` for ``http.text(...)``."""
        function = call.child_by_field_name("function")
        segment = call.parent
        chain = segment.parent if segment is not None else None
        if function is None:
            return None
        if chain is None or chain.type != "child_annotation":
            return [], _text(function)
        parts = _named(chain)
        if parts[-1] != segment or not all(part.type == "simple_identifier" for part in parts[:-1]):
            return None
        return [_text(part) for part in parts[:-1]], _text(function)

    def _callee(self, owner: list[str], name: str, where: Node) -> "tuple[Document, Node] | Declaration | None":
        """Return what ``owner.name(...)`` calls, with the document declaring it, or a C function's declaration.

        A function or lambda (its definition), an object's static function, or an
        object whose constructor it is (its ``struct_assignment``).
        """
        if len(owner) != 1:
            return self._own_callee(name, where) if not owner else None
        method = self._method_node(owner, name, where)
        if method is not None:  # `team.greet(...)`: a method of the object a local holds
            return self, method
        binding = self.bindings().get(owner[0])
        if binding is not None and binding[1].type == "struct_assignment":
            member = next((m for m in _named(binding[1]) if m.type == "struct_field" and _declares(m, name)), None)
            return (self, _named(member)[-1]) if member is not None else None
        module = self._module_of(owner[0])
        if module is None or module in _BUILTIN_FUNCTIONS:
            return None
        if _is_file_module(module):
            found = self._file_member(module, name)
            return (found[0], _named(found[1])[-1]) if found is not None else None
        return self._declaration(module, name)

    def _own_callee(self, name: str, where: Node) -> "tuple[Document, Node] | None":
        """Return what ``name(...)`` calls: a lambda a local holds, or a function or object of this file."""
        local = self._local(name, where)
        if local is not None:
            value = _unwrap(_named(local)[-1]) if local.type == "local_assignment" else None
            return (self, value) if value is not None else None
        binding = self.bindings().get(name)
        return (self, binding[1]) if binding is not None else None

    def _parameter_names(self, owner: list[str], name: str, where: Node) -> list[str] | None:
        """Return the names of the parameters ``owner.name(...)`` takes, in order, when they have names.

        A Bifrost function, a lambda a local holds, an object's static function or
        method, an object's constructor (its stored fields), a C function (its
        declaration's names, in snake_case) or a builtin (``fmt.format``). A function
        value of a function type (``(i32) => null``) has no names.
        """
        if len(owner) == 1 and self._module_of(owner[0]) in _BUILTIN_FUNCTIONS:
            builtin = _BUILTIN_FUNCTIONS[self._module_of(owner[0]) or ""].get(name)
            return _signature_names(builtin.signature) if builtin is not None else None
        callee = self._callee(owner, name, where)
        if isinstance(callee, Declaration):
            if callee.type != "function":
                return None
            return [to_snake_case(parameter) for parameter in callee.parameters]
        return _parameter_names(callee[1]) if callee is not None else None

    def _argument_target(self, node: Node) -> "tuple[str, Location | None, str] | None":
        """For the name of a named argument, describe what it names, where that is declared, and its type.

        ``hits`` in ``state.AppState(hits: 0)`` is a field of ``AppState``;
        ``width`` in ``area(width: 2)`` is a parameter of ``area``.
        """
        identifier = node.parent if node.parent is not None and node.parent.type == "identifier" else node
        argument = identifier.parent
        call = argument.parent if argument is not None else None
        called = self._called(call) if call is not None and call.type == "user_function_call" else None
        if called is None:
            return None
        callee = self._callee(*called, identifier)
        name = _text(identifier)
        where = ".".join([*called[0], called[1]])
        if isinstance(callee, Declaration):
            kind = next((k for p, k in callee.parameters.items() if to_snake_case(p) == name), None)
            shown = _bifrost_type(kind) if kind else ""
            return (f"{name}: {shown}\n// a parameter of {where}", None, shown) if kind else None
        if callee is None:
            return None
        document, value = callee
        if value.type == "struct_assignment":
            declared = next((m for m in _named(value) if m.type == "struct_field" and _declares(m, name)), None)
            described = f"{document._member_signature(declared)}\n// a field of {where}" if declared else ""
        else:
            parameters = next((part for part in _named(value) if part.type == "parameter_list"), None)
            found = [p for p in _named(parameters) if _declares(p, name)] if parameters is not None else []
            declared = found[0] if found else None
            described = f"{' '.join(_text(declared).split())}\n// a parameter of {where}" if declared else ""
        declared_name = _name_of(declared) if declared is not None else None
        if declared is None or declared_name is None:
            return None
        written = next((part for part in _named(declared) if part.type == "type_or_object"), None)
        shown = " ".join(_text(written).split()) if written is not None else ""
        return described, Location(document.path, document.range(declared_name)), shown

    def _named_argument_hint(self, argument: Node) -> list[InlayHint]:
        """Hint a named argument's type after its name, as for a record field: ``Context(width: i32: 800)``.

        The type of the field or parameter it fills; for a builtin's (``tasks.gather(user: f())``),
        the type of its value.
        """
        name, value = argument.child_by_field_name("name"), argument.child_by_field_name("value")
        if name is None or value is None:
            return []
        target = self._argument_target(name)
        kind = target[2] if target is not None and target[2] else self._written_type(value, value)
        return [InlayHint(self.position(name.end_point), f": {kind}")] if kind else []

    def _is_argument_name(self, node: Node) -> bool:
        """Whether ``node`` is the name in a named argument: ``hits`` in ``AppState(hits: 0)``."""
        identifier = node.parent if node.parent is not None and node.parent.type == "identifier" else node
        argument = identifier.parent
        return (
            argument is not None
            and argument.type == "named_argument"
            and argument.child_by_field_name("name") == identifier
        )

    def _record_field_hint(self, field: Node) -> list[InlayHint]:
        """Hint a record field's type after its name, as for a ``let``: ``#{count: i64: 0}``."""
        parts = _named(field)
        if not parts[1:]:  # half typed: no value yet
            return []
        kind = self._record_field_type(field)
        return [InlayHint(self.position(parts[0].end_point), f": {kind}")] if kind is not None else []

    def _dotted(self, node: Node) -> tuple[str, str] | None:
        """For ``module.member`` with the cursor on ``member``, return the module and member."""
        access = self._member_access(node)
        module = self._module_of(access[0]) if access else None
        return (module, access[1]) if module and access else None

    def _object_member(self, node: Node) -> "tuple[Document, Node, str] | None":
        """For ``Object.member`` or ``value.field`` with the cursor on the member, return its ``struct_field``.

        Also the document declaring it (another file's, for ``state.AppState``) and the object's name.
        """
        access = self._member_access(node)
        if access is None:
            return None
        binding = self.bindings().get(access[0])
        found: tuple[Document, Node, str] | None = None
        if binding is not None and binding[1].type == "struct_assignment":
            found = (self, binding[1], access[0])
        if found is None:
            found = self._struct_of(access[0], node)  # `guard.counter`: the struct `guard` reaches
        if found is None:
            return None
        document, struct, owner = found
        for member in struct.named_children:
            identifier = _name_of(member) if member.type == "struct_field" else None
            if identifier is not None and _text(identifier) == access[1]:
                return document, member, owner
        return None

    def _struct_of(self, name: str, node: Node) -> "tuple[Document, Node, str] | None":
        """Return the ``struct`` a local reaches (``Context``, ``mem.Weak[Context]``, a guard of one).

        With the document declaring it, which is another file's for ``state.AppState``, and its type's name.
        """
        kind = self.type_of(name, node)
        return self._struct_named(self._shared_inner(kind)) if kind is not None else None

    def _struct_named(self, kind: str) -> "tuple[Document, Node, str] | None":
        """Find the object type ``Context``, or ``state.AppState`` in the module a file imports as ``state``."""
        binding = self.bindings().get(kind)
        if binding is not None and binding[1].type == "struct_assignment":
            return self, binding[1], kind
        alias, _, member = kind.partition(".")
        spec = self._module_of(alias) if member and "." not in member else None
        found = self._file_member(spec, member) if spec and _is_file_module(spec) else None
        if found is None:
            return None
        value = _named(found[1])[-1]
        return (found[0], value, kind) if value.type == "struct_assignment" else None

    def _member_access(self, node: Node) -> tuple[str, str] | None:
        """For ``owner.member`` with the cursor on ``member``, return both names."""
        part = node
        if part.parent is not None and part.parent.type == "identifier":
            part = part.parent
        if part.parent is not None and part.parent.type == "user_function_call":
            part = part.parent.parent  # the function_call segment
        owner = part.parent if part is not None else None
        if owner is None or owner.type != "child_annotation":
            return None
        parts = _named(owner)
        if len(parts) < _MODULE_MEMBER or parts[1] != part or parts[0].type != "simple_identifier":
            return None
        return _text(parts[0]), _text(node)

    def _module_of(self, binding: str) -> str | None:
        """Return the configured module a top-level ``let x = import(...)`` names."""
        found = self.bindings().get(binding)
        if found is None:
            return None
        text = _text(found[1])
        if not text.startswith("import("):
            return None
        return text.removeprefix("import(").rstrip(")").strip().strip('"')

    def _extern(self, module: str) -> tuple[_Extern, Path | None] | None:
        """Find ``module`` and the file declaring it: a standard module, or one in ``config.yaml``."""
        if module.startswith(std.PREFIX):
            name = module.removeprefix(std.PREFIX)
            if name in std.BUILTIN or name not in std.available():
                return None  # `std:mem` is built into the compiler, not declared in YAML
            return std.load(name), std.path(name)
        try:
            config = self.load_config()
        except Exception:  # noqa: BLE001 - reported by diagnostics
            return None
        extern = next((e for e in config.externs if e.module == module), None)
        return (extern, self.config_path) if extern is not None else None

    def _declaration(self, module: str, member: str) -> Declaration | None:
        found = self._extern(module)
        if found is None:
            return None
        return next((d for d in found[0].declarations if extern_name(d) == member), None)

    def definition(self, position: Position) -> Location | None:
        """Return where the name at ``position`` is defined."""
        node = self.node_at(position)
        if node is None or node.type not in {"simple_identifier", "identifier"}:
            return None
        if self._is_argument_name(node):
            target = self._argument_target(node)
            return target[1] if target is not None else None
        dotted = self._dotted(node)
        if dotted is not None:
            return self._member_location(*dotted)
        found = self._object_member(node)
        if found is not None:
            document, member, _ = found
            identifier = _name_of(member)
            return Location(document.path, document.range(identifier if identifier is not None else member))
        name = _text(node)
        local = self._local(name, node)
        binding = self.bindings().get(name)
        declared = local if local is not None else binding[0] if binding is not None else None
        if declared is None:
            return None
        identifier = _name_of(declared)
        return Location(self.path, self.range(identifier if identifier is not None else declared))

    def _config_location(self, module: str, member: str) -> Location | None:
        """Find ``member``'s declaration (``init_window`` is ``InitWindow``) in the file declaring ``module``."""
        declaration = self._declaration(module, member)
        found = self._extern(module)
        if found is None or found[1] is None or declaration is None:
            return None
        path = found[1]
        member = declaration.name
        in_module = False
        for row, line in enumerate(path.read_text().splitlines()):
            stripped = line.strip().removeprefix("- ")
            if stripped.startswith("module:"):
                in_module = stripped.removeprefix("module:").strip().strip('"') == module
            elif in_module and stripped in {f"name: {member}", f'name: "{member}"'}:
                column = line.index("name:") + len("name: ")
                return Location(path, ((row, column), (row, column + len(member))))
        return None

    def hover(self, position: Position) -> str | None:
        """Return Markdown describing the name at ``position``."""
        node = self.node_at(position)
        if node is None or node.type not in {"simple_identifier", "identifier"}:
            return None
        access = self._hover_access(node)
        if access is not None:
            return access or None
        name = _text(node)
        local = self._local(name, node)
        receiver = self._receiver(node) if name == "super" else None
        if local is not None or receiver is not None:
            described = self._describe_local(local) if local is not None else f"super: {receiver}"
            return _code(described if local is not None else f"{described}\n// the {receiver} this method is called on")
        binding = self.bindings().get(name)
        if binding is None:
            return None
        assignment, value = binding
        if value.type in {"function_definition", "struct_assignment"}:
            return _documented(self._describe_binding(assignment), _doc(value))
        return _documented(_text(assignment), self._module_doc(value))

    def _hover_member_declaration(self, node: Node) -> str | None:
        """Describe an object's member where it is declared: ``static let origin = ...`` in ``struct { ... }``."""
        identifier = node.parent if node.parent is not None and node.parent.type == "identifier" else node
        member = identifier.parent
        if member is None or member.type != "struct_field" or _name_of(member) != identifier:
            return None
        owner = member.parent.parent if member.parent is not None else None  # struct_assignment, then its `let`
        owner_name = _name_of(owner) if owner is not None else None
        where = f"\n// a member of {_text(owner_name)}" if owner_name is not None else ""
        function = next((p for p in _named(member) if p.type == "local_function_definition"), None)
        return _documented(f"{self._member_signature(member)}{where}", _doc(function))

    def _module_doc(self, value: Node) -> str:
        """Return the documentation of the module ``import("file:module")`` names, if it has one."""
        module = self._module_of_value(value)
        found = self._file_module(module) if module and _is_file_module(module) else None
        return _doc(found[1]) if found is not None else ""

    @staticmethod
    def _module_of_value(value: Node) -> str | None:
        text = _text(value)
        return text.removeprefix("import(").rstrip(")").strip().strip('"') if text.startswith("import(") else None

    def _hover_member(self, node: Node) -> str | None:
        """Describe ``module.function`` (an extern) or ``Object.member``."""
        dotted = self._dotted(node)
        if dotted is not None and _is_file_module(dotted[0]):
            found = self._file_member(*dotted)
            if found is None:
                return None
            other, member = found
            described = f"{other._describe_binding(member)}\n// {dotted[0]} ({other.path.name})"
            return _documented(described, _doc(_named(member)[-1]))
        if dotted is not None and dotted[0] in _BUILTIN_FUNCTIONS:
            builtin = _BUILTIN_FUNCTIONS[dotted[0]].get(dotted[1])
            return _code(f"{builtin!r} = {builtin.signature}\n// {builtin.summary}") if builtin else None
        if dotted is not None and dotted[0] == f"{std.PREFIX}mem":
            container = mem.CONTAINERS.get(dotted[1])
            return _code(f"{container!r}[T]\n// {container.summary}") if container else None
        if dotted is not None:
            declaration = self._declaration(*dotted)
            return _code(_extern_signature(dotted[0], declaration)) if declaration else None
        return self._hover_field(node)

    def _hover_field(self, node: Node) -> str | None:
        """Describe ``value.field`` or ``Object.member``, with the object it belongs to."""
        found = self._object_member(node)
        if found is None:
            return self._hover_record_field(node)
        document, member, owner = found
        where = "" if document is self else f" ({document.path.name})"
        function = next((p for p in _named(member) if p.type == "local_function_definition"), None)
        return _documented(f"{self._member_signature(member)}\n// a member of {owner}{where}", _doc(function))

    def _field_path(self, node: Node) -> tuple[str, list[str]] | None:
        """For ``value.a.b`` with the cursor on ``b``, return ``value`` and the fields up to it: ``["a", "b"]``."""
        part = node.parent if node.parent is not None and node.parent.type == "identifier" else node
        owner = part.parent
        if owner is None or owner.type != "child_annotation":
            return None
        parts = _named(owner)
        if part not in parts[1:]:
            return None
        path = parts[: parts.index(part) + 1]
        if any(p.type != "simple_identifier" for p in path):
            return None
        return _text(path[0]), [_text(p) for p in path[1:]]

    def _hover_record_result(self, node: Node) -> str | None:
        """Describe ``Record`` where a function's result is written: the record it returns."""
        written = node.parent if node.parent is not None and node.parent.type == "identifier" else node
        result = written.parent
        function = result.parent if result is not None and result.type == "type_or_object" else None
        if function is None or function.type not in {"function_definition", "local_function_definition"}:
            return ""
        record = self._record_returned(function)
        shape = record or "a record, shaped like what the function returns (once it compiles)"
        return _code(f"{RECORD} = {shape}\n// the result: records of one shape, from every `return`")

    def _hover_access(self, node: Node) -> str | None:
        """Describe a name that is not a plain one: ``module.member``, ``value.field.field``, or a field in ``#{...}``.

        Or the name of a named argument: the field or parameter it names.
        ``None`` for a plain name; ``""`` for one of these with nothing to show.
        """
        if self._is_argument_name(node):
            target = self._argument_target(node)
            return _code(target[0]) if target is not None else ""
        if _text(node) == RECORD:
            return self._hover_record_result(node)
        declared = self._hover_member_declaration(node)
        if declared is not None:
            return declared
        written = self._hover_record_literal(node)
        if written is not None:
            return written
        if self._member_access(node) is not None:
            return self._hover_member(node) or ""
        return self._hover_record_field(node) or "" if self._field_path(node) is not None else None

    def _hover_record_field(self, node: Node) -> str | None:
        """Describe ``found.user`` or ``found.user.id``: a field of a record (or object) a local holds."""
        path = self._field_path(node)
        kind = self._path_type(path[0], path[1], node) if path is not None else None
        if path is None or kind is None:
            return None
        owner = ".".join([path[0], *path[1][:-1]])
        return _code(f"{path[1][-1]}: {kind}\n// a field of {owner}")

    def _hover_record_literal(self, node: Node) -> str | None:
        """Describe a field where a record is written: ``message`` in ``#{message: "hello"}``."""
        identifier = node.parent if node.parent is not None and node.parent.type == "identifier" else node
        field = identifier.parent
        if field is None or field.type != "record_field" or _named(field)[0] != identifier:
            return None
        record = field.parent
        assert record is not None
        kind = self._record_literal_type(record)
        whole = _type_name(kind) if kind is not None else self._written_type(record, record)
        return _code(f"{_text(identifier)}: {self._record_field_type(field) or '?'}\n// a field of {whole}")

    def _record_field_type(self, field: Node) -> str | None:
        """Return the type of a field where a record is written: the compiler's, else as written."""
        record = field.parent
        assert record is not None
        kind = _field_type(self._record_literal_type(record), [_text(_named(field)[0])])
        return _type_name(kind) if kind is not None else self._written_type(_named(field)[-1], field)

    def _record_literal_type(self, record: Node) -> Any:  # noqa: ANN401 - a compiler type
        """Return the compiler's type of a record literal a ``let`` holds (``let p = #{...}``), if it compiled."""
        names: list[str] = []
        current = record
        while True:
            outer = current.parent.parent if current.parent is not None else None  # literal, then expression
            holder = outer.parent if outer is not None else None
            if outer is None or holder is None or holder.type != "record_field":
                break
            names.insert(0, _text(_named(holder)[0]))
            current = holder.parent
            assert current is not None
        statement = _ancestor(current, "local_assignment")
        if statement is None or _unwrap(_named(statement)[-1]).start_byte != current.parent.start_byte:
            return None
        return _field_type(self._compiled_type(statement), names)

    def _written_type(self, value: Node, where: Node) -> str | None:
        """Return the type of an expression as written: literals (``"a"`` is a ``str``), records, calls, names."""
        value = _unwrap(value)
        if value.type == "literal":
            value = _named(value)[0]
        match value.type:
            case "string" | "boolean" | "number":
                number = value.type == "number" and _named(value)[0].type == "float"
                return "f64" if number else {"string": "str", "boolean": "bool", "number": "i64"}[value.type]
            case "record" | "await_expression":
                return self._record_text(self._entries(value, where))
            case "list":
                return self._list_text(value, where)
            case "binary_expression" | "unary_expression":
                return self._operation_type(value, where)
            case "child_annotation" if all(part.type == "simple_identifier" for part in _named(value)):
                return self._field_access_type(value, where)
        return self._type_of_value(value, where)

    def _field_access_type(self, access: Node, where: Node) -> str | None:
        """Return the type of ``point.x``: a field of a record (or object) a local holds."""
        parts = [_text(part) for part in _named(access)]
        return self._path_type(parts[0], parts[1:], where)

    def _path_type(self, name: str, fields: list[str], where: Node) -> str | None:
        """Return the type of ``name.a.b``: from the compiler, or else from the objects' declared fields.

        The compiler knows a record's fields; a guard or ``mem.Weak`` is only a
        pointer to it, so for those (``s.hits``, ``s <- app``) the fields come
        from the object's declaration, as a hover finds them.
        """
        local = self._local(name, where)
        kind = _field_type(self._compiled_type(local), fields) if local is not None else None
        if kind is not None:
            return _type_name(kind)
        found = self._struct_of(name, where)
        written = None
        for wanted in fields:
            if found is None:
                return None
            document, struct, _ = found
            member = next((m for m in _named(struct) if m.type == "struct_field" and _declares(m, wanted)), None)
            declared = next((c for c in _named(member) if c.type == "type_or_object"), None) if member else None
            if declared is None:
                return None
            written = " ".join(_text(declared).split())
            found = document._struct_named(written)
        return written

    def _operation_type(self, operation: Node, where: Node) -> str | None:
        """Return ``bool`` for ``a < b``, and ``a``'s type for ``a + 1`` (an ``f64`` if either side is one)."""
        operator = next((_text(child) for child in operation.children if not child.is_named), "")
        if operator in _LOGICAL:
            return "bool"
        operands = [_unwrap(operand) for operand in _named(operation)]
        written = [
            kind for operand in operands if operand.type != "literal" and (kind := self._written_type(operand, where))
        ]
        if written:
            return written[0]  # a literal takes the type of the value beside it
        kinds = [self._written_type(operand, where) for operand in operands]
        return "f64" if "f64" in kinds else next((kind for kind in kinds if kind), None)

    def _member_signature(self, member: Node) -> str:
        """Return ``static let new = () => Context``, or ``let width: i32`` for a field."""
        function = next((p for p in _named(member) if p.type == "local_function_definition"), None)
        if function is None:
            return " ".join(_text(member).split())
        modifier = "static " if member.child_by_field_name("modifier") is not None else ""
        identifier = _name_of(member)
        name = _text(identifier) if identifier is not None else "?"
        return f"{modifier}let {name} = {self._signature(function)}"

    # -- completion -----------------------------------------------------------------------

    def completions(self, position: Position) -> list[Completion]:
        """Offer modules in ``import("...")``, members after ``name.``, and names in scope otherwise."""
        modules = self._import_completions(position)
        if modules is not None:
            return modules
        row, character = self.point(position)
        line = self._lines[row][:character].decode(errors="replace") if row < len(self._lines) else ""
        before = line.rstrip()
        stem = before[len(before.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")) :]
        prefix = before[: len(before) - len(stem)]
        if prefix.endswith("."):
            owner = prefix[:-1].split()[-1].split("[")[-1].split("(")[-1] if prefix[:-1].strip() else ""
            module = self._module_of(owner)
            if module:
                return self._members(module)
            if owner in self.bindings():
                return self._static_members(owner)
            where = self.root.named_descendant_for_point_range((row, character), (row, character))
            found = self._struct_of(owner, where) if where is not None and "." not in owner else None
            if found is not None:
                return self._fields(found[1])
            return self._record_completions(owner, where) if where is not None and owner else []
        names = [
            Completion(
                name,
                self._describe(value)[0].value,
                self._describe(value)[1],
                documentation=_doc(value) or self._module_doc(value),
            )
            for name, (_, value) in self.bindings().items()
        ]
        node = self.root.named_descendant_for_point_range((row, character), (row, character))
        function = self._enclosing_function(node) if node is not None else None
        if function is not None:
            for candidate in self._locals(function):
                names.append(Completion(candidate, "variable"))
        return (
            names
            + [Completion(keyword, "keyword") for keyword in KEYWORDS]
            + [Completion(kind, "type") for kind in TYPES]
        )

    def _locals(self, function: Node) -> list[str]:
        found: list[str] = []
        stack = [function]
        while stack:
            current = stack.pop()
            if current.type in {"parameter", "local_assignment", "lock"}:
                identifier = _name_of(current)
                if identifier is not None:
                    found.append(_text(identifier))
            stack.extend(current.named_children)
        return sorted(set(found))

    @property
    def project_root(self) -> Path:
        """Where ``import("a.b:module")`` finds ``a/b.bif``: the project's ``src/``, or its folder."""
        return source_root(self.config_path.parent) if self.config_path else self.path.parent

    def _open(self, path: Path) -> "Document | None":
        try:
            text = path.read_text()
        except OSError:
            return None
        return Document(path, text, self.config_path, self.config)

    def _file_module(self, spec: str) -> "tuple[Document, Node] | None":
        """Find ``file:module``'s ``module`` node, and the document it is in."""
        file_part, _, name = spec.partition(":")
        dots = len(file_part) - len(file_part.lstrip("."))
        base = self.project_root if dots == 0 else self.path.parent
        for _ in range(max(dots - 1, 0)):
            base = base.parent
        names = file_part[dots:].split(".")
        if not all(names):
            return None
        other = self._open(base.joinpath(*names[:-1], names[-1] + ".bif"))
        if other is None:
            return None
        module = next(
            (
                item
                for item in _named(other.root)
                if item.type == "module" and _text(item.child_by_field_name("name")) == name
            ),
            None,
        )
        return (other, module) if module is not None else None

    def _member_location(self, module: str, member: str) -> Location | None:
        """Find ``module.member``: in another Bifrost file, or in the config.yaml or YAML that declares it."""
        if not _is_file_module(module):
            return self._config_location(module, member)
        found = self._file_member(module, member)
        identifier = _name_of(found[1]) if found is not None else None
        return Location(found[0].path, found[0].range(identifier)) if found and identifier else None

    def _file_member(self, spec: str, member: str) -> "tuple[Document, Node] | None":
        found = self._file_module(spec)
        if found is None:
            return None
        other, module = found
        for assignment in module.children_by_field_name("member"):
            identifier = _name_of(assignment)
            if identifier is not None and _text(identifier) == member:
                return other, assignment
        return None

    def _describe_binding(self, assignment: Node) -> str:
        """``let greet = (name: str) => null``: a ``let``, with a function shown by its signature."""
        identifier = _name_of(assignment)
        parts = _named(assignment)
        name = _text(identifier) if identifier is not None else "?"
        if len(parts) > 1 and parts[-1].type == "function_definition":
            described = f"let {name} = {self._signature(parts[-1])}"
            record = self._record_returned(parts[-1]) if described.endswith(f"=> {RECORD}") else None
            if record is not None:
                described += f"\n// {RECORD} is {record}"
            if name == "main" and parts[-1].child_by_field_name("async") is not None:
                try:
                    kind = self.load_config().package.type
                except (OSError, ValueError):
                    kind = "sync"
                note = "runs on the event loop" if kind == "async" else "needs `type: async`"
                described += f"\n// async main: {note} (package.type in config.yaml is {kind})"
            return described
        if len(parts) > 1 and parts[-1].type == "struct_assignment":
            members = [self._member_signature(m) for m in _named(parts[-1]) if m.type == "struct_field"]
            return f"let {name} = struct {{\n" + "".join(f"    {member}\n" for member in members) + "}"
        return " ".join(_text(assignment).split())

    def _file_modules(self) -> list[tuple[str, str]]:
        """Return every exported module under the project root, as (``utils.text:greeting``, description)."""
        found = []
        root = self.project_root
        for path in sorted(root.rglob("*.bif")):
            relative = path.relative_to(root)
            if any(part.startswith(".") or part in _SKIPPED_FOLDERS for part in relative.parts[:-1]):
                continue
            if path.resolve() == self.path.resolve():
                continue
            other = self._open(path)
            if other is None:
                continue
            exported = {
                _text(name)
                for item in _named(other.root)
                if item.type == "export_statement"
                for name in item.children_by_field_name("module")
            }
            dotted = ".".join(relative.with_suffix("").parts)
            for item in _named(other.root):
                name = item.child_by_field_name("name") if item.type == "module" else None
                if name is not None and _text(name) in exported:
                    found.append((f"{dotted}:{_text(name)}", _module_description(item) or f"module in {relative}"))
        return found

    def _import_completions(self, position: Position) -> list[Completion] | None:
        """Offer every module ``import("...")`` can name, when the cursor is in its string."""
        row, column = self.point(position)
        line = self._lines[row][:column].decode(errors="replace") if row < len(self._lines) else ""
        opened = re.search(r'\bimport\(\s*"([^"]*)$', line)
        if opened is None:
            return None
        typed = len(opened.group(1).encode("utf-16-le")) // 2
        replace = ((position[0], position[1] - typed), position)
        found = [
            Completion(f"{std.PREFIX}{name}", "module", self._standard_summary(name), replace)
            for name in std.available()
        ]
        try:
            config = self.load_config()
        except Exception:  # noqa: BLE001 - reported by diagnostics
            return found
        files = [Completion(spec, "module", description, replace) for spec, description in self._file_modules()]
        return found + files + [Completion(e.module, "module", e.description, replace) for e in config.externs]

    @staticmethod
    def _standard_summary(name: str) -> str:
        if name == "mem":
            return "memory containers: " + ", ".join(mem.CONTAINERS)
        if name == "fmt":
            return "formatting into owned strings: " + ", ".join(fmt.FUNCTIONS)
        if name == "json":
            return "values as JSON text: " + ", ".join(json.FUNCTIONS)
        if name == "tasks":
            return "running async functions at the same time: " + ", ".join(tasks.FUNCTIONS)
        return std.load(name).description

    def _fields(self, struct: Node) -> list[Completion]:
        """Offer an object's stored fields and methods (not its static functions) after ``value.``."""
        found = []
        for member in struct.named_children:
            identifier = _name_of(member) if member.type == "struct_field" else None
            if identifier is None or member.child_by_field_name("modifier") is not None:
                continue
            function = next((part for part in _named(member) if part.type == "local_function_definition"), None)
            kind = "method" if function is not None else "field"
            found.append(
                Completion(_text(identifier), kind, self._member_signature(member), documentation=_doc(function))
            )
        return found

    def _record_completions(self, owner: str, where: Node) -> list[Completion]:
        """Offer the fields of the record (or object) ``owner`` reaches, after ``results.`` or ``found.user.``."""
        name, *fields = owner.split(".")
        local = self._local(name, where)
        if local is None:
            return []
        kind = _field_type(self._compiled_type(local), fields)
        if kind is not None and kind.kind == "struct":
            return [Completion(field, "field", f"{field}: {_type_name(held)}") for field, held in kind.fields]
        written = self._path_type(name, fields, where) if fields else None
        found = self._struct_named(written) if written is not None else None
        return self._fields(found[1]) if found is not None else self._written_fields(local, fields)

    def _written_fields(self, local: Node, path: list[str]) -> list[Completion]:
        """Offer the fields of a record a ``let`` holds as written, for when the file does not compile.

        ``let p = #{x: 1}`` gives ``x``; ``let found = await tasks.gather(user: f())``
        gives ``user``; ``path`` leads into a record in it (``p.tag.``, ``found.user.``).
        """
        value = _named(local)[-1] if local.type == "local_assignment" else None
        entries = self._entries(value, local) if value is not None else None
        for wanted in path:
            chosen = next((written for name, written in entries or [] if name == wanted), None)
            if chosen is None:
                return []
            entries = self._entries(chosen, local)
            if entries is None:  # not a record: an object, like `User`
                kind = self._written_type(chosen, local)
                found = self._struct_named(kind) if kind is not None else None
                return self._fields(found[1]) if found is not None else []
        return [
            Completion(name, "field", f"{name}: {self._written_type(written, written) or '?'}")
            for name, written in entries or []
        ]

    def _entries(self, value: Node, where: Node) -> list[tuple[str, Node]] | None:
        """Return the fields of the record an expression gives, as written: each name and the value it holds.

        A record literal, ``await tasks.gather(name: f())``, a local holding one,
        or a call of a function written ``=> Record`` (from what it returns).
        """
        value = _unwrap(value)
        if value.type == "await_expression":
            awaited = value.child_by_field_name("value")
            value = _unwrap(awaited) if awaited is not None else value
        found = _written_entries(value)
        if found is not None:
            return found
        if value.type == "identifier":
            local = self._local(_text(value), where)
            held = _named(local)[-1] if local is not None and local.type == "local_assignment" else None
            return self._entries(held, local) if held is not None and held != value else None
        if value.type == "function_call":
            call = _named(value)[0]
            function = call.child_by_field_name("function")
            binding = self.bindings().get(_text(function)) if function is not None else None
            if binding is not None and binding[1].type == "function_definition":
                return self._returned_entries(binding[1])
        return None

    def _returned_entries(self, function: Node) -> list[tuple[str, Node]] | None:
        """Return the fields of the record a ``=> Record`` function returns, from its first ``return`` of one."""
        if self._signature(function).rpartition(" => ")[2] != RECORD or function.id in self._resolving:
            return None
        self._resolving.add(function.id)
        try:
            body = _named(function)[-1]
            returned = [_named(r)[0] for r in _descendants(body, "return_statement", own=True) if _named(r)]
            if body.type != "block_expression":
                returned = [body]
            return next((found for value in returned if (found := self._entries(value, value)) is not None), None)
        finally:
            self._resolving.discard(function.id)

    def _static_members(self, owner: str) -> list[Completion]:
        """Offer an object's static functions after ``Object.``."""
        binding = self.bindings().get(owner)
        if binding is None or binding[1].type != "struct_assignment":
            return []
        found = []
        for member in binding[1].named_children:
            identifier = _name_of(member) if member.type == "struct_field" else None
            if identifier is not None and member.child_by_field_name("modifier") is not None:
                function = next((p for p in _named(member) if p.type == "local_function_definition"), None)
                found.append(
                    Completion(
                        _text(identifier), "function", self._member_signature(member), documentation=_doc(function)
                    )
                )
        return found

    def _members(self, module: str) -> list[Completion]:
        if _is_file_module(module):
            found = self._file_module(module)
            if found is None:
                return []
            other, node = found
            completions = []
            for member in node.children_by_field_name("member"):
                identifier = _name_of(member)
                parts = _named(member)
                if identifier is not None and len(parts) > 1:
                    kind = other._describe(parts[-1])[0].value
                    described, doc = other._describe_binding(member), _doc(parts[-1])
                    completions.append(Completion(_text(identifier), kind, described, documentation=doc))
            return completions
        if module in _BUILTIN_FUNCTIONS:
            return [Completion(f.name, "function", f"{f!r}{f.signature}") for f in _BUILTIN_FUNCTIONS[module].values()]
        if module == f"{std.PREFIX}mem":
            return [
                Completion(container.name, "struct", f"{container!r}[T]: {container.summary}")
                for container in mem.CONTAINERS.values()
            ]
        found = self._extern(module)
        if found is None:
            return []
        return [
            Completion(extern_name(d), d.type, _extern_signature(module, d).split("\n")[0], documentation=d.doc)
            for d in found[0].declarations
        ]


_BINDING_KINDS = {"parameter": "parameter", "struct_field": "field", "lock": "guard", "module": "module"}
_VALUE_KINDS = {
    "struct_assignment": "object",
    "function_definition": "function",
    "local_function_definition": "function",
}


def _kind_of_binding(node: Node) -> str | None:
    """Say what a ``let``, parameter or field binds, for naming messages (``None``: can't tell)."""
    if node.type == "struct_field" and any(p.type == "local_function_definition" for p in _named(node)):
        return "static function" if node.child_by_field_name("modifier") is not None else "method"
    if node.type in _BINDING_KINDS:
        return _BINDING_KINDS[node.type]
    if any(child.is_error for child in node.children):
        # Half-typed: judge by what follows `=` (`struct {` is an object), else not at all.
        value_text = _text(node).partition("=")[2].lstrip()
        return "object" if re.match(r"struct\b", value_text) else None
    parts = _named(node)
    value = parts[-1] if len(parts) > 1 else None
    if value is None:
        return "variable"
    if value.type in _VALUE_KINDS:
        return _VALUE_KINDS[value.type]
    return "module" if _text(value).startswith("import(") else "variable"


# Operators whose result is a ``bool``, whatever they compare.
_LOGICAL = {"<", "<=", ">", ">=", "==", "!=", "&&", "||", "!", "and", "or", "not"}


# How long a record in an inlay hint may get before its other fields are left as `...`.
_HINT_RECORD = 30


def _shortened(hint: InlayHint) -> InlayHint:
    """Shorten the records in a type hint, keeping the whole type for its tooltip."""
    if hint.parameter or "#{" not in hint.label:
        return hint
    short = _short_type(hint.label.removeprefix(": "), nested=False)
    if f": {short}" == hint.label:
        return hint
    return InlayHint(hint.position, f": {short}", tooltip=hint.label.removeprefix(": "))


def _short_type(kind: str, *, nested: bool) -> str:
    """``#{task1: #{task1: str, ok: bool}, task2: str}`` -> ``#{task1: #{task1: str, ...}, ...}``.

    A record shows its fields while they fit (a nested one, only its first),
    and ``...`` for the rest; other types are left as they are.
    """
    if not (kind.startswith("#{") and kind.endswith("}")):
        return kind
    fields = _top_level(kind[2:-1])
    shown: list[str] = []
    for index, written in enumerate(fields):
        name, _, held = written.partition(": ")
        entry = f"{name}: {_short_type(held, nested=True)}"
        more = ", ..." if index < len(fields) - 1 else ""
        if shown and (nested or len("#{" + ", ".join([*shown, entry]) + more + "}") > _HINT_RECORD):
            break
        shown.append(entry)
    rest = ", ..." if len(shown) < len(fields) else ""
    return "#{" + ", ".join(shown) + rest + "}"


def _top_level(text: str) -> list[str]:
    """Split ``a: i64, b: #{c: str, d: f64}`` at its top-level commas."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _field_type(kind: Any, path: list[str]) -> Any:  # noqa: ANN401 - a compiler type
    """Follow ``path`` through the fields of a record or object type; ``None`` where one is missing."""
    for name in path:
        if kind is None or kind.kind != "struct":
            return None
        kind = dict(kind.fields).get(name)
    return kind


def _written_entries(value: Node) -> list[tuple[str, Node]] | None:
    """Return the fields a record gets where it is written: ``#{x: 1}``, or ``tasks.gather(user: f())``'s names."""
    if value.type == "literal":
        value = _named(value)[0]
    if value.type == "record":
        fields = [field for field in _named(value) if field.type == "record_field" and len(_named(field)) > 1]
        return [(_text(_named(field)[0]), _named(field)[-1]) for field in fields]
    if value.type == "child_annotation" and _text(value).split("(")[0].endswith(".gather"):
        call = _named(_named(value)[-1])[0]  # the user_function_call
        arguments = [argument for argument in _named(call) if argument.type == "named_argument"]
        return [
            (_text(name), written)
            for argument in arguments
            if (name := argument.child_by_field_name("name")) is not None
            and (written := argument.child_by_field_name("value")) is not None
        ]
    return None


def _parameter_names(function: Node) -> list[str] | None:
    """Return the names of the parameters a function, lambda or object constructor takes.

    A constructor takes its object's stored fields, in order; ``None`` for anything else.
    """
    if function.type == "struct_assignment":
        members = [m for m in _named(function) if m.type == "struct_field" and _named(m)[-1].type == "type_or_object"]
    elif function.type in {"function_definition", "local_function_definition"}:
        found = next((part for part in _named(function) if part.type == "parameter_list"), None)
        if found is None:
            return None
        members = _named(found)
    else:
        return None
    return [_text(identifier) for member in members if (identifier := _name_of(member)) is not None]


def _signature_names(signature: str) -> list[str]:
    """Return the names of the parameters in a builtin's signature, before any ``...``.

    ``(pattern: str, ...) => mem.Unique[str]`` gives ``["pattern"]``.
    """
    inside, depth = "", 0
    for character in signature:
        depth += {"(": 1, ")": -1}.get(character, 0)
        if depth == 0:
            break
        inside += character
    names = []
    for part in inside[1:].split(","):
        name, colon, _ = part.strip().partition(":")
        if not colon or not name.isidentifier():
            break
        names.append(name)
    return names


def _declares(member: Node, name: str) -> bool:
    """Whether a ``struct_field`` declares ``name``."""
    identifier = _name_of(member)
    return identifier is not None and _text(identifier) == name


def _ancestor(node: Node, kind: str) -> Node | None:
    """Return the nearest ``kind`` node around ``node``, stopping at a function."""
    current = node.parent
    while current is not None and current.type not in {kind, "function_definition", "local_function_definition"}:
        current = current.parent
    return current if current is not None and current.type == kind else None


def _doc(node: Node | None) -> str:
    """Return the documentation of a function, object or module: a ``/* ... */`` first in its body.

    As a Python docstring is: ``let draw = (...) => null { /* Draw the frame */ ... }``,
    ``struct { /* ... */ let x: i32 }``, ``module helper = { /* ... */ ... }``.
    ``//`` comments are only comments.
    """
    if node is None:
        return ""
    body = node
    if node.type in {"function_definition", "local_function_definition"}:
        body = _named(node)[-1] if _named(node) else node
        if body.type != "block_expression":
            return ""  # a one-expression body has nowhere for it
    elif node.type not in {"struct_assignment", "module"}:
        return ""
    children = body.children
    opening = next((index for index, child in enumerate(children) if child.type == "{"), None)
    for child in children[opening + 1 :] if opening is not None else []:  # after `{`: a module's name comes before
        if child.type == "comment":
            return _doc_text(_text(child)) if _text(child).startswith("/*") else ""
        if child.is_named:
            return ""
    return ""


def _doc_text(comment: str) -> str:
    """Strip a block comment to its text: ``/* Draw the frame`` / `` * using ctx */`` is two lines of text."""
    lines = comment.removeprefix("/*").removesuffix("*/").splitlines()
    return "\n".join(line.strip().removeprefix("*").strip() for line in lines).strip()


def _documented(code: str, doc: str) -> str:
    """Return a hover: the code, then its documentation (Markdown) under it."""
    return _code(code) + (f"\n\n{doc}" if doc else "")


def _code(text: str) -> str:
    return f"```bifrost\n{text}\n```"


def _type_name(kind: Any) -> str:  # noqa: ANN401 - a compiler type
    """Write a compiler type as Bifrost does: ``#{sum: i64}`` for a record, ``(i32) => null`` for a function."""
    if kind.kind == "fn":
        return display_types(kind.name)
    if kind.kind == "struct" and kind.name == "record":
        return "#{" + ", ".join(f"{name}: {_type_name(field)}" for name, field in kind.fields) + "}"
    if is_list(kind):
        return f"{_type_name(kind.element)}[]"
    return {"cstr": "str"}.get(kind.name, kind.name)


# Standard modules whose functions the compiler expands itself.
_BUILTIN_FUNCTIONS = {
    f"{std.PREFIX}fmt": fmt.FUNCTIONS,
    f"{std.PREFIX}json": json.FUNCTIONS,
    f"{std.PREFIX}tasks": tasks.FUNCTIONS,
}


def _bifrost_type(kind: str) -> str:
    """Show a config type as Bifrost writes it: ``raylib_Color`` is ``raylib.Color``, ``None`` is ``null``."""
    function = re.fullmatch(r"\(([^()]*)\)\s*=>\s*([^()]+)", kind.strip())
    if function is not None:
        parameters = ", ".join(_bifrost_type(part.strip()) for part in function.group(1).split(",") if part.strip())
        return f"({parameters}) => {_bifrost_type(function.group(2).strip())}"
    if kind == "None":
        return "null"
    if kind == "cstr":
        return "str"
    module, _, name = kind.partition("_")
    module = module.removeprefix(std.PREFIX)  # std:stdio_File would be stdio.File
    return f"{module}.{to_pascal_case(name)}" if name and name[0].isupper() else kind


def _extern_signature(module: str, declaration: Declaration) -> str:
    """Show an extern as Bifrost sees it, with the C symbol it binds."""
    name = f"{module.removeprefix(std.PREFIX)}.{extern_name(declaration)}"
    doc = f"\n// {declaration.doc}" if declaration.doc else ""
    if declaration.type == "struct":
        fields = ", ".join(
            f"let {to_snake_case(field)}: {_bifrost_type(kind)}" for field, kind in declaration.fields.items()
        )
        return f"{name} = struct {{ {fields} }}{doc}\n// C: struct {declaration.name}"
    parameters = [
        f"{to_snake_case(parameter)}: {_bifrost_type(kind)}" for parameter, kind in declaration.parameters.items()
    ]
    if declaration.variadic:
        parameters.append("...")
    result = _bifrost_type(declaration.return_type)
    return f"{name} = ({', '.join(parameters)}) => {result}{doc}\n// C: {declaration.name}"
