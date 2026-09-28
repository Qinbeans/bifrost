"""Generate ``config.yaml`` extern declarations from C and C++ headers.

``traverse`` parses each header with libclang and turns what it declares into
the ``externs`` entries ``Project`` loads: one module per header (named after
the file), with its structs and then its functions. Only declarations written
in the traversed headers are taken, not those of headers they include.

C types map by their size on this platform: integers to ``i8``..``u64``,
``float``/``double`` to ``f32``/``f64``, ``const char *`` to ``cstr``, other
pointers to ``ptr``, enums to their integer type, and structs passed by value to
``module_Name``. What cannot be bound is skipped with a reason: variadic
functions, ``static`` functions (no symbol to link), C++ functions
without ``extern "C"``, structs with array, union or bit-field members, and
anything that uses a skipped struct.
"""

import keyword
import re
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from clang import cindex
from clang.cindex import Cursor, CursorKind, Type, TypeKind

HEADER_LANGUAGES = {".h": "c", ".hpp": "c++", ".hh": "c++", ".hxx": "c++", ".h++": "c++"}

_SIGNED = {TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.SHORT, TypeKind.INT, TypeKind.LONG, TypeKind.LONGLONG}
_UNSIGNED = {
    TypeKind.CHAR_U,
    TypeKind.UCHAR,
    TypeKind.USHORT,
    TypeKind.UINT,
    TypeKind.ULONG,
    TypeKind.ULONGLONG,
    TypeKind.CHAR16,
    TypeKind.CHAR32,
    TypeKind.WCHAR,
}
_FIXED = {TypeKind.VOID: "None", TypeKind.BOOL: "bool", TypeKind.FLOAT: "f32", TypeKind.DOUBLE: "f64"}
_CHARACTERS = {TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.CHAR_U}
_INTEGER_BITS = {8, 16, 32, 64}


class TraverseError(Exception):
    """A header could not be parsed, or the result does not load."""


class _Unsupported(Exception):  # noqa: N818 - internal control flow, never escapes
    """A C type with no ``config.yaml`` equivalent."""


@dataclass(frozen=True)
class Skipped:
    """A declaration that was not bound, and why."""

    name: str
    reason: str


@dataclass
class ModuleExterns:
    """The externs generated from one header."""

    module: str
    header: Path
    structs: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    # Headers included first because this one is not self-contained (rcamera.h needs raylib.h).
    context: list[Path] = field(default_factory=list)
    # Why the header could not be parsed at all; its module is then left as it was.
    error: str | None = None

    def declarations(self) -> list[dict[str, Any]]:
        """Structs first, as functions may take them."""
        return [*self.structs, *self.functions]


def find_headers(paths: Iterable[Path]) -> list[Path]:
    """Expand directories into the headers under them, in a stable order."""
    headers: list[Path] = []
    for path in paths:
        if path.is_dir():
            headers.extend(sorted(p for p in path.rglob("*") if p.suffix in HEADER_LANGUAGES and p.is_file()))
        elif path.is_file():
            headers.append(path)
        else:
            msg = f"{path} does not exist"
            raise TraverseError(msg)
    unique = list(dict.fromkeys(h.resolve() for h in headers))
    if not unique:
        msg = "no C or C++ headers found in " + ", ".join(str(p) for p in paths)
        raise TraverseError(msg)
    return unique


def builtin_includes() -> list[str]:
    """Return ``-isystem`` for clang's own headers (``stdarg.h``, ``stddef.h``, ...).

    The libclang wheel does not ship them, so they come from a ``clang`` on
    ``PATH``, if there is one.
    """
    clang = shutil.which("clang")
    if clang is None:
        return []
    result = subprocess.run([clang, "-print-resource-dir"], capture_output=True, text=True, check=False)  # noqa: S603
    include = Path(result.stdout.strip()) / "include"
    return ["-isystem", str(include)] if result.returncode == 0 and include.is_dir() else []


def module_name(header: Path) -> str:
    """Name a header's module after the file: ``raylib.h`` -> ``raylib``."""
    name = re.sub(r"\W", "_", header.stem)
    return f"_{name}" if name[0].isdigit() else name


class _Traversal:
    """Parses every header, then maps each one's declarations."""

    def __init__(self, headers: list[Path], include_dirs: list[Path], clang_args: list[str]) -> None:
        self.headers = headers
        self.index = cindex.Index.create()
        search = [f"-I{d}" for d in dict.fromkeys([*include_dirs, *(h.parent for h in headers)])]
        self.arguments = [*search, *builtin_includes(), *clang_args]
        self.units: dict[Path, cindex.TranslationUnit] = {}
        self.context: dict[Path, list[Path]] = {}
        self.errors: dict[Path, str] = {}
        for header in headers:
            unit, errors = self.parse(header, [])
            if errors:
                self.errors[header] = errors
            else:
                self.units[header] = unit
        # Some headers only parse after another (rcamera.h uses raylib.h's types
        # without including it): retry them after the ones that parsed alone.
        standalone = list(self.units)
        for header in list(self.errors):
            for context in [standalone, *([other] for other in standalone)]:
                unit, errors = self.parse(header, context)
                if not errors:
                    self.units[header], self.context[header] = unit, context
                    del self.errors[header]
                    break
        # Struct USR -> its config name ("raylib_Color"), or why it was skipped.
        self.structs: dict[str, str] = {}
        self.skipped_structs: dict[str, str] = {}

    def parse(self, header: Path, context: list[Path]) -> tuple[cindex.TranslationUnit, str]:
        """Parse ``header`` after including ``context``; return the unit and any errors."""
        language = HEADER_LANGUAGES.get(header.suffix, "c")
        included = [argument for path in context for argument in ("-include", str(path))]
        unit = self.index.parse(
            str(header),
            args=["-x", language, *self.arguments, *included],
            options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
        )
        errors = [d for d in unit.diagnostics if d.severity >= cindex.Diagnostic.Error]
        if not errors:
            return unit, ""
        shown = "\n".join(f"  {d}" for d in errors[:5])
        more = f"\n  ... and {len(errors) - 5} more" if len(errors) > 5 else ""  # noqa: PLR2004 - errors shown
        hint = ""
        if any("file not found" in d.spelling for d in errors):
            hint = "\n(add search paths with -I; clang's own headers come from a clang on PATH)"
        return unit, f"{shown}{more}{hint}"

    def own_cursors(self, header: Path) -> Iterator[Cursor]:
        """Yield the top-level declarations written in ``header``.

        That includes those inside ``extern "C"`` blocks and namespaces (whose
        functions are then reported as having C++ linkage).
        """
        pending = list(self.units[header].cursor.get_children())
        while pending:
            cursor = pending.pop(0)
            if cursor.location.file is None or Path(cursor.location.file.name).resolve() != header:
                continue
            if cursor.kind in {CursorKind.LINKAGE_SPEC, CursorKind.NAMESPACE}:
                pending[:0] = list(cursor.get_children())
                continue
            yield cursor

    # -- structs ------------------------------------------------------------------

    def struct_names(self, header: Path) -> list[tuple[str, Cursor]]:
        """Name each struct defined in ``header``: its tag, or its typedef if anonymous."""
        definitions: dict[str, Cursor] = {}
        names: dict[str, str] = {}
        for cursor in self.own_cursors(header):
            if cursor.kind == CursorKind.STRUCT_DECL and cursor.is_definition():
                definitions.setdefault(cursor.get_usr(), cursor)
                if not cursor.is_anonymous():
                    names.setdefault(cursor.get_usr(), cursor.spelling)
            elif cursor.kind == CursorKind.TYPEDEF_DECL:
                target = cursor.underlying_typedef_type.get_canonical().get_declaration()
                if target.kind == CursorKind.STRUCT_DECL and target.is_definition():
                    definitions.setdefault(target.get_usr(), target)
                    names.setdefault(target.get_usr(), cursor.spelling)
        return [(names[usr], cursor) for usr, cursor in definitions.items() if usr in names]

    def struct(self, name: str, cursor: Cursor) -> dict[str, Any] | str:
        """Return the struct's declaration, or why it cannot be one."""
        fields: dict[str, str] = {}
        for member in cursor.get_children():
            if member.kind != CursorKind.FIELD_DECL:
                if member.kind in {CursorKind.STRUCT_DECL, CursorKind.UNION_DECL} and member.is_definition():
                    continue  # a nested definition; any member of its type is checked below
                continue
            if member.is_bitfield():
                return f"member {member.spelling} is a bit-field"
            try:
                fields[member.spelling] = self.type(member.type, in_struct=True)
            except _Unsupported as error:
                return f"member {member.spelling}: {error}"
        if not fields:
            return "it has no members"
        return {"name": name, "type": "struct", "fields": fields}

    # -- types --------------------------------------------------------------------

    def type(self, c_type: Type, *, in_struct: bool = False) -> str:
        """Map a C type to a ``config.yaml`` type name."""
        canonical = c_type.get_canonical()
        kind = canonical.kind
        if kind in _FIXED:
            return _FIXED[kind]
        if kind in _SIGNED or kind in _UNSIGNED:
            bits = canonical.get_size() * 8
            if bits not in _INTEGER_BITS:
                msg = f"{c_type.spelling} is {bits} bits"
                raise _Unsupported(msg)
            return f"{'u' if kind in _UNSIGNED else 'i'}{bits}"
        if kind == TypeKind.ENUM:
            return self.type(canonical.get_declaration().enum_type)
        if kind == TypeKind.POINTER:
            pointee = canonical.get_pointee()
            is_text = pointee.get_canonical().kind in _CHARACTERS and pointee.is_const_qualified()
            return "cstr" if is_text else "ptr"
        if kind == TypeKind.RECORD:
            return self.record(canonical)
        if kind in {TypeKind.CONSTANTARRAY, TypeKind.INCOMPLETEARRAY, TypeKind.VARIABLEARRAY}:
            what = "an array member" if in_struct else "an array"
            msg = f"{c_type.spelling} is {what}"
            raise _Unsupported(msg)
        msg = f"{c_type.spelling} has no config type"
        raise _Unsupported(msg)

    def record(self, canonical: Type) -> str:
        declaration = canonical.get_declaration()
        if declaration.kind == CursorKind.UNION_DECL:
            msg = f"{canonical.spelling} is a union"
            raise _Unsupported(msg)
        usr = declaration.get_usr()
        if usr in self.structs:
            return self.structs[usr]
        if usr in self.skipped_structs:
            msg = f"{canonical.spelling} was skipped ({self.skipped_structs[usr]})"
            raise _Unsupported(msg)
        msg = f"{canonical.spelling} is not defined in the traversed headers"
        raise _Unsupported(msg)

    # -- functions ----------------------------------------------------------------

    def function(self, cursor: Cursor, *, cpp: bool) -> dict[str, Any] | str:
        """Return the function's declaration, or why it cannot be one."""
        if cursor.type.kind == TypeKind.FUNCTIONPROTO and cursor.type.is_function_variadic():
            return "it is variadic"
        # Plain `inline` functions are bound: C requires an external definition
        # somewhere (libraylib.a defines raymath's). `static` ones have none.
        if cursor.storage_class == cindex.StorageClass.STATIC:
            return "it is static, so no library defines its symbol"
        if cpp and cursor.mangled_name.removeprefix("_") != cursor.spelling:
            return 'it has C++ linkage; declare it extern "C" to bind it'
        parameters: dict[str, str] = {}
        for position, argument in enumerate(cursor.get_arguments()):
            name = argument.spelling or f"arg{position}"
            if keyword.iskeyword(name) or name in parameters:
                name = f"{name}_{position}"
            try:
                parameters[name] = self.type(argument.type)
            except _Unsupported as error:
                return f"parameter {name}: {error}"
        try:
            result = self.type(cursor.result_type)
        except _Unsupported as error:
            return f"result: {error}"
        return {"name": cursor.spelling, "type": "function", "parameters": parameters, "return": result}

    # -- modules ------------------------------------------------------------------

    def run(self) -> list[ModuleExterns]:
        """Map every header's structs, then every header's functions."""
        modules = [
            ModuleExterns(module_name(h), h, context=self.context.get(h, []), error=self.errors.get(h))
            for h in self.headers
        ]
        parsed = [m for m in modules if m.error is None]
        if not parsed:
            details = "\n".join(f"{m.header}:\n{m.error}" for m in modules)
            msg = f"no header could be parsed:\n{details}"
            raise TraverseError(msg)
        modules_with_units, modules = modules, parsed
        owned = self.claim_structs(modules)
        for module in modules:
            self.declare_structs(module, owned[module.header])
        for module in modules:
            self.declare_functions(module)
        return modules_with_units

    def claim_structs(self, modules: list[ModuleExterns]) -> dict[Path, list[tuple[str, Cursor]]]:
        """Name every struct before mapping any, so any header can use any header's structs.

        A struct several headers define (``raylib.h`` and ``rlgl.h`` both define
        ``Matrix``) belongs to the first; the others refer to it.
        """
        owned: dict[Path, list[tuple[str, Cursor]]] = {m.header: [] for m in modules}
        for module in modules:
            for name, cursor in self.struct_names(module.header):
                if cursor.get_usr() not in self.structs:
                    self.structs[cursor.get_usr()] = f"{module.module}_{name}"
                    owned[module.header].append((name, cursor))
        return owned

    def declare_structs(self, module: ModuleExterns, structs: list[tuple[str, Cursor]]) -> None:
        for name, cursor in structs:
            declaration = self.struct(name, cursor)
            if isinstance(declaration, str):
                del self.structs[cursor.get_usr()]
                self.skipped_structs[cursor.get_usr()] = declaration
                module.skipped.append(Skipped(name, declaration))
            else:
                module.structs.append(declaration)

    def declare_functions(self, module: ModuleExterns) -> None:
        cpp = HEADER_LANGUAGES.get(module.header.suffix) == "c++"
        seen: set[str] = set()
        for cursor in self.own_cursors(module.header):
            if cursor.kind != CursorKind.FUNCTION_DECL or cursor.spelling in seen:
                continue
            seen.add(cursor.spelling)
            declaration = self.function(cursor, cpp=cpp)
            if isinstance(declaration, str):
                module.skipped.append(Skipped(cursor.spelling, declaration))
            else:
                module.functions.append(declaration)


def traverse(
    paths: Iterable[Path], include_dirs: Iterable[Path] = (), clang_args: Iterable[str] = ()
) -> list[ModuleExterns]:
    """Generate one module of externs per header found under ``paths``."""
    return _Traversal(find_headers(paths), list(include_dirs), list(clang_args)).run()


# -- writing config.yaml ----------------------------------------------------------


class _Dumper(yaml.SafeDumper):
    """Indent list items under their key, as ``config.yaml`` is written by hand."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:  # noqa: FBT002, ARG002
        super().increase_indent(flow, False)


def merge(config: dict[str, Any], modules: list[ModuleExterns]) -> dict[str, Any]:
    """Replace (or add) each generated module in ``config``'s externs, keeping the rest."""
    externs = list(config.get("externs") or [])
    for module in modules:
        if module.error is not None:
            continue  # keep what the config already has for a header that did not parse
        existing = next((e for e in externs if e.get("module") == module.module), None)
        entry = {
            "module": module.module,
            "description": (existing or {}).get("description") or f"Generated from {module.header.name}",
            "declarations": module.declarations(),
        }
        if existing is None:
            externs.append(entry)
        else:
            externs[externs.index(existing)] = entry
    return {**config, "externs": externs}


def dump(config: dict[str, Any]) -> str:
    """Write ``config`` as YAML, with a blank line between top-level sections."""
    text = yaml.dump(config, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=120)
    return re.sub(r"\n(?=[^\s-])", "\n\n", text)


__all__ = ["ModuleExterns", "Skipped", "TraverseError", "dump", "find_headers", "merge", "module_name", "traverse"]
