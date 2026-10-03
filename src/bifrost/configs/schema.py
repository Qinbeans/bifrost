from pathlib import Path
from typing import Literal

from mlir_python.codegen import Library, OptLevel
from pydantic import BaseModel, Field, field_validator, model_validator


class _Flags(BaseModel):
    optimization: OptLevel
    linker: str


class _Package(BaseModel):
    name: str
    version: str
    description: str
    # The file `bfc build` compiles when given none, relative to config.yaml: src/<name>/main.bif.
    entry: str | None = None
    # `async` lets `main` be async (`let main = [...] async () => ...`): it runs on the event
    # loop, and the program ends when it is done. `sync` (the default) does not.
    type: Literal["sync", "async"] = "sync"
    # What the project is: an `executable` (`bfc build` compiles its entry; the default), or a
    # `library`: a package other projects use, which `bfc package` makes a .bifpkg of.
    build: Literal["executable", "library"] = "executable"

    @model_validator(mode="after")
    def _library_has_no_entry(self) -> "_Package":
        if self.build == "library" and self.entry is not None:
            msg = "a library has no entry (package.build is library); remove package.entry, or build an executable"
            raise ValueError(msg)
        return self


class _Struct(BaseModel):
    name: str
    type: Literal["struct"]
    fields: dict[str, str]
    # The name Bifrost code uses, when not the C name made PascalCase (`as: Request`).
    bifrost_name: str | None = Field(None, alias="as")
    # One line about it, shown in the editor.
    doc: str = ""


class _Function(BaseModel):
    name: str
    type: Literal["function"]
    parameters: dict[str, str]
    return_type: str = Field(alias="return")
    # Takes further arguments after `parameters`, like C's `printf(const char *, ...)`.
    variadic: bool = False
    # The name Bifrost code uses, when not the C name made snake_case (`as: listen`).
    bifrost_name: str | None = Field(None, alias="as")
    # One line about it, shown in the editor.
    doc: str = ""


Declaration = _Struct | _Function


class _Extern(BaseModel):
    module: str
    description: str
    declarations: list[Declaration]


def source_root(project: Path) -> Path:
    """Return where ``import("a.b:module")`` looks for ``a/b.bif`` in the project at ``project``.

    With a source layout, that is ``src/`` (``import("app.api:users")`` is
    ``src/app/api.bif``, as in Python); otherwise, the project folder itself.
    """
    source = project / "src"
    return source if source.is_dir() else project


# Where a package listed only by version comes from, unless config.yaml names an index `default`:
# the packages in extras/, published by CI.
DEFAULT_INDEX = "https://qinbeans.github.io/bifrost/packages/"


class _PackageSource(BaseModel):
    """Where a package comes from, when not the default index: a named index, a ``.bifpkg`` or a project folder."""

    # A `.bifpkg` file or a project folder, relative to config.yaml.
    path: str | None = None
    # One of the indexes config.yaml names (`index:`).
    index: str | None = None
    # The version it must have: required from an index.
    version: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "_PackageSource":
        if (self.path is None) == (self.index is None):
            msg = "a package comes from a path or an index: give one of path: and index:"
            raise ValueError(msg)
        if self.index is not None and self.version is None:
            msg = f"a package from index {self.index} needs its version:"
            raise ValueError(msg)
        return self


class Config(BaseModel):
    path: Path = Path("build")
    package: _Package
    flags: _Flags
    libraries: list[Library] = []
    externs: list[_Extern] = []
    # The indexes packages come from, by name: a folder (relative to config.yaml), or a URL serving
    # a page of links to .bifpkg files. A package listed only by version comes from `default`, or
    # else DEFAULT_INDEX.
    index: dict[str, str] = {}
    # The packages this project uses: name -> version (from the default index), {index: name,
    # version: ...}, or {path: ...}.
    packages: dict[str, str | _PackageSource] = {}
    # Where `import("a.b:module")` also looks, after the project's own sources: its packages'.
    # Filled in when the packages are resolved (see `bifrost.packages`), never written.
    sources: list[Path] = Field(default_factory=list, exclude=True)
    # The config.yaml declaring each extern module that a package brings (others are this one's).
    declared_in: dict[str, Path] = Field(default_factory=dict, exclude=True)

    @field_validator("index", mode="before")
    @classmethod
    def _named_indexes(cls, index: object) -> object:
        if isinstance(index, str):
            msg = "index: names its indexes (index: {extras: https://...}); a package picks one with `index: extras`"
            raise ValueError(msg)
        return index

    @field_validator("libraries", mode="before")
    @classmethod
    def parse_library_paths(cls, libraries: object) -> object:
        """Convert path-like library entries from YAML into Path objects."""
        if not isinstance(libraries, list):
            return libraries

        return [Path(library) if isinstance(library, str) and "/" in library else library for library in libraries]
