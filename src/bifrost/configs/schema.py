from pathlib import Path
from typing import Literal

from mlir_python.codegen import Library, OptLevel
from pydantic import BaseModel, Field, field_validator


class _Flags(BaseModel):
    optimization: OptLevel
    linker: str


class _Package(BaseModel):
    name: str
    version: str
    description: str
    # The file `bifrost build` compiles when given none, relative to config.yaml: src/<name>/main.bif.
    entry: str | None = None
    # `async` lets `main` be async (`let main = [...] async () => ...`): it runs on the event
    # loop, and the program ends when it is done. `sync` (the default) does not.
    type: Literal["sync", "async"] = "sync"


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


class Config(BaseModel):
    path: Path = Path("build")
    package: _Package
    flags: _Flags
    libraries: list[Library] = []
    externs: list[_Extern] = []

    @field_validator("libraries", mode="before")
    @classmethod
    def parse_library_paths(cls, libraries: object) -> object:
        """Convert path-like library entries from YAML into Path objects."""
        if not isinstance(libraries, list):
            return libraries

        return [Path(library) if isinstance(library, str) and "/" in library else library for library in libraries]
