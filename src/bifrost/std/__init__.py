"""Bifrost's standard modules: extern declarations bundled with the compiler.

Each ``<name>.yaml`` here is one ``externs`` entry, as a project's
``config.yaml`` would write it, for a library every program can link without
configuring it (libc). Bifrost code imports one as ``import("std:<name>")``.
"""

from importlib import resources
from pathlib import Path

import yaml

from bifrost.configs.schema import _Extern

PREFIX = "std:"
# Standard modules built into the compiler rather than declared in YAML.
BUILTIN = {"mem", "fmt", "json", "tasks"}


def available() -> list[str]:
    """Return the names of the standard modules, e.g. ``["mem", "stdio"]``."""
    return sorted(
        BUILTIN
        | {
            entry.name.removesuffix(".yaml")
            for entry in resources.files(__package__).iterdir()
            if entry.name.endswith(".yaml")
        }
    )


def path(name: str) -> Path:
    """Return the file declaring standard module ``name``."""
    return Path(str(resources.files(__package__) / f"{name}.yaml"))


def load(name: str) -> _Extern:
    """Load standard module ``name``, whose ``module`` is ``std:<name>``.

    Raises:
        KeyError: If there is no such standard module.

    """
    if name not in available() or name in BUILTIN:
        raise KeyError(name)
    extern = _Extern(**yaml.safe_load(path(name).read_text()))
    if extern.module != f"{PREFIX}{name}":
        msg = f"{path(name)} declares module {extern.module!r}, expected {PREFIX}{name!r}"
        raise ValueError(msg)
    return extern


__all__ = ["PREFIX", "available", "load", "path"]
