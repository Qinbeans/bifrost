"""Read and change a project's version (``package.version`` in config.yaml), like ``uv version``.

Only the ``version:`` line under ``package:`` is rewritten, so the rest of the
file, comments included, stays as it was.
"""

import re
from enum import StrEnum
from pathlib import Path

import yaml

# `  version: 0.1.0` inside the `package:` block (indented, before the next top-level key).
_VERSION_LINE = re.compile(r"(?m)^(package:[ \t]*\n(?:[ \t]+.*\n|[ \t]*\n)*?[ \t]+version:[ \t]*)(\S+)(.*)$")
_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class VersionError(Exception):
    """The version cannot be read, bumped or written."""


class Bump(StrEnum):
    """The part of a ``major.minor.patch`` version to bump; the parts after it reset to 0."""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


def read(config: Path) -> tuple[str, str]:
    """Return the project's name and version.

    Raises:
        VersionError: If config.yaml has no ``package.version``.

    """
    package = (yaml.safe_load(config.read_text()) or {}).get("package") or {}
    if "version" not in package:
        msg = f"{config} has no package.version"
        raise VersionError(msg)
    return str(package.get("name", "")), str(package["version"])


def bumped(version: str, part: Bump) -> str:
    """Return ``version`` with ``part`` bumped: ``0.1.3`` -> ``0.2.0`` for ``minor``.

    Raises:
        VersionError: If ``version`` is not ``major.minor.patch``.

    """
    found = _SEMVER.fullmatch(version)
    if found is None:
        msg = f"cannot bump {version}: it is not major.minor.patch, like 0.1.0"
        raise VersionError(msg)
    major, minor, patch = (int(number) for number in found.groups())
    match part:
        case Bump.MAJOR:
            return f"{major + 1}.0.0"
        case Bump.MINOR:
            return f"{major}.{minor + 1}.0"
        case Bump.PATCH:
            return f"{major}.{minor}.{patch + 1}"


def write(config: Path, version: str) -> None:
    """Set ``package.version`` in ``config``, changing nothing else.

    Raises:
        VersionError: If ``version`` is not ``major.minor.patch``, or the line is not found.

    """
    if _SEMVER.fullmatch(version) is None:
        msg = f"{version} is not a version: use major.minor.patch, like 1.2.0"
        raise VersionError(msg)
    text = config.read_text()
    updated, count = _VERSION_LINE.subn(lambda found: f"{found[1]}{version}{found[3]}", text, count=1)
    if count == 0:
        msg = f"no `version:` line under `package:` in {config}"
        raise VersionError(msg)
    config.write_text(updated)


__all__ = ["Bump", "VersionError", "bumped", "read", "write"]
