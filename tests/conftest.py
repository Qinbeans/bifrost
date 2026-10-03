"""What every test shares: packages the examples take from an index come from extras/ instead."""

import functools
from pathlib import Path

import pytest
import yaml

from bifrost import packages
from bifrost.configs.schema import _PackageSource

EXTRAS = Path(__file__).parents[1] / "extras"


@functools.cache
def _extras() -> dict[str, Path]:
    """Return each package in extras/, by name: its folder."""
    found = {}
    for config in EXTRAS.glob("*/config.yaml"):
        package = (yaml.load(config.read_text(), Loader=yaml.CSafeLoader) or {}).get("package") or {}
        if package.get("name"):
            found[package["name"]] = config.parent
    return found


@pytest.fixture(autouse=True)
def _packages_from_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve a package in extras/ from its folder, not an index: what CI publishes, offline and unreleased."""
    local = _extras()
    resolve = packages._Resolver.resolve

    def from_extras(
        self: object, name: str, source: str | _PackageSource, folder: Path, indexes: dict[str, str]
    ) -> None:
        if name in local and (isinstance(source, str) or source.index is not None):
            source = _PackageSource(path=str(local[name]))
        resolve(self, name, source, folder, indexes)

    monkeypatch.setattr(packages._Resolver, "resolve", from_extras)
