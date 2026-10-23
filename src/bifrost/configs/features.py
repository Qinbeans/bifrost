"""Features: parts of a package (or project) that are used only when asked for, as Cargo's are.

A ``config.yaml`` names its features, and what each turns on::

    features:
      default: [json, query]        # on unless a user says `default-features: false`
      json: []                      # turns on nothing else
      query: []
      ws:                           # with what only it brings:
        features: [http1]           #   other features it turns on
        modules: [axum.ws]          #   Bifrost files only it may import (src/axum/ws.bif)
        externs: [axum_ws]          #   extern modules declared only with it
        libraries: [./build/lib/libaxum_ws.a, wslay]   # linked only with it

A project picks a package's features where it lists it::

    packages:
      axum:
        version: 0.1.0
        features: [ws]              # besides the defaults
        default-features: false     # or only those listed

Every use of a package asks for some of its features, and it is built with all
of them (each turned on with what it turns on): a package used twice is used
once. A project's own features are its defaults, or what ``bfc build
--features`` gives. A file a feature gates cannot be imported without it: the
error says which feature to turn on.
"""

from pathlib import Path

from bifrost.configs.schema import Config, _Extern, _Feature, source_root

DEFAULT = "default"


class FeatureError(ValueError):
    """A feature asked for that does not exist; the message says what does."""


def defined(config: Config) -> dict[str, _Feature]:
    """Return the features ``config`` defines (not ``default``), each as what it turns on."""
    return {
        name: feature if isinstance(feature, _Feature) else _Feature(features=feature)
        for name, feature in config.features.items()
        if name != DEFAULT
    }


def enabled(config: Config, requested: set[str], *, defaults: bool, owner: str) -> set[str]:
    """Return the features turned on: those ``requested`` (and the defaults, with ``defaults``), with all they enable.

    Raises:
        FeatureError: If one asked for is not defined.

    """
    features = defined(config)
    default = config.features.get(DEFAULT, [])
    pending = [*sorted(requested), *(default if defaults and isinstance(default, list) else [])]
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        if name not in features:
            known = ", ".join(sorted(features)) or "none"
            raise FeatureError(f"{owner} has no feature {name!r} (its features: {known})")
        found.add(name)
        pending.extend(features[name].features)
    return found


def check(config: Config) -> None:
    """Check what ``config``'s features name: features, externs and modules it has.

    Raises:
        FeatureError: At the first that is not.

    """
    features = defined(config)
    default = config.features.get(DEFAULT, [])
    if not isinstance(default, list):
        raise FeatureError("features.default lists features: default: [json, query]")
    modules = {extern.module for extern in config.externs}
    for name, feature in [(DEFAULT, _Feature(features=default)), *features.items()]:
        for other in feature.features:
            if other not in features:
                raise FeatureError(f"feature {name} turns on {other!r}, which is not a feature")
        for extern in feature.externs:
            if extern not in modules:
                raise FeatureError(f"feature {name} gates extern module {extern!r}, which externs: does not declare")


def apply(config: Config, root: Path, on: set[str]) -> tuple[list[_Extern], list[object], dict[Path, str]]:
    """Return what ``config`` (in folder ``root``) brings with the features ``on``: externs, libraries, gated files.

    Externs a feature that is off gates are left out; libraries of features that are on are
    added (relative ones to ``root``); files of features that are off cannot be imported (each
    maps to why).
    """
    features = defined(config)
    off = {name: feature for name, feature in features.items() if name not in on}
    hidden = {extern for feature in off.values() for extern in feature.externs}
    externs = [extern for extern in config.externs if extern.module not in hidden]
    libraries: list[object] = []
    for name in sorted(on):
        for library in features[name].libraries:
            path = root / library if isinstance(library, Path) and not library.is_absolute() else library
            if path not in libraries:
                libraries.append(path)
    gated: dict[Path, str] = {}
    sources = source_root(root)
    owner = config.package.name
    for name, feature in off.items():
        for module in feature.modules:
            path = sources.joinpath(*module.split(".")).with_suffix(".bif").resolve()
            gated[path] = (
                f"{module} is part of {owner}'s feature {name}, which is off; turn it on where {owner} is "
                f"listed: features: [{name}]"
            )
    return externs, libraries, gated


__all__ = ["DEFAULT", "FeatureError", "apply", "check", "defined", "enabled"]
