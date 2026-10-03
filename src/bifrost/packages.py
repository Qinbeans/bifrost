"""Packages: Bifrost code, and the C libraries it binds, that other projects use.

A package is a project whose ``config.yaml`` says ``package.build: library``
(it has no ``entry``: no ``main`` to build). ``bfc package`` makes it a ``.bifpkg``
file, a gzipped tarball named like a Python wheel::

    http_server-0.1.0-x86_64-unknown-linux-gnu.bifpkg     # with C libraries for that target
    greeting-0.2.0-any.bifpkg                             # Bifrost only: any target

holding its ``.bif`` sources (``src/``), its ``config.yaml`` (its externs and
libraries), and the static libraries it links (``lib/``), built for that target.
Bifrost code ships as source, since the compiler needs its signatures,
ownership and what pauses; C code ships compiled.

A project lists the packages it uses in its ``config.yaml``::

    index: https://example.org/bifrost/    # or a folder
    packages:
      http_server: 0.1.0                   # from the index: this target's, else `any`
      greeting:
        path: ../greeting                  # a project folder (while developing), or a .bifpkg

Packages from an index or a ``.bifpkg`` are unpacked into ``build/pkg/<name>/``.
Their externs and libraries join the project's, and ``import("a.b:module")``
looks in their sources after the project's own.
"""

import hashlib
import platform
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from bifrost.configs.schema import Config, _PackageSource, source_root

ARCHIVE = ".bifpkg"
ANY = "any"  # the target of a package with no C libraries
_MARKER = ".bifpkg-source"  # in an unpacked package: the archive it came from


class PackageError(Exception):
    """A package cannot be built, found or used: the message says why."""


@dataclass(frozen=True)
class Package:
    """A package a project uses: where it is, and its configuration."""

    name: str
    root: Path  # the folder holding its config.yaml
    config: Config


def host_triple() -> str:
    """Return this machine's target triple, as packages name it: ``x86_64-unknown-linux-gnu``."""
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    system = platform.system()
    if system == "Darwin":
        return f"{machine}-apple-darwin"
    if system == "Windows":
        return f"{machine}-pc-windows-msvc"
    libc = "musl" if platform.libc_ver()[0] != "glibc" else "gnu"
    return f"{machine}-unknown-linux-{libc}"


def archive_name(name: str, version: str, triple: str) -> str:
    """``http_server-0.1.0-x86_64-unknown-linux-gnu.bifpkg``."""
    return f"{name}-{version}-{triple}{ARCHIVE}"


# -- building ---------------------------------------------------------------------------


def build(project: Path, output: Path | None = None) -> Path:
    """Make the ``.bifpkg`` of the project in folder ``project``; return its path (in ``dist/`` by default).

    Raises:
        PackageError: If the project is not a library, or a library it links has not been built.

    """
    config_path = project / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    package = raw.get("package", {})
    name, version = package.get("name"), package.get("version")
    if not name or not version:
        raise PackageError(f"{config_path} needs package.name and package.version to be packaged")
    if package.get("build") != "library":
        raise PackageError(
            f"{config_path} builds an executable; only a library is packaged (set package.build: library)"
        )
    files: dict[str, Path] = {}
    libraries: list[str] = []
    for library in raw.get("libraries", []):
        if "/" not in str(library):
            libraries.append(library)  # a system library (`m`, `ssl`): linked by name where it is used
            continue
        found = (project / library).resolve()
        if not found.is_file():
            raise PackageError(f"{library} is not built yet ({found}); build it before packaging")
        files[f"lib/{found.name}"] = found
        libraries.append(f"./lib/{found.name}")
    sources = source_root(project)
    for path in sorted(sources.rglob("*.bif")):
        if "build" not in path.relative_to(sources).parts:
            files[f"src/{path.relative_to(sources).as_posix()}"] = path
    raw["libraries"] = libraries
    triple = host_triple() if any(key.startswith("lib/") for key in files) else ANY
    archive = (output or project / "dist") / archive_name(name, version, triple)
    archive.parent.mkdir(parents=True, exist_ok=True)
    staged = archive.with_suffix(".yaml")
    staged.write_text(yaml.safe_dump(raw, sort_keys=False))
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staged, arcname="config.yaml")
            for arcname, path in files.items():
                tar.add(path, arcname=arcname)
    finally:
        staged.unlink()
    return archive


# -- using ----------------------------------------------------------------------------------


def resolve(config: Config, folder: Path, *, install: bool) -> Config:
    """Add the packages ``config`` uses (and theirs) to it: their externs, libraries and sources.

    ``folder`` holds the project's ``config.yaml``. With ``install``, packages from the index
    or a ``.bifpkg`` are fetched and unpacked as needed; without, they must be unpacked already.

    Raises:
        PackageError: If a package cannot be found or used.

    """
    resolver = _Resolver(folder, install=install)
    for name, source in config.packages.items():
        resolver.resolve(name, source, folder, config.index)
    modules = {extern.module for extern in config.externs}
    for package in resolver.found.values():
        for extern in package.config.externs:
            if extern.module in modules:
                raise PackageError(f"package {package.name} declares module {extern.module}, which is already declared")
            modules.add(extern.module)
            config.externs.append(extern)
            config.declared_in[extern.module] = package.root / "config.yaml"
        config.libraries.extend(library for library in package.config.libraries if library not in config.libraries)
        config.sources.append(source_root(package.root))
    return config


class _Resolver:
    """Finds the packages a project uses, and theirs, unpacking them into the project's ``build/pkg/``."""

    def __init__(self, project: Path, *, install: bool) -> None:
        self.project = project
        self.install = install
        self.found: dict[str, Package] = {}  # in the order C libraries link: a package before what it uses

    def resolve(self, name: str, source: str | _PackageSource, folder: Path, index: str | None) -> None:
        """Find package ``name`` as ``source`` says (relative to ``folder``, whose config names ``index``)."""
        if name in self.found:
            return
        if isinstance(source, str):
            root = self.unpack(_from_index(name, source, folder, index, install=self.install), name)
            wanted = source
        else:
            path = (folder / source.path).resolve()
            if path.is_dir():
                root = path  # a project folder: used as it is
            elif path.is_file():
                root = self.unpack(path, name)
            else:
                raise PackageError(f"package {name}: {source.path} is neither a project folder nor a {ARCHIVE} file")
            wanted = source.version
        from bifrost.configs import ConfigBuilder  # noqa: PLC0415 - it resolves packages in turn

        config = ConfigBuilder(root / "config.yaml", resolve=False).build()
        if config.package.name != name:
            raise PackageError(f"package {name} is named {config.package.name} in its config.yaml")
        if config.package.build != "library":
            raise PackageError(f"package {name} builds an executable, not a library (package.build in its config.yaml)")
        if wanted is not None and config.package.version != wanted:
            raise PackageError(f"package {name} is version {config.package.version}, not {wanted}")
        self.found[name] = Package(name, root, config)
        for inner, inner_source in config.packages.items():  # what it uses
            self.resolve(inner, inner_source, root, config.index)

    def unpack(self, archive: Path, name: str) -> Path:
        """Unpack ``archive`` into ``build/pkg/<name>/`` (unless it is there already); return that folder."""
        target = self.project / "build" / "pkg" / name
        marker = target / _MARKER
        if marker.is_file() and marker.read_text() == archive.name:
            return target
        if not self.install:
            raise PackageError(f"package {name} is not installed; run `bfc build` to unpack {archive.name}")
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target, filter="data")
        marker.write_text(archive.name)
        return target


def _from_index(name: str, version: str, folder: Path, index: str | None, *, install: bool) -> Path:
    """Find ``name`` at ``version`` in the index (this target's, else ``any``); return the local archive."""
    if index is None:
        raise PackageError(f"package {name} {version} is listed by version, but config.yaml names no index")
    wanted = [archive_name(name, version, triple) for triple in (host_triple(), ANY)]
    if not re.match(r"^[a-z]+://", index):
        directory = (folder / index).resolve()
        for file in wanted:
            if (directory / file).is_file():
                return directory / file
        raise PackageError(f"package {name} {version} is not in {directory} (looked for {' or '.join(wanted)})")
    cache = _cache() / name
    for file in wanted:
        if (cache / file).is_file():
            return cache / file
    if not install:
        raise PackageError(f"package {name} {version} is not installed; run `bfc build` to fetch it")
    links = _links(index)
    for file in wanted:
        if file in links:
            return _download(links[file], cache / file)
    raise PackageError(f"package {name} {version} is not in {index} (looked for {' or '.join(wanted)})")


def _download(url: str, target: Path) -> Path:
    """Fetch ``url`` into ``target``, checking it against the ``#sha256=`` the index gives (if any).

    Raises:
        PackageError: If what arrived is not what the index lists.

    """
    address, _, fragment = url.partition("#")
    expected = fragment.removeprefix("sha256=") if fragment.startswith("sha256=") else None
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(address) as response, partial.open("wb") as out:  # noqa: S310 - the index's link
        while chunk := response.read(1 << 16):
            digest.update(chunk)
            out.write(chunk)
    if expected is not None and digest.hexdigest() != expected.lower():
        partial.unlink()
        raise PackageError(f"{target.name} from {address} does not match its index's SHA-256; not using it")
    partial.replace(target)
    return target


def _cache() -> Path:
    """Where packages fetched from an index are kept: ``~/.cache/bifrost/packages``."""
    return Path.home() / ".cache" / "bifrost" / "packages"


class _Links(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__()
        self.base = base
        self.found: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href") if tag == "a" else None
        if href:
            url = urllib.parse.urljoin(self.base, href)  # with its #sha256=, if any
            self.found[url.split("#")[0].rsplit("/", 1)[-1]] = url


def _links(index: str) -> dict[str, str]:
    """Read an index page: each linked file's name -> its URL (with the ``#sha256=`` of a flat index)."""
    with urllib.request.urlopen(index) as response:  # noqa: S310 - the index the project names
        page = response.read().decode()
    parser = _Links(index if index.endswith("/") else index + "/")
    parser.feed(page)
    return parser.found


__all__ = ["ANY", "ARCHIVE", "Package", "PackageError", "archive_name", "build", "host_triple", "resolve"]
