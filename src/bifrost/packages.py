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

    index:
      mine: https://example.org/bifrost/   # named indexes: URLs, or folders
    packages:
      http_server: 0.1.0                   # from the default index: this target's, else `any`
      tools:
        index: mine                        # from a named index
        version: 0.3.0
      greeting:
        path: ../greeting                  # a project folder (while developing), or a .bifpkg

A package listed only by version comes from the index named ``default``, or else
``DEFAULT_INDEX``, where the packages in ``extras/`` are published.

``bfc add`` writes these entries (see ``add``). Packages from an index or a
``.bifpkg`` are unpacked into ``build/pkg/<name>/``.
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
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from bifrost.configs.schema import DEFAULT_INDEX, Config, _PackageSource, source_root

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

    def resolve(self, name: str, source: str | _PackageSource, folder: Path, indexes: dict[str, str]) -> None:
        """Find package ``name`` as ``source`` says (relative to ``folder``, whose config names ``indexes``)."""
        if name in self.found:
            return
        if isinstance(source, str) or source.index is not None:
            wanted = source if isinstance(source, str) else source.version
            assert wanted is not None  # an index source has its version (see _PackageSource)
            index = index_location(name, source, indexes)
            root = self.unpack(_from_index(name, wanted, folder, index, install=self.install), name)
        else:
            assert source.path is not None
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


def index_location(name: str, source: str | _PackageSource, indexes: dict[str, str]) -> str:
    """Return where the index package ``name`` comes from is: ``default``'s (or ``DEFAULT_INDEX``), or the one it names.

    Raises:
        PackageError: If it names an index config.yaml does not.

    """
    if isinstance(source, str) or source.index is None:
        return indexes.get("default", DEFAULT_INDEX)
    if source.index not in indexes:
        raise PackageError(f"package {name} comes from index {source.index}, which config.yaml's index: does not name")
    return indexes[source.index]


def _from_index(name: str, version: str, folder: Path, index: str, *, install: bool) -> Path:
    """Find ``name`` at ``version`` in the index (this target's, else ``any``); return the local archive."""
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


# -- adding ------------------------------------------------------------------------------


def versions(name: str, index: str, folder: Path) -> list[str]:
    """Return the versions of ``name`` in ``index`` (relative to ``folder``) for this target or ``any``, oldest first.

    Raises:
        PackageError: If the index cannot be read.

    """
    if re.match(r"^[a-z]+://", index):
        try:
            files = list(_links(index))
        except OSError as error:
            raise PackageError(f"cannot read the index {index}: {error}") from None
    else:
        directory = (folder / index).resolve()
        if not directory.is_dir():
            raise PackageError(f"the index {directory} is not a folder")
        files = [path.name for path in directory.iterdir()]
    found = set()
    for file in files:
        for triple in (host_triple(), ANY):
            suffix = f"-{triple}{ARCHIVE}"
            if file.startswith(f"{name}-") and file.endswith(suffix):
                found.add(file.removeprefix(f"{name}-").removesuffix(suffix))
    return sorted(found, key=_version_key)


def _version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    """Order versions by their numbers: ``0.10.0`` after ``0.9.0``."""
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"[.+-]", version))


def package_name(path: Path) -> str:
    """Return the ``package.name`` of the project folder or ``.bifpkg`` at ``path``.

    Raises:
        PackageError: If ``path`` is neither, or names no package.

    """
    if path.is_dir():
        text = (path / "config.yaml").read_text() if (path / "config.yaml").is_file() else None
    elif path.is_file():
        try:
            with tarfile.open(path, "r:gz") as tar:
                member = tar.extractfile("config.yaml")
                text = member.read().decode() if member is not None else None
        except (tarfile.TarError, KeyError):
            text = None
    else:
        raise PackageError(f"{path} is neither a project folder nor a {ARCHIVE} file")
    name = ((yaml.safe_load(text or "") or {}).get("package") or {}).get("name") if text is not None else None
    if not name:
        raise PackageError(f"{path} has no config.yaml naming package.name")
    return str(name)


def add(
    config_path: Path,
    name: str | None,
    *,
    version: str | None = None,
    path: str | None = None,
    index: str | None = None,
) -> tuple[str, str | _PackageSource]:
    """Add a package to the project's ``config.yaml``, and install it as ``bfc build`` would.

    From ``path`` (a project folder or ``.bifpkg``, relative to config.yaml; its name is read from
    it when ``name`` is None), or else from an index at ``version`` (default: the newest there for
    this target): the default index, or ``index``, one config.yaml names (``NAME``), or one to add
    to it (``NAME=URL``, or a folder). Only the lines of the entry (and of the index) change; if
    the package cannot be installed, config.yaml is left as it was. Return the name and what was
    written.

    Raises:
        PackageError: If the package cannot be found or installed.

    """
    folder = config_path.parent
    original = config_path.read_text()
    indexes: dict[str, str] = (yaml.safe_load(original) or {}).get("index") or {}
    if not isinstance(indexes, dict):
        indexes = {}  # the old `index: url`: the config's own check reports it
    text = original
    source: str | _PackageSource
    if path is not None:
        found = package_name((folder / path).resolve())
        if name is not None and name != found:
            raise PackageError(f"{path} is package {found}, not {name}")
        name, source = found, _PackageSource(path=path, version=version)
    elif name is None:
        raise PackageError("give a package name, or --path")
    else:
        if index is not None:
            text = _choose_index(text, index, indexes)
        source = _newest(name, version, index.partition("=")[0] if index is not None else None, indexes, folder)
    config_path.write_text(_set_entry(text, "packages", name, lambda indent: _entry(name, source, indent)))
    from bifrost.configs import ConfigBuilder  # noqa: PLC0415 - it resolves packages in turn

    try:
        ConfigBuilder(config_path, install=True).build()
    except (PackageError, ValueError) as error:  # ValueError: pydantic's, for a config that does not validate
        config_path.write_text(original)
        raise PackageError(str(error)) from None
    return name, source


def _choose_index(text: str, index: str, indexes: dict[str, str]) -> str:
    """Use ``index``: one config.yaml names (``NAME``), or one to add (``NAME=URL``); return the new ``text``.

    Raises:
        PackageError: If config.yaml names no index ``NAME``.

    """
    chosen, _, location = index.partition("=")
    if location:
        indexes[chosen] = location
        return _set_entry(text, "index", chosen, lambda indent: [f"{indent}{chosen}: {_scalar(location)}"])
    if chosen not in indexes:
        raise PackageError(f"config.yaml names no index {chosen}; add it with --index {chosen}=URL")
    return text


def _newest(
    name: str, version: str | None, chosen: str | None, indexes: dict[str, str], folder: Path
) -> str | _PackageSource:
    """Return the entry of ``name`` from index ``chosen`` (or the default) at ``version``, or else its newest.

    Raises:
        PackageError: If the index does not have it.

    """
    named = chosen is not None and chosen != "default"
    where = index_location(name, _PackageSource(index=chosen, version="") if named else "", indexes)
    available = versions(name, where, folder)
    if not available:
        raise PackageError(f"package {name} is not in {where} for {host_triple()} or {ANY}")
    if version is not None and version not in available:
        raise PackageError(f"package {name} {version} is not in {where} (it has {', '.join(available)})")
    newest = version or available[-1]
    return _PackageSource(index=chosen, version=newest) if named else newest


def _scalar(value: str) -> str:
    """Write ``value`` as YAML reads it back as that string: ``0.1`` is quoted, ``0.1.0`` is not."""
    return yaml.safe_dump(value, default_flow_style=True).removesuffix("\n").removesuffix("\n...")


def _entry(name: str, source: str | _PackageSource, indent: str) -> list[str]:
    if isinstance(source, str):
        return [f"{indent}{name}: {_scalar(source)}"]
    lines = [f"{indent}{name}:"]
    lines += [f"{indent * 2}{key}: {_scalar(value)}" for key, value in source.model_dump(exclude_none=True).items()]
    return lines


def _top_level(lines: list[str], key: str) -> int | None:
    """Return the line of top-level ``key:``, if any."""
    pattern = re.compile(rf"^{re.escape(key)}:(\s|$)")
    return next((number for number, line in enumerate(lines) if pattern.match(line)), None)


def _block_end(lines: list[str], start: int) -> int:
    """Return the line after the block under ``lines[start]``, before the blank lines and comments ending it."""
    end = start + 1
    for number in range(start + 1, len(lines)):
        line = lines[number]
        if line.strip() and not line.startswith((" ", "\t")):
            break
        if line.strip() and not line.lstrip().startswith("#"):
            end = number + 1
    return end


def _set_entry(text: str, key: str, name: str, entry: Callable[[str], list[str]]) -> str:
    """Set ``<key>.<name>`` in config.yaml's ``text`` to the lines ``entry(indent)``, changing no other line.

    Raises:
        PackageError: If ``key:`` is written in flow style (``{a: 1}``).

    """
    lines = text.splitlines()
    start = _top_level(lines, key)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines += [f"{key}:", *entry("  ")]
        return "\n".join(lines) + "\n"
    rest = lines[start].split(":", 1)[1].split("#")[0].strip()
    if rest == "{}":
        lines[start] = f"{key}:"
    elif rest:
        raise PackageError(f"{key}: in config.yaml is written on one line; write it as a block to add to it")
    end = _block_end(lines, start)
    entries = [
        number
        for number in range(start + 1, end)
        if lines[number].strip() and not lines[number].lstrip().startswith("#")
    ]
    indent = " " * (len(lines[entries[0]]) - len(lines[entries[0]].lstrip())) if entries else "  "
    pattern = re.compile(rf"^{re.escape(indent)}{re.escape(name)}:(\s|$)")
    existing = next((number for number in entries if pattern.match(lines[number])), None)
    if existing is None:
        lines[end:end] = entry(indent)
    else:
        after = existing + 1
        while after < end and (not lines[after].strip() or lines[after].startswith(indent + " ")):
            after += 1
        while after > existing + 1 and not lines[after - 1].strip():  # keep blank lines after the entry
            after -= 1
        lines[existing:after] = entry(indent)
    return "\n".join(lines) + "\n"


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


__all__ = [
    "ANY",
    "ARCHIVE",
    "DEFAULT_INDEX",
    "Package",
    "PackageError",
    "add",
    "archive_name",
    "build",
    "host_triple",
    "index_location",
    "package_name",
    "resolve",
    "versions",
]
