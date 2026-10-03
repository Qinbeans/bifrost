"""Packages: a project's Bifrost code (and C libraries) that others use, by path, from a .bifpkg or an index."""

import functools
import hashlib
import importlib.util
import re
import subprocess
import tarfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bifrost import packages
from bifrost.__main__ import cli
from bifrost.configs import ConfigBuilder, source_root
from bifrost.lowering import lower_file
from bifrost.packages import PackageError
from bifrost.project import Project
from bifrost.server.analysis import Document

GREETING = """\
let fmt = import("std:fmt")
let mem = import("std:mem")

module text = {
    let greet = [] (name: str) => mem.Unique[str] "Hello, {name}!"
}

export(text)
"""

APP = """\
let io = import("std:stdio")
let text = import("greeting.text:text")

let main = [io.puts, text.greet] () => null {
    io.puts(text.greet("ada"))
}
"""


def _project(folder: Path, name: str, extra: str = "", *, source: str | None = None) -> Path:
    """Write a project: its config.yaml (with ``extra``), and ``src/<name>/main.bif`` or a library module."""
    (folder / "src" / name).mkdir(parents=True)
    entry = f"  entry: src/{name}/main.bif\n" if source is not None else "  build: library\n"
    (folder / "config.yaml").write_text(
        f"package:\n  name: {name}\n  version: 0.2.0\n  description: ''\n{entry}"
        f"flags: {{optimization: 0, linker: clang}}\n{extra}"
    )
    if source is not None:
        (folder / "src" / name / "main.bif").write_text(source)
    else:
        (folder / "src" / name / "text.bif").write_text(GREETING)
    return folder


def _run(app: Path, *, install: bool = True) -> str:
    config = ConfigBuilder(app / "config.yaml", install=install).build()
    config.path = app / "build"
    config.path.mkdir(parents=True, exist_ok=True)
    project = Project(config)
    assert config.package.entry is not None
    unit = lower_file(project, app / config.package.entry, root=source_root(app))
    with unit.errors():
        executable = project.build()
    return subprocess.run([executable], capture_output=True, text=True, check=True).stdout  # noqa: S603


def test_a_package_by_path(tmp_path: Path) -> None:
    # A project folder, used as it is: its modules import like the app's own.
    _project(tmp_path / "greeting", "greeting")
    app = _project(tmp_path / "app", "app", "packages:\n  greeting:\n    path: ../greeting\n", source=APP)
    assert _run(app) == "Hello, ada!\n"


def test_a_package_from_an_index(tmp_path: Path) -> None:
    # `bfc package` makes greeting-0.2.0-any.bifpkg (no C libraries: any target); the app finds it
    # by version in an index folder and unpacks it into build/pkg/greeting.
    archive = packages.build(_project(tmp_path / "greeting", "greeting"), tmp_path / "index")
    assert archive.name == "greeting-0.2.0-any.bifpkg"
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ["config.yaml", "src/greeting/text.bif"]
    app = _project(tmp_path / "app", "app", "index: ../index\npackages:\n  greeting: 0.2.0\n", source=APP)
    assert _run(app) == "Hello, ada!\n"
    assert (app / "build" / "pkg" / "greeting" / "src" / "greeting" / "text.bif").is_file()


def test_packaging_copies_built_libraries(tmp_path: Path) -> None:
    library = _project(tmp_path / "lib", "lib", "libraries: [./build/liblib.a, m]\n")
    with pytest.raises(PackageError, match=re.escape("liblib.a is not built yet")):
        packages.build(library)
    (library / "build").mkdir()
    (library / "build" / "liblib.a").write_bytes(b"!<arch>\n")
    archive = packages.build(library)
    assert archive.name == f"lib-0.2.0-{packages.host_triple()}.bifpkg"
    with tarfile.open(archive) as tar:
        assert "lib/liblib.a" in tar.getnames()
        config = tar.extractfile("config.yaml")
        assert config is not None
        assert "- ./lib/liblib.a\n- m\n" in config.read().decode()


def _package_index() -> object:
    """``tools/package_index.py``: what CI publishes the index with."""
    path = Path(__file__).resolve().parent.parent / "tools" / "package_index.py"
    spec = importlib.util.spec_from_file_location("package_index", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the method's own name
        pass


@contextmanager
def _serve(folder: Path) -> Iterator[str]:
    """Serve ``folder`` over HTTP; yield its URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.parametrize("tampered", [False, True])
def test_a_package_from_a_url_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tampered: bool) -> None:
    # The index CI publishes: a page of links to the releases' archives, each with its SHA-256,
    # which `bfc build` checks before using what it downloaded.
    archive = packages.build(_project(tmp_path / "greeting", "greeting"), tmp_path / "site" / "download")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if tampered:
        archive.write_bytes(archive.read_bytes() + b"\0")
    monkeypatch.setattr(packages, "_cache", lambda: tmp_path / "cache")
    index: Any = _package_index()
    with _serve(tmp_path / "site") as url:
        asset = {"name": archive.name, "browser_download_url": f"{url}download/{archive.name}"}
        releases = [{"draft": False, "assets": [asset | {"digest": f"sha256:{digest}"}]}]
        (tmp_path / "site" / "index.html").write_text(index.render(index.archive_links(releases)))
        app = _project(tmp_path / "app", "app", f"index: {url}\npackages:\n  greeting: 0.2.0\n", source=APP)
        if not tampered:
            assert _run(app) == "Hello, ada!\n"
            return
        with pytest.raises(PackageError, match=re.escape("does not match its index's SHA-256; not using it")):
            ConfigBuilder(app / "config.yaml", install=True).build()
    assert list((tmp_path / "cache" / "greeting").iterdir()) == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            "packages:\n  greeting: 0.2.0\n",
            "package greeting 0.2.0 is listed by version, but config.yaml names no index",
        ),
        ("index: ../index\npackages:\n  greeting: 9.9.9\n", "package greeting 9.9.9 is not in"),
        ("packages:\n  greeting:\n    path: ../greeting\n    version: 1.0.0\n", "package greeting is version 0.2.0"),
        ("packages:\n  other:\n    path: ../greeting\n", "package other is named greeting in its config.yaml"),
        ("packages:\n  greeting:\n    path: ../nowhere\n", "neither a project folder nor a .bifpkg file"),
    ],
)
def test_errors(tmp_path: Path, extra: str, message: str) -> None:
    _project(tmp_path / "greeting", "greeting")
    (tmp_path / "index").mkdir()
    app = _project(tmp_path / "app", "app", extra, source=APP)
    with pytest.raises(PackageError, match=re.escape(message)):
        ConfigBuilder(app / "config.yaml", install=True).build()


def test_only_a_library_is_a_package(tmp_path: Path) -> None:
    # `package.build: library` makes a project a package; an executable is neither packaged nor used as one.
    app = _project(tmp_path / "app", "app", source=APP)
    with pytest.raises(PackageError, match=re.escape("builds an executable; only a library is packaged")):
        packages.build(app)
    user = _project(tmp_path / "user", "user", "packages:\n  app:\n    path: ../app\n", source=APP)
    with pytest.raises(PackageError, match=re.escape("package app builds an executable, not a library")):
        ConfigBuilder(user / "config.yaml", install=True).build()
    (app / "config.yaml").write_text(
        (app / "config.yaml").read_text().replace("  entry:", "  build: library\n  entry:")
    )
    with pytest.raises(ValueError, match=re.escape("a library has no entry")):
        ConfigBuilder(app / "config.yaml").build()


def test_a_library_builds_no_executable(tmp_path: Path) -> None:
    library = _project(tmp_path / "greeting", "greeting")
    result = CliRunner().invoke(cli, ["build", "--config", str(library / "config.yaml")])
    assert result.exit_code == 1
    assert "is a library (package.build: library), which builds no executable" in " ".join(result.output.split())


def test_the_editor_does_not_install(tmp_path: Path) -> None:
    # Unpacking is `bfc build`'s; until then the editor says so, and still checks the rest.
    packages.build(_project(tmp_path / "greeting", "greeting"), tmp_path / "index")
    app = _project(tmp_path / "app", "app", "index: ../index\npackages:\n  greeting: 0.2.0\n", source=APP)
    with pytest.raises(
        PackageError, match=re.escape("not installed; run `bfc build` to unpack greeting-0.2.0-any.bifpkg")
    ):
        ConfigBuilder(app / "config.yaml").build()
    main = app / "src" / "app" / "main.bif"
    messages = [d.message for d in Document.open(main, main.read_text()).diagnostics()]
    assert "package greeting is not installed; run `bfc build` to unpack greeting-0.2.0-any.bifpkg" in messages
    _run(app)  # installs it
    assert [d.message for d in Document.open(main, main.read_text()).diagnostics()] == []
