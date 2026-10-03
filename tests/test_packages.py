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
    # by version in an index folder it names, and unpacks it into build/pkg/greeting.
    archive = packages.build(_project(tmp_path / "greeting", "greeting"), tmp_path / "index")
    assert archive.name == "greeting-0.2.0-any.bifpkg"
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ["config.yaml", "src/greeting/text.bif"]
    extra = "index:\n  local: ../index\npackages:\n  greeting:\n    index: local\n    version: 0.2.0\n"
    app = _project(tmp_path / "app", "app", extra, source=APP)
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
        app = _project(tmp_path / "app", "app", f"index:\n  default: {url}\npackages:\n  greeting: 0.2.0\n", source=APP)
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
            "packages:\n  greeting:\n    index: mine\n    version: 0.2.0\n",
            "package greeting comes from index mine, which config.yaml's index: does not name",
        ),
        ("index:\n  default: ../index\npackages:\n  greeting: 9.9.9\n", "package greeting 9.9.9 is not in"),
        (
            "index:\n  mine: ../index\npackages:\n  greeting:\n    index: mine\n",
            "a package from index mine needs its version:",
        ),
        ("packages:\n  greeting:\n    version: 0.2.0\n", "give one of path: and index:"),
        ("index: ../index\n", "index: names its indexes"),
        ("packages:\n  greeting:\n    path: ../greeting\n    version: 1.0.0\n", "package greeting is version 0.2.0"),
        ("packages:\n  other:\n    path: ../greeting\n", "package other is named greeting in its config.yaml"),
        ("packages:\n  greeting:\n    path: ../nowhere\n", "neither a project folder nor a .bifpkg file"),
    ],
)
def test_errors(tmp_path: Path, extra: str, message: str) -> None:
    _project(tmp_path / "greeting", "greeting")
    (tmp_path / "index").mkdir()
    app = _project(tmp_path / "app", "app", extra, source=APP)
    with pytest.raises((PackageError, ValueError), match=re.escape(message)):
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


def _add(app: Path, *arguments: str) -> str:
    """Run ``bfc add`` in ``app``; return its output, on one line."""
    result = CliRunner().invoke(cli, ["add", *arguments, "--config", str(app / "config.yaml")])
    output = " ".join(result.output.split())
    assert result.exit_code == 0, output
    return output


def test_add_from_an_index(tmp_path: Path) -> None:
    # The newest version for this target (by number: 0.10.0 after 0.9.0), installed as `bfc build` would.
    greeting = _project(tmp_path / "greeting", "greeting")
    for version in ("0.9.0", "0.10.0"):
        config = greeting / "config.yaml"
        config.write_text(re.sub(r"version: \S+", f"version: {version}", config.read_text()))
        packages.build(greeting, tmp_path / "index")
    app = _project(tmp_path / "app", "app", source=APP)
    before = (app / "config.yaml").read_text()
    assert "added greeting 0.10.0 from index local" in _add(app, "greeting", "--index", "local=../index")
    assert (app / "config.yaml").read_text() == before + (
        "\nindex:\n  local: ../index\n\npackages:\n  greeting:\n    index: local\n    version: 0.10.0\n"
    )
    assert (app / "build" / "pkg" / "greeting" / "src" / "greeting" / "text.bif").is_file()
    assert _run(app, install=False) == "Hello, ada!\n"

    _add(app, "greeting@0.9.0", "--index", "local")
    assert (app / "config.yaml").read_text().endswith("packages:\n  greeting:\n    index: local\n    version: 0.9.0\n")
    result = CliRunner().invoke(
        cli, ["add", "greeting@1.0.0", "--index", "local", "--config", str(app / "config.yaml")]
    )
    assert result.exit_code == 1
    assert "package greeting 1.0.0 is not in ../index (it has 0.9.0, 0.10.0)" in " ".join(result.output.split())
    assert (app / "config.yaml").read_text().endswith("packages:\n  greeting:\n    index: local\n    version: 0.9.0\n")


def test_add_keeps_the_rest_of_the_config(tmp_path: Path) -> None:
    # Only the entry's lines change: comments, other entries and keys stay as they were.
    _project(tmp_path / "greeting", "greeting")
    _project(tmp_path / "other", "other")
    app = _project(tmp_path / "app", "app", source=APP)
    config = app / "config.yaml"
    head = config.read_text()
    config.write_text(
        head + "# What it uses:\npackages:\n    other:\n        path: ../other   # mine\n\n# The end\nlibraries: [m]\n"
    )
    assert "added greeting from ../greeting" in _add(app, "--path", "../greeting")
    assert config.read_text() == head + (
        "# What it uses:\npackages:\n    other:\n        path: ../other   # mine\n"
        "    greeting:\n        path: ../greeting\n\n# The end\nlibraries: [m]\n"
    )
    _add(app, "other", "--path", "../greeting/../other")  # replaces its entry
    assert config.read_text() == head + (
        "# What it uses:\npackages:\n    other:\n        path: ../greeting/../other\n"
        "    greeting:\n        path: ../greeting\n\n# The end\nlibraries: [m]\n"
    )
    assert _run(app) == "Hello, ada!\n"


def test_add_uses_the_published_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Without --index, `bfc add` uses the published index, which config.yaml need not name.
    packages.build(_project(tmp_path / "greeting", "greeting"), tmp_path / "index")
    monkeypatch.setattr(packages, "DEFAULT_INDEX", str(tmp_path / "index"))
    app = _project(tmp_path / "app", "app", "packages: {}\n", source=APP)
    _add(app, "greeting")
    assert (app / "config.yaml").read_text().endswith("\npackages:\n  greeting: 0.2.0\n")
    assert "index" not in (app / "config.yaml").read_text()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["nothing", "--index", "local=../index"], "package nothing is not in ../index for"),
        (["greeting", "--index", "mine"], "config.yaml names no index mine; add it with --index mine=URL"),
        (["--path", "../nowhere"], "nowhere is neither a project folder nor a .bifpkg file"),
        (["other", "--path", "../greeting"], "../greeting is package greeting, not other"),
        (["--path", "../app"], "package app builds an executable, not a library"),
    ],
)
def test_add_errors(tmp_path: Path, arguments: list[str], message: str) -> None:
    _project(tmp_path / "greeting", "greeting")
    (tmp_path / "index").mkdir()
    app = _project(tmp_path / "app", "app", source=APP)
    before = (app / "config.yaml").read_text()
    result = CliRunner().invoke(cli, ["add", *arguments, "--config", str(app / "config.yaml")])
    assert result.exit_code == 1
    assert message in " ".join(result.output.split())
    assert (app / "config.yaml").read_text() == before


def test_the_editor_does_not_install(tmp_path: Path) -> None:
    # Unpacking is `bfc build`'s; until then the editor says so, and still checks the rest.
    packages.build(_project(tmp_path / "greeting", "greeting"), tmp_path / "index")
    app = _project(tmp_path / "app", "app", "index:\n  default: ../index\npackages:\n  greeting: 0.2.0\n", source=APP)
    with pytest.raises(
        PackageError, match=re.escape("not installed; run `bfc build` to unpack greeting-0.2.0-any.bifpkg")
    ):
        ConfigBuilder(app / "config.yaml").build()
    main = app / "src" / "app" / "main.bif"
    messages = [d.message for d in Document.open(main, main.read_text()).diagnostics()]
    assert "package greeting is not installed; run `bfc build` to unpack greeting-0.2.0-any.bifpkg" in messages
    _run(app)  # installs it
    assert [d.message for d in Document.open(main, main.read_text()).diagnostics()] == []
