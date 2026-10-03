"""The examples compile against their config.yaml (linking needs their C libraries built, so it is not done here)."""

from pathlib import Path

import pytest

from bifrost import packages
from bifrost.configs import Config, ConfigBuilder, source_root
from bifrost.configs.schema import _PackageSource
from bifrost.lowering import lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

EXAMPLES = Path(__file__).parents[1] / "examples"
EXTRAS = Path(__file__).parents[1] / "extras"


def _config(project: Path) -> Config:
    """Return the example's config, taking its packages from an index from extras/ instead (where CI builds them)."""
    config = ConfigBuilder(project / "config.yaml", resolve=False).build()
    local = {
        ConfigBuilder(path, resolve=False).build().package.name: path.parent for path in EXTRAS.glob("*/config.yaml")
    }
    config.packages = {
        name: _PackageSource(path=str(local[name])) if isinstance(source, str) or source.index is not None else source
        for name, source in config.packages.items()
    }
    return packages.resolve(config, project, install=False)


@pytest.mark.parametrize("project", [EXAMPLES / "http", EXAMPLES / "raylib", EXAMPLES / "async"])
def test_example_compiles(project: Path) -> None:
    config = _config(project)
    assert config.package.entry is not None
    compiled = Project(config)
    unit = lower_file(compiled, project / config.package.entry, root=source_root(project))
    with unit.errors():
        _ = compiled.program.mlir


def test_library_paths_are_relative_to_the_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "package: {name: a, version: '0', description: ''}\n"
        "flags: {optimization: 0, linker: clang}\n"
        "libraries: [./build/lib/liba.a, m]\n"
    )
    config = ConfigBuilder(tmp_path / "config.yaml").build()
    assert config.libraries == [tmp_path / "build" / "lib" / "liba.a", "m"]


def test_fields_of_an_object_from_another_file() -> None:
    # api/hits.bif locks a mem.Weak[state.AppState]; AppState is declared in state.bif.
    path = EXAMPLES / "http" / "src" / "http" / "api" / "hits.bif"
    source = path.read_text()
    document = Document.open(path, source)
    row = source.splitlines().index("        s.hits = s.hits + 1")
    assert document.hover((row, 10)) == "```bifrost\nlet hits: i64\n// a member of state.AppState (state.bif)\n```"
    assert [c.label for c in document.completions((row, 10))] == ["hits", "notes"]
    location = document.definition((row, 10))
    assert location is not None
    assert location.path.name == "state.bif"
