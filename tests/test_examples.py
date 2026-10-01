"""The examples compile against their config.yaml (linking needs their C libraries built, so it is not done here)."""

from pathlib import Path

import pytest

from bifrost.configs import ConfigBuilder, source_root
from bifrost.lowering import lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize("project", [EXAMPLES / "http", EXAMPLES / "raylib", EXAMPLES / "async"])
def test_example_compiles(project: Path) -> None:
    config = ConfigBuilder(project / "config.yaml").build()
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
