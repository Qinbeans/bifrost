import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bifrost.__main__ import cli
from bifrost.configs import ConfigBuilder


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    cwd = Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


def test_init_creates_a_project_that_builds_and_runs(workspace: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "shop", "-d", "A shop API"])
    assert result.exit_code == 0, result.output
    project = workspace / "shop"
    assert sorted(p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file()) == [
        ".gitignore",
        "README.md",
        "config.yaml",
        "src/shop/main.bif",
    ]
    config = ConfigBuilder(project / "config.yaml").build()
    assert (config.package.name, config.package.description) == ("shop", "A shop API")
    assert config.package.entry == "src/shop/main.bif"
    assert "build/" in (project / ".gitignore").read_text()

    # Modules import each other from src/, as in Python: shop.utils is src/shop/utils.bif.
    (project / "src" / "shop" / "utils.bif").write_text(
        'let fmt = import("std:fmt")\nlet mem = import("std:mem")\n\nmodule text = {\n'
        "    let price = [fmt.format] (dollars: i32, cents: i32) => mem.Unique[str]\n"
        '        fmt.format("$%d.%02d", dollars, cents)\n'
        "}\n\nexport(text)\n"
    )
    (project / "src" / "shop" / "main.bif").write_text(
        'let stdio = import("std:stdio")\nlet text = import("shop.utils:text")\n\n'
        "let main = [text.price, stdio.puts] () => null {\n"
        "    let shown = text.price(19, 99)\n    stdio.puts(shown)\n}\n"
    )
    os.chdir(project)
    built = runner.invoke(cli, ["build"])  # the entry in config.yaml, into build/
    assert built.exit_code == 0, built.output
    output = subprocess.run([project / "build" / "shop"], capture_output=True, text=True, check=True)  # noqa: S603
    assert output.stdout == "$19.99\n"


@pytest.mark.parametrize(("name", "message"), [("My-Api", "not a project name"), ("_hidden", "not a project name")])
def test_init_rejects_bad_names(workspace: Path, name: str, message: str) -> None:
    result = CliRunner().invoke(cli, ["init", name])
    assert result.exit_code == 1
    assert message in result.output
    assert not (workspace / name).exists()


def test_init_does_not_overwrite(workspace: Path) -> None:
    (workspace / "shop").mkdir()
    result = CliRunner().invoke(cli, ["init", "shop"])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_build_without_entry_says_how(workspace: Path) -> None:
    (workspace / "config.yaml").write_text(
        "package:\n  name: x\n  version: '0'\n  description: ''\n\nflags:\n  optimization: 0\n  linker: clang\n"
    )
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 1
    assert "set package.entry" in result.output
