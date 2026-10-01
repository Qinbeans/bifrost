"""``bfc version``: print, set or bump ``package.version`` in config.yaml, like ``uv version``."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bifrost.__main__ import cli
from bifrost.versioning import Bump, VersionError, bumped

CONFIG = """\
package:
  name: shop
  # bumped on every release
  version: 0.1.3  # semver
  description: A shop

flags:
  optimization: 2
  linker: clang
"""


def _run(config: Path, *arguments: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli, ["version", *arguments, "--config", str(config)])
    return result.exit_code, result.output


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return path


def test_prints_the_version(config: Path) -> None:
    assert _run(config) == (0, "shop 0.1.3\n")
    assert _run(config, "--short") == (0, "0.1.3\n")


@pytest.mark.parametrize(("part", "expected"), [("major", "1.0.0"), ("minor", "0.2.0"), ("patch", "0.1.4")])
def test_bumps_only_the_version_line(config: Path, part: str, expected: str) -> None:
    assert _run(config, "--bump", part) == (0, f"shop 0.1.3 => {expected}\n")
    assert config.read_text() == CONFIG.replace("version: 0.1.3", f"version: {expected}")


def test_sets_the_version(config: Path) -> None:
    assert _run(config, "2.0.0", "--short") == (0, "2.0.0\n")
    assert "  version: 2.0.0  # semver\n" in config.read_text()


def test_dry_run_writes_nothing(config: Path) -> None:
    assert _run(config, "--bump", "minor", "--dry-run") == (0, "shop 0.1.3 => 0.2.0\n")
    assert config.read_text() == CONFIG


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["1.2"], "1.2 is not a version"),
        (["1.0.0", "--bump", "patch"], "give a version or --bump, not both"),
    ],
)
def test_errors(config: Path, arguments: list[str], message: str) -> None:
    code, output = _run(config, *arguments)
    assert code == 1
    assert message in output
    assert config.read_text() == CONFIG


def test_bumping_needs_major_minor_patch() -> None:
    with pytest.raises(VersionError, match=r"not major\.minor\.patch"):
        bumped("1.0", Bump.PATCH)
