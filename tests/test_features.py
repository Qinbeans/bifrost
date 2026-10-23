"""Features: what a package (or project) brings only when asked for, as Cargo's do."""

import subprocess
from pathlib import Path

import pytest

from bifrost.configs import ConfigBuilder
from bifrost.configs.features import FeatureError
from bifrost.lowering import BifrostError, lower_file
from bifrost.packages import PackageError
from bifrost.project import Project

PACKAGE = """
package:
  name: greet
  version: 0.1.0
  description: greetings
  build: library
flags:
  optimization: 0
  linker: clang
features:
  default: [polite]
  polite: []
  loud:
    features: [polite]
    modules: [greet.loud]
    externs: [shout]
    libraries: [./lib/libshout.a]
externs:
  - module: shout
    description: upper case
    declarations:
      - {name: shout_upper, as: upper, type: function, parameters: {text: cstr}, return: cstr}
"""

SHOUT = r"""
#include <ctype.h>
static char buffer[64];
const char *shout_upper(const char *text)
{
    int i = 0;
    for (; text[i] && i < 63; i++)
        buffer[i] = (char)toupper((unsigned char)text[i]);
    buffer[i] = 0;
    return buffer;
}
"""

LOUD = """
let io = import("std:stdio")
let shout = import("shout")

module loud = {
    let hello = [io.puts, shout.upper] (name: str) => null io.puts(shout.upper("hello, {name}"))
}

export(loud)
"""

PLAIN = """
let io = import("std:stdio")

module plain = {
    let hello = [io.printf] (name: str) => null io.printf("hello, %s\\n", name)
}

export(plain)
"""


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "greet"
    (root / "src" / "greet").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "config.yaml").write_text(PACKAGE)
    (root / "src" / "greet" / "loud.bif").write_text(LOUD)
    (root / "src" / "greet" / "plain.bif").write_text(PLAIN)
    (root / "shout.c").write_text(SHOUT)
    subprocess.run(["clang", "-c", "-fPIC", "-o", root / "shout.o", root / "shout.c"], check=True)  # noqa: S603, S607
    subprocess.run(["ar", "rcs", root / "lib" / "libshout.a", root / "shout.o"], check=True)  # noqa: S603, S607
    return root


def _app(tmp_path: Path, entry: str, main: str) -> Path:
    app = tmp_path / "app"
    (app / "src" / "app").mkdir(parents=True)
    config = f"""
package:
  name: app
  version: 0.1.0
  description: uses greet
  entry: src/app/main.bif
flags:
  optimization: 0
  linker: clang
packages:
  greet:
{entry}
"""
    (app / "config.yaml").write_text(config)
    (app / "src" / "app" / "main.bif").write_text(main)
    return app


def _build(app: Path) -> list[str]:
    config = ConfigBuilder(app / "config.yaml").build()
    config.path = app / "build"
    config.path.mkdir(exist_ok=True)
    project = Project(config)
    unit = lower_file(project, app / "src" / "app" / "main.bif", root=app / "src")
    with unit.errors():
        executable = project.build()
    return subprocess.run([executable], capture_output=True, text=True, check=True).stdout.splitlines()  # noqa: S603


LOUD_MAIN = """
let loud = import("greet.loud:loud")
let plain = import("greet.plain:plain")

let main = [loud.hello, plain.hello] () => null {
    loud.hello("ada")
    plain.hello("bob")
}
"""


def test_a_feature_brings_its_modules_externs_and_libraries(tmp_path: Path) -> None:
    # With `loud`, its file imports, its extern module is declared and its library linked.
    _package(tmp_path)
    app = _app(tmp_path, "    path: ../greet\n    features: [loud]", LOUD_MAIN)
    assert _build(app) == ["HELLO, ADA", "hello, bob"]
    config = ConfigBuilder(app / "config.yaml").build()
    assert any(extern.module == "shout" for extern in config.externs)


def test_without_the_feature_its_file_cannot_be_imported(tmp_path: Path) -> None:
    # Without it, the file is not there to import (the error says which feature brings it), its
    # extern module is not declared, and its library not linked; the defaults are still on.
    _package(tmp_path)
    app = _app(tmp_path, "    path: ../greet", LOUD_MAIN)
    config = ConfigBuilder(app / "config.yaml").build()
    assert not any(extern.module == "shout" for extern in config.externs)
    assert not any("libshout" in str(library) for library in config.libraries)
    with pytest.raises(BifrostError) as error:
        _build(app)
    assert "greet.loud is part of greet's feature loud, which is off; turn it on where greet is listed" in str(
        error.value.msg
    )


def test_a_feature_that_does_not_exist(tmp_path: Path) -> None:
    _package(tmp_path)
    app = _app(tmp_path, "    path: ../greet\n    features: [quiet]", LOUD_MAIN)
    with pytest.raises(PackageError) as error:
        ConfigBuilder(app / "config.yaml").build()
    assert str(error.value) == "package greet has no feature 'quiet' (its features: loud, polite)"


def test_features_turn_on_others_and_unify(tmp_path: Path) -> None:
    # `loud` turns on `polite`; without the defaults, only what is asked for (and what it turns on).
    root = _package(tmp_path)
    from bifrost.configs import features  # noqa: PLC0415

    config = ConfigBuilder(root / "config.yaml", resolve=False).build()
    assert features.enabled(config, {"loud"}, defaults=False, owner="greet") == {"loud", "polite"}
    assert features.enabled(config, set(), defaults=True, owner="greet") == {"polite"}
    assert features.enabled(config, set(), defaults=False, owner="greet") == set()
    with pytest.raises(FeatureError):
        features.enabled(config, {"nope"}, defaults=False, owner="greet")
