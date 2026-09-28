from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bifrost.__main__ import cli
from bifrost.configs import Config
from bifrost.configs.traverse import TraverseError, dump, merge, traverse
from bifrost.project import Project

SHAPES = """\
typedef enum { RED, GREEN } Hue;
typedef struct Point { int x; int y; } Point;
typedef struct { float w, h; } Size;
typedef struct Named { char name[8]; } Named;
typedef Point Position;
struct Opaque;

void draw(Point at, Size size, Hue hue);
Position origin(void);
const char *title(unsigned char *buffer, long long count, double scale, _Bool flag);
void log_line(const char *format, ...);
void rename_it(Named named);
void take(struct Opaque *opaque);
static inline int helper(int a) { return a; }
void keyword(int lambda, int);
"""

GEOMETRY = """\
typedef struct Point { int x; int y; } Point;
Point midpoint(Point a, Point b);
"""

CPP = """\
namespace shapes { int area(int w, int h); }
extern "C" int c_area(int w, int h);
"""


@pytest.fixture
def headers(tmp_path: Path) -> Path:
    (tmp_path / "shapes.h").write_text(SHAPES)
    (tmp_path / "geometry.h").write_text(GEOMETRY)
    (tmp_path / "widgets.hpp").write_text(CPP)
    return tmp_path


def _by_name(declarations: list[dict]) -> dict[str, dict]:
    return {d["name"]: d for d in declarations}


def test_maps_structs_and_functions(headers: Path) -> None:
    (shapes,) = traverse([headers / "shapes.h"])
    declarations = _by_name(shapes.declarations())
    assert shapes.module == "shapes"
    assert declarations["Point"]["fields"] == {"x": "i32", "y": "i32"}
    assert declarations["Size"]["fields"] == {"w": "f32", "h": "f32"}  # an anonymous struct, named by its typedef
    assert declarations["draw"]["parameters"] == {"at": "shapes_Point", "size": "shapes_Size", "hue": "u32"}
    assert declarations["origin"] == {"name": "origin", "type": "function", "parameters": {}, "return": "shapes_Point"}
    assert declarations["title"]["parameters"] == {"buffer": "ptr", "count": "i64", "scale": "f64", "flag": "bool"}
    assert declarations["title"]["return"] == "cstr"
    assert declarations["take"]["parameters"] == {"opaque": "ptr"}
    assert declarations["keyword"]["parameters"] == {"lambda_0": "i32", "arg1": "i32"}
    # Structs come before the functions that take them.
    names = [d["name"] for d in shapes.declarations()]
    assert names.index("Point") < names.index("draw")


def test_skips_what_cannot_be_bound(headers: Path) -> None:
    (shapes,) = traverse([headers / "shapes.h"])
    reasons = {s.name: s.reason for s in shapes.skipped}
    assert reasons["Named"] == "member name: char[8] is an array member"
    assert reasons["log_line"] == "it is variadic"
    assert "static" in reasons["helper"]
    assert "Named was skipped" in reasons["rename_it"]


def test_a_struct_defined_twice_belongs_to_the_first_header(headers: Path) -> None:
    shapes, geometry = traverse([headers / "shapes.h", headers / "geometry.h"])
    assert "Point" in _by_name(shapes.declarations())
    assert "Point" not in _by_name(geometry.declarations())
    assert _by_name(geometry.declarations())["midpoint"]["return"] == "shapes_Point"


def test_cpp_binds_only_extern_c(headers: Path) -> None:
    (widgets,) = traverse([headers / "widgets.hpp"])
    assert [d["name"] for d in widgets.functions] == ["c_area"]
    assert "C++ linkage" in {s.name: s.reason for s in widgets.skipped}["area"]


def test_directories_are_searched(headers: Path) -> None:
    assert [m.module for m in traverse([headers])] == ["geometry", "shapes", "widgets"]


def test_reports_parse_errors(tmp_path: Path) -> None:
    (tmp_path / "broken.h").write_text('#include "missing.h"\n')
    with pytest.raises(TraverseError, match=r"missing\.h"):
        traverse([tmp_path / "broken.h"])


def test_headers_that_need_another_are_parsed_after_it(tmp_path: Path) -> None:
    (tmp_path / "base.h").write_text("typedef struct Vec { float x; } Vec;\n")
    (tmp_path / "extra.h").write_text("Vec scale(Vec v, float by);\n")  # uses Vec without including base.h
    base, extra = traverse([tmp_path / "base.h", tmp_path / "extra.h"])
    assert extra.context == [(tmp_path / "base.h").resolve()]
    assert extra.functions == [
        {"name": "scale", "type": "function", "parameters": {"v": "base_Vec", "by": "f32"}, "return": "base_Vec"}
    ]
    assert base.structs[0]["name"] == "Vec"


def test_a_header_that_never_parses_is_left_out(tmp_path: Path) -> None:
    (tmp_path / "good.h").write_text("int answer(void);\n")
    (tmp_path / "bad.h").write_text("Unknown thing(void);\n")
    good, bad = traverse([tmp_path / "good.h", tmp_path / "bad.h"])
    assert good.error is None
    assert bad.error is not None
    assert "unknown type name 'Unknown'" in bad.error
    config = {"externs": [{"module": "bad", "description": "hand-written", "declarations": []}]}
    assert [e["module"] for e in merge(config, [good, bad])["externs"]] == ["bad", "good"]


def test_merge_replaces_generated_modules_and_loads(headers: Path) -> None:
    config = {
        "package": {"name": "t", "version": "0", "description": ""},
        "flags": {"optimization": 0, "linker": "clang"},
        "libraries": [],
        "externs": [
            {"module": "other", "description": "kept", "declarations": []},
            {"module": "shapes", "description": "Mine", "declarations": []},
        ],
    }
    merged = merge(config, traverse([headers / "shapes.h"]))
    assert [e["module"] for e in merged["externs"]] == ["other", "shapes"]
    assert merged["externs"][1]["description"] == "Mine"
    Project(Config(**yaml.safe_load(dump(merged))))  # every generated extern declares


def test_command_updates_config(headers: Path, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "package:\n  name: t\n  version: '0'\n  description: ''\n\nflags:\n  optimization: 0\n  linker: clang\n\n"
        "libraries: []\n\nexterns: []\n"
    )
    runner = CliRunner()
    arguments = ["config", "traverse", "-c", str(config), "-i", f"{headers / 'shapes.h'},{headers / 'geometry.h'}"]
    dry = runner.invoke(cli, [*arguments, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "externs: []" in config.read_text()
    result = runner.invoke(cli, [*arguments, "--verbose"])
    assert result.exit_code == 0, result.output
    assert "skipped log_line: it is variadic" in result.output
    modules = [e["module"] for e in yaml.safe_load(config.read_text())["externs"]]
    assert modules == ["shapes", "geometry"]
