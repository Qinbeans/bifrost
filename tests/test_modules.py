import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

TEXT = r"""
let stdio = import("std:stdio")

module greeting = {
    /* Greets people */
    let Person = struct {
        let age: i32,
        static let new = [] (age: i32) => Person Person(age: age)
    }
    let excitement = 3
    let greet = [stdio.printf, shout] (name: str) => null {
        stdio.printf("Hello, %s", name)
        shout(excitement)
    }
    let shout = [stdio.printf, this] (times: i32) => null {
        if times > 0 {
            stdio.printf("!")
            shout(times - 1)
        } else {
            stdio.printf("\n")
        }
    }
    let age_next_year = [Person.new] (age: i32) => i32 {
        let person = Person.new(age + 1)
        return person.age
    }
}

module internal = {
    let secret = () => i32 7
}

export(greeting)
"""

MAIN = r"""
let stdio = import("std:stdio")
let greeting = import("utils.text:greeting")

let main = [greeting.greet, greeting.age_next_year, stdio.printf] () => null {
    greeting.greet("Bifrost")
    let p = greeting.Person(age: 30)
    stdio.printf("%d next year, excitement %d\n", greeting.age_next_year(p.age), greeting.excitement)
}
"""


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "text.bif").write_text(TEXT)
    (tmp_path / "main.bif").write_text(MAIN)
    (tmp_path / "config.yaml").write_text(
        "package:\n  name: mods\n  version: '0'\n  description: ''\n\nflags:\n  optimization: 0\n  linker: clang\n\n"
        "libraries: []\n\nexterns: []\n"
    )
    return tmp_path


def _project(root: Path) -> Project:
    return Project(
        Config(
            path=root,
            package={"name": "mods", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _lower(root: Path, source: str) -> Project:
    path = root / "case.bif"
    path.write_text(source)
    project = _project(root)
    lower_file(project, path, root=root)
    return project


def test_modules_build_and_run(project_root: Path) -> None:
    project = _project(project_root)
    lower_file(project, project_root / "main.bif", root=project_root)
    output = subprocess.run([project.build()], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert output == "Hello, Bifrost!!!\n31 next year, excitement 3\n"


def test_relative_imports(project_root: Path) -> None:
    (project_root / "utils" / "rel.bif").write_text(
        'let greeting = import(".text:greeting")\nmodule rel = {\n    let n = () => i32 greeting.excitement\n}\n'
        "export(rel)\n"
    )
    _lower(project_root, 'let r = import("utils.rel:rel")\nlet main = [r.n] () => null {\n    let v = r.n()\n}\n')


@pytest.mark.parametrize(
    ("files", "source", "message"),
    [
        ({}, 'let x = import("utils.text:internal")\n', "text.bif does not export module 'internal'"),
        ({}, 'let x = import("utils.nothing:greeting")\n', "cannot find nothing.bif for 'utils.nothing'"),
        ({}, 'let x = import("std.io:io")\n', "'std.io' is not a file name"),
        (
            {
                "a.bif": 'let b = import("b:bm")\nmodule am = {\n    let one = () => i32 1\n}\nexport(am)\n',
                "b.bif": 'let a = import("a:am")\nmodule bm = {\n    let two = () => i32 2\n}\nexport(bm)\n',
            },
            'let a = import("a:am")\n',
            "import cycle: a.bif -> b.bif -> a.bif",
        ),
        (
            {"broken.bif": "module bad = {\n    let f = () => i32 missing()\n}\nexport(bad)\n"},
            'let bad = import("broken:bad")\n',
            "in broken.bif:2: 'missing' is not defined",
        ),
        (
            {},
            'let g = import("utils.text:greeting")\nlet main = () => null g.greet("x")\n',
            "add it to the dependency list: [g.greet]",
        ),
        ({}, "let x = 1\nexport(x)\n", "'x' is not a module in this file"),
    ],
)
def test_module_errors(project_root: Path, files: dict[str, str], source: str, message: str) -> None:
    for name, text in files.items():
        (project_root / name).write_text(text)
    with pytest.raises(BifrostError) as error:
        _lower(project_root, source)
    assert message in error.value.msg


def test_editor_follows_imports(project_root: Path) -> None:
    document = Document.open(project_root / "main.bif", MAIN)
    assert [d.message for d in document.diagnostics() if not d.unnecessary] == []
    row = MAIN.splitlines().index('    greeting.greet("Bifrost")')
    assert document.hover((row, 14)) == (
        "```bifrost\nlet greet = (name: str) => null\n// utils.text:greeting (text.bif)\n```"
    )
    definition = document.definition((row, 14))
    assert definition is not None
    assert definition.path == project_root / "utils" / "text.bif"
    typing = Document.open(project_root / "main.bif", MAIN.replace('    greeting.greet("Bifrost")', "    greeting."))
    members = [c.label for c in typing.completions((row, len("    greeting.")))]
    assert members == ["Person", "excitement", "greet", "shout", "age_next_year"]
    importing = Document.open(project_root / "x.bif", 'let g = import("ut\n')
    offered = {(c.label, c.detail) for c in importing.completions((0, len('let g = import("ut')))}
    assert ("utils.text:greeting", "Greets people") in offered
    assert not any(label.endswith(":internal") for label, _ in offered)  # not exported


def test_editor_knows_the_module_file(project_root: Path) -> None:
    document = Document.open(project_root / "utils" / "text.bif", TEXT)
    assert [d.message for d in document.diagnostics() if not d.unnecessary] == []
    outline = [(s.name, s.kind.value, s.detail) for s in document.symbols() if s.name in {"greeting", "internal"}]
    assert outline == [("greeting", "module", "Greets people"), ("internal", "module", "module")]
    row = TEXT.splitlines().index("        shout(excitement)")
    assert document.hover((row, 10)) == "```bifrost\nlet shout = (times: i32) => null\n```"


def test_formats_modules() -> None:
    source = "module helper={\n/* Says hi */\nlet n=1\n}\nexport( helper )\n"
    assert format_source(source) == "module helper = {\n    /* Says hi */\n    let n = 1\n}\nexport(helper)\n"
