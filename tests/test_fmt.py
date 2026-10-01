import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document

HELPER = """
let fmt = import("std:fmt")
let mem = import("std:mem")

module helper = {
    let greet = [fmt.format] (name: str) => mem.Unique[str] {
        return fmt.format("Hello, %s!", name)
    }
    let shout = [fmt.format] (text: mem.Unique[str], times: i32) => mem.Unique[str] {
        let louder = fmt.format("%s x%d, with a tail long enough to grow past sixty-four bytes", text, times)
        return louder
    }
}

export(helper)
"""

MAIN = """
let stdio = import("std:stdio")
let fmt = import("std:fmt")
let helper = import("helper:helper")

let main = [helper.greet, helper.shout, stdio.puts, fmt.format] () => null {
    let message = helper.greet("World")
    stdio.puts(message)
    let loud = helper.shout(message, 3)
    stdio.puts(loud)
    let i = 0
    while i < 3 {
        let line = fmt.format("line %d of %d", i + 1, 3)
        stdio.puts(line)
        let i = i + 1
    }
    stdio.puts(fmt.format("lent, then freed: %d", i))
    if i > 2 {
        let done = fmt.format("done after %d, %.1f%%", i, 99.5)
        stdio.puts(done)
        return
    }
}
"""

# Counts the program's calls to the allocator, to show every string is freed once.
COUNTER = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
static long allocations, frees;
static void *(*real_malloc)(size_t);
static void *(*real_realloc)(void *, size_t);
static void (*real_free)(void *);
static void init(void) {
    if (!real_malloc) {
        real_malloc = dlsym(RTLD_NEXT, "malloc");
        real_realloc = dlsym(RTLD_NEXT, "realloc");
        real_free = dlsym(RTLD_NEXT, "free");
    }
}
void *malloc(size_t n) { init(); allocations++; return real_malloc(n); }
void *realloc(void *p, size_t n) { init(); if (!p) allocations++; return real_realloc(p, n); }
void free(void *p) { init(); if (p) frees++; real_free(p); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "%ld %ld\n", allocations, frees); }
"""


def _project(root: Path) -> Project:
    return Project(
        Config(
            path=root,
            package={"name": "fmt_demo", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def test_formats_and_frees_every_string(tmp_path: Path) -> None:
    (tmp_path / "helper.bif").write_text(HELPER)
    (tmp_path / "main.bif").write_text(MAIN)
    project = _project(tmp_path)
    lower_file(project, tmp_path / "main.bif", root=tmp_path)
    executable = project.build()
    counter = tmp_path / "count.so"
    (tmp_path / "count.c").write_text(COUNTER)
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", counter, tmp_path / "count.c", "-ldl"], check=True)  # noqa: S603, S607
    environment = {"LD_PRELOAD": str(counter)}
    result = subprocess.run([executable], capture_output=True, text=True, check=True, env=environment)  # noqa: S603
    assert result.stdout.splitlines() == [
        "Hello, World!",
        "Hello, World! x3, with a tail long enough to grow past sixty-four bytes",
        "line 1 of 3",
        "line 2 of 3",
        "line 3 of 3",
        "lent, then freed: 3",
        "done after 3, 99.5%",
    ]
    allocations, frees = map(int, result.stderr.split())
    # Seven strings, each freed once; libc allocates one more buffer for stdout, which it keeps.
    assert (allocations, frees) == (8, 7)


HEAD = (
    'let fmt = import("std:fmt")\nlet stdio = import("std:stdio")\nlet mem = import("std:mem")\n'
    "let take = (t: mem.Unique[str]) => null {}\n"
)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            'let f = [fmt.format, take, stdio.puts] () => null {\n    let t = fmt.format("x")\n    take(t)\n'
            "    stdio.puts(t)\n}\n",
            "t was moved on line 7, so it is no longer here",
        ),
        (
            'let f = [fmt.format, take] (x: i32) => null {\n    let t = fmt.format("x")\n    if x > 0 {\n'
            "        take(t)\n    }\n}\n",
            "t is moved only on some paths",
        ),
        (
            'let f = [fmt.format, take] (x: i32) => null {\n    let t = fmt.format("x")\n    while x > 0 {\n'
            "        take(t)\n    }\n}\n",
            "t is moved inside a loop",
        ),
        (
            'let f = () => null {\n    let t: mem.Unique[str] = "x"\n}\n',
            "only an owned value can start a mem.Unique[str]",
        ),
        ('let f = () => null {\n    let t = fmt.format("a")\n}\n', "add it to the dependency list: [fmt.format]"),
    ],
)
def test_ownership_errors(tmp_path: Path, source: str, message: str) -> None:
    path = tmp_path / "case.bif"
    path.write_text(HEAD + source)
    with pytest.raises(BifrostError) as error:
        lower_file(_project(tmp_path), path)
    assert message in error.value.msg


@pytest.mark.parametrize(
    "source",
    [
        'let f = [fmt.format, take] () => null {\n    take(fmt.format("x %d", 1))\n}\n',
        'let f = [fmt.format, take] (x: i32) => null {\n    let t = fmt.format("x")\n    if x > 0 {\n'
        "        take(t)\n    } else {\n        take(t)\n    }\n}\n",
        # Only lent, it is freed after the statement.
        'let f = [fmt.format, stdio.puts] () => null {\n    stdio.puts(fmt.format("x"))\n}\n',
        # A new value replaces the old, which is freed.
        'let f = [fmt.format] () => null {\n    let t = fmt.format("a")\n    let t = fmt.format("b")\n}\n',
    ],
)
def test_valid_moves(tmp_path: Path, source: str) -> None:
    path = tmp_path / "case.bif"
    path.write_text(HEAD + source)
    lower_file(_project(tmp_path), path)


def test_editor_knows_fmt(tmp_path: Path) -> None:
    source = 'let fmt = import("std:fmt")\nlet f = [fmt.format] () => null {\n    let t = fmt.format("%d", 1)\n}\n'
    document = Document.open(tmp_path / "x.bif", source)
    assert [(h.position[0], h.label) for h in document.inlay_hints() if not h.parameter] == [(2, ": mem.Unique[str]")]
    assert "fmt.format = (pattern: str, ...) => mem.Unique[str]" in (document.hover((2, 17)) or "")
    typing = Document.open(tmp_path / "x.bif", source.replace('fmt.format("%d", 1)', "fmt."))
    assert [c.label for c in typing.completions((2, len("    let t = fmt.")))] == ["format"]
