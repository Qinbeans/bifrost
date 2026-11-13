"""A dependency list names the methods a function calls, by their object's type, and the language's own functions.

``team.size()`` needs ``[Team.size]`` (the type, not the value: ``Box.get`` for every ``Box[T]``,
``models.User.label`` for an imported one), a list's functions ``[list.sorted]``, a Result's
``[Result.is_ok]``, and the prelude's ``copy`` ``[copy]``. Each is checked both ways, like a function:
calling one that is not listed is an error, and so is listing one that is never called.
"""

from pathlib import Path

import pytest
from mlir_python.codegen import OptLevel

from bifrost.configs import Config
from bifrost.formatter import format_source
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project
from bifrost.server.analysis import Document, Severity

TEAM = """
let Team = struct {
    let name: str,
    let members: str[],
    let size = [super] () => i64 len(super.members),
    let twice = [super, Team.size] () => i64 super.size() * 2,
    static let solo = [] (name: str) => Team Team(name: name, members: #[name])
}
"""

BOX = """
let Box[T] = struct {
    let value: T,
    let get = [super] () => T super.value,
    let doubled = [super, Box.get] () => T super.get() + super.get()
}
"""

PETS = """
let Cat = struct { let kind: "cat", let lives: i64, let legs = [] () => i64 4 }
let Bird = struct { let kind: "bird", let wings: i64, let legs = [] () => i64 2 }
let Pet = Cat | Bird on "kind" {
    let sound = [super] () => i64 match super { Cat(_): 1, Bird(_): 2 }
}
"""

PEOPLE = """
let Person = struct { let name: str, let age: i64, let older = [super] () => i64 super.age + 1 }
let Employee = struct { let id: i64 } & Person
"""

MODELS = """
module models = {
    let User = struct {
        let id: i64,
        let label = [super] () => i64 super.id * 10
    }
}
export(models)
"""


def _project(root: Path) -> Project:
    return Project(
        Config(
            path=root,
            package={"name": "deps", "version": "0", "description": ""},
            flags={"optimization": OptLevel.O0, "linker": "clang"},
            libraries=[],
            externs=[],
        )
    )


def _check(root: Path, source: str) -> None:
    """Lower and type-check ``source`` as ``main.bif`` (with ``models.bif`` next to it), raising its error."""
    (root / "models.bif").write_text(MODELS)
    path = root / "main.bif"
    path.write_text(source)
    project = _project(root)
    unit = lower_file(project, path, root=root)
    with unit.errors():
        _ = project.program.mlir


@pytest.mark.parametrize(
    "source",
    [
        # A method, by its object's type; also one calling another on super, and a static function.
        TEAM
        + """let main = [Team.size, Team.twice, Team.solo] () => null {
    let team = Team.solo("ada")
    let n = team.size() + team.twice()
}""",
        # A method of a parameter's object, and one through a pipe.
        TEAM + "let count = [Team.size] (team: Team) => i64 team |> .size()",
        # A generic object's method, by its bare name, for each type it is made for.
        BOX
        + """let main = [Box.get, Box.doubled] () => null {
    let a = Box(value: 2)
    let b = Box(value: 2.5)
    let n = a.get() + a.doubled()
    let x = b.get()
}""",
        # An imported object's method, by the type as this file reaches it.
        """let models = import("models:models")
let label = [models.User.label] (user: models.User) => i64 user.label()""",
        # A union's own function, and one each of its objects has.
        PETS + "let noise = [Pet.sound, Pet.legs] (pet: Pet) => i64 pet.sound() + pet.legs()",
        # A base's method, by the base declaring it.
        PEOPLE + "let next = [Person.older] (e: Employee) => i64 e.older()",
        # A list's functions, called on it and through pipes.
        """let main = [list.sorted, list.contains, list.map, list.filter, list.index_of, list.reversed] () => null {
    let xs = #[3, 1, 2]
    let ys = xs.sorted()
    let found = xs.contains(2)
    let zs = xs |> .map([] (x: i64) => i64 x * 2) |> .filter([] (x: i64) => bool x > 2)
    let at = xs.index_of(1)
    let back = xs |> .reversed()
}""",
        """let main = [list.sort, list.reverse] () => null {
    let xs = #[3, 1, 2]
    let g <- xs
    g.sort()
    g.reverse()
    g -> xs
}""",
        # A lambda lists what it calls itself.
        """let main = [list.sorted] () => null {
    let xs = #[3, 1, 2]
    let f = [list.contains] (ys: i64[]) => bool ys.contains(1)
    let ys = xs.sorted()
}""",
        # A Result's methods.
        """let r = [] () => Result[i64, str] ok(1)
let main = [r, Result.is_ok, Result.is_err] () => null {
    let a = r().is_ok()
    let b = r().is_err()
}""",
        # The prelude's copy.
        TEAM
        + """let keep = [copy] (team: Team) => Team copy(team)
let lists = [copy] (xs: i64[]) => i64[] copy(xs)""",
        # A function of this file named copy shadows the prelude's: the entry is that function.
        """let copy = [] (x: i64) => i64 x + 1
let main = [copy] () => null {
    let n = copy(1)
}""",
        # A method calling itself lists this, or itself.
        """let Node = struct {
    let depth: i64,
    let down = [super, this] (n: i64) => i64 if n > 0 { super.down(n - 1) } else { super.depth }
}""",
        # What stays exempt: ok/err, conversions, len, interpolation, spreads, constructors, function values.
        TEAM
        + """let apply = [] (f: (x: i64) => i64, x: i64) => i64 f(x)
let main = [] () => null {
    let r: Result[i64, str] = ok(i64(i32(2)))
    let team = Team(name: "a", members: #[])
    let xs = #[...team.members, "b"]
    let n = len(xs)
    let text = "{n}"
}""",
    ],
)
def test_listed(tmp_path: Path, source: str) -> None:
    _check(tmp_path, source)


WRAP = """
let A = struct { let n: i64, let get = [super] () => i64 super.n }
let B = struct { let m: i64, let get = [super] () => i64 super.m }
"""


@pytest.mark.parametrize(
    "source",
    [
        # In a generic, a method of a type parameter's value is listed by the parameter: each
        # instance's is the method of the type it is for.
        WRAP
        + """let Wrap[T] = struct {
    let inner: T,
    let get = [super, T.get] () => i64 super.inner.get()
}
let main = [Wrap.get] () => null {
    let x = Wrap(inner: A(n: 1)).get() + Wrap(inner: B(m: 2)).get()
}""",
        WRAP
        + """let fetch[T] = [T.get] (x: T) => i64 x.get()
let main = [fetch] () => null {
    let x = fetch(A(n: 1)) + fetch(B(m: 2))
}""",
        # A method an object declares in place of its base's is its own.
        """let Person = struct { let name: str, let label = [super] () => str super.name }
let Employee = struct { let id: str, let label = [super] () => str super.id } & Person
let main = [Employee.label, Person.label] () => null {
    let s = Employee(id: "E1", name: "Bob").label()
    let t = Person(name: "Ann").label()
}""",
        # A struct (or a parameter) named list does not hide a list's functions.
        """let list = struct { let n: i64 }
let main = [list.sorted] () => null {
    let xs = #[3, 1]
    let ys = xs.sorted()
}""",
        "let f = [list.sorted] (list: i64[]) => i64[] list.sorted()",
    ],
)
def test_listed_edges(tmp_path: Path, source: str) -> None:
    _check(tmp_path, source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            TEAM + 'let main = [Team.solo] () => null {\n    let n = Team.solo("a").size()\n}',
            "main calls 'Team.size' without depending on it; add it to the dependency list: [Team.size]",
        ),
        (
            TEAM + "let count = [] (team: Team) => i64 team |> .size()",
            "count calls 'Team.size' without depending on it; add it to the dependency list: [Team.size]",
        ),
        (
            # super.size() needs Team.size too, not only super.
            """let Team = struct {
    let members: str[],
    let size = [super] () => i64 len(super.members),
    let twice = [super] () => i64 super.size() * 2
}""",
            "Team.twice calls 'Team.size' without depending on it; add it to the dependency list: [Team.size]",
        ),
        (
            BOX + "let main = [] () => null {\n    let n = Box(value: 1).get()\n}",
            "main calls 'Box.get' without depending on it; add it to the dependency list: [Box.get]",
        ),
        (
            'let models = import("models:models")\nlet label = [] (user: models.User) => i64 user.label()',
            "label calls 'models.User.label' without depending on it; add it to the dependency list: "
            "[models.User.label]",
        ),
        (
            PETS + "let noise = [Pet.sound] (pet: Pet) => i64 pet.sound() + pet.legs()",
            "noise calls 'Pet.legs' without depending on it; add it to the dependency list: [Pet.legs]",
        ),
        (
            PETS + "let noise = [] (pet: Pet) => i64 pet.sound()",
            "noise calls 'Pet.sound' without depending on it; add it to the dependency list: [Pet.sound]",
        ),
        (
            PEOPLE + "let next = [] (e: Employee) => i64 e.older()",
            "next calls 'Person.older' without depending on it; add it to the dependency list: [Person.older]",
        ),
        (
            PEOPLE + "let next = [Employee.older] (e: Employee) => i64 e.older()",
            "older is Person's method (Employee is built on Person); list it by it: [Person.older]",
        ),
        (
            "let main = [] () => null {\n    let xs = #[2, 1]\n    let ys = xs.sorted()\n}",
            "main calls 'list.sorted' without depending on it; add it to the dependency list: [list.sorted]",
        ),
        (
            "let main = [list.sorted] () => null {\n    let xs = #[2, 1] |> .sorted() |> .reversed()\n}",
            "main calls 'list.reversed' without depending on it; add it to the dependency list: [list.reversed]",
        ),
        (
            "let has = [] (xs: i64[]) => bool xs.contains(1)",
            "has calls 'list.contains' without depending on it; add it to the dependency list: [list.contains]",
        ),
        (
            "let main = [list.sorted] () => null {\n    let f = [] (ys: i64[]) => i64[] ys.sorted()\n}",
            "calls 'list.sorted' without depending on it; add it to the dependency list: [list.sorted]",
        ),
        (
            "let r = [] () => Result[i64, str] ok(1)\nlet main = [r] () => null {\n    let a = r().is_ok()\n}",
            "main calls 'Result.is_ok' without depending on it; add it to the dependency list: [Result.is_ok]",
        ),
        (
            "let keep = [] (xs: i64[]) => i64[] copy(xs)",
            "keep calls 'copy' without depending on it; add it to the dependency list: [copy]",
        ),
        # Listed, never called.
        (TEAM + "let main = [Team.size] () => null {}", "'Team.size' is a dependency of main but never called"),
        (BOX + "let main = [Box.get] () => null {}", "'Box.get' is a dependency of main but never called"),
        ("let main = [list.sorted] () => null {}", "'list.sorted' is a dependency of main but never called"),
        ("let main = [copy] () => null {}", "'copy' is a dependency of main but never called"),
        (
            "let main = [Result.is_ok] () => null {}",
            "'Result.is_ok' is a dependency of main but never called",
        ),
        (
            # A local named copy hides the prelude's: calling it calls the local, so the entry is unused.
            "let main = [copy] () => null {\n    let copy = [] (x: i64) => i64 x\n    let n = copy(1)\n}",
            "'copy' is a dependency of main but never called",
        ),
        # Named by a value, not its type.
        (
            TEAM + "let count = [team.size] (team: Team) => i64 team.size()",
            "list a method by its object's type, not a value (team): [Team.size]",
        ),
        (
            TEAM + 'let main = [team.size] () => null {\n    let team = Team(name: "a", members: #[])\n'
            "    let n = team.size()\n}",
            "list a method by its object's type, not a value (team): [Team.size]",
        ),
        # Not a method or function at all.
        (TEAM + "let main = [Team.nothing] () => null {}", "'Team.nothing' is not defined"),
        ("let main = [list.push] () => null {}", "a list has no function push; its functions are list.contains"),
        ("let main = [Result.unwrap] () => null {}", "a Result has no method unwrap"),
        # Calling a method on the type is still an error.
        (
            TEAM + 'let main = [Team.size] () => null {\n    let n = Team.size(Team(name: "a", members: #[]))\n}',
            "Team.size is a method, called on a Team (value.size(...)); call it on one",
        ),
    ],
)
def test_unlisted(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError, match=None) as raised:
        _check(tmp_path, source)
    assert message in raised.value.msg


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            # Each instance is checked; one not listing its type parameter's method says which entry to write,
            # rather than a mangled name at the call.
            WRAP
            + """let Wrap[T] = struct {
    let inner: T,
    let get = [super, A.get] () => i64 super.inner.get()
}
let main = [Wrap.get] () => null {
    let x = Wrap(inner: A(n: 1)).get() + Wrap(inner: B(m: 2)).get()
}""",
            "in Wrap[B]: Wrap[B].get calls 'B.get' without depending on it; add it to the dependency list: [T.get]",
        ),
        (
            WRAP
            + """let Wrap[T] = struct {
    let inner: T,
    let k: i64,
    let get = [super, T.get] () => i64 super.k
}
let main = [Wrap.get] () => null {
    let x = Wrap(inner: A(n: 1), k: 2).get()
}""",
            "'T.get' is a dependency of Wrap[A].get but never called",
        ),
        (
            # An error in a generic object's method is reported there, each time it is used.
            """let Box[T] = struct {
    let items: T[],
    let mapped = [super] (f: (x: T) => T) => Box[T] Box(items: super.items.map(f))
}
let double = [] (x: i64) => i64 x * 2
let main = [Box.mapped, double] () => null {
    let c = Box(items: #[1]).mapped(double)
}""",
            "Box[i64].mapped calls 'list.map' without depending on it; add it to the dependency list: [list.map]",
        ),
        (
            # ... and in a generic function whose result another generic's call needs.
            """let Box[T] = struct { let value: T }
let make[T] = [] (value: T) => Box[T] { return Box(value: copy(value)) }
let unbox[T] = [] (b: Box[T]) => T { return b.value }
let main = [make, unbox] () => null {
    let n = make(3) |> unbox
}""",
            "make[i64] calls 'copy' without depending on it; add it to the dependency list: [copy]",
        ),
        (
            # An object's own method, in place of its base's, is listed by it.
            """let Person = struct { let name: str, let label = [super] () => str super.name }
let Employee = struct { let id: str, let label = [super] () => str super.id } & Person
let main = [Person.label] () => null {
    let s = Employee(id: "E1", name: "Bob").label()
}""",
            "main calls 'Employee.label' without depending on it; add it to the dependency list: [Employee.label]",
        ),
        (
            "let f = [xs.sorted] (xs: i64[]) => i64[] xs.sorted()",
            "list a method by its object's type, not a value (xs): [list.sorted]",
        ),
    ],
)
def test_unlisted_edges(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(BifrostError, match=None) as raised:
        _check(tmp_path, source)
    assert message in raised.value.msg


def test_formatter_keeps_entries() -> None:
    source = "let f = [Team.size,list.sorted ,copy, models.User.label, Box.get] () => null {}\n"
    assert format_source(source) == "let f = [Team.size, list.sorted, copy, models.User.label, Box.get] () => null {}\n"


EDITED = (
    TEAM
    + """let main = [Team.solo, Team.size, list.sorted, copy] () => null {
    let team = Team.solo("ada")
    let n = team.size()
    let names = team.members.sorted()
    let again = copy(team)
}
"""
)


def _document(root: Path, source: str) -> Document:
    (root / "config.yaml").write_text((Path(__file__).parents[1] / "examples" / "async" / "config.yaml").read_text())
    return Document.open(root / "main.bif", source)


def test_entries_in_the_editor(tmp_path: Path) -> None:
    # The editor reports what lowering does, and describes the entries.
    document = _document(tmp_path, EDITED)
    assert [d.message for d in document.diagnostics() if d.severity == Severity.ERROR] == []
    row = EDITED.splitlines().index(next(line for line in EDITED.splitlines() if line.startswith("let main")))
    line = EDITED.splitlines()[row]
    assert document.hover((row, line.index("Team.size") + 6)) == (
        "```bifrost\nlet size = () => i64\n// a member of Team\n```"
    )
    assert document.hover((row, line.index("list.sorted") + 6)) == (
        "```bifrost\nlet sorted = () => T[]\n"
        "// a list's function: a new list of copies of its items, smallest first\n```"
    )
    assert "a deep copy of x" in (document.hover((row, line.index("copy") + 1)) or "")
    missing = EDITED.replace(", Team.size,", ",")
    errors = [d.message for d in _document(tmp_path, missing).diagnostics() if d.severity == Severity.ERROR]
    assert errors == ["main calls 'Team.size' without depending on it; add it to the dependency list: [Team.size]"]


@pytest.mark.parametrize(
    ("typed", "offered"),
    [
        ("let f = [Team.", [("solo", "function"), ("size", "method"), ("twice", "method")]),
        ("let f = [io.printf,\n    list.", [(name, "function") for name in ["contains", "filter", "index_of", "map"]]),
        ("let f = [Result.", [("is_err", "function"), ("is_ok", "function")]),
    ],
)
def test_completing_entries(tmp_path: Path, typed: str, offered: list[tuple[str, str]]) -> None:
    # In a dependency list, an object's name offers its methods too (listed by its type), and
    # `list.` and `Result.` the language's own.
    source = TEAM + typed + "\n"
    row = source.count("\n") - 1
    completions = _document(tmp_path, source).completions((row, len(source.splitlines()[row])))
    assert [(c.label, c.kind) for c in completions][: len(offered)] == offered


def test_completing_copy_in_a_list(tmp_path: Path) -> None:
    source = TEAM + "let f = [co\n"
    row = source.count("\n") - 1
    labels = [c.label for c in _document(tmp_path, source).completions((row, len("let f = [co")))]
    assert {"copy", "list", "Result"} <= set(labels)
    elsewhere = TEAM + "let f = [] () => null {\n    co\n"
    labels = [c.label for c in _document(tmp_path, elsewhere).completions((elsewhere.count("\n") - 1, 6))]
    assert "copy" not in labels  # only offered where it is listed


def _at(source: str, needle: str, offset: int = 0) -> tuple[int, int]:
    """Return the (row, column) of ``needle``'s first occurrence in ``source``, moved ``offset`` along it."""
    index = source.index(needle) + offset
    return source[:index].count("\n"), index - (source.rfind("\n", 0, index) + 1)


@pytest.mark.parametrize(
    ("typed", "offered"),
    [
        # A union's static and own functions, and the methods each of its objects has.
        ("let f = [Pet.", [("tom", "function"), ("sound", "method"), ("legs", "method")]),
        # After a comma with no space.
        ("let f = [copy,Team.", [("solo", "function"), ("size", "method"), ("twice", "method")]),
        ("let f = [copy,list.", [("contains", "function")]),
    ],
)
def test_completing_more_entries(tmp_path: Path, typed: str, offered: list[tuple[str, str]]) -> None:
    pets = PETS.replace("Bird(_): 2 }\n", "Bird(_): 2 },\n    static let tom = [] () => Pet Cat(lives: 9)\n")
    source = TEAM + pets + typed
    row = source.count("\n")
    completions = _document(tmp_path, source + "\n").completions((row, len(source.splitlines()[row])))
    assert [(c.label, c.kind) for c in completions][: len(offered)] == offered


UNIONS_EDITED = (
    PETS
    + PEOPLE
    + """let noise = [Pet.sound, Pet.legs, Person.older, copy] (pet: Pet, e: Employee) => i64 {
    let again = copy(pet)
    return pet.sound() + pet.legs() + e.older()
}
"""
)


def test_union_and_base_entries_in_the_editor(tmp_path: Path) -> None:
    # A union's methods, and a base's, hover and go to their declarations, in the list and where called.
    document = _document(tmp_path, UNIONS_EDITED)
    assert [d.message for d in document.diagnostics() if d.severity == Severity.ERROR] == []
    sound = "```bifrost\nlet sound = () => i64\n// a member of Pet\n```"
    legs = "```bifrost\nlet legs = () => i64\n// a member of each of Pet's objects (Cat, Bird)\n```"
    older = "```bifrost\nlet older = () => i64\n// a member of Person\n```"
    for needle, hover in [
        ("Pet.sound", sound),
        ("Pet.legs", legs),
        ("Person.older,", older),
        ("pet.sound", sound),
        ("pet.legs", legs),
        ("e.older", older),
    ]:
        assert document.hover(_at(UNIONS_EDITED, needle, needle.index(".") + 1)) == hover, needle
    declared = document.definition(_at(UNIONS_EDITED, "Pet.legs", 4))
    assert declared is not None
    assert declared.range[0] == _at(UNIONS_EDITED, "legs = [] () => i64 4")
    assert document.definition(_at(UNIONS_EDITED, "e.older", 2)).range[0] == _at(UNIONS_EDITED, "older = [super]")
    assert "a deep copy of x" in (document.hover(_at(UNIONS_EDITED, "copy(pet)")) or "")  # called, too


def test_language_owners_in_the_editor(tmp_path: Path) -> None:
    source = "let main = [list.sorted, Result.is_ok, copy] () => null {\n    let copy = [] (x: i64) => i64 x\n}\n"
    document = _document(tmp_path, source)
    assert "a list's functions" in (document.hover(_at(source, "list.sorted")) or "")
    assert "a Result's methods" in (document.hover(_at(source, "Result.is_ok")) or "")
    assert "a deep copy" not in (document.hover(_at(source, "copy = [")) or "")  # the local hides the prelude's


def test_type_parameter_entry_in_the_editor(tmp_path: Path) -> None:
    source = """let Wrap[T] = struct {
    let inner: T,
    let get = [super, T.get] () => i64 super.inner.get()
}
"""
    hover = _document(tmp_path, source).hover(_at(source, "T.get", 2))
    assert hover == "```bifrost\nT.get\n// the get method of each type T stands for\n```"


def test_entry_errors_span_the_entry(tmp_path: Path) -> None:
    # An error at an entry underlines all of it, not only the name of its type.
    source = TEAM + "let main = [list.sorted, Team.size] () => null {\n    let xs = #[1] |> .sorted()\n}\n"
    errors = [d for d in _document(tmp_path, source).diagnostics() if d.severity == Severity.ERROR]
    assert [d.message for d in errors] == ["'Team.size' is a dependency of main but never called"]
    start = _at(source, "list.sorted, Team.size", len("list.sorted, "))
    assert errors[0].range == (start, (start[0], start[1] + len("Team.size")))


def test_lambda_names_a_method_by_a_local_around_it(tmp_path: Path) -> None:
    source = (
        TEAM
        + 'let main = [] () => null {\n    let team = Team(name: "a", members: #[])\n'
        + "    let f = [team.size] () => i64 team.size()\n}"
    )
    with pytest.raises(BifrostError) as raised:
        _check(tmp_path, source)
    assert "list a method by its object's type, not a value (team): [Team.size]" in raised.value.msg
