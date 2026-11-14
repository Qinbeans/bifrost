"""``std:sync``: channels (``sync.Channel``) and condvars (``sync.CondVar``, a channel with one slot).

The programs are built and run, the threaded ones on POSIX threads (declared
here, as a project's ``config.yaml`` would). Memory is checked by preloading a
library that counts ``malloc`` and ``free``, and with valgrind where it is
installed. Whether a sender or a receiver waits is seen from another thread,
which polls an ``mem.Atomic`` it writes after each send or receive.
"""

import collections
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from bifrost.configs import Config
from bifrost.configs.schema import _Extern, _Function
from bifrost.lowering import BifrostError, lower_file
from bifrost.project import Project

RUNS = 50  # each threaded program runs this many times: every message, every time

# Counts allocations; reports how many are still live when the program exits.
COUNTER_C = r"""
#include <stdio.h>
#include <stddef.h>
extern void *__libc_malloc(size_t);
extern void __libc_free(void *);
static long live;
void *malloc(size_t size) { __atomic_add_fetch(&live, 1, __ATOMIC_SEQ_CST); return __libc_malloc(size); }
void free(void *pointer) { if (pointer) __atomic_sub_fetch(&live, 1, __ATOMIC_SEQ_CST); __libc_free(pointer); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "live allocations: %ld\n", live); }
"""


def _project(tmp_path: Path) -> Project:
    thread = {"thread": "ptr", "attributes": "i64", "start": "(ptr) => None", "argument": "ptr"}
    declarations = [
        _Function.model_validate(
            {"name": "pthread_create", "as": "create", "type": "function", "parameters": thread, "return": "i32"}
        ),
        _Function.model_validate(
            {
                "name": "pthread_join",
                "as": "join",
                "type": "function",
                "parameters": {"thread": "u64", "result": "i64"},
                "return": "i32",
            }
        ),
    ]
    sleep = _Function.model_validate(
        {"name": "usleep", "as": "usleep", "type": "function", "parameters": {"microseconds": "u32"}, "return": "i32"}
    )
    return Project(
        Config(
            path=tmp_path,
            package={"name": "channels", "version": "0", "description": ""},
            flags={"optimization": 0, "linker": "clang"},
            externs=[
                _Extern(module="pthread", description="POSIX threads", declarations=declarations),
                _Extern(module="unistd", description="Sleeping", declarations=[sleep]),
            ],
        )
    )


def _build(tmp_path: Path, program: str) -> Path:
    source = tmp_path / "main.bif"
    source.write_text(program)
    project = _project(tmp_path)
    unit = lower_file(project, source)
    with unit.errors():
        return project.build()


def _run(executable: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([executable], capture_output=True, text=True, check=True, timeout=60, env=env)  # noqa: S603


def _leftover(stderr: str) -> int:
    """Return how many allocations the counter saw still live at exit.

    At most one is stdio's own: the buffer glibc may (or, depending on the C library and how
    the program ends, may not) have allocated for stdout and still hold. Anything more is
    something the program leaked.
    """
    match = re.search(r"live allocations: (-?\d+)", stderr)
    assert match is not None, stderr
    return int(match.group(1))


def _counting(tmp_path: Path) -> dict[str, str]:
    """Return an environment that preloads the allocation counter, built with the C compiler at hand."""
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler to build the allocation counter")
    (tmp_path / "count.c").write_text(COUNTER_C)
    library = tmp_path / "libcount.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-o", library, tmp_path / "count.c"], check=True)  # noqa: S603
    return {**os.environ, "LD_PRELOAD": str(library)}


def _valgrind(executable: Path) -> None:
    command = ["valgrind", "-q", "--leak-check=full", "--errors-for-leak-kinds=all", "--error-exitcode=9", executable]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)  # noqa: S603
    assert result.returncode == 0, result.stderr


HEADER = """\
let io = import("std:stdio")
let fmt = import("std:fmt")
let mem = import("std:mem")
let sync = import("std:sync")
let pthread = import("pthread")
let unistd = import("unistd")

let Thread = struct { let id: u64 }
let Flag = struct { let step: i64 }
"""


# -- one thread ----------------------------------------------------------------------

ORDER = (
    HEADER
    + r"""
let main = [io.printf, channel, condvar] () => null {
    let jobs: sync.Channel[i32] = channel(4)
    7 -> jobs
    8 -> jobs
    let n: i32 = 9
    n -> jobs
    10 -> jobs
    let a <- jobs
    let b <- jobs
    11 -> jobs
    let c <- jobs
    let d <- jobs
    let e: i32 <- jobs
    io.printf("%d %d %d %d %d\n", a, b, c, d, e)
    let ready: sync.CondVar[bool] = condvar()
    true -> ready
    let go <- ready
    false -> ready
    let again <- ready
    io.printf("%d %d\n", go, again)
}
"""
)


def test_messages_come_out_oldest_first(tmp_path: Path) -> None:
    executable = _build(tmp_path, ORDER)
    result = _run(executable, env=_counting(tmp_path))
    assert result.stdout.splitlines() == ["7 8 9 10 11", "1 0"]
    assert _leftover(result.stderr) <= 1  # only stdout's buffer: both channels were freed


OWNED = (
    HEADER
    + r"""
let Counter = struct { let hits: i64 }
let Team = struct { let name: str, let members: i64[] }

let main = [io.printf, io.puts, fmt.format, channel] () => null {
    let names: sync.Channel[mem.Unique[str]] = channel(3)
    let first = fmt.format("job %d", 1)
    first -> names
    fmt.format("job %d", 2) -> names
    fmt.format("job %d", 3) -> names
    let got <- names
    io.puts(got)
    let lists: sync.Channel[i64[]] = channel(2)
    let xs = #[1, 2, 3]
    xs -> lists
    #[4, 5] -> lists
    let ys <- lists
    io.printf("%lld %lld\n", ys[0], ys[2])
    let teams: sync.Channel[Team] = channel(2)
    Team(name: "a", members: #[1, 2]) -> teams
    Team(name: "b", members: #[3]) -> teams
    let team <- teams
    io.printf("%s %lld\n", team.name, team.members[1])
    let points: sync.Channel[#{x: i64, y: i64}] = channel(1)
    #{x: 1, y: 2} -> points
    let p <- points
    io.printf("%lld %lld\n", p.x, p.y)
    let counters: sync.Channel[mem.Atomic[Counter]] = channel(2)
    let counter: mem.Atomic[Counter] = Counter(hits: 41)
    counter -> counters
    counter -> counters
    let c <- counters
    let g <- c
    g.hits = g.hits + 1
    io.printf("%lld\n", g.hits)
    g -> c
    let fns: sync.Channel[(x: i64) => i64] = channel(1)
    let k = #[10]
    let add = [k] (x: i64) => i64 x + k[0]
    add -> fns
    let f <- fns
    io.printf("%lld\n", f(5))
}
"""
)


def test_owned_messages_move_through_and_are_freed_once(tmp_path: Path) -> None:
    # Received ones are freed with their receiver's scope; the rest (job 2 and 3, #[4, 5], team b and a
    # counter's owner) with the channel's last owner.
    executable = _build(tmp_path, OWNED)
    result = _run(executable, env=_counting(tmp_path))
    assert result.stdout.splitlines() == ["job 1", "1 3", "a 2", "1 2", "42", "15"]
    assert _leftover(result.stderr) <= 1  # only stdout's buffer


@pytest.mark.skipif(shutil.which("valgrind") is None, reason="needs valgrind")
def test_owned_messages_leave_nothing_behind(tmp_path: Path) -> None:
    _valgrind(_build(tmp_path, OWNED))


OWNERS = (
    HEADER
    + r"""
let make = [channel] (slots: i64) => sync.Channel[mem.Unique[str]] channel(slots)

let fill = [fmt.format] (names: sync.Channel[mem.Unique[str]], n: i64) => null {
    let i = 0
    while i < n {
        fmt.format("name %lld", i) -> names
        let i = i + 1
    }
}

let same = [] (names: sync.Channel[mem.Unique[str]]) => sync.Channel[mem.Unique[str]] names

let main = [io.puts, make, fill, same] () => null {
    let names = make(8)
    let other = names
    fill(other, 3)
    let again = same(names)
    let first <- again
    io.puts(first)
}
"""
)


def test_channels_have_several_owners(tmp_path: Path) -> None:
    executable = _build(tmp_path, OWNERS)
    result = _run(executable, env=_counting(tmp_path))
    assert result.stdout.splitlines() == ["name 0"]
    assert _leftover(result.stderr) <= 1  # the last of three owners freed it, and names 1 and 2


def test_a_negative_number_of_slots_stops_the_program(tmp_path: Path) -> None:
    program = HEADER + (
        "let main = [channel] () => null {\n    let n = 0 - 3\n    let jobs: sync.Channel[i64] = channel(n)\n}\n"
    )
    executable = _build(tmp_path, program)
    result = subprocess.run([executable], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode != 0
    assert "main.bif:12: channel(n) takes 0 or more slots, and n is negative" in result.stderr


def _lower(tmp_path: Path, program: str) -> Project:
    source = tmp_path / "main.bif"
    source.write_text(program)
    project = _project(tmp_path)
    unit = lower_file(project, source)
    with unit.errors():
        _ = project.program.mlir
    return project


CELL = "let Box = struct { let n: i64 }\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # Sending and receiving wait, which an async function must not do, nor one holding a guard.
        (
            "let f = [channel] async () => null {\n"
            "    let jobs: sync.Channel[i64] = channel(1)\n    let j <- jobs\n}\n",
            "f is async, and it receives from a sync.Channel: the receiver may wait, which blocks the event loop",
        ),
        (
            "let f = [channel] async () => null {\n    let jobs: sync.Channel[i64] = channel(1)\n    1 -> jobs\n}\n",
            "f is async, and it sends to a sync.Channel: the sender may wait, which blocks the event loop",
        ),
        (
            "let f = [channel] () => null {\n    let c: mem.Atomic[Box] = Box(n: 1)\n"
            "    let jobs: sync.Channel[i64] = channel(1)\n    let g <- c\n    let j <- jobs\n    g -> c\n}\n",
            "receives from jobs while g holds c",
        ),
        (
            "let f = [channel] () => null {\n    let c: mem.Atomic[Box] = Box(n: 1)\n"
            "    let jobs: sync.Channel[i64] = channel(1)\n    let g <- c\n    1 -> jobs\n    g -> c\n}\n",
            "sends to jobs while g holds c",
        ),
        # What a message may be.
        (
            "let f = [channel] () => null {\n    let jobs: sync.Channel[mem.Weak[Box]] = channel(1)\n}\n",
            "a sync.Channel cannot carry a mem.Weak: it is a lend, which cannot outlive the call that lends it",
        ),
        (
            "let f = [channel] () => null {\n    let jobs: sync.Channel[mem.Shared[Box]] = channel(1)\n}\n",
            "a sync.Channel cannot carry a mem.Shared, which is for one thread; send a mem.Atomic instead",
        ),
        (
            "let f = [channel] () => null {\n    let jobs: sync.Channel[i64] = channel(1)\n    true -> jobs\n}\n",
            "jobs carries i64 messages, but this is bool",
        ),
        # A message sent has moved.
        (
            "let f = [channel, fmt.format, io.puts] () => null {\n"
            "    let jobs: sync.Channel[mem.Unique[str]] = channel(1)\n"
            '    let s = fmt.format("x%d", 1)\n    s -> jobs\n    io.puts(s)\n}\n',
            "s was moved on line 14, so it is no longer here",
        ),
        # channel() and condvar() are the prelude's, listed like copy.
        (
            "let f = [] () => null {\n    let jobs: sync.Channel[i64] = channel(1)\n}\n",
            "f calls 'channel' without depending on it; add it to the dependency list: [channel]",
        ),
        ("let f = [channel] () => null {\n    let x = 1\n}\n", "'channel' is a dependency of f but never called"),
        (
            "let f = [channel] () => null {\n    let jobs = channel(1)\n}\n",
            "channel() makes a sync.Channel; write the type of its messages where it goes",
        ),
    ],
)
def test_channel_errors(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(BifrostError) as error:
        _lower(tmp_path, HEADER + CELL + body)
    assert message in error.value.msg


def test_a_guard_given_back_before_a_receive_and_taken_again_after_stays_two_guards(tmp_path: Path) -> None:
    # The receive between them is a call: the cell passes never let a lock span it.
    program = (
        HEADER + CELL + "let f = [channel] () => null {\n    let c: mem.Atomic[Box] = Box(n: 1)\n"
        "    let jobs: sync.Channel[i64] = channel(1)\n    1 -> jobs\n"
        "    let g <- c\n    g.n = 2\n    g -> c\n    let j <- jobs\n    let h <- c\n    h.n = j\n    h -> c\n}\n"
    )
    linked = str(_lower(tmp_path, program).program.linked())
    assert linked.count("call @bifrost_atomic_lock") == 2
    assert linked.count("call @bifrost_atomic_unlock") == 2


# -- threads -------------------------------------------------------------------------

THREADS = (
    HEADER
    + r"""
let step = [] (flag: mem.Weak[Flag]) => i64 {
    let g <- flag
    let at = g.step
    g -> flag
    return at
}

let mark = [] (flag: mem.Weak[Flag], at: i64) => null {
    let g <- flag
    g.step = at
    g -> flag
}

// Print how far the other thread has got.
let show = [step, io.printf] (flag: mem.Weak[Flag]) => null {
    io.printf("%lld\n", step(flag))
}

"""
)


def _settle(at: str) -> str:
    """Bifrost lines for main: wait until the other thread has marked ``at``, give it 50 ms more, show how far it got.

    Each look takes a guard of its own, so the other thread can take the mutex in between (lending
    the mem.Atomic to a function would hold it for the whole wait).
    """
    return (
        "    let seen = 0\n"
        f"    while seen < {at} {{\n"
        "        unistd.usleep(1000)\n"
        "        let g <- flag\n"
        "        let seen = g.step\n"
        "        g -> flag\n"
        "    }\n"
        "    unistd.usleep(50000)\n"
        "    show(flag)\n"
    )


def _sender_waits(make: str, sends: int) -> str:
    """Return a program whose sender thread sends ``sends`` messages on a channel made by ``make``.

    The channel carries that mem.Atomic: main sends it first, and the sender's first receive
    takes it. Main prints how far the sender got on its own, receives everything, and prints
    the count again.
    """
    maker = make.split("(", maxsplit=1)[0]  # what main calls: channel or condvar
    return (
        THREADS
        + rf"""
let send = [mark] (pipe: sync.Channel[mem.Atomic[Flag]]) => null {{
    let flag <- pipe
    let i = 1
    while i <= {sends} {{
        flag -> pipe
        mark(flag, i)
        let i = i + 1
    }}
}}

let main = [pthread.create, pthread.join, send, show, unistd.usleep, {maker}] (slots: i64) => null {{
    let pipe: sync.Channel[mem.Atomic[Flag]] = {make}
    let flag: mem.Atomic[Flag] = Flag(step: 0)
    let sender: mem.Unique[Thread] = Thread(id: 0)
    pthread.create(sender, 0, send, pipe)
    flag -> pipe
{_settle("slots")}    let i = 0
    while i < {sends} {{
        let message <- pipe
        let i = i + 1
    }}
    let s <- sender
    pthread.join(s.id, 0)
    s -> sender
    show(flag)
}}
"""
    )


@pytest.mark.parametrize(
    ("make", "slots", "sends"),
    [
        ("channel(2)", 2, 4),  # the third send waits until main receives
        ("channel(1)", 1, 2),
        ("condvar()", 1, 2),  # one slot: the second send waits
        ("channel(0)", 0, 2),  # no slots: even the first waits for a receiver
        ("channel()", 0, 1),
    ],
)
def test_the_sender_waits_only_while_every_slot_holds_an_unread_message(
    tmp_path: Path, make: str, slots: int, sends: int
) -> None:
    # `main` here takes the slots it expects the sender to fill on its own (a Bifrost main takes no
    # parameters): written into the program instead.
    program = _sender_waits(make, sends).replace("(slots: i64) => null {", f"() => null {{\n    let slots = {slots}")
    executable = _build(tmp_path, program)
    for _ in range(3):
        result = _run(executable, env=_counting(tmp_path))
        assert result.stdout.splitlines() == [str(slots), str(sends)]
        assert _leftover(result.stderr) <= 1  # the channel and the mem.Atomic were freed


RECEIVER = (
    THREADS
    + r"""
// Takes the mem.Atomic, says it is waiting (1), receives another message (2).
let receive = [mark] (pipe: sync.Channel[mem.Atomic[Flag]]) => null {
    let flag <- pipe
    mark(flag, 1)
    let next <- pipe
    mark(flag, 2)
}

let main = [pthread.create, pthread.join, receive, show, unistd.usleep, channel] () => null {
    let pipe: sync.Channel[mem.Atomic[Flag]] = channel(4)
    let flag: mem.Atomic[Flag] = Flag(step: 0)
    let receiver: mem.Unique[Thread] = Thread(id: 0)
    pthread.create(receiver, 0, receive, pipe)
    flag -> pipe
    settle(flag, 1)
    flag -> pipe
    let r <- receiver
    pthread.join(r.id, 0)
    r -> receiver
    show(flag)
}
""".replace("    settle(flag, 1)\n", _settle("1"))
)


def test_the_receiver_waits_until_a_message_is_sent(tmp_path: Path) -> None:
    executable = _build(tmp_path, RECEIVER)
    for _ in range(3):
        result = _run(executable, env=_counting(tmp_path))
        assert result.stdout.splitlines() == ["1", "2"]
        assert _leftover(result.stderr) <= 1


# Producers send 1 to MESSAGES each; consumers print every message they receive, until a 0.
WORKERS = (
    HEADER
    + r"""
let produce = [] (jobs: sync.Channel[i64]) => null {
    let i = 1
    while i <= MESSAGES {
        i -> jobs
        let i = i + 1
    }
}

let consume = [io.printf] (jobs: sync.Channel[i64]) => null {
    let done = false
    while !done {
        let job <- jobs
        if job == 0 {
            let done = true
        } else {
            io.printf("%lld\n", job)
        }
    }
}

let start = [pthread.create] (worker: mem.Weak[Thread], body: (ptr) => null, jobs: sync.Channel[i64]) => null {}

let main = [pthread.create, pthread.join, produce, consume, channel] () => null {
    let jobs: sync.Channel[i64] = channel(SLOTS)
    let c1: mem.Unique[Thread] = Thread(id: 0)
    let c2: mem.Unique[Thread] = Thread(id: 0)
    let c3: mem.Unique[Thread] = Thread(id: 0)
    let p1: mem.Unique[Thread] = Thread(id: 0)
    let p2: mem.Unique[Thread] = Thread(id: 0)
    let p3: mem.Unique[Thread] = Thread(id: 0)
    pthread.create(c1, 0, consume, jobs)
    pthread.create(c2, 0, consume, jobs)
    pthread.create(c3, 0, consume, jobs)
    pthread.create(p1, 0, produce, jobs)
    pthread.create(p2, 0, produce, jobs)
    pthread.create(p3, 0, produce, jobs)
    let g1 <- p1
    pthread.join(g1.id, 0)
    g1 -> p1
    let g2 <- p2
    pthread.join(g2.id, 0)
    g2 -> p2
    let g3 <- p3
    pthread.join(g3.id, 0)
    g3 -> p3
    let stop: i64 = 0
    stop -> jobs
    stop -> jobs
    stop -> jobs
    let h1 <- c1
    pthread.join(h1.id, 0)
    h1 -> c1
    let h2 <- c2
    pthread.join(h2.id, 0)
    h2 -> c2
    let h3 <- c3
    pthread.join(h3.id, 0)
    h3 -> c3
}
"""
).replace(
    "let start = [pthread.create] (worker: mem.Weak[Thread], body: (ptr) => null, "
    "jobs: sync.Channel[i64]) => null {}\n\n",
    "",
)

PRODUCERS = 3
MESSAGES = 2000


@pytest.mark.parametrize("slots", [0, 1, 8])
def test_every_message_reaches_exactly_one_receiver(tmp_path: Path, slots: int) -> None:
    program = WORKERS.replace("MESSAGES", str(MESSAGES)).replace("SLOTS", str(slots))
    executable = _build(tmp_path, program)
    expected = collections.Counter(dict.fromkeys(range(1, MESSAGES + 1), PRODUCERS))
    for _ in range(RUNS):
        result = _run(executable)
        assert collections.Counter(int(line) for line in result.stdout.split()) == expected


# Two producers send owned strings; the consumer takes only some. The rest are still queued when
# the last owner (main) lets go of the channel, which frees them.
LEFT_OVER = (
    HEADER
    + r"""
let produce = [fmt.format] (names: sync.Channel[mem.Unique[str]]) => null {
    let i = 0
    while i < 20 {
        fmt.format("name %lld", i) -> names
        let i = i + 1
    }
}

let consume = [io.puts] (names: sync.Channel[mem.Unique[str]]) => null {
    let i = 0
    while i < 10 {
        let name <- names
        let i = i + 1
    }
    io.puts("took 10")
}

let main = [pthread.create, pthread.join, produce, consume, channel] () => null {
    let names: sync.Channel[mem.Unique[str]] = channel(64)
    let a: mem.Unique[Thread] = Thread(id: 0)
    let b: mem.Unique[Thread] = Thread(id: 0)
    let c: mem.Unique[Thread] = Thread(id: 0)
    pthread.create(c, 0, consume, names)
    pthread.create(a, 0, produce, names)
    pthread.create(b, 0, produce, names)
    let ga <- a
    pthread.join(ga.id, 0)
    ga -> a
    let gb <- b
    pthread.join(gb.id, 0)
    gb -> b
    let gc <- c
    pthread.join(gc.id, 0)
    gc -> c
}
"""
)


def test_queued_messages_are_freed_with_the_last_owner(tmp_path: Path) -> None:
    executable = _build(tmp_path, LEFT_OVER)
    for _ in range(RUNS):
        result = _run(executable, env=_counting(tmp_path))
        assert result.stdout.splitlines() == ["took 10"]
        assert _leftover(result.stderr) <= 1  # 30 names were still queued


@pytest.mark.skipif(shutil.which("valgrind") is None, reason="needs valgrind")
def test_threads_leave_nothing_behind(tmp_path: Path) -> None:
    _valgrind(_build(tmp_path, LEFT_OVER))
    _valgrind(_build(tmp_path, WORKERS.replace("MESSAGES", "200").replace("SLOTS", "0")))
