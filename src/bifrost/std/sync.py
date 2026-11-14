"""``std:sync``: channels, for passing values between threads.

Imported like any standard module (``let sync = import("std:sync")``), but
built into the compiler: its types are checked at compile time, and compile
to the runtime in ``sync_runtime``.

- ``sync.Channel[T]``: messages of type ``T``, in the order they were sent,
  from any number of senders to any number of receivers, on any threads.
  ``channel(n)`` makes one with ``n`` slots (``let jobs: sync.Channel[i32] =
  channel(4)``). ``7 -> jobs`` sends: the sender returns at once while a slot
  is free, and waits only when every slot holds an unread message. ``let job
  <- jobs`` receives: the receiver takes the oldest message at once if there
  is one, and waits only when there are none. Each message goes to exactly
  one receiver. ``channel(0)`` (or ``channel()``) has no slots: the sender
  waits until a receiver takes its message.
- ``sync.CondVar[T]``: a ``sync.Channel[T]`` with exactly one slot, made by
  ``condvar()``. It is the same type under another name, for readability:
  it goes wherever a ``sync.Channel[T]`` goes, and the compiler names it
  ``sync.Channel`` in its messages.

A message moves into the channel (an owned local sent is not used again),
and the receiver owns what it receives. Numbers, booleans, strings and
records of those are copied. A ``mem.Atomic`` may be sent (the channel holds
another owner of it until it is received); a ``mem.Weak`` (a lend, which
cannot outlive its call) and a ``mem.Shared`` (one thread's) may not.

A channel has several owners, counted atomically like a ``mem.Atomic``'s:
``let other = jobs`` makes another, and the last one frees the channel and
the messages still queued in it. It is passed to C's ``void *`` (a thread's
argument) as is, like a ``mem.Atomic``.

``channel`` and ``condvar`` are the prelude's, like ``copy``: no import is
needed to call them, a function lists them (``[channel]``, ``[condvar]``),
and a name of the file hides them. The type of the messages comes from where
the channel goes: ``let jobs: sync.Channel[i32] = channel(4)``, or a returned
one, from the function's result type.

Threads: ``spawn(work(jobs))`` runs ``work(jobs)`` on a new OS thread (an
async ``work`` on an event loop of that thread's own, to the end), and
gives a ``sync.Thread[R]`` (``R`` is what ``work`` returns). ``await t``
gives its result: in code that is not async, the caller waits there until
the thread is done (only if it is not done yet); in an async function, only
that task pauses, and the event loop runs the others meanwhile. A thread is
awaited, or let go on purpose with ``tasks.ignore(t)`` (its result, which
must own nothing, is dropped); one that is neither is a compile error.
``spawn`` is the prelude's, like ``channel`` (``[spawn]``). What the call is
given moves into the thread: owned values (the thread frees them when the
call returns), copies, and another owner of each ``mem.Atomic`` and channel.
A ``mem.Shared`` (one thread's) and a ``mem.Weak`` (a lend, which cannot
outlive its call) cannot be given to one.
"""

from mlir_python.lang import Fn, Token, i64, ptr

from bifrost.std.mem import Container

CHANNEL = Container(
    "Channel",
    "messages between threads, oldest first: the sender waits only if every slot holds an unread message",
    supported=True,
    module="sync",
)
CONDVAR = Container(
    "CondVar",
    "a sync.Channel with one slot (condvar()): the receiver waits until there is a message",
    supported=True,
    module="sync",
)

CONTAINERS = {container.name: container for container in (CHANNEL, CONDVAR)}

# The prelude's functions that make them: `channel(4)`, `condvar()`.
CHANNEL_FUNCTION = "channel"
CONDVAR_FUNCTION = "condvar"
FUNCTIONS = {
    CHANNEL_FUNCTION: "a new sync.Channel with n slots (0, the default: the sender waits for a receiver)",
    CONDVAR_FUNCTION: "a new sync.CondVar: a sync.Channel with one slot",
}

# The prelude's function that starts a thread: `let t = spawn(work(jobs))`.
SPAWN_FUNCTION = "spawn"
SPAWN = "runs a call on a new OS thread (an async one on its own event loop); `await` the thread for its result"
THREAD = "sync.Thread"


# What a spawn compiles to: thread_runtime.c (and, for `await t` in async code, async_runtime.c).
def thread_new(result: i64, arguments: i64) -> ptr: ...
def thread_result(thread: ptr) -> ptr: ...
def thread_arguments(thread: ptr) -> ptr: ...
def thread_start(thread: ptr, entry: Fn[[ptr], None]) -> None: ...
def thread_join(thread: ptr) -> None: ...
def thread_free(thread: ptr) -> None: ...
def thread_ignore(thread: ptr) -> None: ...
def thread_state(thread: ptr) -> ptr: ...
def async_watch(state: ptr) -> Token: ...


THREAD_RUNTIME = {
    function.__name__: function
    for function in (
        thread_new,
        thread_result,
        thread_arguments,
        thread_start,
        thread_join,
        thread_free,
        thread_ignore,
        thread_state,
        async_watch,
    )
}
for _name, _function in THREAD_RUNTIME.items():
    _function.__name__ = f"bifrost_{_name}"

__all__ = [
    "CHANNEL",
    "CHANNEL_FUNCTION",
    "CONDVAR",
    "CONDVAR_FUNCTION",
    "CONTAINERS",
    "FUNCTIONS",
    "SPAWN",
    "SPAWN_FUNCTION",
    "THREAD",
    "THREAD_RUNTIME",
]
