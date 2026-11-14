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
"""

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

__all__ = ["CHANNEL", "CHANNEL_FUNCTION", "CONDVAR", "CONDVAR_FUNCTION", "CONTAINERS", "FUNCTIONS"]
