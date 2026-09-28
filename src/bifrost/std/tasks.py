"""``std:tasks``: running async functions at the same time.

Imported like any standard module (``let tasks = import("std:tasks")``), but
built into the compiler.

- ``tasks.gather(name: f(x), other: g(y), ...)``: starts every call, waits
  until all are done, and returns a record of their results by name
  (``found.name``, ``found.other``). It pauses, so it is awaited, in an async
  function: ``let found = await tasks.gather(user: fetch_user(id), orders:
  fetch_orders(id))``. Each argument is a call of an async function; one
  returning nothing is passed without a name. The calls run on one thread,
  taking turns at their ``await``s.
- ``tasks.ignore(f(x))``: starts a call without waiting for it, where not
  waiting is deliberate (a call that pauses must otherwise be awaited). It
  runs on alongside the caller, at the caller's ``await``s; nothing waits for
  it, so it is cut off if ``main`` ends first. Its call returns nothing, and
  takes numbers, booleans and string literals: anything else it borrowed could
  be gone before it finishes.
"""

from mlir_python.lang import Token

from bifrost.std.fmt import Builtin

GATHER = Builtin(
    "gather",
    "(name: f(...), ...) => #{name: ..., ...}",
    "runs async calls at the same time; awaited, it gives their results by name",
    module="tasks",
)

IGNORE = Builtin(
    "ignore",
    "(f(...)) => null",
    "starts an async call without waiting for it: deliberately not awaited",
    module="tasks",
)

FUNCTIONS = {builtin.name: builtin for builtin in (GATHER, IGNORE)}


# ``bifrost_async_forget`` (async_runtime.c): let go of a started task's token; the task runs on.
def forget(token: Token) -> None: ...


forget.__name__ = "bifrost_async_forget"

__all__ = ["FUNCTIONS", "GATHER", "IGNORE", "forget"]
