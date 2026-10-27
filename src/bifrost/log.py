"""What the compiler says as it works: each step of a build, on stderr (``  Downloading raylib-6.0-any.bifpkg``).

The verb is right-aligned, as Cargo writes its steps, so that what each step works
on lines up below the one before. Only the command line says them (``enable``):
the editor's server and the tests compile quietly.
"""

from rich.console import Console
from rich.markup import escape

_CONSOLE = Console(stderr=True, highlight=False, soft_wrap=True)
_WIDTH = 12  # the verbs' column
_ENABLED = [False]


def enable() -> None:
    """Say the steps from here on (``bfc`` does, for the commands that build)."""
    _ENABLED[0] = True


def step(verb: str, what: str) -> None:
    """Say that the build does ``verb`` to ``what``: ``step("Compiling", "src/main.bif")``."""
    if _ENABLED[0]:
        _CONSOLE.print(f"[bold green]{verb:>{_WIDTH}}[/bold green] {escape(what)}")


def note(what: str) -> None:
    """Say something about the step before, indented under what it works on (a command it runs)."""
    if _ENABLED[0]:
        _CONSOLE.print(f"{'':>{_WIDTH}} [dim]{escape(what)}[/dim]")


__all__ = ["enable", "note", "step"]
