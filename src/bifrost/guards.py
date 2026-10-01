"""Check guards: compile-time locks on pointers and values.

``let guard <- ctx`` locks ``ctx``: until ``guard -> ctx`` releases it, ``ctx``
cannot be used, and its fields are reached (and changed) only through
``guard``. A guard on a whole value (``mem.Atomic[i64]``) reads it as ``guard``
and writes it with ``guard = guard + 1``. The release must be in the scope that took the lock, or a scope
nested in it, and every path through that scope must release exactly once.
A guard is not held across a call that can pause (a function that waits for
something, see ``lowering``): while it waits, other code runs, and could
change what the guard holds.

A held value (a ``mem.Unique`` local that owns it, or a ``mem.Weak`` parameter
it is lent to) is only reached through a guard, or lent to a call. A
``mem.Shared`` or ``mem.Atomic`` may also be given another owner
(``let other = counter``) or returned.

All of this is checked here, before lowering. A guard compiles to the pointer
(or value) it locks; only a ``mem.Shared`` or ``mem.Atomic`` lock also runs
code (see ``bifrost.std.mem_runtime``).
"""

from collections.abc import Callable
from dataclasses import dataclass

from tree_sitter import Node

_CELLS = {"mem.Shared", "mem.Atomic"}  # containers that may have several owners


class GuardError(Exception):
    """A guard rule is broken at ``node``."""

    def __init__(self, node: Node, message: str) -> None:
        super().__init__(message)
        self.node = node
        self.message = message


@dataclass(frozen=True)
class _Lock:
    guard: str
    source: str
    node: Node  # the `let guard <- source` statement
    block: Node  # the block it was taken in


@dataclass
class _State:
    """What holds what on one path through a function."""

    held: dict[str, _Lock]  # guard -> its lock
    released: dict[str, Node]  # guard -> where it was last released

    def copy(self) -> "_State":
        return _State(dict(self.held), dict(self.released))

    def locker_of(self, source: str) -> _Lock | None:
        return next((lock for lock in self.held.values() if lock.source == source), None)


def _text(node: Node) -> str:
    return (node.text or b"").decode()


def _named(node: Node) -> list[Node]:
    return [child for child in node.named_children if child.type != "comment"]


def _unwrap(node: Node) -> Node:
    while node.type in {"expression", "condition", "parenthesized_expression", "getter_owner"}:
        node = _named(node)[0]
    return node


def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _target(node: Node) -> tuple[Node, list[Node]]:
    """Split an assignment target into its root name and the indexes it reads on the way.

    ``g.items[i].name`` gives ``g`` and ``[i]``; ``g[0]`` gives ``g`` and ``[0]``.
    """
    node = _unwrap(node)
    if node.type == "get_expression":
        owner, index = _named(node)
        root, indexes = _target(owner)
        return root, [*indexes, index]
    if node.type == "child_annotation":
        return _target(_named(node)[0])
    return node, []


def _release_hint(lock: _Lock) -> str:
    return f"`{lock.guard} -> {lock.source}`"


class _Checker:
    def __init__(
        self,
        held: dict[str, str],
        pauses: Callable[[Node], bool],
        values: set[str],
        lends: Callable[[str, str], bool],
    ) -> None:
        self.held = held  # name -> its container, e.g. "mem.Weak"
        self.pauses = pauses  # whether a call can pause
        self.values = values  # held names whose value is not an object: a guard on one is read as a value
        self.lends = lends  # whether calling a method (name, method) lends the held name: it changes its object

    # -- statements ---------------------------------------------------------------

    def block(self, node: Node, state: _State) -> _State | None:
        """Check a block; return the state after it, or ``None`` if it always returns."""
        current: _State | None = state
        for item in _named(node):
            if current is None:
                break  # after a return: unreachable
            current = self.item(item, current)
        if current is not None:
            for lock in current.held.values():
                if lock.block == node:
                    msg = (
                        f"{lock.guard} still holds {lock.source} at the end of the scope it was locked in; "
                        f"release it with {_release_hint(lock)}"
                    )
                    raise GuardError(lock.node, msg)
        return current

    def item(self, item: Node, state: _State) -> _State | None:
        match item.type:
            case "lock":
                return self.lock(item, state)
            case "release":
                return self.release(item, state)
            case "field_assignment":
                self.field_assignment(item, state)
            case "guard_assignment":
                self.guard_assignment(item, state)
            case "local_assignment":
                parts = _named(item)
                name = _text(parts[0])
                if name in state.held:
                    raise GuardError(parts[0], f"{name} is a guard; give the value another name")
                self.uses(parts[-1], state)
            case "return_statement":
                return self.return_(item, state)
            case "expression":
                return self.statement(_unwrap(item), state)
            case _:
                self.uses(item, state)
        return state

    def statement(self, node: Node, state: _State) -> _State | None:
        match node.type:
            case "if":
                return self.if_(node, state)
            case "while":
                return self.while_(node, state)
            case "forall":
                return self.forall(node, state)
            case "match_expression":
                return self.match(node, state)
        self.uses(node, state)
        return state

    def lock(self, node: Node, state: _State) -> _State:
        guard = _text(node.child_by_field_name("guard"))
        source_node = _unwrap(node.child_by_field_name("source"))
        if source_node.type != "identifier":
            raise GuardError(source_node, "lock a name: `let guard <- ctx`")
        source = _text(source_node)
        if source in state.held:
            raise GuardError(source_node, f"{source} is a guard; lock the value it holds instead")
        locker = state.locker_of(source)
        if locker is not None:
            raise GuardError(source_node, f"{source} is already locked by {locker.guard} (line {_line(locker.node)})")
        self.name(source_node, state, field_access=False, lent=True)
        if guard in state.held:
            raise GuardError(node, f"{guard} already holds {state.held[guard].source}")
        block = node.parent
        assert block is not None
        state.held[guard] = _Lock(guard, source, node, block)
        state.released.pop(guard, None)
        return state

    def release(self, node: Node, state: _State) -> _State:
        guard = _text(node.child_by_field_name("guard"))
        source = _text(_unwrap(node.child_by_field_name("source")))
        lock = state.held.get(guard)
        if lock is None:
            if guard in state.released:
                raise GuardError(node, f"{guard} was already released on line {_line(state.released[guard])}")
            raise GuardError(node, f"{guard} is not a guard; lock something first: `let {guard} <- {source}`")
        if lock.source != source:
            raise GuardError(node, f"{guard} holds {lock.source}, not {source}; release it with {_release_hint(lock)}")
        if not (lock.block.start_byte <= node.start_byte and node.end_byte <= lock.block.end_byte):
            raise GuardError(node, f"{guard} must be released in the scope it was locked in (line {_line(lock.node)})")
        del state.held[guard]
        state.released[guard] = node
        return state

    def field_assignment(self, node: Node, state: _State) -> None:
        target = node.child_by_field_name("target")
        root, indexes = _target(target)
        name = _text(root)
        if name not in state.held:
            self.not_held(root, name, state)
            locker = state.locker_of(name)
            if locker is not None:
                raise GuardError(
                    root,
                    f"{name} is locked by {locker.guard} (line {_line(locker.node)}); change it through {locker.guard}",
                )
            what = "items" if target.type == "get_expression" else "fields"
            raise GuardError(root, f"{what} change only through a guard: lock {name} first (`let guard <- {name}`)")
        for index in indexes:
            self.uses(index, state)
        self.uses(node.child_by_field_name("value"), state)

    def guard_assignment(self, node: Node, state: _State) -> None:
        target = node.child_by_field_name("guard")
        name = _text(target)
        lock = state.held.get(name)
        if lock is None:
            self.not_held(target, name, state)
            raise GuardError(target, f"{name} is not a guard; give a new value a name with `let {name} = ...`")
        if lock.source not in self.values:
            raise GuardError(target, f"{name} holds an object; change its fields ({name}.x = ...) instead")
        self.uses(node.child_by_field_name("value"), state)

    def return_(self, node: Node, state: _State) -> None:
        self.uses(node, state)
        if state.held:
            lock = next(iter(state.held.values()))
            msg = (
                f"returns while {lock.guard} still holds {lock.source} (locked on line {_line(lock.node)}); "
                f"release it first with {_release_hint(lock)}"
            )
            raise GuardError(node, msg)

    # -- control flow -------------------------------------------------------------

    def if_(self, node: Node, state: _State) -> _State | None:
        condition, body, *rest = _named(node)
        self.uses(condition, state)
        line = _line(node)
        paths = [(self.block(body, state.copy()), f"when the condition on line {line} is true")]
        if rest and rest[0].type == "if":
            paths.append((self.if_(rest[0], state.copy()), f"when it is false (the else-if on line {_line(rest[0])})"))
        elif rest:
            paths.append((self.block(rest[0], state.copy()), f"when the condition on line {line} is false"))
        else:
            paths.append((state.copy(), f"when the condition on line {line} is false (there is no else)"))
        return self.merge(paths, node)

    def forall(self, node: Node, state: _State) -> _State:
        _, iterated, body = _named(node)
        return self.loop(node, iterated, body, state)

    def while_(self, node: Node, state: _State) -> _State:
        condition, body = _named(node)
        return self.loop(node, condition, body, state)

    def loop(self, node: Node, head: Node, body: Node, state: _State) -> _State:
        """Check a loop, whose body runs zero times or many: it must release every guard it takes."""
        self.uses(head, state)
        after = self.block(body, state.copy())
        if after is not None:
            for guard, lock in state.held.items():
                if guard not in after.held:
                    msg = (
                        f"{guard} is released inside a loop, which may run zero times or many; release it after "
                        f"the loop, or lock and release {lock.source} inside each iteration"
                    )
                    raise GuardError(after.released.get(guard, node), msg)
        return state

    def match(self, node: Node, state: _State) -> _State | None:
        scrutinee, *arms = _named(node)
        self.uses(scrutinee, state)
        paths: list[tuple[_State | None, str]] = []
        has_default = False
        for arm in arms:
            condition, body = _named(arm)
            if _unwrap(condition).type == "default_var":
                has_default = True
            else:
                self.uses(condition, state)
            label = f"in the `{' '.join(_text(condition).split())}` arm (line {_line(arm)})"
            branch = state.copy()
            if body.type == "block_expression":
                paths.append((self.block(body, branch), label))
            else:
                paths.append((self.item(body, branch), label))
        if not has_default:
            paths.append((state.copy(), f"when no arm of the match on line {_line(node)} matches (there is no _)"))
        return self.merge(paths, node)

    @staticmethod
    def merge(paths: list[tuple[_State | None, str]], node: Node) -> _State | None:
        """Join the paths out of a branch: each must hold the same guards."""
        live = [(state, label) for state, label in paths if state is not None]
        if not live:
            return None
        held = set(live[0][0].held)
        for state, _ in live[1:]:
            if set(state.held) != held:
                guard = min(held.symmetric_difference(state.held))
                holding = [label for s, label in live if guard in s.held]
                releasing = [label for s, label in live if guard not in s.held]
                lock = next(s.held[guard] for s, _ in live if guard in s.held)
                msg = (
                    f"{guard} is released only on some paths: it still holds {lock.source} {' and '.join(holding)}, "
                    f"but is released {' and '.join(releasing)}. Release it on every path, or on none"
                )
                raise GuardError(node, msg)
        merged = live[0][0].copy()
        for state, _ in live[1:]:
            merged.released.update(state.released)
        return merged

    # -- names --------------------------------------------------------------------

    def uses(self, node: Node | None, state: _State) -> None:
        """Check every name ``node`` reads."""
        if node is None:
            return
        match node.type:
            case "local_function_definition":
                return  # a lambda is a function of its own, checked when it is lowered
            case "identifier":
                self.name(node, state, field_access=False, lent=self.is_argument(node))
                return
            case "get_expression":
                self.index_read(node, state)
                return
            case "child_annotation":
                self.chain(node, state)
                return
            case "function_call":
                self.arguments(node, state)
                self.pause(node, state)
                return
        for child in _named(node):
            self.uses(child, state)

    def index_read(self, node: Node, state: _State) -> None:
        """Check ``g[i]``, which reaches into what ``g`` holds, as ``g.x`` does."""
        owner, index = _named(node)
        inner = _unwrap(owner)
        if inner.type == "identifier":
            self.name(inner, state, field_access=True, lent=False)
        else:
            self.uses(inner, state)
        self.uses(index, state)

    def chain(self, node: Node, state: _State) -> None:
        """Check ``a.b.f(x)``: what it starts from, and its calls' arguments."""
        first, *rest = _named(node)
        if first.type == "get_expression":
            self.uses(first, state)
        elif self.lent_to(node):
            self.name(first, state, field_access=False, lent=True)  # `ctx.bump()` lends ctx
        else:
            self.name(first, state, field_access=bool(rest), lent=False)
        for part in rest:
            if part.type == "function_call":
                self.arguments(part, state)
        if rest and rest[-1].type == "function_call":
            self.pause(node, state)

    def lent_to(self, node: Node) -> bool:
        """Whether ``node`` calls a method that changes its object on a held name (``ctx.bump()``), lending it."""
        first, *rest = _named(node)
        if len(rest) != 1 or rest[0].type != "function_call":
            return False
        called = _named(rest[0])[0].child_by_field_name("function")
        return called is not None and self.lends(_text(first), _text(called))

    def pause(self, call: Node, state: _State) -> None:
        """Reject a call that can pause while a guard is held."""
        if not state.held or not self.pauses(call):
            return
        lock = next(iter(state.held.values()))
        callee = _text(call).split("(")[0]
        msg = (
            f"{lock.guard} holds {lock.source} across {callee}(...), which can pause: other code runs "
            f"meanwhile and could change it. Release it first ({_release_hint(lock)}), and lock again after"
        )
        raise GuardError(call, msg)

    def arguments(self, call: Node, state: _State) -> None:
        inner = _named(call)[0]
        for argument in _named(inner):
            if argument.type == "expression":
                self.uses(argument, state)
            elif argument.type == "named_argument":
                self.uses(argument.child_by_field_name("value"), state)

    @staticmethod
    def is_argument(identifier: Node) -> bool:
        """Whether ``identifier`` is a whole call argument: `draw(ctx)`."""
        parent = identifier.parent
        while parent is not None and parent.type in {"expression", "parenthesized_expression"}:
            parent = parent.parent
        return parent is not None and parent.type in {"user_function_call", "builtin_call", "named_argument"}

    def name(self, node: Node, state: _State, *, field_access: bool, lent: bool) -> None:
        name = _text(node)
        if name in state.held:
            if not (field_access or lent or state.held[name].source in self.values):
                msg = (
                    f"{name} is a guard: use it to reach fields ({name}.x) or items ({name}[i]), "
                    "or lend it to a call, not as a value"
                )
                raise GuardError(node, msg)
            return
        locker = state.locker_of(name)
        if locker is not None:
            msg = (
                f"{name} is locked by {locker.guard} (line {_line(locker.node)}); "
                f"use {locker.guard} until {_release_hint(locker)}"
            )
            raise GuardError(node, msg)
        self.not_held(node, name, state)
        if name in self.held and field_access:
            msg = f"{name} is a {self.held[name]}; lock it (`let guard <- {name}`) to reach its fields"
            raise GuardError(node, msg)
        if name in self.held and not lent and not (self.held[name] in _CELLS and self.is_whole_value(node)):
            msg = f"{name} is a {self.held[name]}: lend it to a call or lock it, rather than copy it"
            raise GuardError(node, msg)

    @staticmethod
    def is_whole_value(identifier: Node) -> bool:
        """Whether ``identifier`` is all of a ``let``'s value or a returned one: `let other = counter`."""
        parent = identifier.parent
        while parent is not None and parent.type in {"expression", "parenthesized_expression"}:
            parent = parent.parent
        returned = {"return_statement", "function_definition", "local_function_definition"}  # or a body
        return parent is not None and parent.type in {"local_assignment", *returned}

    @staticmethod
    def not_held(node: Node, name: str, state: _State) -> None:
        if name in state.released:
            msg = f"{name} was released on line {_line(state.released[name])}; lock again to use it"
            raise GuardError(node, msg)


def check(
    body: Node,
    held: dict[str, str],
    pauses: Callable[[Node], bool] = lambda _: False,
    values: frozenset[str] = frozenset(),
    lends: Callable[[str, str], bool] = lambda _name, _method: False,
) -> None:
    """Check the guards in a function ``body``; ``held`` maps its ``mem`` parameters and locals to their container.

    ``pauses`` tells whether a call (a ``function_call`` or ``child_annotation``) can pause;
    ``values`` names the held values that are not objects, whose guards read and write them whole;
    ``lends`` tells whether calling a method on a held name lends it (the method changes its object).

    Raises:
        GuardError: At the first rule broken.

    """
    checker = _Checker(held, pauses, set(values), lends)
    state = _State({}, {})
    if body.type == "block_expression":
        checker.block(body, state)
    else:
        checker.uses(body, state)


__all__ = ["GuardError", "check"]
