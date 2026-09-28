"""Check owned values, and plan where they are freed.

An owned value (``mem.Unique[str]``, e.g. from ``fmt.format``) has one owner: a
``let`` or a parameter. It is freed where its owner's scope ends, on every path
(and before each ``return``), unless ownership moves on first:

- ``let other = message`` moves it to ``other``;
- ``return message`` moves it to the caller;
- passing it to a ``mem.Unique[str]`` parameter moves it to the callee.

Passing it anywhere else (``stdio.puts(message)``) only lends it.

A ``mem.Shared`` or ``mem.Atomic`` local (a cell) is an owner too, released
where its scope ends, but ``let other = counter`` copies it: both are owners,
and the value lives until the last is released. ``return counter`` still
moves it to the caller, and a cell parameter is borrowed, never released.

The rules that keep this sound, all checked here at compile time:

- a moved value is not used again;
- a value is moved on every path through a branch, or on none, and never
  inside a loop, so where it is freed does not depend on the path taken;
- an owned result is always given an owner (bound, returned or moved), since
  nothing else would free it.

``check`` returns the plan the lowering follows: which owners to free at the
end of each block, and before each ``return``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from tree_sitter import Node


class OwnershipError(Exception):
    """An ownership rule is broken at ``node``."""

    def __init__(self, node: Node, message: str) -> None:
        super().__init__(message)
        self.node = node
        self.message = message


@dataclass(frozen=True)
class Oracle:
    """What the compiler knows about calls and types, which the check needs."""

    owned_call: Callable[[Node], bool]  # does this call expression return an owned value?
    moved_arguments: Callable[[Node], set[int]]  # which positional arguments of this call are moved?
    owned_type: Callable[[Node], bool]  # is this type node ``mem.Unique[str]``?
    cell_call: Callable[[Node], bool] = lambda _: False  # does this call return a mem.Shared or mem.Atomic?
    cell_type: Callable[[Node], bool] = lambda _: False  # is this type node a mem.Shared or mem.Atomic?


@dataclass
class Plan:
    """Where the lowering frees owned values: at the end of a block, before a return (by node id)."""

    at_end: dict[int, list[str]] = field(default_factory=dict)
    before_return: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class _State:
    alive: dict[str, tuple[Node, Node]]  # owner -> (where it became one, the block that frees it)
    moved: dict[str, Node]  # owner -> where it was moved

    def copy(self) -> "_State":
        return _State(dict(self.alive), dict(self.moved))


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


def _call_arguments(node: Node) -> list[Node] | None:
    """For a call expression (``f(x)``, ``a.f(x)``), return its positional arguments."""
    call = node
    if call.type == "child_annotation":
        call = _named(call)[-1]
    if call.type != "function_call":
        return None
    inner = _named(call)[0]
    return [argument for argument in _named(inner) if argument.type == "expression"]


class _Checker:
    def __init__(self, oracle: Oracle, cells: set[str]) -> None:
        self.oracle = oracle
        self.plan = Plan()
        self.cells = set(cells)  # names holding a mem.Shared or mem.Atomic: parameters, then owners

    # -- statements -------------------------------------------------------------

    def block(self, node: Node, state: _State) -> _State | None:
        current: _State | None = state
        for item in _named(node):
            if current is None:
                break
            current = self.item(item, current)
        if current is not None:
            for name, (_, owner_block) in list(current.alive.items()):
                if owner_block == node:
                    self.plan.at_end.setdefault(node.id, []).append(name)
                    del current.alive[name]
        return current

    def item(self, item: Node, state: _State) -> _State | None:
        match item.type:
            case "local_assignment":
                self.local_assignment(item, state)
            case "return_statement":
                return self.return_(item, state)
            case "expression":
                return self.statement(_unwrap(item), state)
            case "lock" | "release" | "field_assignment" | "guard_assignment":
                for child in _named(item):
                    self.expression(child, state)
        return state

    def local_assignment(self, item: Node, state: _State) -> None:
        parts = _named(item)
        name, value = _text(parts[0]), _unwrap(parts[-1])
        declared = item.child_by_field_name("type")
        if name in state.alive:
            where = _line(state.alive[name][0])
            raise OwnershipError(parts[0], f"{name} already owns a value (line {where}); give the new one another name")
        block = item.parent
        assert block is not None
        if value.type == "identifier" and _text(value) in self.cells:
            self.name(value, state)  # another owner of the same cell
            state.alive[name] = (item, block)
            self.cells.add(name)
        elif value.type == "identifier" and _text(value) in state.alive:
            self.move(value, state, item)
            state.alive[name] = (item, block)
        elif self.oracle.owned_call(value):
            self.expression(value, state, owned=True)
            state.alive[name] = (item, block)
            if self.oracle.cell_call(value):
                self.cells.add(name)
        elif declared is not None and self.oracle.owned_type(declared):
            msg = f"only an owned value can start a {_text(declared)}: a call that returns one, like fmt.format(...)"
            raise OwnershipError(value, msg)
        elif declared is not None and self.oracle.cell_type(declared):
            self.expression(value, state)  # the value moves into a new cell
            state.alive[name] = (item, block)
            self.cells.add(name)
        else:
            self.expression(value, state)
        state.moved.pop(name, None)

    def return_(self, node: Node, state: _State) -> None:
        values = _named(node)
        value = _unwrap(values[0]) if values else None
        if value is not None and value.type == "identifier" and _text(value) in state.alive:
            self.move(value, state, node)
        elif value is not None:
            self.expression(value, state, owned=self.oracle.owned_call(value))
        if state.alive:
            self.plan.before_return[node.id] = list(state.alive)

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
        self.expression(node, state)
        return state

    # -- control flow -----------------------------------------------------------

    def if_(self, node: Node, state: _State) -> _State | None:
        condition, body, *rest = _named(node)
        self.expression(condition, state)
        paths = [(self.block(body, state.copy()), f"when the condition on line {_line(node)} is true")]
        if rest and rest[0].type == "if":
            paths.append((self.if_(rest[0], state.copy()), f"in the else-if on line {_line(rest[0])}"))
        elif rest:
            paths.append((self.block(rest[0], state.copy()), f"when the condition on line {_line(node)} is false"))
        else:
            paths.append((state.copy(), f"when the condition on line {_line(node)} is false (there is no else)"))
        return self.merge(paths, node)

    def forall(self, node: Node, state: _State) -> _State:
        _, iterated, body = _named(node)
        return self.loop(node, iterated, body, state)

    def while_(self, node: Node, state: _State) -> _State:
        condition, body = _named(node)
        return self.loop(node, condition, body, state)

    def loop(self, node: Node, head: Node, body: Node, state: _State) -> _State:
        """Check a loop, whose body runs zero times or many: nothing alive before it may move inside it."""
        self.expression(head, state)
        after = self.block(body, state.copy())
        if after is not None:
            for name in state.alive:
                if name not in after.alive:
                    msg = (
                        f"{name} is moved inside a loop, which may run zero times or many; "
                        "move it after the loop, or give it an owner inside each iteration"
                    )
                    raise OwnershipError(after.moved.get(name, node), msg)
        return state

    def match(self, node: Node, state: _State) -> _State | None:
        scrutinee, *arms = _named(node)
        self.expression(scrutinee, state)
        paths: list[tuple[_State | None, str]] = []
        has_default = False
        for arm in arms:
            condition, body = _named(arm)
            if _unwrap(condition).type == "default_var":
                has_default = True
            else:
                self.expression(condition, state)
            label = f"in the `{' '.join(_text(condition).split())}` arm (line {_line(arm)})"
            branch = state.copy()
            paths.append(
                (self.block(body, branch) if body.type == "block_expression" else self.item(body, branch), label)
            )
        if not has_default:
            paths.append((state.copy(), f"when no arm of the match on line {_line(node)} matches"))
        return self.merge(paths, node)

    @staticmethod
    def merge(paths: list[tuple[_State | None, str]], node: Node) -> _State | None:
        live = [(state, label) for state, label in paths if state is not None]
        if not live:
            return None
        alive = set(live[0][0].alive)
        for state, _ in live[1:]:
            if set(state.alive) != alive:
                name = min(alive.symmetric_difference(state.alive))
                moving = [label for s, label in live if name not in s.alive]
                keeping = [label for s, label in live if name in s.alive]
                msg = (
                    f"{name} is moved only on some paths: {' and '.join(moving)}, but not "
                    f"{' and '.join(keeping)}. Move it on every path, or on none"
                )
                raise OwnershipError(node, msg)
        merged = live[0][0].copy()
        for state, _ in live[1:]:
            merged.moved.update(state.moved)
        return merged

    # -- expressions ------------------------------------------------------------

    def move(self, value: Node, state: _State, where: Node) -> None:
        name = _text(value)
        del state.alive[name]
        state.moved[name] = where

    def expression(self, node: Node | None, state: _State, *, owned: bool = False) -> None:
        """Check the names ``node`` reads; ``owned``: it may be a call returning an owned value."""
        if node is None:
            return
        node = _unwrap(node)
        if node.type == "local_function_definition":
            return  # a lambda is a function of its own, checked when it is lowered
        arguments = _call_arguments(node)
        if arguments is not None:
            self.call(node, arguments, state, owned=owned)
        elif node.type == "identifier":
            self.name(node, state)
        elif node.type == "child_annotation":
            self.name(_named(node)[0], state)
            for part in _named(node)[1:]:
                if part.type == "function_call":
                    for argument in _named(_named(part)[0]):
                        self.expression(argument, state)
        else:
            for child in _named(node):
                self.expression(child, state)

    def call(self, node: Node, arguments: list[Node], state: _State, *, owned: bool) -> None:
        """Check a call: its owned result must have an owner, and its moved arguments move."""
        if not owned and self.oracle.owned_call(node):
            cell = self.oracle.cell_call(node)
            what, name = ("a mem.Shared or mem.Atomic", "value") if cell else ("an owned string", "text")
            msg = (
                f"{_text(node).split('(')[0]}(...) returns {what}, which nothing would free; "
                f"give it an owner first (`let {name} = ...`), then use that"
            )
            raise OwnershipError(node, msg)
        moved = self.oracle.moved_arguments(node)
        for index, argument in enumerate(arguments):
            inner = _unwrap(argument)
            if index in moved and inner.type == "identifier" and _text(inner) in state.alive:
                self.move(inner, state, argument)
            else:
                self.expression(argument, state, owned=index in moved)
        if node.type == "child_annotation":
            self.name(_named(node)[0], state)

    @staticmethod
    def name(node: Node, state: _State) -> None:
        name = _text(node)
        if name in state.moved and name not in state.alive:
            raise OwnershipError(node, f"{name} was moved on line {_line(state.moved[name])}, so it is no longer here")


def check(body: Node, owned_parameters: set[str], oracle: Oracle, cell_parameters: set[str] = frozenset()) -> Plan:
    """Check the owned values in a function ``body``; return where the lowering frees them.

    ``cell_parameters`` are the ``mem.Shared`` and ``mem.Atomic`` parameters,
    which ``let`` copies rather than moves.

    Raises:
        OwnershipError: At the first rule broken.

    """
    checker = _Checker(oracle, cell_parameters)
    if body.type != "block_expression":
        if owned_parameters:
            raise OwnershipError(body, "a function taking a mem.Unique[str] needs a block body, to free it")
        checker.expression(body, _State({}, {}), owned=oracle.owned_call(_unwrap(body)))
        return checker.plan
    state = _State(dict.fromkeys(sorted(owned_parameters), (body, body)), {})
    checker.block(body, state)
    return checker.plan


__all__ = ["Oracle", "OwnershipError", "Plan", "check"]
