"""Check owned values, and plan where they are freed.

An owned value has one owner, a ``let`` or a parameter, and is freed where
its owner's scope ends, on every path (and before each ``return``), unless
ownership moves on first. Owned values are:

- strings from ``fmt.format`` and ``json.encode`` (``mem.Unique[str]``);
- lists (``T[]``), and records and objects holding one (at any depth).

Ownership moves:

- ``let other = message`` moves it to ``other``;
- ``return message`` moves it to the caller;
- passing it to a ``mem.Unique[str]`` parameter moves it to the callee;
- putting it in a new list, record or object that is kept (``let row =
  #{scores: scores}``, ``#[...rows, row]``) moves it into that value.

Passing it anywhere else (``stdio.puts(message)``, ``total(xs)``) only lends
it. A new list or record that is only lent (``http.json(ctx, 200, #{users:
users})``) borrows what it holds, and is freed after the statement.

Reading an owned value out of another (``rows[0]``, ``team.members``, the
``row`` of ``forall row in rows``) gives a view of it: usable, but not an
owner, so it cannot be kept (returned, or put in a list or record); keep a
copy (``#[...team.members]``) instead. While a view is alive (to the end of
its block), what it reads from is not moved or replaced.

``let xs = #[...xs, x]`` gives ``xs`` a new value; the old one is freed
first, or, when the new list starts with all of the old one, grown in place.

A ``mem.Shared`` or ``mem.Atomic`` local (a cell) is an owner too, released
where its scope ends, but ``let other = counter`` copies it: both are owners,
and the value lives until the last is released. ``return counter`` still
moves it to the caller, and a cell parameter is borrowed, never released.

The rules that keep this sound, all checked here at compile time:

- a moved value is not used again;
- a value is moved on every path through a branch, or on none, and never
  inside a loop, so where it is freed does not depend on the path taken;
- a cell is always given an owner (bound, returned or
  moved), since nothing else would free it.

``check`` returns the plan the lowering follows: which owners to free at the
end of each block, before each ``return``, and before each ``let`` that
replaces one.
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

    owned_call: Callable[[Node], bool]  # does this call expression return an owned string or cell?
    moved_arguments: Callable[[Node], set[int]]  # which positional arguments of this call are moved?
    owned_type: Callable[[Node], bool]  # is this type node ``mem.Unique[str]``?
    cell_call: Callable[[Node], bool] = lambda _: False  # does this call return a mem.Shared or mem.Atomic?
    cell_type: Callable[[Node], bool] = lambda _: False  # is this type node a mem.Shared or mem.Atomic?
    owned_value: Callable[[Node], bool] = lambda _: False  # a list, owned string, or record holding one?
    copy: Callable[[Node], str] = lambda _: ""  # how to write a copy of this expression, if it can be copied
    constructs: Callable[[Node], bool] = lambda _: False  # does this call build an object (its arguments move in)?
    changes: Callable[[Node], bool] = lambda _: False  # does this call (`a.f(x)`) change the object it is called on?
    # Called at each `let` (and `forall`) before what follows is checked, so that
    # the types of later expressions, which the checks above ask about, are known.
    bind: Callable[[str, Node], None] = lambda _name, _node: None


@dataclass
class Plan:
    """Where the lowering frees owned values: at the end of a block, before a return (by node id)."""

    at_end: dict[int, list[str]] = field(default_factory=dict)
    before_return: dict[int, list[str]] = field(default_factory=dict)
    # A `let` (by node id) giving a new value to an owner whose old value is still alive:
    # the lowering frees the old one (or grows it in place, for `let xs = #[...xs, x]`).
    replaced: dict[int, str] = field(default_factory=dict)


@dataclass
class _State:
    alive: dict[str, tuple[Node, Node]]  # owner -> (where it became one, the block that frees it)
    moved: dict[str, Node]  # owner -> where it was moved
    views: dict[str, tuple[str, Node, Node]] = field(default_factory=dict)  # view -> (its owner, where, block)

    def copy(self) -> "_State":
        return _State(dict(self.alive), dict(self.moved), dict(self.views))


def _text(node: Node) -> str:
    return (node.text or b"").decode()


def _named(node: Node) -> list[Node]:
    return [child for child in node.named_children if child.type != "comment"]


def _unwrap(node: Node) -> Node:
    while node.type in {"expression", "condition", "parenthesized_expression", "getter_owner"}:
        node = _named(node)[0]
    return node


def _awaited(node: Node) -> Node:
    """``await f(x)`` -> ``f(x)``; other expressions are themselves."""
    node = _unwrap(node)
    if node.type == "await_expression":
        value = node.child_by_field_name("value")
        return _unwrap(value) if value is not None else node
    return node


def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _call_parts(node: Node) -> Node | None:
    """For a call expression (``f(x)``, ``a.f(x)``), return the ``user_function_call`` with its arguments."""
    call = node
    if call.type == "child_annotation":
        call = _named(call)[-1]
    if call.type != "function_call":
        return None
    inner = _named(call)[0]
    return inner if inner.type == "user_function_call" else None


def _call_arguments(node: Node) -> list[Node] | None:
    """For a call expression (``f(x)``, ``a.f(x)``), return its positional arguments."""
    call = node
    if call.type == "child_annotation":
        call = _named(call)[-1]
    if call.type != "function_call":
        return None
    inner = _named(call)[0]
    return [argument for argument in _named(inner) if argument.type == "expression"]


def _named_values(node: Node) -> list[Node]:
    """For a call expression, the values of its named arguments (``width: 800``)."""
    inner = _call_parts(node)
    if inner is None:
        return []
    values = [argument.child_by_field_name("value") for argument in _named(inner) if argument.type == "named_argument"]
    return [value for value in values if value is not None]


def _writes_item(target: Node) -> bool:
    """Whether an assignment target goes through a list's item: ``g[0] = x``, ``g.users[i].name = x``."""
    node = _unwrap(target)
    if node.type == "get_expression":
        return True
    return node.type == "child_annotation" and _writes_item(_named(node)[0])


def _root(node: Node) -> str | None:
    """Return the local a read starts from: ``rows`` for ``rows[0].scores``; ``None`` for anything else."""
    node = _unwrap(node)
    if node.type == "identifier":
        return _text(node)
    if node.type == "get_expression":
        return _root(_named(node)[0])
    if node.type == "child_annotation":
        parts = _named(node)
        if all(part.type == "simple_identifier" for part in parts[1:]):
            return _text(parts[0]) if parts[0].type == "simple_identifier" else _root(parts[0])
    return None


class _Checker:
    def __init__(self, oracle: Oracle, cells: set[str], parameters: set[str]) -> None:
        self.oracle = oracle
        self.plan = Plan()
        self.cells = set(cells)  # names holding a mem.Shared or mem.Atomic: parameters, then owners
        self.parameters = parameters
        self.guards: dict[str, str] = {}  # guard -> the local it locks
        self.lent: set[str] = set()  # the mem.Weak parameters

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
            for name, (_, _, view_block) in list(current.views.items()):
                if view_block == node:
                    del current.views[name]
        return current

    def item(self, item: Node, state: _State) -> _State | None:
        match item.type:
            case "local_assignment":
                self.local_assignment(item, state)
            case "return_statement":
                return self.return_(item, state)
            case "expression":
                return self.statement(_unwrap(item), state)
            case "field_assignment":
                self.field_assignment(item, state)
            case "lock" | "release" | "guard_assignment":
                for child in _named(item):
                    self.expression(child, state)
                self.guard(item)
        return state

    def local_assignment(self, item: Node, state: _State) -> None:
        parts = _named(item)
        name, value = _text(parts[0]), _unwrap(parts[-1])
        declared = item.child_by_field_name("type")
        self.oracle.bind(name, item)
        block = item.parent
        assert block is not None
        if name in state.alive and name not in self.cells:
            self.replace(name, value, item, state)
            return
        if name in state.alive:
            where = _line(state.alive[name][0])
            raise OwnershipError(parts[0], f"{name} already owns a value (line {where}); give the new one another name")
        state.views.pop(name, None)
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
            # The value moves into a new cell.
            if self.oracle.owned_value(value):
                self.keep(value, state)
            else:
                self.expression(value, state)
            state.alive[name] = (item, block)
            self.cells.add(name)
        else:
            self.value(name, value, item, state)
        state.moved.pop(name, None)

    def value(self, name: str, value: Node, item: Node, state: _State) -> None:
        """``let name = value`` for the rest: a new list or record it owns, a view, or a plain value."""
        block = item.parent
        assert block is not None
        if self.fresh(value):
            self.keep(value, state)
            state.alive[name] = (item, block)
        elif self.oracle.owned_value(value):
            self.expression(value, state)  # a view of what it reads
            self.view(name, value, item, block, state)
        else:
            self.expression(value, state)

    def replace(self, name: str, value: Node, item: Node, state: _State) -> None:
        """``let name = value`` where ``name`` owns a value already: the new one replaces it."""
        self.unused(name, state, item, "replaced")
        entry = state.alive[name]
        if value.type == "identifier" and _text(value) in state.alive and _text(value) != name:
            self.move(value, state, item)
        elif self.fresh(value):
            self.keep(value, state)
        elif self.oracle.owned_call(value):
            self.expression(value, state, owned=True)
        else:
            where = _line(entry[0])
            msg = f"{name} already owns a value (line {where}); give the new one another name"
            raise OwnershipError(_named(item)[0], msg)
        if name in state.alive:
            self.plan.replaced[item.id] = name  # its old value is freed (or grown) first
        state.alive[name] = entry
        state.moved.pop(name, None)

    def owner(self, name: str | None, state: _State) -> str | None:
        """Return what a name reads through: a guard's local, cell or lent value; ``None`` if nothing owns it here.

        A cell's value is its cells', and a ``mem.Weak``'s its caller's: a guard on
        one may replace a list in it (the old one is freed).
        """
        name = self.guards.get(name, name) if name is not None else None
        if name is not None and (name in self.cells or name in self.lent or name in state.alive):
            return name
        return None

    def guard(self, item: Node) -> None:
        """Follow which local each guard locks, so that a field changed through it changes that local's."""
        guard, source = item.child_by_field_name("guard"), item.child_by_field_name("source")
        if item.type == "lock" and guard is not None and source is not None:
            self.guards[_text(guard)] = _root(source) or ""
        elif item.type == "release" and guard is not None:
            self.guards.pop(_text(guard), None)

    def view(self, name: str, value: Node, item: Node, block: Node, state: _State) -> None:
        """Make ``name`` a view of what ``value`` reads, which is then not moved while the view is alive."""
        root = _root(value)
        while root is not None and root in state.views:
            root = state.views[root][0]
        root = self.guards.get(root, root) if root is not None else None
        if root is not None and (root in state.alive or root in self.cells or root in self.lent):
            state.views[name] = (root, item, block)

    def field_assignment(self, item: Node, state: _State) -> None:
        target, value = item.child_by_field_name("target"), item.child_by_field_name("value")
        if target is None or value is None:
            return
        owned = self.oracle.owned_value(value)
        if owned or _writes_item(target):
            root = self.owner(_root(target), state)
            if root is None:
                raise OwnershipError(target, self.not_owner(target, state, replacing=owned))
        if owned:
            assert root is not None
            self.unused(root, state, item, "changed")
            self.keep(value, state)
            self.name(target, state)
        else:
            self.expression(target, state)
            self.expression(value, state)

    def not_owner(self, target: Node, state: _State, *, replacing: bool) -> str:
        """Say why a change through ``target`` is not this function's to make."""
        name = _root(target) or _text(target)
        source = self.guards.get(name, name)
        if source in state.views:
            what = f"{source} is a view of {state.views[source][0]}"
            return f"{what}, so its items cannot change; change them through {state.views[source][0]}"
        if source in self.parameters:
            return (
                f"{source} is lent to this function, so the items of its lists cannot change here "
                f"(they are its caller's); take it as a mem.Weak to change it, or change a copy"
            )
        if replacing:
            return f"{name} is not an owner here, so a list in it cannot be replaced"
        return f"{name} is not an owner here, so the items of its lists cannot change"

    def return_(self, node: Node, state: _State) -> None:
        values = _named(node)
        value = _unwrap(values[0]) if values else None
        if value is not None and value.type == "identifier" and _text(value) in state.alive:
            self.move(value, state, node)
        elif value is not None and self.oracle.owned_value(value):
            self.keep(value, state)
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
        identifier, iterated, body = _named(node)
        self.oracle.bind(_text(identifier), node)
        # The loop reads the list it goes through, which is not moved or replaced meanwhile.
        inside = state.copy()
        self.view(_text(identifier), iterated, node, body, inside)
        return self.loop(node, iterated, body, state, inside)

    def while_(self, node: Node, state: _State) -> _State:
        condition, body = _named(node)
        return self.loop(node, condition, body, state, state.copy())

    def loop(self, node: Node, head: Node, body: Node, state: _State, inside: _State) -> _State:
        """Check a loop, whose body runs zero times or many: nothing alive before it may move inside it."""
        self.expression(head, state)
        after = self.block(body, inside)
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

    def fresh(self, node: Node) -> bool:
        """Whether an expression makes a new list, or a new record or object holding one."""
        core = _awaited(node)
        if core.type == "literal":
            inner = _named(core)[0]
            return inner.type == "list" or (inner.type == "record" and self.oracle.owned_value(core))
        if core.type == "get_expression" and _named(core)[1].type in {"spread_between", "rest_of", "spread_action"}:
            return True  # a slice: a new list
        # `await f()` has the type of what it waits for.
        return _call_arguments(core) is not None and self.oracle.owned_value(_unwrap(node))

    def keep(self, node: Node, state: _State) -> None:
        """Check a list, record or object that is kept (bound, returned, or put in another): what it holds moves in."""
        core = _awaited(node)
        if core.type == "identifier" and _text(core) in state.alive:
            self.move(core, state, core)
            return
        if core.type == "literal" and _named(core)[0].type in {"list", "record"}:
            self.literal(_named(core)[0], state)
            return
        if core.type == "get_expression":  # a slice copies what it reads
            self.expression(core, state)
            return
        arguments = _call_arguments(core)
        if arguments is not None and self.oracle.constructs(core):
            for argument in [*arguments, *_named_values(core)]:
                self.part(argument, state)
            return
        if arguments is not None:
            self.expression(node, state, owned=True)  # a call making a new one; its arguments are lent
            return
        raise OwnershipError(core, self.borrowed(core, state))

    def literal(self, inner: Node, state: _State) -> None:
        """Check a ``#[...]`` or ``#{...}`` that is kept: its items or fields move in."""
        if inner.type == "record":
            values = [record_field.child_by_field_name("value") for record_field in _named(inner)]
            for value in values:
                if value is not None:
                    self.part(value, state)
            return
        for item in _named(inner):
            if item.type == "expression":
                self.part(item, state)
            else:
                for child in _named(item):  # `...xs` copies xs; `#[a...b]` counts
                    self.expression(child, state)

    def part(self, node: Node, state: _State) -> None:
        """Check a value put in a list, record or object that is kept."""
        inner = _unwrap(node)
        if self.oracle.owned_value(inner):
            self.keep(inner, state)
            return
        if inner.type == "identifier" and _text(inner) in self.cells:
            msg = f"{_text(inner)} is a mem.Shared or mem.Atomic, which cannot be kept in a list, record or object yet"
            raise OwnershipError(inner, msg)
        self.expression(inner, state)

    def borrowed(self, node: Node, state: _State) -> str:
        """Explain why ``node``, a list or a record holding one, cannot be kept here, and what to do."""
        text = " ".join(_text(node).split())
        root = _root(node)
        if root is not None and root in state.moved and root not in state.alive:
            return f"{root} was moved on line {_line(state.moved[root])}, so it is no longer here"
        if node.type == "identifier" and root in self.parameters:
            source = f"{text} is a parameter, lent by the caller"
        elif node.type == "identifier" and root in state.views:
            source = f"{text} reads from {state.views[root][0]}"
        elif root is not None and root != text:
            source = f"{text} is part of {root}"
        else:
            source = f"{text} is not an owner"
        copy = self.oracle.copy(node)
        if copy:
            return f"{source}, so it cannot be kept here; keep a copy: {copy}"
        return f"{source}, so it cannot be kept here (a record or object holding a list cannot be copied yet)"

    def unused(self, name: str, state: _State, where: Node, verb: str) -> None:
        """Reject moving or replacing ``name`` while a view of it is alive."""
        for view, (owner, since, _) in state.views.items():
            if owner == name:
                msg = (
                    f"{name} is read by {view} (line {_line(since)}) until the end of its block, "
                    f"so it cannot be {verb} here"
                )
                raise OwnershipError(where, msg)

    def move(self, value: Node, state: _State, where: Node) -> None:
        name = _text(value)
        self.unused(name, state, where, "moved")
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
            self.chain(node, state)
        else:
            for child in _named(node):
                self.expression(child, state)

    def chain(self, node: Node, state: _State) -> None:
        """Check ``a.b.f(x)`` or ``users[0].name``: what it starts from, and the arguments of its calls."""
        first = _named(node)[0]
        if first.type == "get_expression":
            self.expression(first, state)
        else:
            self.name(first, state)
        for part in _named(node)[1:]:
            if part.type == "function_call":
                for argument in _named(_named(part)[0]):
                    self.expression(argument, state)

    def call(self, node: Node, arguments: list[Node], state: _State, *, owned: bool) -> None:
        """Check a call: its owned result must have an owner, and its moved arguments move."""
        if not owned and self.oracle.cell_call(node):
            msg = (
                f"{_text(node).split('(')[0]}(...) returns a mem.Shared or mem.Atomic, which nothing would release; "
                "give it an owner first (`let value = ...`), then use that"
            )
            raise OwnershipError(node, msg)
        moved = self.oracle.moved_arguments(node)
        for index, argument in enumerate(arguments):
            inner = _unwrap(argument)
            if index in moved and inner.type == "identifier" and _text(inner) in state.alive:
                self.move(inner, state, argument)
            elif index in moved and self.oracle.owned_value(inner) and not self.oracle.owned_call(inner):
                raise OwnershipError(inner, self.borrowed(inner, state))  # given away, but not ours to give
            else:
                self.expression(argument, state, owned=index in moved)
        for value in _named_values(node):
            self.expression(value, state)
        if node.type == "child_annotation":
            self.name(_named(node)[0], state)
            if self.oracle.changes(node):
                self.changed(node, state)

    def changed(self, node: Node, state: _State) -> None:
        """Check that the object ``node`` calls a changing method on (which may replace its lists) is ours."""
        first = _named(node)[0]
        root = _text(first) if first.type == "simple_identifier" else _root(first)
        owner = self.owner(root, state)
        source = self.guards.get(root, root) if root is not None else None
        if owner is not None:
            self.unused(owner, state, node, "changed")
            return
        call = _named(_named(node)[-1])[0]
        method = _text(call.child_by_field_name("function") or call)
        if source == "super":
            msg = f"{method} changes its object, but this method reads super as a copy; lock super here to change it"
            raise OwnershipError(first, msg)
        if source in state.views:
            msg = (
                f"{method} changes its object, but {source} is a view of {state.views[source][0]} (a copy of an item); "
                f"change it through a guard on {state.views[source][0]}, like g[i].{method}()"
            )
            raise OwnershipError(first, msg)
        if source in self.parameters:
            msg = (
                f"{method} changes its object, but {source} is a copy given to this function, so its caller would "
                f"not see the change; take it as a mem.Weak to change the caller's"
            )
            raise OwnershipError(first, msg)

    @staticmethod
    def name(node: Node, state: _State) -> None:
        name = _root(node) or _text(node) if node.type == "child_annotation" else _text(node)
        if name in state.moved and name not in state.alive:
            raise OwnershipError(node, f"{name} was moved on line {_line(state.moved[name])}, so it is no longer here")


def check(
    body: Node,
    owned_parameters: set[str],
    oracle: Oracle,
    cell_parameters: set[str] = frozenset(),
    parameters: dict[str, bool] | None = None,
) -> Plan:
    """Check the owned values in a function ``body``; return where the lowering frees them.

    ``cell_parameters`` are the ``mem.Shared`` and ``mem.Atomic`` parameters,
    which ``let`` copies rather than moves; ``parameters`` are all of them
    (lent by the caller, for messages), each with whether it is a ``mem.Weak``.

    Raises:
        OwnershipError: At the first rule broken.

    """
    parameters = parameters or {}
    checker = _Checker(oracle, cell_parameters, set(parameters))
    checker.lent = {name for name, weak in parameters.items() if weak}
    if body.type != "block_expression":
        if owned_parameters:
            raise OwnershipError(body, "a function taking a mem.Unique[str] needs a block body, to free it")
        state = _State({}, {})
        if oracle.owned_value(_unwrap(body)):
            checker.keep(body, state)
        else:
            checker.expression(body, state, owned=oracle.owned_call(_unwrap(body)))
        return checker.plan
    state = _State(dict.fromkeys(sorted(owned_parameters), (body, body)), {})
    checker.block(body, state)
    return checker.plan


__all__ = ["Oracle", "OwnershipError", "Plan", "check"]
