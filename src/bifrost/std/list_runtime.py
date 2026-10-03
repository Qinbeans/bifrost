"""What lists compile to: items on the heap, after a header with their count.

A list value is a pointer to its first item, right after a header::

    [length: i64][capacity: i64][item 0][item 1] ...

so it passes as is wherever C takes an array (``T *``). Items are any type:
numbers, strings, records, objects, or other lists. A list has one owner,
which frees it (see ``bifrost.ownership``); freeing a list of lists, or of
records holding lists, frees those too (the drop functions of
``bifrost.lowering.owned``).

``capacity`` is how many items fit before the list must grow: appending to a
list whose old value is not used again (``let xs = #[...xs, x]``) grows it in
place, doubling its capacity when full, so that is amortized O(1).

These functions are compiled once, into their own module, and linked into any
program that uses a list.
"""

from mlir_python.lang import Module, Ptr, cstr, i32, i64, ptr, u64

runtime = Module("bifrost_list")

HEADER = 16  # bytes before the first item
MINIMUM = 4  # the smallest capacity a growing list takes


@runtime.extern(name="malloc")
def c_malloc(size: u64) -> ptr: ...


@runtime.extern(name="realloc")
def c_realloc(pointer: ptr, size: u64) -> ptr: ...


@runtime.extern(name="free")
def c_free(pointer: ptr) -> None: ...


@runtime.extern(name="memcpy")
def c_memcpy(target: ptr, source: ptr, size: u64) -> ptr: ...


@runtime.extern(name="strlen")
def c_strlen(text: cstr) -> u64: ...


@runtime.extern(name="strcmp")
def c_strcmp(first: cstr, second: cstr) -> i32: ...


@runtime.extern(name="write")
def c_write(descriptor: i32, data: cstr, size: u64) -> i64: ...


@runtime.extern(name="abort")
def c_abort() -> None: ...


@runtime.function
def bifrost_list_fail(message: cstr) -> None:
    """Write ``message`` to stderr (unbuffered, so it is never lost) and stop the program."""
    c_write(2, message, c_strlen(message))
    c_abort()


@runtime.function
def bifrost_list_new(length: i64, size: i64) -> ptr:
    """Make a list of ``length`` items of ``size`` bytes each, not yet written; return its first item's address."""
    capacity = max(length, 1)
    header = Ptr[i64](c_malloc(u64(HEADER + capacity * size)))
    header[0] = length
    header[1] = capacity
    return header + 2


@runtime.function
def bifrost_list_length(items: ptr) -> i64:
    return (Ptr[i64](items) - 2)[0]


@runtime.function
def bifrost_list_set_length(items: ptr, length: i64) -> None:
    """Keep the first ``length`` items written (at most its capacity), as ``filter`` does."""
    (Ptr[i64](items) - 2)[0] = length


@runtime.function
def bifrost_string_compare(first: cstr, second: cstr) -> i32:
    """Order two strings as C's ``strcmp`` does: negative, zero (equal), or positive."""
    return c_strcmp(first, second)


@runtime.function
def bifrost_list_grow(items: ptr, more: i64, size: i64) -> ptr:
    """Make room for ``more`` items at the end of the list (in place when they fit); return its items.

    The list's length grows by ``more``; the new items are not yet written.
    When full, its capacity doubles (at least), so appending one at a time
    costs O(1) on average. The list may move: use the address returned.
    """
    header = Ptr[i64](items) - 2
    length = header[0] + more
    capacity = header[1]
    if length > capacity:
        capacity = max(capacity * 2, length, MINIMUM)
        header = Ptr[i64](c_realloc(header, u64(HEADER + capacity * size)))
        header[1] = capacity
    header[0] = length
    return header + 2


@runtime.function
def bifrost_list_copy(target: ptr, source: ptr, count: i64, size: i64) -> None:
    """Copy ``count`` items of ``size`` bytes from ``source`` to ``target`` (as they are: a shallow copy)."""
    if count > 0:
        c_memcpy(target, source, u64(count * size))


@runtime.function
def bifrost_list_index(items: ptr, index: i64, message: cstr) -> i64:
    """Return where ``xs[index]`` is: from the start, or from the end if negative; out of range stops the program."""
    length = (Ptr[i64](items) - 2)[0]
    position = index
    if position < 0:
        position = position + length
    if position < 0 or position >= length:
        bifrost_list_fail(message)
    return position


@runtime.function
def bifrost_list_bound(items: ptr, index: i64, given: i64, fallback: i64) -> i64:
    """Return where a slice ``xs[a...b]`` starts or ends, as Python does.

    Negative counts from the end, and anything past either end stops there.
    ``given`` is 0 when the bound is left out (``xs[a...]``).
    """
    length = (Ptr[i64](items) - 2)[0]
    if given == 0:
        return fallback
    position = index
    if position < 0:
        position = position + length
    return min(max(position, 0), length)


@runtime.function
def bifrost_string_copy(text: cstr) -> cstr:
    """Return a copy of ``text`` that its holder owns (what spreading or slicing a list of owned strings makes)."""
    size = c_strlen(text) + u64(1)
    copied = c_malloc(size)
    c_memcpy(copied, text, size)
    return cstr(copied)


@runtime.function
def bifrost_string_free(text: cstr) -> None:
    c_free(text)


@runtime.function
def bifrost_env_new(size: i64) -> ptr:
    """Allocate what a closure captured (see ``bifrost.owned.closure_of``); out of memory stops the program."""
    found = c_malloc(u64(size))
    if found == ptr(0):
        bifrost_list_fail("out of memory\n")
    return found


@runtime.function
def bifrost_env_free(env: ptr) -> None:
    """Free what a closure captured (what its captures own is freed first, by its drop function)."""
    c_free(env)


@runtime.function
def bifrost_list_free(items: ptr) -> None:
    """Free the list itself (what its items own is freed first, by its drop function)."""
    c_free(Ptr[i64](items) - 2)
