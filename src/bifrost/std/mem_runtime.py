"""What ``mem.Shared`` and ``mem.Atomic`` compile to: cells with a header.

A ``mem.Shared[T]`` or ``mem.Atomic[T]`` value is a pointer to its ``T``, on
the heap, right after a header that counts its owners::

    mem.Shared:   [count: i64][locked: i64][T ...]
    mem.Atomic:   [mutex: 64 bytes][count: i64][unused: i64][T ...]

The count is always the ``i64`` 16 bytes before the value, so the pointer
passes as is wherever a ``mem.Weak[T]`` or a C ``void *`` goes. A
``mem.Shared`` is used by one thread: its count is a plain integer, and its
lock only catches a second guard taken while the first is held. A
``mem.Atomic`` may be used by several: its count changes atomically, and its
guard locks a real mutex (``pthread_mutex_t``, at most 64 bytes on the
platforms Bifrost targets).

These functions are compiled once, into their own module, and linked into any
program that uses a cell.
"""

from mlir_python.lang import Module, Ptr, atomic_add, cstr, i32, i64, ptr, u8, u64

runtime = Module("bifrost_mem")

SHARED_HEADER = 16  # bytes before the value
ATOMIC_HEADER = 80
MUTEX_SIZE = 64


@runtime.extern(name="malloc")
def c_malloc(size: u64) -> ptr: ...


@runtime.extern(name="free")
def c_free(pointer: ptr) -> None: ...


@runtime.extern(name="strlen")
def c_strlen(text: cstr) -> u64: ...


@runtime.extern(name="write")
def c_write(descriptor: i32, data: cstr, size: u64) -> i64: ...


@runtime.extern(name="abort")
def c_abort() -> None: ...


@runtime.extern(name="pthread_mutex_init")
def c_mutex_init(mutex: ptr, attributes: i64) -> i32: ...  # attributes: always NULL (0), the default mutex


@runtime.extern(name="pthread_mutex_destroy")
def c_mutex_destroy(mutex: ptr) -> i32: ...


@runtime.extern(name="pthread_mutex_lock")
def c_mutex_lock(mutex: ptr) -> i32: ...


@runtime.extern(name="pthread_mutex_unlock")
def c_mutex_unlock(mutex: ptr) -> i32: ...


# -- mem.Shared -------------------------------------------------------------------


@runtime.function
def bifrost_shared_new(size: i64) -> ptr:
    """Allocate a cell for a ``size``-byte value, with one owner; return the value's address."""
    header = Ptr[i64](c_malloc(u64(size + SHARED_HEADER)))
    header[0] = 1
    header[1] = 0
    return header + 2


@runtime.function
def bifrost_shared_retain(value: ptr) -> None:
    header = Ptr[i64](value) - 2
    header[0] = header[0] + 1


@runtime.function
def bifrost_shared_release(value: ptr) -> None:
    """Drop one owner; the last frees the cell."""
    header = Ptr[i64](value) - 2
    count = header[0] - 1
    header[0] = count
    if count == 0:
        c_free(header)


@runtime.function
def bifrost_shared_unref(value: ptr) -> i64:
    """Drop one owner, and return how many are left, without freeing: at 0, what the value holds is freed first."""
    header = Ptr[i64](value) - 2
    count = header[0] - 1
    header[0] = count
    return count


@runtime.function
def bifrost_shared_destroy(value: ptr) -> None:
    c_free(Ptr[i64](value) - 2)


@runtime.function
def bifrost_shared_lock(value: ptr, message: cstr) -> None:
    """Take the guard; a second one while it is held stops the program with ``message``."""
    header = Ptr[i64](value) - 2
    if header[1] != 0:
        c_write(2, message, c_strlen(message))
        c_abort()
    header[1] = 1


@runtime.function
def bifrost_shared_unlock(value: ptr) -> None:
    header = Ptr[i64](value) - 2
    header[1] = 0


# -- mem.Atomic -------------------------------------------------------------------


@runtime.function
def bifrost_atomic_new(size: i64) -> ptr:
    """Allocate a cell for a ``size``-byte value, with one owner and an unlocked mutex."""
    cell = Ptr[u8](c_malloc(u64(size + ATOMIC_HEADER)))
    c_mutex_init(cell, 0)
    count = Ptr[i64](cell + MUTEX_SIZE)
    count[0] = 1
    return cell + ATOMIC_HEADER


@runtime.function
def bifrost_atomic_retain(value: ptr) -> None:
    atomic_add(Ptr[i64](value) - 2, 1)


@runtime.function
def bifrost_atomic_release(value: ptr) -> None:
    """Drop one owner, from any thread; the last destroys the mutex and frees the cell."""
    if atomic_add(Ptr[i64](value) - 2, -1) == 1:
        cell = Ptr[u8](value) - ATOMIC_HEADER
        c_mutex_destroy(cell)
        c_free(cell)


@runtime.function
def bifrost_atomic_unref(value: ptr) -> i64:
    """Drop one owner, from any thread, and return how many are left, without freeing (see bifrost_shared_unref)."""
    return atomic_add(Ptr[i64](value) - 2, -1) - 1


@runtime.function
def bifrost_atomic_destroy(value: ptr) -> None:
    cell = Ptr[u8](value) - ATOMIC_HEADER
    c_mutex_destroy(cell)
    c_free(cell)


@runtime.function
def bifrost_atomic_lock(value: ptr) -> None:
    c_mutex_lock(Ptr[u8](value) - ATOMIC_HEADER)


@runtime.function
def bifrost_atomic_unlock(value: ptr) -> None:
    c_mutex_unlock(Ptr[u8](value) - ATOMIC_HEADER)
