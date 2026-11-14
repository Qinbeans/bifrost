"""What ``sync.Channel`` (and ``sync.CondVar``, a channel with one slot) compiles to.

A channel value is a pointer to one block on the heap::

    [mutex: 64 bytes][not empty: 64 bytes][not full: 64 bytes]
    [owners: i64][slots: i64][size: i64][sent: i64][taken: i64][unbuffered: i64]
    [message 0][message 1] ... [message slots - 1]

The mutex guards everything after the owners; the two conditions
(``pthread_cond_t``, at most 64 bytes on the platforms Bifrost targets) are
where a receiver waits for a message (``not empty``) and a sender for a free
slot (``not full``). The messages are a ring buffer of ``slots`` items of
``size`` bytes each, inline: message ``k`` (counting every message ever
sent) is in slot ``k % slots``. ``sent - taken`` is how many are queued.

``channel(0)`` is unbuffered: it has one slot, and a sender also waits, after
putting its message there, until a receiver has taken it (a rendezvous).

The owners are counted atomically, like a ``mem.Atomic``'s: any thread may
make another or drop one. The last owner frees the channel; the generated
code first frees the messages still queued (``bifrost_channel_queued`` and
``bifrost_channel_next``), with their types' drop functions.

Sending is ``bifrost_channel_send_begin`` (wait for a free slot, holding the
mutex; return the slot), a store of the message there, then
``bifrost_channel_send_end`` (count it, wake a receiver, let go of the mutex).
Receiving is the same with ``bifrost_channel_receive_begin`` and
``bifrost_channel_receive_end``. The message is copied in or out by the
generated code, which knows its type.

These functions are compiled once, into their own module, and linked into any
program that uses a channel.
"""

from mlir_python.lang import Module, Ptr, atomic_add, cstr, i32, i64, ptr, u8, u64

runtime = Module("bifrost_sync")

LOCK_SIZE = 64  # bytes for a pthread_mutex_t or a pthread_cond_t
NOT_EMPTY = LOCK_SIZE
NOT_FULL = 2 * LOCK_SIZE
FIELDS = 3 * LOCK_SIZE // 8  # the index of the owners, in i64s
OWNERS, SLOTS, SIZE, SENT, TAKEN, UNBUFFERED = range(FIELDS, FIELDS + 6)
HEADER = 3 * LOCK_SIZE + 6 * 8  # bytes before the first message: 240, a multiple of 16


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


@runtime.extern(name="pthread_cond_init")
def c_cond_init(condition: ptr, attributes: i64) -> i32: ...  # attributes: always NULL (0)


@runtime.extern(name="pthread_cond_destroy")
def c_cond_destroy(condition: ptr) -> i32: ...


@runtime.extern(name="pthread_cond_wait")
def c_cond_wait(condition: ptr, mutex: ptr) -> i32: ...


@runtime.extern(name="pthread_cond_signal")
def c_cond_signal(condition: ptr) -> i32: ...


@runtime.extern(name="pthread_cond_broadcast")
def c_cond_broadcast(condition: ptr) -> i32: ...


@runtime.function
def bifrost_channel_new(capacity: i64, size: i64, message: cstr) -> ptr:
    """Make a channel of ``capacity`` slots for ``size``-byte messages, with one owner.

    A negative ``capacity`` stops the program with ``message``; 0 makes an unbuffered channel.
    """
    if capacity < 0:
        c_write(2, message, c_strlen(message))
        c_abort()
    slots = max(capacity, 1)
    channel = Ptr[u8](c_malloc(u64(HEADER + slots * size)))
    c_mutex_init(channel, 0)
    c_cond_init(channel + NOT_EMPTY, 0)
    c_cond_init(channel + NOT_FULL, 0)
    fields = Ptr[i64](channel)
    fields[OWNERS] = 1
    fields[SLOTS] = slots
    fields[SIZE] = size
    fields[SENT] = 0
    fields[TAKEN] = 0
    fields[UNBUFFERED] = 0
    if capacity == 0:
        fields[UNBUFFERED] = 1
    return channel


@runtime.function
def bifrost_channel_retain(channel: ptr) -> None:
    atomic_add(Ptr[i64](channel) + OWNERS, 1)


@runtime.function
def bifrost_channel_destroy(channel: ptr) -> None:
    """Free the channel itself (what its queued messages own is freed first, by the generated code)."""
    bytes_ = Ptr[u8](channel)
    c_cond_destroy(bytes_ + NOT_FULL)
    c_cond_destroy(bytes_ + NOT_EMPTY)
    c_mutex_destroy(bytes_)
    c_free(bytes_)


@runtime.function
def bifrost_channel_release(channel: ptr) -> None:
    """Drop one owner, from any thread; the last frees the channel (its messages own nothing)."""
    if atomic_add(Ptr[i64](channel) + OWNERS, -1) == 1:
        bifrost_channel_destroy(channel)


@runtime.function
def bifrost_channel_unref(channel: ptr) -> i64:
    """Drop one owner, from any thread, and return how many are left, without freeing anything."""
    return atomic_add(Ptr[i64](channel) + OWNERS, -1) - 1


@runtime.function
def bifrost_channel_queued(channel: ptr) -> i64:
    """How many messages wait to be received (for the last owner, which no other thread can reach)."""
    fields = Ptr[i64](channel)
    return fields[SENT] - fields[TAKEN]


@runtime.function
def bifrost_channel_next(channel: ptr) -> ptr:
    """Take the oldest queued message's slot, without waiting or locking (for the last owner, to free it)."""
    fields = Ptr[i64](channel)
    taken = fields[TAKEN]
    fields[TAKEN] = taken + 1
    return Ptr[u8](channel) + HEADER + (taken % fields[SLOTS]) * fields[SIZE]


@runtime.function
def bifrost_channel_send_begin(channel: ptr) -> ptr:
    """Lock the channel and return the slot for the next message: the sender waits while every slot holds one."""
    bytes_ = Ptr[u8](channel)
    fields = Ptr[i64](channel)
    c_mutex_lock(bytes_)
    while fields[SENT] - fields[TAKEN] == fields[SLOTS]:
        c_cond_wait(bytes_ + NOT_FULL, bytes_)
    return bytes_ + HEADER + (fields[SENT] % fields[SLOTS]) * fields[SIZE]


@runtime.function
def bifrost_channel_send_end(channel: ptr) -> None:
    """Queue the message just stored, wake a receiver, and unlock (unbuffered: once a receiver has taken it)."""
    bytes_ = Ptr[u8](channel)
    fields = Ptr[i64](channel)
    sent = fields[SENT] + 1
    fields[SENT] = sent
    c_cond_signal(bytes_ + NOT_EMPTY)
    if fields[UNBUFFERED] != 0:
        while fields[TAKEN] < sent:
            c_cond_wait(bytes_ + NOT_FULL, bytes_)
    c_mutex_unlock(bytes_)


@runtime.function
def bifrost_channel_receive_begin(channel: ptr) -> ptr:
    """Lock the channel and return the oldest message's slot: the receiver waits while none is queued."""
    bytes_ = Ptr[u8](channel)
    fields = Ptr[i64](channel)
    c_mutex_lock(bytes_)
    while fields[SENT] == fields[TAKEN]:
        c_cond_wait(bytes_ + NOT_EMPTY, bytes_)
    return bytes_ + HEADER + (fields[TAKEN] % fields[SLOTS]) * fields[SIZE]


@runtime.function
def bifrost_channel_receive_end(channel: ptr) -> None:
    """Free the slot just read, wake a sender waiting for it, and unlock."""
    bytes_ = Ptr[u8](channel)
    fields = Ptr[i64](channel)
    fields[TAKEN] = fields[TAKEN] + 1
    if fields[UNBUFFERED] != 0:
        c_cond_broadcast(bytes_ + NOT_FULL)  # the sender waiting for its message to be taken, and the next one
    else:
        c_cond_signal(bytes_ + NOT_FULL)
    c_mutex_unlock(bytes_)
