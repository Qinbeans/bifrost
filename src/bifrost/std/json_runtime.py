r"""What ``json.encode`` compiles to: calls that append to a growable text buffer.

The compiler expands ``json.encode(#{id: 7, name: n})`` into::

    buffer = stack(JsonBuffer)
    bifrost_json_begin(buffer)
    bifrost_json_raw(buffer, "{\\"id\\": ")
    bifrost_json_int(buffer, 7)
    bifrost_json_raw(buffer, ", \\"name\\": ")
    bifrost_json_string(buffer, n)
    bifrost_json_raw(buffer, "}")
    text = bifrost_json_end(buffer)      # malloc'd; its owner frees it

These functions are compiled once, into their own module, and linked into any
program that encodes JSON.
"""

from mlir_python.lang import Module, Ptr, cstr, f64, i32, i64, ptr, stack, struct, u8, u64

runtime = Module("bifrost_json")

# Bytes that JSON strings escape.
QUOTE, BACKSLASH, NEWLINE, RETURN, TAB, SPACE = 34, 92, 10, 13, 9, 32


@struct
class JsonBuffer:
    data: cstr
    length: i64
    capacity: i64


@runtime.extern(name="malloc")
def c_malloc(size: u64) -> cstr: ...


@runtime.extern(name="realloc")
def c_realloc(pointer: cstr, size: u64) -> cstr: ...


@runtime.extern(name="strlen")
def c_strlen(text: ptr) -> u64: ...


@runtime.extern(name="snprintf")
def c_snprintf(buffer: ptr, size: u64, pattern: cstr, *args) -> i32: ...  # noqa: ANN002


@runtime.function
def bifrost_json_begin(buffer: Ptr[JsonBuffer]) -> None:
    data = c_malloc(u64(64))
    Ptr[u8](data)[0] = 0
    buffer[0] = JsonBuffer(data, 0, 64)


@runtime.function
def bifrost_json_reserve(buffer: Ptr[JsonBuffer], extra: i64) -> None:
    current = buffer[0]
    needed = current.length + extra + 1
    if needed > current.capacity:
        capacity = current.capacity * 2
        while capacity < needed:
            capacity = capacity * 2
        buffer[0] = JsonBuffer(c_realloc(current.data, u64(capacity)), current.length, capacity)


@runtime.function
def bifrost_json_byte(buffer: Ptr[JsonBuffer], byte: u8) -> None:
    bifrost_json_reserve(buffer, 1)
    current = buffer[0]
    out = Ptr[u8](current.data)
    out[current.length] = byte
    out[current.length + 1] = 0
    buffer[0] = JsonBuffer(current.data, current.length + 1, current.capacity)


@runtime.function
def bifrost_json_raw(buffer: Ptr[JsonBuffer], text: ptr) -> None:
    count = i64(c_strlen(text))
    bifrost_json_reserve(buffer, count)
    current = buffer[0]
    out = Ptr[u8](current.data)
    source = Ptr[u8](text)
    for index in range(count):
        out[current.length + index] = source[index]
    out[current.length + count] = 0
    buffer[0] = JsonBuffer(current.data, current.length + count, current.capacity)


@runtime.function
def bifrost_json_string(buffer: Ptr[JsonBuffer], text: cstr) -> None:
    bifrost_json_byte(buffer, QUOTE)
    source = Ptr[u8](text)
    index = 0
    while source[index] != 0:
        byte = source[index]
        if byte == QUOTE:
            bifrost_json_raw(buffer, '\\"')
        elif byte == BACKSLASH:
            bifrost_json_raw(buffer, "\\\\")
        elif byte == NEWLINE:
            bifrost_json_raw(buffer, "\\n")
        elif byte == RETURN:
            bifrost_json_raw(buffer, "\\r")
        elif byte == TAB:
            bifrost_json_raw(buffer, "\\t")
        elif byte < SPACE:
            escaped = stack(u8, 8)
            c_snprintf(escaped, u64(8), "\\u%04x", i32(byte))
            bifrost_json_raw(buffer, escaped)
        else:
            bifrost_json_byte(buffer, byte)
        index = index + 1
    bifrost_json_byte(buffer, QUOTE)


@runtime.function
def bifrost_json_int(buffer: Ptr[JsonBuffer], value: i64) -> None:
    text = stack(u8, 32)
    c_snprintf(text, u64(32), "%lld", value)
    bifrost_json_raw(buffer, text)


@runtime.function
def bifrost_json_float(buffer: Ptr[JsonBuffer], value: f64) -> None:
    text = stack(u8, 40)
    c_snprintf(text, u64(40), "%.15g", value)
    bifrost_json_raw(buffer, text)


@runtime.function
def bifrost_json_bool(buffer: Ptr[JsonBuffer], value: bool) -> None:
    if value:
        bifrost_json_raw(buffer, "true")
    else:
        bifrost_json_raw(buffer, "false")


@runtime.function
def bifrost_json_end(buffer: Ptr[JsonBuffer]) -> cstr:
    return buffer[0].data


__all__ = [
    "JsonBuffer",
    "bifrost_json_begin",
    "bifrost_json_bool",
    "bifrost_json_end",
    "bifrost_json_float",
    "bifrost_json_int",
    "bifrost_json_raw",
    "bifrost_json_string",
    "runtime",
]
