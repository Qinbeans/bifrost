/* std:fs's runtime: reading and writing whole files (see std/fs.py).
 *
 * A read gives a reading: the text, or why there is none, each a new string
 * (malloc'd: the program owns whichever it takes, and frees it). Memory comes
 * from malloc and realloc only, as the decoding runtime's does.
 */
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
    char *text;  /* the file's text, until taken */
    char *error; /* why there is none, until taken */
} Reading;

/* A new string of `a` and `b`. */
static char *joined(const char *a, const char *b)
{
    size_t left = strlen(a), right = strlen(b);
    char *out = malloc(left + right + 1);
    if (out == NULL)
        return NULL;
    memcpy(out, a, left);
    memcpy(out + left, b, right + 1);
    return out;
}

/* `path: what went wrong`, the C library's words for errno. */
static char *failure(const char *path, const char *what)
{
    char *prefix = joined(path, ": ");
    if (prefix == NULL)
        return NULL;
    char *out = joined(prefix, what);
    free(prefix);
    return out;
}

/* Whether `text` (of `len` bytes) is UTF-8: what read_to_string requires, as Rust's does. */
static bool is_utf8(const unsigned char *text, size_t len)
{
    size_t i = 0;
    while (i < len) {
        unsigned char c = text[i];
        size_t more;
        unsigned int code;
        if (c < 0x80) {
            i++;
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            more = 1;
            code = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            more = 2;
            code = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            more = 3;
            code = c & 0x07;
        } else {
            return false;
        }
        if (i + more >= len) /* cut off */
            return false;
        for (size_t k = 1; k <= more; k++) {
            if ((text[i + k] & 0xC0) != 0x80)
                return false;
            code = (code << 6) | (text[i + k] & 0x3F);
        }
        /* overlong forms, surrogates, and past U+10FFFF */
        if ((more == 1 && code < 0x80) || (more == 2 && code < 0x800) || (more == 3 && code < 0x10000) ||
            (code >= 0xD800 && code <= 0xDFFF) || code > 0x10FFFF)
            return false;
        i += more + 1;
    }
    return true;
}

/* Read all of `path`; the reading holds its text, or why it could not. */
void *bifrost_fs_read(const char *path)
{
    Reading *reading = malloc(sizeof *reading);
    if (reading == NULL)
        return NULL;
    reading->text = NULL;
    reading->error = NULL;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        reading->error = failure(path, strerror(errno));
        return reading;
    }
    /* Read until the end, growing as it goes: a pipe or /proc file has no size to ask for. */
    size_t len = 0, capacity = 4096;
    char *text = malloc(capacity);
    while (text != NULL) {
        if (len + 1 == capacity) {
            char *grown = realloc(text, capacity * 2);
            if (grown == NULL) {
                free(text);
                text = NULL;
                break;
            }
            text = grown;
            capacity *= 2;
        }
        size_t got = fread(text + len, 1, capacity - len - 1, file);
        len += got;
        if (got == 0) {
            if (ferror(file)) {
                reading->error = failure(path, strerror(errno));
                free(text);
                text = NULL;
            }
            break;
        }
    }
    fclose(file);
    if (text == NULL) {
        if (reading->error == NULL)
            reading->error = failure(path, "out of memory");
        return reading;
    }
    text[len] = '\0';
    if (memchr(text, '\0', len) != NULL) {
        reading->error = failure(path, "holds a NUL byte, so it is not text");
        free(text);
    } else if (!is_utf8((const unsigned char *)text, len)) {
        reading->error = failure(path, "is not valid UTF-8");
        free(text);
    } else {
        reading->text = text;
    }
    return reading;
}

bool bifrost_fs_failed(void *reading) { return reading == NULL || ((Reading *)reading)->text == NULL; }

/* The text read: the caller owns it. */
char *bifrost_fs_text(void *reading)
{
    char *text = ((Reading *)reading)->text;
    ((Reading *)reading)->text = NULL;
    return text;
}

/* Why the read failed: the caller owns it. */
char *bifrost_fs_error(void *reading)
{
    if (reading == NULL)
        return joined("out of memory", "");
    char *error = ((Reading *)reading)->error;
    ((Reading *)reading)->error = NULL;
    return error != NULL ? error : joined("out of memory", "");
}

/* Free a reading, and whatever it holds that was not taken. */
void bifrost_fs_done(void *reading)
{
    if (reading == NULL)
        return;
    free(((Reading *)reading)->text);
    free(((Reading *)reading)->error);
    free(reading);
}

/* Write `text` to `path`, replacing what it held (creating it if need be): NULL, or why it could not (owned). */
char *bifrost_fs_write(const char *path, const char *text)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL)
        return failure(path, strerror(errno));
    size_t len = strlen(text);
    bool wrote = fwrite(text, 1, len, file) == len;
    int error = errno;
    if (fclose(file) != 0 && wrote) {
        wrote = false;
        error = errno;
    }
    return wrote ? NULL : failure(path, strerror(error));
}

bool bifrost_fs_is_null(const char *text) { return text == NULL; }

/* Whether `path` names something: a file or a folder. */
bool bifrost_fs_exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0;
}
