/* Bifrost's decoding runtime: what `json.decode` and `form.decode` compile to call.
 *
 * The compiler writes one decoder per type (see bifrost.decoding): it walks the type's
 * fields and calls these to read the text, value by value. A reader remembers the first
 * error; once it has one, every call returns a zero value and reads nothing, so a decoder
 * runs to its end and then asks whether it worked (bifrost_decode_failed), freeing what it
 * made if not. Strings come back malloc'd: what they are decoded into owns them.
 *
 * JSON follows RFC 8259 (escapes, \uXXXX with surrogate pairs as UTF-8, numbers); unknown
 * object fields are skipped. Forms are application/x-www-form-urlencoded (`a=1&b=x+y%21`),
 * which is also how the HTTP server hands over a path's parameters and a query.
 */
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *text;
    size_t at;
    char *error;       /* the first error, malloc'd; NULL while all is well */
    char *key;         /* the object key or form field being read */
    size_t key_size;
    /* forms: the pairs, decoded */
    char **names;
    char **values;
    size_t count;
    size_t current;    /* the pair a value is read from */
    const char *field; /* the field being read, for messages */
    /* JSON: for each object or array being read, whether its first item is still to come */
    bool first[512];
    int depth;
} Reader;

/* malloc'd copies (every allocation goes through malloc, as Bifrost's own do) */
static char *copy_text(const char *text)
{
    size_t size = strlen(text) + 1;
    return memcpy(malloc(size), text, size);
}

static Reader *new_reader(const char *text)
{
    Reader *r = malloc(sizeof(Reader));  /* each field set: malloc and memset would compile to calloc */
    r->text = text != NULL ? text : "";
    r->at = 0;
    r->error = NULL;
    r->key = NULL;
    r->key_size = 0;
    r->names = NULL;
    r->values = NULL;
    r->count = 0;
    r->current = 0;
    r->field = NULL;
    r->depth = 0;
    return r;
}

static void fail(Reader *r, const char *format, const char *detail)
{
    if (r->error != NULL)
        return;
    size_t size = strlen(format) + (detail != NULL ? strlen(detail) : 0) + 64;
    r->error = malloc(size);
    snprintf(r->error, size, format, detail != NULL ? detail : "");
}

static void fail_at(Reader *r, const char *what)
{
    if (r->error != NULL)
        return;
    char where[96];
    if (r->text[r->at] == '\0')
        snprintf(where, sizeof where, "%s at the end of the text", what);
    else
        snprintf(where, sizeof where, "%s at %zu, near '%.12s'", what, r->at, r->text + r->at);
    fail(r, "%s", where);
}

void *bifrost_decode_json(const char *text)
{
    return new_reader(text);
}

void bifrost_decode_free(void *reader)
{
    Reader *r = reader;
    free(r->error);
    free(r->key);
    for (size_t i = 0; i < r->count; i++) {
        free(r->names[i]);
        free(r->values[i]);
    }
    free(r->names);
    free(r->values);
    free(r);
}

bool bifrost_decode_failed(void *reader)
{
    return ((Reader *)reader)->error != NULL;
}

/* The error, as a string its caller owns. */
char *bifrost_decode_error(void *reader)
{
    Reader *r = reader;
    return copy_text(r->error != NULL ? r->error : "");
}

void bifrost_decode_missing(void *reader, const char *field)
{
    fail(reader, "missing field `%s`", field);
}

void bifrost_decode_duplicate(void *reader, const char *field)
{
    fail(reader, "duplicate field `%s`", field);
}

/* -- JSON ------------------------------------------------------------------- */

static void space(Reader *r)
{
    while (r->text[r->at] == ' ' || r->text[r->at] == '\t' || r->text[r->at] == '\n' || r->text[r->at] == '\r')
        r->at++;
}

static bool expect(Reader *r, char c, const char *what)
{
    space(r);
    if (r->text[r->at] != c) {
        fail_at(r, what);
        return false;
    }
    r->at++;
    return true;
}

/* After the whole value: nothing but whitespace may follow. */
void bifrost_decode_json_end(void *reader)
{
    Reader *r = reader;
    if (r->error != NULL)
        return;
    space(r);
    if (r->text[r->at] != '\0')
        fail_at(r, "trailing characters");
}

static void put_utf8(char **out, uint32_t code)
{
    unsigned char *o = (unsigned char *)*out;
    if (code < 0x80) {
        *o++ = (unsigned char)code;
    } else if (code < 0x800) {
        *o++ = (unsigned char)(0xC0 | (code >> 6));
        *o++ = (unsigned char)(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        *o++ = (unsigned char)(0xE0 | (code >> 12));
        *o++ = (unsigned char)(0x80 | ((code >> 6) & 0x3F));
        *o++ = (unsigned char)(0x80 | (code & 0x3F));
    } else {
        *o++ = (unsigned char)(0xF0 | (code >> 18));
        *o++ = (unsigned char)(0x80 | ((code >> 12) & 0x3F));
        *o++ = (unsigned char)(0x80 | ((code >> 6) & 0x3F));
        *o++ = (unsigned char)(0x80 | (code & 0x3F));
    }
    *out = (char *)o;
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

static bool hex4(Reader *r, uint32_t *code)
{
    *code = 0;
    for (int i = 0; i < 4; i++) {
        int digit = hex_digit(r->text[r->at]);
        if (digit < 0) {
            fail_at(r, "invalid \\u escape");
            return false;
        }
        *code = *code * 16 + (uint32_t)digit;
        r->at++;
    }
    return true;
}

/* A JSON string, unescaped, malloc'd (NULL after an error). Decoded text is never longer than its source. */
static char *json_string(Reader *r)
{
    if (!expect(r, '"', "expected a string"))
        return NULL;
    size_t start = r->at;
    while (r->text[r->at] != '"' && r->text[r->at] != '\0')
        r->at += r->text[r->at] == '\\' && r->text[r->at + 1] != '\0' ? 2 : 1;
    if (r->text[r->at] != '"') {
        fail_at(r, "unterminated string");
        return NULL;
    }
    char *out = malloc(r->at - start + 1), *o = out;
    r->at = start;
    while (r->text[r->at] != '"') {
        char c = r->text[r->at++];
        if ((unsigned char)c < 0x20) {
            r->at--;
            fail_at(r, "control character in string");
            free(out);
            return NULL;
        }
        if (c != '\\') {
            *o++ = c;
            continue;
        }
        char e = r->text[r->at++];
        switch (e) {
        case '"': *o++ = '"'; break;
        case '\\': *o++ = '\\'; break;
        case '/': *o++ = '/'; break;
        case 'b': *o++ = '\b'; break;
        case 'f': *o++ = '\f'; break;
        case 'n': *o++ = '\n'; break;
        case 'r': *o++ = '\r'; break;
        case 't': *o++ = '\t'; break;
        case 'u': {
            uint32_t code, low;
            if (!hex4(r, &code)) {
                free(out);
                return NULL;
            }
            if (code >= 0xD800 && code <= 0xDBFF && r->text[r->at] == '\\' && r->text[r->at + 1] == 'u') {
                r->at += 2;
                if (!hex4(r, &low)) {
                    free(out);
                    return NULL;
                }
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            }
            put_utf8(&o, code);
            break;
        }
        default:
            r->at--;
            fail_at(r, "invalid escape");
            free(out);
            return NULL;
        }
    }
    r->at++;
    *o = '\0';
    return out;
}

char *bifrost_decode_json_string(void *reader)
{
    Reader *r = reader;
    if (r->error != NULL)
        return copy_text("");
    char *text = json_string(r);
    return text != NULL ? text : copy_text("");
}

static bool number_char(char c)
{
    return (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
}

int64_t bifrost_decode_json_int(void *reader)
{
    Reader *r = reader;
    if (r->error != NULL)
        return 0;
    space(r);
    const char *start = r->text + r->at;
    char *end;
    errno = 0;
    long long value = strtoll(start, &end, 10);
    if (end == start || (*end != '\0' && number_char(*end))) {
        fail_at(r, "expected an integer");
        return 0;
    }
    if (errno == ERANGE) {
        fail_at(r, "integer out of range");
        return 0;
    }
    r->at += (size_t)(end - start);
    return value;
}

double bifrost_decode_json_float(void *reader)
{
    Reader *r = reader;
    if (r->error != NULL)
        return 0;
    space(r);
    const char *start = r->text + r->at;
    char *end;
    double value = strtod(start, &end);
    if (end == start || *start == '+' || *start == '.' || (start[0] == '0' && start[1] >= '0' && start[1] <= '9')) {
        fail_at(r, "expected a number");
        return 0;
    }
    r->at += (size_t)(end - start);
    return value;
}

static bool word(Reader *r, const char *text)
{
    size_t size = strlen(text);
    if (strncmp(r->text + r->at, text, size) != 0)
        return false;
    r->at += size;
    return true;
}

bool bifrost_decode_json_bool(void *reader)
{
    Reader *r = reader;
    if (r->error != NULL)
        return false;
    space(r);
    if (word(r, "true"))
        return true;
    if (!word(r, "false"))
        fail_at(r, "expected true or false");
    return false;
}

/* Whether the next value is null (it is then read). */
bool bifrost_decode_json_null(void *reader)
{
    Reader *r = reader;
    if (r->error != NULL)
        return false;
    space(r);
    return word(r, "null");
}

static void skip(Reader *r, int depth);

static bool enter(Reader *r, char c, const char *what)
{
    if (r->error != NULL || !expect(r, c, what))
        return false;
    if (r->depth == (int)(sizeof r->first)) {
        fail_at(r, "nested too deeply");
        return false;
    }
    r->first[r->depth++] = true;
    return true;
}

/* Before each key or item: whether one follows (after a comma, unless it is the first), or `close` ends it. */
static bool next(Reader *r, char close, const char *what)
{
    if (r->error != NULL || r->depth == 0)
        return false;
    space(r);
    if (r->text[r->at] == close) {
        r->at++;
        r->depth--;
        return false;
    }
    if (!r->first[r->depth - 1] && !expect(r, ',', what))
        return false;
    r->first[r->depth - 1] = false;
    return true;
}

bool bifrost_decode_json_object(void *reader)
{
    return enter(reader, '{', "expected an object");
}

/* Read the next key of an object (`"name":`), or its end (`}`): whether there was a key. */
bool bifrost_decode_json_key(void *reader)
{
    Reader *r = reader;
    if (!next(r, '}', "expected ',' or '}'"))
        return false;
    free(r->key);
    r->key = json_string(r);
    if (r->key == NULL || !expect(r, ':', "expected ':'"))
        return false;
    return true;
}

bool bifrost_decode_key_is(void *reader, const char *name)
{
    Reader *r = reader;
    return r->error == NULL && r->key != NULL && strcmp(r->key, name) == 0;
}

/* Skip the value of a key no field has. */
void bifrost_decode_json_skip(void *reader)
{
    skip(reader, 0);
}

bool bifrost_decode_json_array(void *reader)
{
    return enter(reader, '[', "expected an array");
}

/* Before each item of an array: whether one follows (after a comma), or the array ends (`]`). */
bool bifrost_decode_json_more(void *reader)
{
    return next(reader, ']', "expected ',' or ']'");
}

static void skip(Reader *r, int depth)
{
    if (r->error != NULL)
        return;
    if (depth > 512) {
        fail_at(r, "nested too deeply");
        return;
    }
    space(r);
    char c = r->text[r->at];
    if (c == '"') {
        free(json_string(r));
    } else if (c == '{') {
        r->at++;
        space(r);
        if (r->text[r->at] == '}') {
            r->at++;
            return;
        }
        do {
            free(json_string(r));
            if (!expect(r, ':', "expected ':'"))
                return;
            skip(r, depth + 1);
            space(r);
        } while (r->error == NULL && r->text[r->at] == ',' && ++r->at);
        expect(r, '}', "expected ',' or '}'");
    } else if (c == '[') {
        r->at++;
        space(r);
        if (r->text[r->at] == ']') {
            r->at++;
            return;
        }
        do {
            skip(r, depth + 1);
            space(r);
        } while (r->error == NULL && r->text[r->at] == ',' && ++r->at);
        expect(r, ']', "expected ',' or ']'");
    } else if (word(r, "true") || word(r, "false") || word(r, "null")) {
        return;
    } else if (number_char(c)) {
        bifrost_decode_json_float(r);
    } else {
        fail_at(r, "expected a value");
    }
}

/* -- forms (application/x-www-form-urlencoded) ------------------------------- */

static char *form_decoded(const char *text, size_t size)
{
    char *out = malloc(size + 1), *o = out;
    for (size_t i = 0; i < size; i++) {
        if (text[i] == '+') {
            *o++ = ' ';
        } else if (text[i] == '%' && i + 2 < size && hex_digit(text[i + 1]) >= 0 && hex_digit(text[i + 2]) >= 0) {
            *o++ = (char)(hex_digit(text[i + 1]) * 16 + hex_digit(text[i + 2]));
            i += 2;
        } else {
            *o++ = text[i];
        }
    }
    *o = '\0';
    return out;
}

void *bifrost_decode_form(const char *text)
{
    Reader *r = new_reader(text);
    const char *at = r->text;
    while (*at != '\0') {
        const char *end = strchr(at, '&');
        size_t size = end != NULL ? (size_t)(end - at) : strlen(at);
        if (size > 0) {
            const char *equals = memchr(at, '=', size);
            size_t name_size = equals != NULL ? (size_t)(equals - at) : size;
            r->names = realloc(r->names, (r->count + 1) * sizeof(char *));
            r->values = realloc(r->values, (r->count + 1) * sizeof(char *));
            r->names[r->count] = form_decoded(at, name_size);
            r->values[r->count] = equals != NULL ? form_decoded(equals + 1, size - name_size - 1) : copy_text("");
            r->count++;
        }
        at += size + (end != NULL ? 1 : 0);
    }
    return r;
}

/* Select the field `name` to read a value of (the last one given, as serde does): whether it is there. */
bool bifrost_decode_form_field(void *reader, const char *name)
{
    Reader *r = reader;
    r->field = name;
    for (size_t i = r->count; i > 0; i--) {
        if (strcmp(r->names[i - 1], name) == 0) {
            r->current = i - 1;
            return r->error == NULL;
        }
    }
    return false;
}

/* For a form decoded into one value (a path's single parameter): select its only pair. */
void bifrost_decode_form_single(void *reader)
{
    Reader *r = reader;
    r->field = NULL;
    if (r->count != 1) {
        char detail[64];
        snprintf(detail, sizeof detail, "%zu", r->count);
        fail(r, "expected 1 value, got %s", detail);
        return;
    }
    r->current = 0;
}

static const char *form_value(Reader *r)
{
    return r->error == NULL && r->current < r->count ? r->values[r->current] : NULL;
}

static void form_invalid(Reader *r, const char *what, const char *value)
{
    if (r->error != NULL)
        return;
    size_t size = strlen(what) + strlen(value) + (r->field != NULL ? strlen(r->field) : 0) + 64;
    r->error = malloc(size);
    if (r->field != NULL)
        snprintf(r->error, size, "field `%s`: %s, not '%s'", r->field, what, value);
    else
        snprintf(r->error, size, "%s, not '%s'", what, value);
}

int64_t bifrost_decode_form_int(void *reader)
{
    Reader *r = reader;
    const char *value = form_value(r);
    if (value == NULL)
        return 0;
    char *end;
    errno = 0;
    long long number = strtoll(value, &end, 10);
    if (*value == '\0' || *end != '\0' || errno == ERANGE) {
        form_invalid(r, "expected an integer", value);
        return 0;
    }
    return number;
}

double bifrost_decode_form_float(void *reader)
{
    Reader *r = reader;
    const char *value = form_value(r);
    if (value == NULL)
        return 0;
    char *end;
    double number = strtod(value, &end);
    if (*value == '\0' || *end != '\0') {
        form_invalid(r, "expected a number", value);
        return 0;
    }
    return number;
}

bool bifrost_decode_form_bool(void *reader)
{
    Reader *r = reader;
    const char *value = form_value(r);
    if (value == NULL)
        return false;
    if (strcmp(value, "true") == 0)
        return true;
    if (strcmp(value, "false") != 0)
        form_invalid(r, "expected true or false", value);
    return false;
}

char *bifrost_decode_form_string(void *reader)
{
    const char *value = form_value(reader);
    return copy_text(value != NULL ? value : "");
}
