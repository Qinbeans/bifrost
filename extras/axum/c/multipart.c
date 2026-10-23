/*
 * multipart/form-data (axum's `multipart` feature): what a form with files submits.
 *
 * The body is read whole (it is, before a handler runs) and split into fields, each with
 * its headers (name, file name, content type) and data. A handler goes through them:
 *
 *     while form.next_field() {
 *         io.printf("%s: %d bytes\n", form.name(), form.len())
 *     }
 *
 * A Multipart lives with its request.
 */

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "axum.h"

typedef struct { void *raw; } AxumMultipart;

typedef struct {
    char *name, *file_name, *content_type;
    Headers headers;
    char *data;
    size_t len;
} Field;

typedef struct {
    Field *fields;
    size_t count, capacity;
    size_t next;     /* the field next_field moves to */
    Field *current;
    char *error;
} Multipart;

static void multipart_free(void *data)
{
    Multipart *m = data;
    for (size_t i = 0; i < m->count; i++) {
        Field *f = m->fields + i;
        free(f->name);
        free(f->file_name);
        free(f->content_type);
        for (size_t j = 0; j < f->headers.count; j++) {
            free(f->headers.items[j].name);
            free(f->headers.items[j].value);
        }
        free(f->headers.items);
        free(f->data);
    }
    free(m->fields);
    free(m->error);
    free(m);
}

static const char *find(const char *haystack, size_t size, const char *needle, size_t needle_len)
{
    if (needle_len == 0 || size < needle_len)
        return NULL;
    for (size_t i = 0; i + needle_len <= size; i++)
        if (haystack[i] == needle[0] && memcmp(haystack + i, needle, needle_len) == 0)
            return haystack + i;
    return NULL;
}

/* A parameter of a header value: `name="field"` of `form-data; name="field"; filename="a.txt"`. */
static char *parameter(const char *value, const char *name)
{
    size_t name_len = strlen(name);
    const char *at = value;
    while ((at = strchr(at, ';')) != NULL) {
        at++;
        while (*at == ' ' || *at == '\t')
            at++;
        if (strncasecmp(at, name, name_len) != 0 || at[name_len] != '=')
            continue;
        at += name_len + 1;
        if (*at == '"') {
            at++;
            const char *end = at;
            while (*end != '\0' && *end != '"')
                end += *end == '\\' && end[1] != '\0' ? 2 : 1;
            char *text = axum_strndup(at, (size_t)(end - at)), *o = text;
            for (const char *c = text; *c; c++) /* unescape \" and \\ */
                *o++ = *c == '\\' && c[1] ? *++c : *c;
            *o = '\0';
            return text;
        }
        return axum_strndup(at, strcspn(at, "; \t"));
    }
    return NULL;
}

static bool fail(Multipart *m, const char *message)
{
    m->error = axum_strdup(message);
    return false;
}

static bool parse(Multipart *m, const char *body, size_t size, const char *boundary)
{
    char delimiter[256];
    int delimiter_len = snprintf(delimiter, sizeof delimiter, "--%s", boundary);
    if (delimiter_len <= 2 || (size_t)delimiter_len >= sizeof delimiter)
        return fail(m, "Invalid `boundary` for `multipart/form-data` request");
    const char *at = find(body, size, delimiter, (size_t)delimiter_len), *end = body + size;
    if (at == NULL)
        return fail(m, "Error parsing `multipart/form-data` request");
    for (;;) {
        at += delimiter_len;
        if (end - at >= 2 && at[0] == '-' && at[1] == '-')
            return true; /* the closing delimiter */
        if (end - at < 2 || at[0] != '\r' || at[1] != '\n')
            return fail(m, "Error parsing `multipart/form-data` request");
        at += 2;
        const char *headers_end = find(at, (size_t)(end - at), "\r\n\r\n", 4);
        if (headers_end == NULL && !(end - at >= 2 && at[0] == '\r' && at[1] == '\n'))
            return fail(m, "Error parsing `multipart/form-data` request");
        if (m->count == m->capacity) {
            m->capacity = m->capacity ? m->capacity * 2 : 8;
            m->fields = realloc(m->fields, m->capacity * sizeof(Field));
        }
        Field *f = m->fields + m->count++;
        memset(f, 0, sizeof *f);
        const char *line = at;
        const char *body_start = headers_end != NULL ? headers_end + 4 : at + 2;
        while (headers_end != NULL && line < headers_end) {
            const char *line_end = find(line, (size_t)(headers_end + 2 - line), "\r\n", 2);
            const char *colon = memchr(line, ':', (size_t)(line_end - line));
            if (colon != NULL) {
                const char *value = colon + 1;
                while (value < line_end && (*value == ' ' || *value == '\t'))
                    value++;
                if (f->headers.count == f->headers.capacity) {
                    f->headers.capacity = f->headers.capacity ? f->headers.capacity * 2 : 4;
                    f->headers.items = realloc(f->headers.items, f->headers.capacity * sizeof(Header));
                }
                Header *h = f->headers.items + f->headers.count++;
                h->name = axum_strndup(line, (size_t)(colon - line));
                for (char *c = h->name; *c; c++)
                    *c = (char)tolower((unsigned char)*c);
                h->value = axum_strndup(value, (size_t)(line_end - value));
                if (strcmp(h->name, "content-disposition") == 0) {
                    f->name = parameter(h->value, "name");
                    f->file_name = parameter(h->value, "filename");
                } else if (strcmp(h->name, "content-type") == 0) {
                    f->content_type = axum_strdup(h->value);
                }
            }
            line = line_end + 2;
        }
        char closing[260];
        int closing_len = snprintf(closing, sizeof closing, "\r\n%s", delimiter);
        const char *next = find(body_start, (size_t)(end - body_start), closing, (size_t)closing_len);
        if (next == NULL)
            return fail(m, "Error parsing `multipart/form-data` request: a field has no end");
        f->len = (size_t)(next - body_start);
        f->data = axum_strndup(body_start, f->len);
        at = next + 2;
    }
}

/* The request's form fields (multipart/form-data); axum_multipart_error says if it is not one. */
AxumMultipart axum_multipart_of(AxumRequest request)
{
    Request *r = request.raw;
    Multipart *m = axum_zalloc(sizeof(Multipart));
    axum_on_release(r, multipart_free, m);
    const char *content_type = "";
    for (size_t i = 0; i < r->headers.count; i++)
        if (strcmp(r->headers.items[i].name, "content-type") == 0)
            content_type = r->headers.items[i].value;
    if (strncasecmp(content_type, "multipart/form-data", 19) != 0) {
        fail(m, "Invalid `boundary` for `multipart/form-data` request");
        return (AxumMultipart){m};
    }
    char *boundary = parameter(content_type, "boundary");
    if (boundary == NULL) {
        fail(m, "Invalid `boundary` for `multipart/form-data` request");
        return (AxumMultipart){m};
    }
    parse(m, r->body, r->body_len, boundary);
    free(boundary);
    return (AxumMultipart){m};
}

/* Why the body is not multipart/form-data ("" when it is). */
const char *axum_multipart_error(AxumMultipart multipart)
{
    Multipart *m = multipart.raw;
    return m->error != NULL ? m->error : "";
}

/* Move to the next field: whether there is one. */
bool axum_multipart_next(AxumMultipart multipart)
{
    Multipart *m = multipart.raw;
    if (m->error != NULL || m->next >= m->count) {
        m->current = NULL;
        return false;
    }
    m->current = m->fields + m->next++;
    return true;
}

static Field *current(AxumMultipart multipart)
{
    return ((Multipart *)multipart.raw)->current;
}

const char *axum_field_name(AxumMultipart multipart)
{
    Field *f = current(multipart);
    return f != NULL && f->name != NULL ? f->name : "";
}

const char *axum_field_file_name(AxumMultipart multipart)
{
    Field *f = current(multipart);
    return f != NULL && f->file_name != NULL ? f->file_name : "";
}

const char *axum_field_content_type(AxumMultipart multipart)
{
    Field *f = current(multipart);
    return f != NULL && f->content_type != NULL ? f->content_type : "";
}

const char *axum_field_text(AxumMultipart multipart)
{
    Field *f = current(multipart);
    return f != NULL ? f->data : "";
}

int64_t axum_field_len(AxumMultipart multipart)
{
    Field *f = current(multipart);
    return f != NULL ? (int64_t)f->len : 0;
}

const char *axum_field_header(AxumMultipart multipart, const char *name)
{
    Field *f = current(multipart);
    for (size_t i = 0; f != NULL && i < f->headers.count; i++)
        if (strcasecmp(f->headers.items[i].name, name) == 0)
            return f->headers.items[i].value;
    return "";
}
