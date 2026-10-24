/* axum.serve_file: tower-http's ServeFile, one file as a service.
 *
 * GET and HEAD answer with the file: its type from its extension (h2o's table:
 * text/html, image/png, ..., application/octet-stream otherwise), Last-Modified,
 * and Accept-Ranges. If-Modified-Since gives a 304 when it has not changed, and
 * If-Unmodified-Since a 412 when it has; a single `Range: bytes=...` a 206 of
 * that part (a 416 when it is past the end; several ranges get the whole file).
 * A file that is missing, or is not a regular file, is a 404; other methods a
 * 405. The file is read on each request, so a change shows at once.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "axum.h"

typedef struct {
    char *path;
    char *mime;
} ServeFile;

static h2o_mimemap_t *mimemap(void)
{
    static h2o_mimemap_t *map;
    if (map == NULL)
        map = h2o_mimemap_create();
    return map;
}

/* The type of the file at `path`, from its extension. */
static char *mime_of(const char *path)
{
    const char *slash = strrchr(path, '/');
    const char *dot = strrchr(slash != NULL ? slash + 1 : path, '.');
    h2o_iovec_t extension = dot != NULL ? h2o_iovec_init(dot + 1, strlen(dot + 1)) : h2o_iovec_init("", 0);
    h2o_mimemap_type_t *type = h2o_mimemap_get_type_by_extension(mimemap(), extension);
    if (type == NULL || type->type != H2O_MIMEMAP_TYPE_MIMETYPE)
        return axum_strdup("application/octet-stream");
    return axum_strndup(type->data.mimetype.base, type->data.mimetype.len);
}

static const char *header_of(Request *r, const char *name)
{
    for (size_t i = 0; i < r->headers.count; i++)
        if (strcmp(r->headers.items[i].name, name) == 0)
            return r->headers.items[i].value;
    return NULL;
}

/* An HTTP date (`Sun, 06 Nov 1994 08:49:37 GMT`) as seconds; false when it is not one. */
static bool parse_date(const char *text, time_t *out)
{
    struct tm tm;
    if (text == NULL || h2o_time_parse_rfc1123(text, strlen(text), &tm) != 0)
        return false;
    *out = timegm(&tm);
    return true;
}

typedef enum { RANGE_NONE, RANGE_ONE, RANGE_UNSATISFIABLE } RangeKind;

/* Read `Range: bytes=a-b` (or `a-`, or `-n`, the last n) of a file of `size` bytes: [*start, *end]. */
static RangeKind parse_range(const char *text, int64_t size, int64_t *start, int64_t *end)
{
    if (text == NULL || strncmp(text, "bytes=", 6) != 0 || strchr(text, ',') != NULL)
        return RANGE_NONE; /* not bytes, or several: the whole file (RFC 9110 lets a server ignore Range) */
    const char *at = text + 6;
    char *after;
    if (*at == '-') {
        errno = 0;
        int64_t last = strtoll(at + 1, &after, 10);
        if (errno != 0 || after == at + 1 || *after != '\0' || last < 0)
            return RANGE_NONE;
        if (last == 0 || size == 0)
            return RANGE_UNSATISFIABLE;
        *start = last >= size ? 0 : size - last;
        *end = size - 1;
        return RANGE_ONE;
    }
    errno = 0;
    *start = strtoll(at, &after, 10);
    if (errno != 0 || after == at || *after != '-' || *start < 0)
        return RANGE_NONE;
    at = after + 1;
    if (*at == '\0') {
        *end = size - 1;
    } else {
        errno = 0;
        *end = strtoll(at, &after, 10);
        if (errno != 0 || *after != '\0' || *end < *start)
            return RANGE_NONE;
        if (*end >= size)
            *end = size - 1;
    }
    return *start >= size ? RANGE_UNSATISFIABLE : RANGE_ONE;
}

/* Read `len` bytes of `file` from `start` into a new buffer; NULL when it cannot. */
static char *read_part(FILE *file, int64_t start, int64_t len)
{
    char *buffer = axum_alloc((size_t)len + 1);
    if (fseeko(file, (off_t)start, SEEK_SET) != 0 || fread(buffer, 1, (size_t)len, file) != (size_t)len) {
        free(buffer);
        return NULL;
    }
    buffer[len] = '\0';
    return buffer;
}

static void *serve_file(AxumRequest request, void *data, AxumResponse *out)
{
    Request *r = request.raw;
    ServeFile *serve = data;
    if (strcmp(r->method, "GET") != 0 && strcmp(r->method, "HEAD") != 0) {
        *out = axum_response(405);
        axum_set_header(*out, "allow", "GET,HEAD");
        return axum_done_token();
    }
    FILE *file = fopen(serve->path, "rb");
    struct stat info;
    if (file == NULL || fstat(fileno(file), &info) != 0 || !S_ISREG(info.st_mode)) {
        if (file != NULL)
            fclose(file);
        *out = axum_response(404);
        return axum_done_token();
    }
    char modified[H2O_TIMESTR_RFC1123_LEN + 1];
    struct tm gmt;
    gmtime_r(&info.st_mtime, &gmt);
    h2o_time2str_rfc1123(modified, &gmt);
    time_t since;
    if (parse_date(header_of(r, "if-unmodified-since"), &since) && info.st_mtime > since) {
        fclose(file);
        *out = axum_response(412);
        return axum_done_token();
    }
    if (parse_date(header_of(r, "if-modified-since"), &since) && info.st_mtime <= since) {
        fclose(file);
        *out = axum_response(304);
        axum_set_header(*out, "last-modified", modified);
        return axum_done_token();
    }
    int64_t size = (int64_t)info.st_size, start = 0, end = size - 1;
    RangeKind range = parse_range(header_of(r, "range"), size, &start, &end);
    if (range == RANGE_UNSATISFIABLE) {
        fclose(file);
        char content_range[64];
        snprintf(content_range, sizeof content_range, "bytes */%" PRId64, size);
        *out = axum_response(416);
        axum_set_header(*out, "content-range", content_range);
        return axum_done_token();
    }
    int64_t len = range == RANGE_ONE ? end - start + 1 : size;
    char *body = read_part(file, start, len);
    fclose(file);
    if (body == NULL) {
        *out = axum_response(500);
        return axum_done_token();
    }
    *out = axum_response(range == RANGE_ONE ? 206 : 200);
    Response *response = out->raw;
    response->body = body;
    response->len = (size_t)len;
    axum_set_header(*out, "content-type", serve->mime);
    axum_set_header(*out, "last-modified", modified);
    axum_set_header(*out, "accept-ranges", "bytes");
    if (range == RANGE_ONE) {
        char content_range[96];
        snprintf(content_range, sizeof content_range, "bytes %" PRId64 "-%" PRId64 "/%" PRId64, start, end, size);
        axum_set_header(*out, "content-range", content_range);
    }
    return axum_done_token();
}

/* ServeFile::new(path): a service answering with the file at `path` (relative to where the server runs). */
AxumService axum_serve_file(const char *path)
{
    ServeFile *serve = axum_alloc(sizeof(ServeFile));
    serve->path = axum_strdup(path);
    serve->mime = mime_of(path);
    Service *service = axum_alloc(sizeof(Service));
    *service = (Service){serve_file, serve};
    return (AxumService){service};
}

/* ServeFile::new_with_mime: the same, with its type given (`text/markdown; charset=utf-8`). */
AxumService axum_serve_file_with_mime(const char *path, const char *mime)
{
    AxumService service = axum_serve_file(path);
    ServeFile *serve = ((Service *)service.raw)->data;
    free(serve->mime);
    serve->mime = axum_strdup(mime);
    return service;
}
