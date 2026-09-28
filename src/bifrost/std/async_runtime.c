/* Bifrost's async runtime: MLIR's async runtime API (mlir/ExecutionEngine/AsyncRuntime.h)
 * on one thread, driven by whatever event loop the program runs.
 *
 * A function that pauses compiles to a coroutine. When it waits on something not
 * ready yet, the runtime keeps it as a waiter of that token or value; when the
 * token completes (a C library calls mlirAsyncRuntimeEmplaceToken, e.g. when a
 * timer fires), its waiters move to the ready queue. Ready coroutines run when the
 * event loop calls bifrost_async_run_ready(), or when code blocks on a result
 * (main calling a function that pauses): blocking runs ready coroutines, and, when
 * none are left, asks the event loop to wait for events (the poller it set with
 * bifrost_async_set_poller).
 *
 * It also keeps timers, for std:time's sleep: an event loop bounds its wait by
 * bifrost_async_next_timer(), and bifrost_async_run_ready() fires those due. With
 * no event loop, blocking code sleeps until the next timer.
 *
 * Everything runs on the thread that runs the event loop, so nothing is locked.
 * Bifrost compiles this file with the program's linker and links it in when the
 * program uses async.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*Resume)(void *);

typedef struct Waiter {
    void *handle;
    Resume resume;
    struct Waiter *next;
} Waiter;

/* A token, a value (with storage), or a group (of tokens, `pending` not yet ready). */
typedef struct Object {
    int64_t refs;
    bool ready, error;
    int64_t pending;
    struct Object *group; /* the group this token was added to, if any */
    Waiter *waiters;
    char *storage;
} Object;

/* A token std:time's sleep returns, completed at `deadline` (ms on the monotonic clock). */
typedef struct Timer {
    int64_t deadline;
    struct Object *token;
    struct Timer *next; /* timers are kept sorted by deadline */
} Timer;

static Waiter *ready_head, *ready_tail;
static Timer *timers;
static void (*poller)(void *);
static void *poller_context;

static void *allocate(size_t size)
{
    void *memory = calloc(1, size);
    if (memory == NULL) {
        fputs("bifrost: out of memory in the async runtime\n", stderr);
        abort();
    }
    return memory;
}

static void schedule(void *handle, Resume resume)
{
    Waiter *waiter = allocate(sizeof(Waiter));
    waiter->handle = handle;
    waiter->resume = resume;
    if (ready_tail != NULL)
        ready_tail->next = waiter;
    else
        ready_head = waiter;
    ready_tail = waiter;
}

static Object *create(int64_t refs)
{
    Object *object = allocate(sizeof(Object));
    object->refs = refs;
    return object;
}

static void wake(Object *object)
{
    Waiter *waiter = object->waiters;
    object->waiters = NULL;
    while (waiter != NULL) {
        Waiter *next = waiter->next;
        schedule(waiter->handle, waiter->resume);
        free(waiter);
        waiter = next;
    }
}

static void complete(Object *object, bool error)
{
    object->ready = true;
    object->error = error;
    Object *group = object->group;
    if (group != NULL) {
        group->error |= error;
        if (--group->pending == 0) {
            group->ready = true;
            wake(group);
        }
    }
    wake(object);
}

static void await_and_execute(Object *object, void *handle, Resume resume)
{
    if (object->ready) {
        schedule(handle, resume);
        return;
    }
    Waiter *waiter = allocate(sizeof(Waiter));
    waiter->handle = handle;
    waiter->resume = resume;
    waiter->next = object->waiters;
    object->waiters = waiter;
}

/* -- timers ------------------------------------------------------------------------ */

static int64_t now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

void mlirAsyncRuntimeDropRef(void *pointer, int64_t count);

void mlirAsyncRuntimeEmplaceToken(void *token);

/* Complete the tokens of the timers that are due. */
static void fire_timers(void)
{
    int64_t now = now_ms();
    while (timers != NULL && timers->deadline <= now) {
        Timer *timer = timers;
        timers = timer->next;
        mlirAsyncRuntimeEmplaceToken(timer->token);
        free(timer);
    }
}

void *mlirAsyncRuntimeCreateToken(void);

/* std:time's sleep: a token that completes after `milliseconds`. */
void *bifrost_time_sleep(int64_t milliseconds)
{
    Object *token = mlirAsyncRuntimeCreateToken();
    Timer *timer = allocate(sizeof(Timer));
    timer->deadline = now_ms() + (milliseconds > 0 ? milliseconds : 0);
    timer->token = token;
    Timer **at = &timers;
    while (*at != NULL && (*at)->deadline <= timer->deadline)
        at = &(*at)->next;
    timer->next = *at;
    *at = timer;
    return token;
}

/* Milliseconds since some fixed point: for measuring time. */
int64_t bifrost_time_now(void) { return now_ms(); }

/* -- for event loops ---------------------------------------------------------------- */

/* Milliseconds until the next timer is due (0 if one is), or -1 if there is none:
   an event loop waits at most this long for events. */
int64_t bifrost_async_next_timer(void)
{
    if (timers == NULL)
        return -1;
    int64_t left = timers->deadline - now_ms();
    return left > 0 ? left : 0;
}

/* Run the coroutines that are ready (and those whose timers are due), including
   those they make ready. */
void bifrost_async_run_ready(void)
{
    fire_timers();
    while (ready_head != NULL) {
        Waiter *waiter = ready_head;
        ready_head = waiter->next;
        if (ready_head == NULL)
            ready_tail = NULL;
        Resume resume = waiter->resume;
        void *handle = waiter->handle;
        free(waiter);
        resume(handle);
    }
}

/* Whether coroutines are ready to run: an event loop should not sleep then. */
bool bifrost_async_pending(void) { return ready_head != NULL || bifrost_async_next_timer() == 0; }

/* How to wait for events when code blocks on a result and nothing is ready:
   `poll(context)` returns after handling some. */
void bifrost_async_set_poller(void (*poll)(void *), void *context)
{
    poller = poll;
    poller_context = context;
}

/* tasks.ignore: the caller lets go of a task's token, not waiting for it; the task
   runs on (its token lives until the task completes it). */
void bifrost_async_forget(void *token) { mlirAsyncRuntimeDropRef(token, 1); }

static void block_on(Object *object)
{
    while (!object->ready) {
        bifrost_async_run_ready();
        if (object->ready)
            break;
        if (poller != NULL) {
            poller(poller_context);
            continue;
        }
        int64_t wait = bifrost_async_next_timer();
        if (wait < 0) {
            fputs("bifrost: waiting for something that nothing can complete (no event loop is running)\n", stderr);
            abort();
        }
        struct timespec pause = {wait / 1000, (wait % 1000) * 1000000};
        nanosleep(&pause, NULL);
    }
}

/* -- MLIR's async runtime API -------------------------------------------------------- */

void mlirAsyncRuntimeAddRef(void *object, int64_t count) { ((Object *)object)->refs += count; }

void mlirAsyncRuntimeDropRef(void *pointer, int64_t count)
{
    Object *object = pointer;
    object->refs -= count;
    if (object->refs == 0) {
        free(object->storage);
        free(object);
    }
}

/* A token or value starts with two references, as in MLIR's own runtime: one for
   whoever waits on it, and one that completing it drops, so it lives until it is
   complete even when nothing waits on it (tasks.ignore). */
void *mlirAsyncRuntimeCreateToken(void) { return create(2); }

void *mlirAsyncRuntimeCreateValue(int64_t size)
{
    Object *value = create(2);
    value->storage = allocate(size > 0 ? (size_t)size : 1);
    return value;
}

void *mlirAsyncRuntimeCreateGroup(int64_t size)
{
    (void)size;
    Object *group = create(1);
    group->ready = true;
    return group;
}

int64_t mlirAsyncRuntimeAddTokenToGroup(void *token_pointer, void *group_pointer)
{
    Object *token = token_pointer, *group = group_pointer;
    if (!token->ready) {
        if (token->group != NULL) {
            fputs("bifrost: a token was added to a second group\n", stderr);
            abort();
        }
        token->group = group;
        group->ready = false;
        group->pending++;
    }
    group->error |= token->error;
    return group->pending;
}

/* Complete a token or value, and drop the reference it had until then. */
static void finish(Object *object, bool error)
{
    complete(object, error);
    mlirAsyncRuntimeDropRef(object, 1);
}

void mlirAsyncRuntimeEmplaceToken(void *token) { finish(token, false); }
void mlirAsyncRuntimeEmplaceValue(void *value) { finish(value, false); }
void mlirAsyncRuntimeSetTokenError(void *token) { finish(token, true); }
void mlirAsyncRuntimeSetValueError(void *value) { finish(value, true); }
bool mlirAsyncRuntimeIsTokenError(void *token) { return ((Object *)token)->error; }
bool mlirAsyncRuntimeIsValueError(void *value) { return ((Object *)value)->error; }
bool mlirAsyncRuntimeIsGroupError(void *group) { return ((Object *)group)->error; }
void mlirAsyncRuntimeAwaitToken(void *token) { block_on(token); }
void mlirAsyncRuntimeAwaitValue(void *value) { block_on(value); }
void mlirAsyncRuntimeAwaitAllInGroup(void *group) { block_on(group); }
char *mlirAsyncRuntimeGetValueStorage(void *value) { return ((Object *)value)->storage; }
void mlirAsyncRuntimeExecute(void *handle, Resume resume) { schedule(handle, resume); }

void mlirAsyncRuntimeAwaitTokenAndExecute(void *token, void *handle, Resume resume)
{
    await_and_execute(token, handle, resume);
}

void mlirAsyncRuntimeAwaitValueAndExecute(void *value, void *handle, Resume resume)
{
    await_and_execute(value, handle, resume);
}

void mlirAsyncRuntimeAwaitAllInGroupAndExecute(void *group, void *handle, Resume resume)
{
    await_and_execute(group, handle, resume);
}

int64_t mlirAsyncRuntimGetNumWorkerThreads(void) { return 1; } /* sic: MLIR's name */

void mlirAsyncRuntimePrintCurrentThreadId(void) {}
