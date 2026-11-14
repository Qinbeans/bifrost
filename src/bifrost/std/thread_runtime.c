/* What `spawn(f(x))` (a sync.Thread) compiles to: f(x) run on a new OS thread.
 *
 * A thread is one block on the heap:
 *
 *     [pthread_t][state][entry][result size][f's result][f's arguments]
 *
 * The generated code allocates it (bifrost_thread_new, with the result's and arguments' sizes),
 * stores the arguments, and starts it (bifrost_thread_start). The new thread runs
 * `entry(thread)`, a function the compiler generates for that spawn: it
 * calls f with the arguments, stores the result, and frees what the thread owned.
 *
 * Then whoever holds the thread either waits for it (`await t`: bifrost_thread_join,
 * read the result, bifrost_thread_free) or lets it go (`tasks.ignore(t)`:
 * bifrost_thread_ignore). `state` says which of the two sides is done last, so that
 * one frees the block: RUNNING, then DONE when f has returned, or IGNORED when nothing
 * will wait; a thread ending after it was ignored frees itself.
 *
 * In async code, `await t` pauses only the task waiting: the event loop watches the
 * state (bifrost_async_watch, in async_runtime.c), and completes the task's token
 * when it is DONE.
 *
 * Bifrost compiles this file with the program's linker and links it in when the
 * program spawns a thread.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { RUNNING, DONE, IGNORED };

typedef struct Thread {
    pthread_t id;
    int64_t state;
    void (*entry)(void *);
    int64_t result_size; /* rounded up to 16: the arguments start 16-byte aligned, like the result */
} Thread;

static int64_t rounded(int64_t size) { return size > 0 ? (size + 15) / 16 * 16 : 0; }

/* A new thread, not started, with room (zeroed) for a result and arguments of these sizes. */
void *bifrost_thread_new(int64_t result_size, int64_t arguments_size)
{
    size_t size = sizeof(Thread) + (size_t)rounded(result_size) + (size_t)rounded(arguments_size);
    Thread *thread = calloc(1, size);
    if (thread == NULL) {
        fputs("bifrost: out of memory starting a thread\n", stderr);
        abort();
    }
    thread->result_size = rounded(result_size);
    return thread;
}

/* Where f's result goes. */
void *bifrost_thread_result(void *thread) { return (char *)thread + sizeof(Thread); }

/* Where f's arguments go. */
void *bifrost_thread_arguments(void *pointer)
{
    Thread *thread = pointer;
    return (char *)thread + sizeof(Thread) + thread->result_size;
}

static void *run(void *pointer)
{
    Thread *thread = pointer;
    thread->entry(thread);
    int64_t before = __atomic_exchange_n(&thread->state, DONE, __ATOMIC_ACQ_REL);
    if (before == IGNORED)
        free(thread); /* nothing waits: the thread was detached */
    return NULL;
}

/* Run `entry(thread)` on a new OS thread: it reads the arguments and stores the result. */
void bifrost_thread_start(void *pointer, void (*entry)(void *))
{
    Thread *thread = pointer;
    thread->entry = entry;
    int failed = pthread_create(&thread->id, NULL, run, thread);
    if (failed) {
        fprintf(stderr, "bifrost: could not start a thread (error %d)\n", failed);
        abort();
    }
}

/* Wait for the thread to end (`await t` in code that is not async). */
void bifrost_thread_join(void *pointer)
{
    Thread *thread = pointer;
    pthread_join(thread->id, NULL);
}

/* The thread's state, which the event loop watches (`await t` in async code). */
int64_t *bifrost_thread_state(void *pointer) { return &((Thread *)pointer)->state; }

/* Free a thread that was waited for, once its result was read. */
void bifrost_thread_free(void *pointer) { free(pointer); }

/* `tasks.ignore(t)`: nothing will wait for the thread. It runs on, and frees itself when it ends. */
void bifrost_thread_ignore(void *pointer)
{
    Thread *thread = pointer;
    pthread_detach(thread->id);
    int64_t before = __atomic_exchange_n(&thread->state, IGNORED, __ATOMIC_ACQ_REL);
    if (before == DONE)
        free(thread); /* it ended already */
}
