# threads

`spawn(f(x))` runs a call on a new OS thread. `main` here is plain (`type: sync` in
`config.yaml`), and spawns two async functions: each runs on an event loop of its own
thread, so `sum_it` and `timer` sleep at the same time. `await job` waits for a thread
and gives its result (`sum_it`'s sum); `await job2` waits for `timer` to end.

## Build and run

```bash
bfc build
./build/threads
```

`bfc build` compiles `src/threads/main.bif` (the `entry` in `config.yaml`).
Modules in `src/` import each other from there: `import("threads.utils:text")`
is the module `text` of `src/threads/utils.bif`.
