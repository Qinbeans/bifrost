# async

An async `main`, allowed by `type: async` in `config.yaml`: it awaits `std:time`'s
`sleep` (a call that pauses), and counts with a `mem.Atomic[i64]` through a guard on
the whole value (`count = count + 1`).

## Build and run

```bash
bfc build
./build/async
```

`bfc build` compiles `src/async/main.bif` (the `entry` in `config.yaml`).
Modules in `src/` import each other from there: `import("async.utils:text")`
is the module `text` of `src/async/utils.bif`.
