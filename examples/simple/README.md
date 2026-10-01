# simple

The simple project

## Build and run

```bash
bifrost build
./build/simple
```

`bifrost build` compiles `src/simple/main.bif` (the `entry` in `config.yaml`).
Modules in `src/` import each other from there: `import("simple.utils:text")`
is the module `text` of `src/simple/utils.bif`.
