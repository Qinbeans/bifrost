# json

The json project

## Build and run

```bash
bfc build
./build/json
```

`bfc build` compiles `src/json/main.bif` (the `entry` in `config.yaml`).
Modules in `src/` import each other from there: `import("json.utils:text")`
is the module `text` of `src/json/utils.bif`.
