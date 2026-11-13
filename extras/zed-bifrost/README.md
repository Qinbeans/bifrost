# zed-bifrost

[Zed](https://zed.dev) support for [Bifrost](https://github.com/Qinbeans/bifrost) (`.bif`) files.

## Features

- Syntax highlighting from the [tree-sitter-bifrost](https://github.com/Qinbeans/tree-sitter-bifrost) grammar: keywords, primitive types (`i32`, `f64`, `str`, ...), built-ins, literals, function and object names
- Line (`//`) and block (`/* */`) comments, bracket matching and auto-closing, auto-indent inside `{ }`
- Outline and symbol navigation from the grammar
- Text objects for functions, objects (structs, unions, modules) and comments (vim mode); quotes are not auto-closed inside strings and comments

## Language server

Diagnostics (including naming: snake_case variables and functions, PascalCase objects), formatting, outline, go-to-definition, hover, completion and inlay hints come from the Bifrost language server (`bfc lsp`). The extension picks the `bfc` executable in this order:

1. `lsp.bifrost-lsp.binary.path` from your Zed settings;
2. `.venv/bin/bfc` at the root of the project, or in one of its immediate (non-hidden) subfolders;
3. `bfc` on the project shell's `PATH` (so an activated virtualenv, `direnv`, or a global install also works).

Zed extensions cannot check whether files exist, and Zed does not index git-ignored folders such as `.venv`. So on Linux and macOS the extension starts the server through a short `/bin/sh` script that does this lookup when the server launches. If it finds no `bfc`, the server exits and **zed: open log** shows why.

On Windows only `.venv\Scripts\bfc.exe` at the project root is checked, before `PATH`, and only when Zed can see `.venv\pyvenv.cfg`. Zed usually can't, because `.venv` is ignored, so `bfc` on `PATH` is used if there is one. Otherwise the extension tries `.venv\Scripts\bfc.exe` anyway. Subfolders are not searched. Set the path explicitly when the default lookup picks the wrong `bfc`:

```jsonc
// ~/.config/zed/settings.json or .zed/settings.json in the project
{
  "lsp": {
    "bifrost-lsp": {
      "binary": {
        "path": "/path/to/project/.venv/bin/bfc",
        // Optional; replaces the default ["lsp"], so include "lsp" yourself.
        "arguments": ["lsp"],
        // Optional; merged over the project's shell environment.
        "env": {}
      },
      // Optional; passed through to the server.
      "initialization_options": {},
      "settings": {}
    }
  }
}
```

Restart the server with the **editor: restart language server** command. **zed: open log** shows errors if it fails to start.

Zed turns inlay hints off by default; enable them with `"inlay_hints": { "enabled": true }`.

Unlike the VS Code extension, this one does not watch `**/config.yaml` (Zed extensions cannot declare file watchers). Reopen or save a `.bif` file after editing `config.yaml`.

## Install as a dev extension

Requires Rust installed via `rustup` (Zed builds the extension for `wasm32-wasip2`).

1. Open the command palette and run **zed: install dev extension**.
2. Choose this folder (`extras/zed-bifrost`).

Zed compiles the extension and the grammar (pinned by `rev` in `extension.toml`). After changing the grammar, push it, update `rev`, and reinstall the extension (or use **Rebuild** on the extensions page).

## Development

```sh
cargo build --release --target wasm32-wasip2
cargo clippy --release --target wasm32-wasip2
```

`src/lib.rs` resolves and launches the language server; `languages/bifrost/` holds the language config and tree-sitter queries.
