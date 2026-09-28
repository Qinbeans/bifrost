# vscode-bifrost

Syntax highlighting for [Bifrost](https://github.com/Qinbeans/bifrost) (`.bif`) files.

## Features

- Keywords, primitive types (`i32`, `f64`, `str`, ...), built-ins (`import`, `len`, ...), and literals (`0x1f`, `0o7`, `0b1`, `1.5`, `"text"`)
- Function names in `let name = (...) => ...` definitions and at call sites
- Objects: the name in `let Name = struct { ... }`, and PascalCase names wherever they are used (`Context`, `rl.Color`)
- Line (`//`) and block (`/* */`) comments, bracket matching, and auto-indent inside `{ }`

## Language server

Diagnostics (including naming: snake_case variables and functions, PascalCase objects), formatting, outline, go-to-definition, hover and completion come from the Bifrost language server (`bifrost lsp`). The extension runs the `bifrost` executable from a `.venv` in a workspace folder (or one of its subfolders), else `bifrost` on `PATH`; set `bifrost.server.path` to use another. **Bifrost: Restart Language Server** restarts it.

## Development

Build the client with `pnpm run build`, and package with `pnpm run package`.


Open this folder in VS Code and press `F5` to launch an Extension Development Host with the extension loaded.
