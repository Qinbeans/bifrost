# Bifrost

A programming language designed around the fundamental distinction between stateful and stateless computation. Rather than treating this as an implicit concern, Bifrost makes it a first-class feature of the language: a function declares everything it depends on, and the compiler holds it to that.

```bifrost
let rl = import("raylib")
let stdio = import("std:stdio")

let Context = struct {
    let window_width: i32,
    let window_height: i32,
    let window_title: str,
    static let new = [] () => Context {
        return Context(window_width: 800, window_height: 600, window_title: "Hello Raylib")
    }
}

let draw = [rl.begin_drawing, rl.clear_background, rl.end_drawing] () => null {
    rl.begin_drawing()
    rl.clear_background(rl.Color(0, 0, 0, 255))
    rl.end_drawing()
}

let main = [rl.init_window, rl.close_window, rl.window_should_close, draw, Context.new, stdio.printf] () => null {
    let ctx = Context.new()
    rl.init_window(ctx.window_width, ctx.window_height, ctx.window_title)
    stdio.printf("window is %dx%d\n", ctx.window_width, ctx.window_height)
    while !rl.window_should_close() {
        draw()
    }
    rl.close_window()
}
```

## Overview

Bifrost source is parsed with [tree-sitter-bifrost](https://github.com/Qinbeans/tree-sitter-bifrost), checked, and lowered to [mlir-python](https://github.com/Qinbeans/mlir-python), which compiles it through MLIR and LLVM to a native executable. C libraries are called directly: their declarations live in a project's `config.yaml`, and can be generated from their headers.

## The language

### Dependency lists

The list in front of a function names every function it may call, and nothing else:

```bifrost
let square = (a: i32) => i32 a * a
let area = [square] (side: i32) => i32 square(side)
```

The compiler checks it both ways: calling a function that is not listed is an error, and so is listing one that is never called. A function without a list calls nothing, and `[]` says so explicitly. `this` lets a function call itself. What a function can reach is always visible in its signature, which keeps functions small and their effects explicit.

### Functions are values

A function is a value like any other. Its type is written like its head, `(x: i32) => i32` (parameter names are optional: `(i32) => i32`), and it can be passed, stored in a `let` or an object field, and called:

```bifrost
let double = [] (x: i32) => i32 x * 2
let apply = [] (f: (x: i32) => i32, x: i32) => i32 f(x)

let main = [apply, double] () => null {
    apply(double, 21)                              // 42
    apply([] (x: i32) => i32 x + 1, 41)            // a lambda: 42
}
```

Naming a function without calling it counts as depending on it, so `main` lists `double`; calling a function value (`f(x)` in `apply`) needs no entry, because its type already says what it is. A lambda has its own dependency list, and cannot use the locals of the function around it yet (pass them as parameters). Function values are C function pointers, so C libraries can call back into Bifrost.

### Objects

`struct` defines an object. `let name: Type` members are stored fields, laid out as a C struct, so objects pass to and from C unchanged. `static let` members are functions called on the type (`Context.new()`) and take no space in the object. Objects are built with named arguments: `Context(window_width: 800, ...)`.

### Owners, guards and shared values

`std:mem` holds the containers that say how a value is held. A `mem.Unique[T]` local is the one owner of its value; a `mem.Weak[T]` parameter is that value lent to a call. Either is reached only through a guard, a lock that exists only at compile time.

```bifrost
let mem = import("std:mem")

let tick = [] (ctx: mem.Weak[Context]) => null {
    let guard <- ctx                     // lock: ctx is reached only through guard
    guard.counter = guard.counter + 1    // read and write through the guard
    guard -> ctx                         // release
}

let main = [tick, Context.new] () => null {
    let ctx: mem.Unique[Context] = Context.new()
    tick(ctx)                            // lends ctx; the change is visible afterwards
}
```

The compiler checks that:

- the release is in the scope that took the lock, or a scope nested in it, and every path releases exactly once (an `if` without `else`, a `match` arm, a loop body or an early `return` that skips or repeats the release is an error);
- a held value is not used while it is locked, and a guard is not used after it is released;
- fields change only through a guard, so every mutation is inside a visible lock;
- a lent value is only lent on or locked: it is never copied, returned or stored in an object, so it cannot outlive the call that lends it.

None of this reaches the generated code: a `mem.Weak[T]` parameter is a plain pointer, and a guard compiles to what it locks. Raw `ptr` exists only in C extern declarations.

A value with several owners is a `mem.Shared[T]`, or a `mem.Atomic[T]` when several threads own it. Each `let` of one is another owner, and the value is freed when the last owner's scope ends:

```bifrost
let make_counter = [] () => mem.Shared[Counter] {
    let counter: mem.Shared[Counter] = Counter(hits: 0)   // a new value on the heap
    return counter                                        // the caller owns it now
}

let bump = [] (counter: mem.Shared[Counter]) => null {    // borrowed for the call
    let g <- counter
    g.hits = g.hits + 1
    g -> counter
}

let main = [make_counter, bump] () => null {
    let counter = make_counter()
    let other = counter                  // a second owner of the same counter
    bump(other)                          // counter sees it: they own one value
}                                        // both owners end: the counter is freed
```

These cost a little at run time: each is a pointer to its value, after a count of its owners. Guards are checked at run time too:

- locking a `mem.Atomic` waits for its mutex, and its owners are counted atomically, so any thread may lock it, keep it or let it go;
- a `mem.Shared` is for one thread: its count is a plain integer, and taking a second guard while another owner's is held stops the program with the line that tried;
- lending one to a `mem.Weak` parameter locks it for the call. Passing one to C's `void *` hands over the pointer instead, for C code that passes it back to functions that take a `mem.Shared` or `mem.Atomic` and lock it themselves, such as a web server's handlers or a thread's.

A guard on a whole value, such as a `mem.Atomic[i64]`, reads it as the guard's name and writes it with `guard = ...`:

```bifrost
let counter: mem.Atomic[i64] = 0
let count <- counter
count = count + 1                        // written through the lock
count -> counter
```

Objects cannot hold one yet.

### Formatting and owned strings

`std:fmt` formats like `printf` into a string allocated to fit, and hands back its ownership:

```bifrost
let fmt = import("std:fmt")
let mem = import("std:mem")

let greet = [fmt.format] (name: str) => mem.Unique[str] {
    return fmt.format("Hello, %s!", name)
}

let main = [greet, stdio.puts] () => null {
    let message = greet("World")    // main owns it now
    stdio.puts(message)             // lent: puts only reads it
}                                   // freed here
```

An owned string (`mem.Unique[str]`) is freed where its owner's scope ends, on every path, unless ownership moves first: `return message`, `let other = message`, or passing it to a `mem.Unique[str]` parameter. The compiler checks that a moved string is not used again, that it moves on every path through a branch or on none (never inside a loop), and that every owned result gets an owner, so nothing leaks and nothing is freed twice. There is no `malloc` or `free` to call yourself.

### Records and JSON

A record is a value with named fields, written where it is used: `#{id: 7, name: "user 7", admin: false}`. Its field types come from its values, and records with the same fields have the same type. `std:json` turns records and objects into JSON text:

```bifrost
let stdio = import("std:stdio")
let json = import("std:json")

let User = struct { let id: i64, let admin: bool }

let main = [json.encode, stdio.puts] () => null {
    let user = User(id: 7, admin: false)                    // an object
    let body = json.encode(#{user: user, roles: 2})         // mem.Unique[str], freed with its owner
    stdio.puts(body)                                        // {"user": {"id": 7, "admin": false}, "roles": 2}
}
```

The compiler writes the conversion from the field names and types, so there is no reflection at run time: strings are escaped, numbers and bools written as they are, and nested records and objects become nested JSON objects. A C function can take JSON directly: a parameter declared `json` in `config.yaml` is given a record or object encoded at the call (and freed after it), or a `str` as JSON text already. That is how the HTTP example responds: `http.json(ctx, 200, #{sum: a + b})`.

### Lists

A list is written `#[4, 8, 15]` and typed `i64[]`. Index it with `xs[i]` (`xs[-1]` is the last item; an index out of range stops the program), take `len(xs)`, and loop over it with `forall`. `#[a...b]` is the range from `a` up to `b`, `b` excluded; `forall` counts through a range without making a list:

```bifrost
let total = [] (xs: i64[]) => i64 {
    let sum = 0
    forall x in xs {
        let sum = sum + x
    }
    return sum
}

let main = [total, io.printf] () => null {
    let scores = #[4, 8, 15, 16, 23, 42]
    io.printf("%lld of %lld\n", total(scores), len(scores))
    forall i in #[0...3] {
        io.printf("%lld\n", scores[i])
    }
}
```

Lists hold integers, floats or bools, for now. A list is freed once nothing uses it, in async functions too; functions borrow the lists passed to them and give the lists they return to their caller.

### HTTP servers

`examples/http` is a JSON API on [h2o](https://github.com/h2o/h2o), serving HTTP/1.1, HTTP/2 and, with a certificate, HTTP/3. Its `http` library is the project's own: a small C wrapper (`c/http.c`) built with CMake and declared in its `config.yaml`, as any C library is. Routes take handlers, fasthttp style, since functions are values:

```bifrost
let http = import("http")

let get_user = [http.param_int, http.error, http.json] (ctx: http.Context) => null {
    let id = http.param_int(ctx, "id", 0)
    if id < 1 {
        http.error(ctx, 404, "no such user")
        return
    }
    http.json(ctx, 200, #{id: id})
}

let main = [http.app, http.get, http.run, get_user] () => i32 {
    let app = http.app()
    http.get(app, "/users/:id", get_user)
    http.get(app, "/health", [http.text] (ctx: http.Context) => null http.text(ctx, 200, "ok"))
    return http.run(app, 8080)      // $PORT, or 8080; HTTPS and HTTP/3 with $TLS_CERT and $TLS_KEY
}
```

State the handlers share is owned by `main` and passed to the server: `http.run_with(app, 8080, shared)` gives it to handlers registered with `http.get_with`, which take it as `mem.Shared[AppState]` and reach it through a guard. The example's README shows how to build and try it.

### Async functions

A function that waits (for a timer, the network, a database) is `async`, and `await`s each call that pauses; while it waits, other work runs (an HTTP server serves other requests). A call pauses if it calls an `async` function, or a C function that returns a `token` (declared so in `config.yaml`, as `http.sleep` is, or `std:time`'s `sleep`):

```bifrost
let slow = [http.query_int, http.sleep, http.json] async (ctx: http.Context) => null {
    let ms = http.query_int(ctx, "ms", 100)
    await http.sleep(ms)                // /health and others are served meanwhile
    http.json(ctx, 200, #{slept: ms})
}
```

The compiler knows which calls pause, so it checks the keywords rather than trusting them: a call that pauses without `await`, an `await` on a call that does not pause, and an `await` outside an `async` function are errors. `main` may be `async` when the project says so, with `type: async` in `config.yaml`'s `package` (the default is `sync`); the program runs it on the event loop and ends when it is done:

```bifrost
let time = import("std:time")

let main = [time.now, time.sleep, stdio.printf] async () => null {
    let started = time.now()
    await time.sleep(100)
    stdio.printf("slept %lld ms\n", time.now() - started)
}
```

Async functions compile to coroutines (the MLIR async dialect, lowered to LLVM coroutines). The runtime (`std/async_runtime.c`, compiled into programs that use it) runs them all on one thread, so shared state needs no locks, but a guard cannot be held across an `await`: other code could change what it holds meanwhile, so the compiler asks to release it first and lock again after.

Awaiting one call after another runs them one at a time. `std:tasks`'s `gather` starts several at once and waits for all of them, like Python's `asyncio.gather`; named calls give a record of their results:

```bifrost
let tasks = import("std:tasks")

let page = [tasks.gather, fetch_user, fetch_orders, audit] async (id: i64) => null {
    let found = await tasks.gather(user: fetch_user(id), orders: fetch_orders(id), audit(id))
    // found.user and found.orders; it took as long as the slowest call, not all three
}
```

Each argument is a call that pauses. One that returns a value is named; one that returns nothing (`audit`) is passed without a name. The calls take turns at their `await`s on the one thread, and cannot borrow a `mem.Weak` (a `mem.Shared` or `mem.Atomic` is fine).

To start a call without waiting for it, deliberately, pass it to `tasks.ignore`, e.g. `tasks.ignore(audit(id))`. It runs alongside the caller, taking turns at the `await`s, but nothing waits for it, so it is cut off if `main` ends first. Because it may outlive the caller, it returns nothing and takes only numbers, booleans and string literals.

A C library that runs an event loop lets the runtime use it: it calls `bifrost_async_run_ready()` after handling events, waits at most `bifrost_async_next_timer()` milliseconds (for `std:time`), and calls `bifrost_async_set_poller(poll, context)` so that blocking code can turn the loop. A function passed where C expects a `(...) => token` callback, such as an HTTP handler, is started by C, which gets the token to wait on; plain functions are wrapped so they fit too. `examples/async` and `examples/http` show both.

### C libraries and standard modules

`import("raylib")` brings in a module declared in `config.yaml`. Its functions take Bifrost names: `InitWindow` is `rl.init_window`, and structs keep PascalCase (`rl.Color`). The C symbols are unchanged.

`import("std:stdio")` brings in a standard module bundled with the compiler (`printf`, `puts`, `fopen`, ...), with no configuration needed. `import("std:mem")` brings in the memory containers above, and `import("std:fmt")` formatting.

### Modules

A file groups its code into modules, and exports the ones other files may use:

```bifrost
// src/app/text.bif
let stdio = import("std:stdio")

module greeting = {
    /* Greets people */
    let greet = [stdio.printf] (name: str) => null stdio.printf("Hello, %s!\n", name)
}

export(greeting)
```

```bifrost
// main.bif
let greeting = import("app.text:greeting")

let main = [greeting.greet] () => null greeting.greet("Bifrost")
```

`import("file:module")` finds `file` with Python-style dots from the project's `src/` folder (or, without one, the folder with `config.yaml`): in a project `app`, `app.utils` is `src/app/utils.bif`. A leading `.` looks next to the importing file instead, and each further `.` one folder up. A module's members see each other by name; other files call them through their dependency lists, like any function. Import cycles are errors. The block comment at the top of a module is its description, shown in the editor.

### Conventions

Variables, functions, parameters and fields are `snake_case`; objects are `PascalCase`. The language server flags names that break this.

## Tooling

| Command | What it does |
|---|---|
| `bifrost init my_api` | Create a project: `config.yaml`, `.gitignore`, `README.md`, `src/my_api/main.bif` |
| `bifrost build` (or `bifrost build file.bif -c config.yaml -o out`) | Compile the project's entry (or a given file) to a native executable in `build/` |
| `bifrost fmt main.bif` (`--check` to only report) | Format source files; only ever changes whitespace |
| `bifrost lsp` | Run the language server (diagnostics, formatting, outline, hover, go-to-definition, completion) |
| `bifrost config traverse -c config.yaml -i path/to/include/` | Generate extern declarations in `config.yaml` from C/C++ headers |

[vscode-bifrost](../vscode-bifrost) adds syntax highlighting and connects VS Code to the language server.

## Project configuration

A project's `config.yaml` names the executable, the optimization level and linker, the libraries to link, and the C declarations Bifrost code can import:

```yaml
package:
  name: hello_raylib
  version: 0.1.0
  description: A simple example using Raylib
  type: sync                 # or async: main may be async, and runs on the event loop

flags:
  optimization: 2
  linker: clang

libraries:
  - ./build/lib/libraylib.a
  - m

externs:
  - module: raylib
    description: Raylib module
    declarations:
      - name: Color
        type: struct
        fields: { r: u8, g: u8, b: u8, a: u8 }
      - name: InitWindow
        type: function
        parameters: { width: i32, height: i32, title: cstr }
        return: None
```

Library paths are relative to `config.yaml`. A parameter that takes a C callback has a function type, `handler: "(http_Context) => None"`: Bifrost passes it a function (or a lambda), and C calls it back. A declaration's `as:` gives it a Bifrost name other than the default, and `doc:` a line shown in the editor.

Writing `externs` by hand is rarely needed: `bifrost config traverse` reads the headers with libclang and generates them, one module per header. It reports each declaration it skips and why (variadic functions, structs with array members, C++ functions without `extern "C"`, ...).

## Architecture

```
bifrost/
├── src/bifrost/
│   ├── __main__.py        # CLI: build, fmt, lsp, config traverse
│   ├── syntax.py          # Describes syntax errors in terms of the source
│   ├── guards.py          # Checks locks and releases along every path
│   ├── ownership.py       # Checks moves of owned values, and plans where they are freed
│   ├── lowering/          # Checks a CST (dependencies, names), loads imported modules, and lowers to Python AST
│   ├── project/           # Declares configured and standard externs; builds the executable
│   ├── configs/           # config.yaml schema, and header traversal (libclang)
│   ├── std/               # Standard modules: stdio.yaml, and mem, fmt and json (built in)
│   ├── naming.py          # snake_case / PascalCase checks and conversions
│   ├── formatter/         # The formatter
│   └── server/            # The language server (pygls) and its analysis
├── examples/              # hello_raylib.bif, its config.yaml, and a CMake file that builds raylib
└── tests/
```

### Compilation pipeline

1. Parse the source with tree-sitter-bifrost. Syntax errors are reported with the token that caused them and what was expected.
2. Bind the top-level names: imported modules, objects, constants and functions.
3. Check each function (guards, dependency lists, names in scope) and lower it to a Python function definition.
4. Hand the functions and the declared externs to mlir-python, which type-checks them and emits MLIR.
5. mlir-python compiles to LLVM IR and links the executable with the configured linker and libraries.

Errors from every step, including type errors found by mlir-python, point at the `.bif` source.

## Installation

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and `clang` on `PATH` (the default linker; `config traverse` also uses its built-in headers).

```bash
uv sync
```

`mlir-python` and `tree-sitter-bifrost` are installed from the package index at `https://qinbeans.github.io/mlir-python/`, configured in `pyproject.toml`.

## Usage

Start a project:

```bash
bifrost init my_api
cd my_api
bifrost build          # compiles package.entry (src/my_api/main.bif) into build/
./build/my_api
```

A project uses the source layout:

```
my_api/
├── .gitignore
├── README.md
├── config.yaml          # package (name, version, description, entry), flags, libraries, externs
└── src/
    └── my_api/
        └── main.bif     # the entry point; other files here import as my_api.<file>
```

Read or change the version in config.yaml, like `uv version`:

```bash
bifrost version                    # my_api 0.1.0
bifrost version --bump minor       # my_api 0.1.0 => 0.2.0 (also major, patch)
bifrost version 1.0.0 --dry-run    # show the change without writing it
bifrost version --short            # 0.1.0
```

The raylib example needs raylib built first:

```bash
cd examples/raylib
cmake -S . -B build && cmake --build build   # builds build/lib/libraylib.a
bifrost build
./build/hello_raylib
```

## Development

```bash
uv run pytest          # run the tests
uv run ruff check      # lint
uv run ruff format     # format
```

## Technology Stack

- **tree-sitter**: parsing, with the grammar in [tree-sitter-bifrost](https://github.com/Qinbeans/tree-sitter-bifrost)
- **mlir-python**: MLIR/LLVM code generation and linking
- **pygls**: the language server
- **libclang**: reading C/C++ headers
- **Pydantic**: configuration validation
- **Python 3.13**, **uv**, **Ruff**, **Pytest**

## Roadmap

- **Closures**: lambdas that capture the locals around them
- **Methods**: struct functions without `static`, reading the instance through `[super]`
- **More of `std:mem`**: owned values other than strings (lists, buffers), and objects that hold a `mem.Shared` or `mem.Atomic`
- **String interpolation**: `"Hello, {name}!"`, compiled to a checked `fmt.format`
- **Opaque C types** in extern declarations, such as `FILE`
- **Trailing commas**
- **More of lists**: strings and objects in lists, lists in records (and so in JSON), and spreading (`#[...xs, 4]`)
- **Quick fixes** in the language server, such as adding a missing dependency

## Project Goals

Bifrost aims to:

1. **Make statefulness explicit**: what a function depends on and changes is part of its signature
2. **Leverage MLIR**: use modern compiler infrastructure for optimizations
3. **Check at compile time, not run time**: dependency lists and pointer guards cost nothing in the generated code
4. **Call C directly**: C libraries are usable as soon as their headers are traversed
5. **Support cross-platform**: work on Windows, Linux and macOS (currently tested on Linux)

## Contributing

This is an experimental language project. Contributions are welcome!

Areas for contribution:

- Language design and syntax
- Standard modules
- Optimization passes
- Documentation and examples
- Test coverage

## Author

Ryan Fong (ryan.lawrence.fong@gmail.com)
