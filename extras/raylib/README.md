# raylib

[raylib](https://www.raylib.com) 6.0 for Bifrost: windows, drawing, input and audio, for games and graphics. It is a Bifrost [package](../../README.md#packages): `examples/raylib` uses it.

It gives a project the `raylib` module, declared in `config.yaml` (generated from raylib's headers with `bfc config traverse`): C names become Bifrost ones, so `InitWindow` is `raylib.init_window`, and structs keep theirs (`raylib.Color`).

```yaml
# a project's config.yaml
packages:
  raylib:
    path: ../raylib      # this folder; or `raylib: 0.1.0`, from the index (see the main README)
```

```bifrost
let rl = import("raylib")

let main = [rl.init_window, rl.window_should_close, rl.begin_drawing, rl.clear_background, rl.end_drawing, rl.close_window] () => null {
    rl.init_window(800, 450, "Hello from Bifrost")
    while !rl.window_should_close() {
        rl.begin_drawing()
        rl.clear_background(rl.Color(r: 245, g: 245, b: 245, a: 255))
        rl.end_drawing()
    }
    rl.close_window()
}
```

## Build and package

raylib needs CMake, a C compiler, and on Linux the X11 (or Wayland) and OpenGL development files. Build it once; projects using this folder link `build/lib/libraylib.a`:

```bash
cmake -S . -B build && cmake --build build   # raylib 6.0, fetched by CMake
bfc package                                  # dist/raylib-0.1.0-<target>.bifpkg
```
