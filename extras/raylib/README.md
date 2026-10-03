# raylib

[raylib](https://www.raylib.com) 6.0 for Bifrost: windows, drawing, input and audio, for games and graphics. It is a Bifrost [package](../../README.md#packages): `examples/raylib` uses it.

It gives a project the `raylib` module, declared in `config.yaml` (generated from raylib's headers with `bfc config traverse`): C names become Bifrost ones, so `InitWindow` is `raylib.init_window`, and structs keep theirs (`raylib.Color`).

```yaml
# a project's config.yaml
packages:
  raylib:
    path: ../raylib      # this folder; or `raylib: 0.1.2`, from the published index (see the main README)
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

On Linux, one build runs natively on both Wayland and X11: GLFW picks Wayland when the program runs under it, and loads either's libraries when the program starts. Building it needs CMake, a C compiler, `wayland-scanner`, and the development files of X11 (with Xrandr, Xinerama, Xcursor and Xi), Wayland, xkbcommon and OpenGL (`apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libwayland-dev libxkbcommon-dev libgl1-mesa-dev`). Projects using it, from the index or this folder, link X11 (`dnf install libX11-devel`, or `apt install libx11-dev`): raylib's clipboard-image code calls it directly. Build it once; projects using this folder link `build/lib/libraylib.a`:

```bash
cmake -S . -B build && cmake --build build   # raylib 6.0, fetched by CMake
bfc package                                  # dist/raylib-0.1.2-<target>.bifpkg
```
