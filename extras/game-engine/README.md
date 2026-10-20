# game_engine

A small game engine for Bifrost, on the [raylib](../raylib) package, written in Bifrost. It is a Bifrost [package](../../README.md#packages): `examples/breakout` uses it.

A game is an app of handlers, as an HTTP server is an app of routes, but each handler is a pure function of the frame and the world:

- an **update** (`game.on_update`, `game.on_key`, `game.on_click`, `game.every`, `game.on_start`) takes the frame and the world, and returns the next world;
- a **draw** (`game.on_draw`) takes the frame and the world, and draws it with raylib.

The app is a value, `game.App[W]`, generic over the world `W`. Each `game.on_*` returns a new app, so it is built with pipes:

```bifrost
let rl = import("raylib")
let game = import("game_engine.game:game")
let keys = import("game_engine.keys:keys")

let World = struct { let x: f32, let jumps: i64 }

let update = [game.down] (f: game.Frame, w: World) => World {
    let speed = 0.0
    if game.down(keys.RIGHT) {
        let speed = 200.0
    }
    return World(x: w.x + speed * f.dt, jumps: w.jumps)
}

let jump = [] (_f: game.Frame, w: World) => World World(x: w.x, jumps: w.jumps + 1)

let draw = [rl.draw_rectangle] (_f: game.Frame, w: World) => null {
    rl.draw_rectangle(i32(w.x), 200, 40, 40, rl.Color(r: 230, g: 41, b: 55, a: 255))
}

let main = [game.app, game.on_update, game.on_key, game.on_draw, game.run, update, jump, draw] () => i32 {
    return game.app("Walk", 800, 450, World(x: 0.0, jumps: 0))
        |> game.on_update(update)
        |> game.on_key(keys.SPACE, jump)
        |> game.on_draw(draw)
        |> game.run
}
```

## What it has

- `game.game`: `app(title, width, height, world)`; `fps`, `background`; the handlers above; `run(app)`, which opens the window and runs the game until it is closed (Escape closes it). Each frame runs the handlers whose time it is, in the order they were added, then clears the screen and runs the draws. A `game.Frame` has `dt` (seconds the last frame took), `time`, `number`, `width`, `height` and `mouse`. `game.down(key)` and `game.pressed(key)` read the keyboard.
- `game_engine.keys`: raylib's key codes, `keys.SPACE`, `keys.LEFT`, `keys.A`, `keys.N1`.
- `game_engine.physics`: pure helpers on raylib's `Rectangle` and `Vector2`: `overlaps`, `moved`, `kept_in`, `center`, `around`, `step`, `clamp`.

Since a game's rules are pure, they can be tested without a window: make a world, and step it.

## Use and package

```yaml
# a project's config.yaml
packages:
  game_engine: 0.1.0     # from the index; or `path:` this folder
```

It uses raylib 0.1.2, whose library is linked into the app (for its target). It is all Bifrost, so `bfc package` makes one `dist/game_engine-0.1.0-any.bifpkg`.
