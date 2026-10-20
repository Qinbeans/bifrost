# Breakout

Breakout, on the [game_engine](../../extras/game-engine) package: a paddle, a ball, and five rows of bricks. Arrows steer, space serves (and starts again once the game is over).

- `world.bif`: what the game is at any moment (the paddle, the ball, the bricks, the score), as values.
- `rules.bif`: the rules, as pure functions: `step` makes the next world from the last, `serve` serves.
- `main.bif`: the handlers the engine runs, each a thin call of the rules, and the app, built with pipes.

## Build and run

It uses the engine and raylib by path, as they are developed in this repository, so raylib's C library is built first (see [its README](../../extras/raylib/README.md)):

```bash
bfc build
./build/breakout
```
