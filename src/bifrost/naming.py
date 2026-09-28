"""Bifrost's naming conventions, and translating C names into them.

Variables, parameters, fields and functions are ``snake_case``; objects
(structs) are ``PascalCase``. Externs keep their C symbols but are named the
Bifrost way: ``raylib.InitWindow`` is ``rl.init_window``, ``rlVertexBuffer`` is
``RlVertexBuffer``.
"""

import keyword
import re

from bifrost.configs.schema import Declaration as _Declaration

_SNAKE = re.compile(r"_?[a-z][a-z0-9]*(_[a-z0-9]+)*_?")
_PASCAL = re.compile(r"[A-Z][a-zA-Z0-9]*")
# Word boundaries in camelCase and PascalCase: `getFPS` -> get|FPS, `HTTPServer` -> HTTP|Server.
_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b|_|\d)|[A-Z]?[a-z]+|[A-Z]+|\d+[a-zA-Z]?(?![a-z])")


def is_snake_case(name: str) -> bool:
    """``draw_text``, ``x``, ``vector2``; a leading or trailing ``_`` is allowed."""
    return name == "_" or _SNAKE.fullmatch(name) is not None


def is_pascal_case(name: str) -> bool:
    """``Color``, ``RenderTexture2D``."""
    return _PASCAL.fullmatch(name) is not None


def _words(name: str) -> list[str]:
    return [word for part in name.split("_") for word in _WORDS.findall(part)]


def to_snake_case(name: str) -> str:
    """``InitWindow`` -> ``init_window``, ``GetFPS`` -> ``get_fps``, ``Vector2Add`` -> ``vector2_add``."""
    words = _words(name)
    if not words:
        return name
    snake = words[0].lower()
    for word in words[1:]:
        # Keep digits on the word before them: Vector2Add -> vector2_add, not vector_2_add.
        snake += word.lower() if word[0].isdigit() else f"_{word.lower()}"
    return f"{snake}_" if keyword.iskeyword(snake) else snake


def to_pascal_case(name: str) -> str:
    """``rlVertexBuffer`` -> ``RlVertexBuffer``, ``float3`` -> ``Float3``, ``Color`` stays."""
    if is_pascal_case(name):
        return name
    return "".join(word[0].upper() + word[1:] for word in _words(name)) or name


def extern_name(declaration: _Declaration) -> str:
    """Name a configured extern in Bifrost: its ``as`` name, else structs PascalCase and functions snake_case."""
    if declaration.bifrost_name:
        return declaration.bifrost_name
    return to_pascal_case(declaration.name) if declaration.type == "struct" else to_snake_case(declaration.name)


__all__ = ["extern_name", "is_pascal_case", "is_snake_case", "to_pascal_case", "to_snake_case"]
