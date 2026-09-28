import pytest

from bifrost.naming import is_pascal_case, is_snake_case, to_pascal_case, to_snake_case


@pytest.mark.parametrize(
    ("name", "snake"),
    [
        ("InitWindow", "init_window"),
        ("GetFPS", "get_fps"),
        ("Vector2Add", "vector2_add"),
        ("DrawText3D", "draw_text3d"),
        ("rlLoadTexture", "rl_load_texture"),
        ("ColorToHSV", "color_to_hsv"),
        ("TextToUTF8", "text_to_utf8"),
        ("drawText", "draw_text"),
        ("draw_text", "draw_text"),
        ("Assert", "assert_"),  # a keyword gets a trailing underscore
    ],
)
def test_to_snake_case(name: str, snake: str) -> None:
    assert to_snake_case(name) == snake
    assert is_snake_case(snake)


@pytest.mark.parametrize(
    ("name", "pascal"),
    [("Color", "Color"), ("rlVertexBuffer", "RlVertexBuffer"), ("float3", "Float3"), ("my_type", "MyType")],
)
def test_to_pascal_case(name: str, pascal: str) -> None:
    assert to_pascal_case(name) == pascal
    assert is_pascal_case(pascal)


def test_checks() -> None:
    assert [is_snake_case(n) for n in ["x", "_", "_private", "vector2_add", "drawText", "Draw", "MAX"]] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert [is_pascal_case(n) for n in ["Color", "Camera3D", "color", "My_Type"]] == [True, True, False, False]
