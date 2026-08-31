from gui_agent.models import GROUNDING_PROMPT, box_center, parse_box


# --- 解析模型输出 -----------------------------------------------------------


def test_parse_json_form():
    assert parse_box('{"bbox_2d": [10, 20, 110, 220]}') == (10, 20, 110, 220)


def test_parse_json_with_surrounding_text():
    text = '好的，我找到了。\n{"bbox_2d": [5, 6, 7, 8], "label": "保存按钮"}\n希望有帮助。'
    assert parse_box(text) == (5, 6, 7, 8)


def test_parse_qwen_box_tokens():
    assert parse_box("<|box_start|>(100,200),(300,400)<|box_end|>") == (100, 200, 300, 400)


def test_parse_bare_list():
    assert parse_box("[12.5, 20, 40, 60.5]") == (12.5, 20, 40, 60.5)


def test_parse_returns_none_on_failure():
    for bad in ["找不到这个元素", "", "bbox_2d: 不知道", "[1, 2]"]:
        assert parse_box(bad) is None, f"{bad!r} 不该解析出框"


def test_json_takes_precedence_over_bare_list():
    """两种形式同时出现时，取带 bbox_2d 键的那个。"""
    text = '参考 [1,1,2,2]，答案是 {"bbox_2d": [30, 40, 50, 60]}'
    assert parse_box(text) == (30, 40, 50, 60)


# --- 中心点 -----------------------------------------------------------------


def test_box_center():
    assert box_center((0, 0, 100, 200)) == (50, 100)
    assert box_center((10, 10, 30, 30)) == (20, 20)


# --- 提示词 -----------------------------------------------------------------


def test_prompt_fills_instruction():
    p = GROUNDING_PROMPT.format(instruction="保存按钮")
    assert "保存按钮" in p
    assert "bbox_2d" in p, "要求 JSON 格式才好解析"


def test_prompt_braces_survive_format():
    """模板里 JSON 的花括号要写成双花括号，否则 format 会报错或吞掉。"""
    p = GROUNDING_PROMPT.format(instruction="x")
    assert '{"bbox_2d": [x1, y1, x2, y2]}' in p


# --- 解析的边界情况 ---------------------------------------------------------


def test_parse_multiline_json():
    assert parse_box('{\n  "bbox_2d": [\n    1, 2, 3, 4\n  ]\n}') == (1, 2, 3, 4)


def test_parse_json_with_spaces_and_label():
    text = '{ "label" : "保存" , "bbox_2d" : [ 10 , 20 , 30 , 40 ] }'
    assert parse_box(text) == (10, 20, 30, 40)


def test_parse_takes_first_four_when_more_numbers():
    assert parse_box('{"bbox_2d": [1, 2, 3, 4, 5, 6]}') == (1, 2, 3, 4)


def test_parse_rejects_three_numbers():
    assert parse_box('{"bbox_2d": [1, 2, 3]}') is None


def test_parse_float_coords():
    assert parse_box('{"bbox_2d": [1.5, 2.25, 3.75, 4.5]}') == (1.5, 2.25, 3.75, 4.5)


def test_parse_box_tokens_with_spaces():
    assert parse_box("<|box_start|>(100, 200),(300, 400)<|box_end|>") == (100, 200, 300, 400)


def test_parse_ignores_prose_without_numbers():
    assert parse_box("屏幕上没有找到这个元素，请换一个描述。") is None


def test_parse_empty_and_whitespace():
    assert parse_box("") is None and parse_box("   \n  ") is None


# --- 中心点 -----------------------------------------------------------------


def test_box_center_with_floats():
    assert box_center((1.0, 2.0, 4.0, 8.0)) == (2.5, 5.0)


def test_box_center_of_degenerate_box():
    assert box_center((5, 5, 5, 5)) == (5, 5)


# --- 提示词 -----------------------------------------------------------------


def test_prompt_has_no_leftover_placeholder():
    p = GROUNDING_PROMPT.format(instruction="保存按钮")
    assert "{" not in p.replace('{"bbox_2d": [x1, y1, x2, y2]}', "")


def test_prompt_handles_instruction_with_braces():
    p = GROUNDING_PROMPT.format(instruction="设置 {高级}")
    assert "设置 {高级}" in p


# --- JPEG 编码 --------------------------------------------------------------


def test_encode_jpeg_produces_jpeg_magic_bytes():
    import numpy as np

    from gui_agent.models import encode_jpeg

    data = encode_jpeg(np.zeros((20, 20, 3), dtype=np.uint8))
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def test_encode_jpeg_quality_affects_size():
    import numpy as np

    from gui_agent.models import encode_jpeg

    img = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    assert len(encode_jpeg(img, quality=95)) > len(encode_jpeg(img, quality=20))


def test_default_model_is_a_qwen_vl():
    from gui_agent.models import DEFAULT_MODEL

    assert "Qwen" in DEFAULT_MODEL and "VL" in DEFAULT_MODEL
