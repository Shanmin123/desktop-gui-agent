import pytest

from gui_agent.models import GROUNDING_PROMPT, box_center, parse_box


# --- 解析模型输出 -----------------------------------------------------------


def test_解析json形式():
    assert parse_box('{"bbox_2d": [10, 20, 110, 220]}') == (10, 20, 110, 220)


def test_解析带前后废话的json():
    text = '好的，我找到了。\n{"bbox_2d": [5, 6, 7, 8], "label": "保存按钮"}\n希望有帮助。'
    assert parse_box(text) == (5, 6, 7, 8)


def test_解析qwen的box标记形式():
    assert parse_box("<|box_start|>(100,200),(300,400)<|box_end|>") == (100, 200, 300, 400)


def test_解析裸列表():
    assert parse_box("[12.5, 20, 40, 60.5]") == (12.5, 20, 40, 60.5)


def test_解析失败返回None():
    for bad in ["找不到这个元素", "", "bbox_2d: 不知道", "[1, 2]"]:
        assert parse_box(bad) is None, f"{bad!r} 不该解析出框"


def test_json优先于裸列表():
    """两种形式同时出现时，取带 bbox_2d 键的那个。"""
    text = '参考 [1,1,2,2]，答案是 {"bbox_2d": [30, 40, 50, 60]}'
    assert parse_box(text) == (30, 40, 50, 60)


# --- 中心点 -----------------------------------------------------------------


def test_中心点():
    assert box_center((0, 0, 100, 200)) == (50, 100)
    assert box_center((10, 10, 30, 30)) == (20, 20)


# --- 提示词 -----------------------------------------------------------------


def test_提示词填入指令():
    p = GROUNDING_PROMPT.format(instruction="保存按钮")
    assert "保存按钮" in p
    assert "bbox_2d" in p, "要求 JSON 格式才好解析"


def test_提示词的花括号没被format吃掉():
    """模板里 JSON 的花括号要写成双花括号，否则 format 会报错或吞掉。"""
    p = GROUNDING_PROMPT.format(instruction="x")
    assert '{"bbox_2d": [x1, y1, x2, y2]}' in p
