"""微调数据的构建：描述提取、裁剪窗口、坐标换算。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_finetune_data import (
    ACTION_TEMPLATE,
    VIEW_H,
    VIEW_W,
    crop_box,
    describe,
)


# --- 元素描述 ---------------------------------------------------------------


def test_strips_tag_and_action():
    assert describe("[heading]  CAR -> CLICK", {}) == "CAR"


def test_strips_type_value():
    s = describe("[combobox]  Enter pick up city, airport name, or airport code. -> TYPE: x", {})
    assert s == "Enter pick up city, airport name, or airport code."


def test_falls_back_to_aria_label_when_text_empty():
    assert describe("[input]   -> CLICK", {"aria_label": "搜索框"}) == "搜索框"


@pytest.mark.parametrize("attrs,want", [
    ({"title": "关闭"}, "关闭"),
    ({"alt": "图标"}, "图标"),
    ({"aria_label": "A", "title": "B"}, "A"),   # aria_label 优先
])
def test_fallback_order(attrs, want):
    assert describe("[div]  -> CLICK", attrs) == want


def test_returns_empty_when_nothing_usable():
    assert describe("[div]   -> CLICK", {}) == ""
    assert describe(None, {}) == ""


def test_handles_repr_without_tag_prefix():
    assert describe("保存按钮 -> CLICK", {}) == "保存按钮"


# --- 裁剪窗口 ---------------------------------------------------------------


def test_window_is_viewport_sized_and_centred():
    x0, y0, x1, y1 = crop_box(600, 1000, 100, 40, 1280, 5429)
    assert (x1 - x0, y1 - y0) == (VIEW_W, VIEW_H)
    # 目标中心 (650, 1020) 应落在窗口中间附近
    assert x0 <= 650 <= x1 and y0 <= 1020 <= y1
    assert abs((y0 + y1) / 2 - 1020) < 2


def test_window_clamped_at_top():
    """目标贴着整页顶部时，窗口不能越到负坐标。"""
    x0, y0, x1, y1 = crop_box(700, 0, 110, 60, 1280, 5429)
    assert y0 == 0 and (y1 - y0) == VIEW_H


def test_window_clamped_at_bottom():
    H = 5429
    x0, y0, x1, y1 = crop_box(100, H - 20, 50, 20, 1280, H)
    assert y1 == H and (y1 - y0) == VIEW_H


def test_window_never_exceeds_image():
    for W, H in [(1280, 5429), (800, 600), (1280, 720)]:
        x0, y0, x1, y1 = crop_box(10, 10, 20, 20, W, H)
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H


def test_narrow_image_does_not_produce_negative_origin():
    """整页比窗口还窄时，x0 要停在 0。"""
    x0, y0, x1, y1 = crop_box(10, 10, 5, 5, 600, 400)
    assert x0 == 0 and y0 == 0 and x1 == 600 and y1 == 400


# --- 坐标换算 ---------------------------------------------------------------


def test_bbox_translates_into_crop_coordinates():
    """裁剪后的 bbox = 原坐标减去窗口左上角。"""
    bx, by, bw, bh = 600, 1000, 100, 40
    x0, y0, _, _ = crop_box(bx, by, bw, bh, 1280, 5429)
    cb = [bx - x0, by - y0, bx - x0 + bw, by - y0 + bh]
    assert cb[2] - cb[0] == bw and cb[3] - cb[1] == bh
    assert all(v >= 0 for v in cb)


# --- 动作提示词 -------------------------------------------------------------


def test_action_template_asks_for_normalized_point():
    p = ACTION_TEMPLATE.format(instruction="打开浏览器")
    assert "打开浏览器" in p
    assert "point" in p and "归一化" in p


def test_action_template_has_no_element_list():
    """训练目标是让模型直接给坐标，给了元素清单它又会去用编号。"""
    p = ACTION_TEMPLATE.format(instruction="x")
    assert "element" not in p and "编号" not in p


def test_action_template_braces_survive_format():
    p = ACTION_TEMPLATE.format(instruction="x")
    assert '{"thought": "为什么这么做", "action": {"type": "click", "point": [0.5, 0.5]}}' in p


def test_grounding_prompt_is_the_inference_one():
    """定位样本的提示词必须和推理时用的一字不差，否则学的东西迁移不过去。"""
    from gui_agent.models import GROUNDING_PROMPT

    assert "bbox_2d" in GROUNDING_PROMPT.format(instruction="x")


# --- 产出的样本 -------------------------------------------------------------


def test_built_samples_are_wellformed():
    p = Path(__file__).resolve().parents[1] / "data" / "finetune" / "train.jsonl"
    if not p.is_file():
        pytest.skip("还没构建微调数据")
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    assert rows
    for r in rows[:200]:
        assert r["kind"] in ("action", "grounding")
        assert r["prompt"] and r["response"]
        assert Path(r["image"]).is_file()
        body = json.loads(r["response"])
        if r["kind"] == "grounding":
            b = body["bbox_2d"]
            assert len(b) == 4 and b[0] <= b[2] and b[1] <= b[3]
        else:
            assert "action" in body and "type" in body["action"]


def test_action_samples_carry_valid_actions():
    """回答里的动作要能被 schema 解析回来，否则训出来的格式是错的。"""
    from gui_agent.schema import Action

    p = Path(__file__).resolve().parents[1] / "data" / "finetune" / "train.jsonl"
    if not p.is_file():
        pytest.skip("还没构建微调数据")
    n = 0
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "action":
            continue
        Action.from_dict(json.loads(r["response"])["action"])
        n += 1
        if n >= 300:
            break
    assert n > 0
