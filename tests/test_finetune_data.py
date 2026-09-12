"""微调数据的构建：描述提取、裁剪窗口、坐标换算。"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_finetune_data import (
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


def test_action_prompt_is_the_production_one():
    """动作样本的提示词必须就是推理时那一份，否则学到的迁移不过去。

    第一版用了一个简化模板：训练时没有元素清单，推理时有，还被要求「优先用
    element 编号」。模型没学过怎么用编号，于是写出 14.0 这种东西。
    """
    from gui_agent.chain import render_prompt
    from gui_agent.schema import Element, ScreenState

    state = ScreenState(1024, 768, elements=[
        Element(id=0, bbox=(0.1, 0.1, 0.2, 0.2), text="文件")])
    p = render_prompt("打开浏览器", state, [])
    assert "打开浏览器" in p and "[0] 文件" in p and "element" in p


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
        assert r["kind"] in ("action", "grounding", "plan")
        assert r["prompt"] and r["response"]
        assert Path(r["image"]).is_file()
        body = json.loads(r["response"])
        if r["kind"] == "grounding":
            b = body["bbox_2d"]
            assert len(b) == 4 and b[0] <= b[2] and b[1] <= b[3]
        elif r["kind"] == "plan":
            # 拆解样本的回答是子任务数组，每项一句话
            assert isinstance(body, list) and body
            assert all(isinstance(s, str) and s.strip() for s in body)
        else:
            assert "action" in body and "type" in body["action"]
            # thought 不能是空串：第一版就是空串，等于教模型别写理由
            assert body.get("thought", "").strip()


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
        act = json.loads(r["response"])["action"]
        if "element" in act:
            # 编号必须出现在提示词列出的清单里，否则是在教模型输出它看不到的编号
            shown = {int(m) for m in re.findall(r"^\s*\[(\d+)\]", r["prompt"], re.M)}
            assert isinstance(act["element"], int) and act["element"] in shown,                 f"编号 {act['element']} 不在提示词的元素清单里"
        else:
            Action.from_dict(act)
        n += 1
        if n >= 300:
            break
    assert n > 0


# --- 定位目标必须写在模型的坐标空间里 ---------------------------------------


def test_to_model_space_scales_box_like_smart_resize():
    """推理时预测框是除以 smart_resize 后的尺寸来归一化的，训练目标得在同一空间。

    1280x720 在 --max-pixels 640 下被缩到 924x504（0.722 倍），目标框不跟着缩
    就整体大了 39%，ScreenSpot 从 71.6% 掉到 30.2%。
    """
    from qwen_vl_utils.vision_process import smart_resize

    from build_finetune_data import to_model_space

    for blocks in (640, 1280):
        rh, rw = smart_resize(720, 1280, factor=28,
                              min_pixels=256 * 28 * 28, max_pixels=blocks * 28 * 28)
        box = to_model_space([0.0, 0.0, 1280.0, 720.0], 1280, 720, blocks)
        assert box == pytest.approx([0.0, 0.0, rw, rh])


def test_to_model_space_keeps_the_box_inside_the_resized_image():
    from qwen_vl_utils.vision_process import smart_resize

    from build_finetune_data import to_model_space

    rh, rw = smart_resize(720, 1280, factor=28,
                          min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
    x1, y1, x2, y2 = to_model_space([100.0, 50.0, 300.0, 90.0], 1280, 720, 640)
    assert 0 <= x1 < x2 <= rw and 0 <= y1 < y2 <= rh


def test_built_grounding_boxes_are_in_model_space():
    """落盘的定位样本，框不能超出 smart_resize 后的尺寸。"""
    from pathlib import Path as _P

    from qwen_vl_utils.vision_process import smart_resize
    from PIL import Image

    p = _P(__file__).resolve().parents[1] / "data" / "finetune" / "train.jsonl"
    if not p.is_file():
        pytest.skip("还没构建微调数据")
    n = 0
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "grounding":
            continue
        with Image.open(r["image"]) as im:
            w, h = im.size
        b = json.loads(r["response"])["bbox_2d"]
        ok = False
        for blocks in (640, 1280):
            rh, rw = smart_resize(h, w, factor=28, min_pixels=256 * 28 * 28,
                                  max_pixels=blocks * 28 * 28)
            if b[2] <= rw + 1 and b[3] <= rh + 1:
                ok = True
                break
        assert ok, f"框 {b} 超出了任何一档 max_pixels 下的尺寸（图 {w}x{h}）"
        n += 1
        if n >= 80:
            break
