"""微调数据的构建：描述提取、裁剪窗口、坐标换算。"""

import json
import re
from collections import Counter
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
            if "point" in body:      # 归一化口径：0~1 的点
                assert len(body["point"]) == 2
                assert all(0.0 <= v <= 1.0 for v in body["point"])
            else:                    # 像素口径：左上角要在右下角之前
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
        body = json.loads(r["response"])
        if "point" in body:
            # 归一化口径不需要 smart_resize 换算，这条检查不适用
            assert all(0.0 <= v <= 1.0 for v in body["point"])
            n += 1
            if n >= 80:
                break
            continue
        b = body["bbox_2d"]
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


def test_norm_coords_grounding_targets_are_ratios():
    """归一化口径下目标是 0~1 的点，不再需要 smart_resize 换算。"""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "bf", _P(__file__).resolve().parents[1] / "scripts" / "build_finetune_data.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)

    from gui_agent.models import GROUNDING_PROMPT_NORM

    # 提示词和推理时那份必须一字不差
    assert "point" in GROUNDING_PROMPT_NORM.format(instruction="x")
    # 构建函数签名里要有这个开关
    import inspect

    assert "norm_coords" in inspect.signature(bf.grounding_samples).parameters


def test_pixel_grounding_targets_are_integers():
    """像素口径的框要写成整数：基座模型自己吐的就是整数。

    写成 234.6 等于在「换坐标制」之外又改了一处输出约定，两件事的影响会混在一起，
    对照实验就说不清是哪一个造成的。
    """
    import json as _json
    from pathlib import Path as _P

    for name in ("finetune", "finetune_px"):
        p = _P(__file__).resolve().parents[1] / "data" / name / "train.jsonl"
        if not p.is_file():
            continue
        n = 0
        for line in p.open(encoding="utf-8"):
            r = _json.loads(line)
            if r["kind"] != "grounding":
                continue
            body = _json.loads(r["response"])
            if "bbox_2d" not in body:      # 归一化口径不适用
                continue
            assert all(isinstance(v, int) for v in body["bbox_2d"]), r["response"]
            n += 1
            if n >= 100:
                break


# --- 两段式样本的类型比例 ---------------------------------------------------


def _bf():
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "bf2", _P(__file__).resolve().parents[1] / "scripts" / "build_finetune_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_screenagent(tmp_path, counts):
    d = tmp_path / "data" / "screenagent"
    d.mkdir(parents=True)
    with (d / "train.jsonl").open("w", encoding="utf-8") as f:
        for t, n in counts.items():
            for _ in range(n):
                f.write(json.dumps({"action": {"type": t}}) + "\n")
    return tmp_path


def _samples(counts):
    out = []
    for t, n in counts.items():
        for i in range(n):
            out.append({"kind": "action",
                        "response": json.dumps({"thought": "x", "action": {"type": t}})})
    return out


def test_balance_follows_the_full_sets_share(tmp_path, monkeypatch):
    """抽完之后各类的占比要贴着整份数据的占比。"""
    import random as _r

    bf = _bf()
    monkeypatch.setattr(bf, "ROOT", _fake_screenagent(
        tmp_path, {"click": 400, "hotkey": 200, "type": 200, "wait": 200}))
    # 点击只收得上来 100 条，别的都富余
    got = bf.balance_types(_samples({"click": 100, "hotkey": 200, "type": 200, "wait": 200}),
                           "train", _r.Random(0))
    share = {t: sum(1 for r in got
                    if json.loads(r["response"])["action"]["type"] == t) / len(got)
             for t in ("click", "hotkey", "type", "wait")}
    assert share["click"] == pytest.approx(0.4, abs=0.03)
    for t in ("hotkey", "type", "wait"):
        assert share[t] == pytest.approx(0.2, abs=0.03)


def test_a_starved_minor_type_does_not_shrink_everything(tmp_path, monkeypatch):
    """少数类只剩几条时，总量不能被它拖垮——它本来就收不上来。"""
    import random as _r

    bf = _bf()
    monkeypatch.setattr(bf, "ROOT", _fake_screenagent(
        tmp_path, {"click": 400, "hotkey": 200, "type": 200, "left_double": 80}))
    got = bf.balance_types(
        _samples({"click": 100, "hotkey": 200, "type": 200, "left_double": 2}),
        "train", _r.Random(0))
    # 按 left_double 定总量的话只剩 10 条上下；按主要类别定是 100/0.4 = 250 上下
    assert len(got) > 200
    kinds = Counter(json.loads(r["response"])["action"]["type"] for r in got)
    assert kinds["left_double"] == 2        # 有多少收多少
    assert kinds["click"] == 100


def test_balance_never_invents_samples(tmp_path, monkeypatch):
    """抽样只能少不能多，每条都要来自原始列表。"""
    import random as _r

    bf = _bf()
    monkeypatch.setattr(bf, "ROOT", _fake_screenagent(
        tmp_path, {"click": 400, "hotkey": 200, "type": 400}))
    src = _samples({"click": 50, "hotkey": 60, "type": 70})
    got = bf.balance_types(list(src), "train", _r.Random(0))
    assert len(got) <= len(src)
    assert all(any(g is s for s in src) for g in got)


def test_responses_fit_under_the_generation_cap():
    """训练目标不能比推理时允许生成的长度还长。

    原来上限是 128 token，而 4~5% 的回答本身就超过 128：生成到一半被截断，JSON
    收不了尾，两段式那轮 353 条里有 24 条因此记成解析失败。上限改成 256 之后，
    这条检查保证以后重建数据也不会再越过去。
    """
    import json as _json
    from pathlib import Path as _P

    from transformers import AutoTokenizer

    from gui_agent.models import MAX_NEW_TOKENS

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    for name in ("finetune", "finetune_2sb", "finetune_2sh"):
        p = _P(__file__).resolve().parents[1] / "data" / name / "train.jsonl"
        if not p.is_file():
            continue
        n = [len(tok(_json.loads(l)["response"])["input_ids"]) for l in p.open(encoding="utf-8")]
        over = [v for v in n if v > MAX_NEW_TOKENS]
        assert len(over) / len(n) <= 0.01, \
            f"{name}: {len(over)}/{len(n)} 条回答超过 {MAX_NEW_TOKENS} token，最长 {max(n)}"


# --- 历史要跨截图累积 ---------------------------------------------------------


def test_history_groups_by_session_not_by_screenshot():
    """真机推理时「已执行」一直在，训练样本也得有：按截图分组会让三分之二的样本没有历史。"""
    from build_finetune_data import session_groups

    rows = [{"session_id": "s1", "image": "a.png"}, {"session_id": "s1", "image": "a.png"},
            {"session_id": "s1", "image": "b.png"}, {"session_id": "s2", "image": "c.png"}]
    assert [len(g) for g in session_groups(rows)] == [3, 1]
    assert [r["image"] for r in session_groups(rows)[0]] == ["a.png", "a.png", "b.png"]


def test_per_image_history_reproduces_the_delivered_dataset():
    """交付的 q35_2sp / lora_2sp 用的是按截图分组那一版，要能复现。"""
    from build_finetune_data import session_groups

    rows = [{"session_id": "s1", "image": "a.png"}, {"session_id": "s1", "image": "b.png"}]
    assert [len(g) for g in session_groups(rows, per_image_history=True)] == [1, 1]
