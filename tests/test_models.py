import numpy as np
import pytest

from gui_agent.models import (
    GROUNDING_PROMPT,
    LocalQwenVL,
    OpenAICompatVLM,
    box_center,
    parse_box,
)


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
    from gui_agent.models import encode_jpeg

    data = encode_jpeg(np.zeros((20, 20, 3), dtype=np.uint8))
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def test_encode_jpeg_quality_affects_size():
    from gui_agent.models import encode_jpeg

    img = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    assert len(encode_jpeg(img, quality=95)) > len(encode_jpeg(img, quality=20))


def test_default_model_is_qwen35_with_its_coordinate_space_registered():
    """默认基座是 Qwen3.5-4B；它的定位坐标口径得已经登记，否则 locate 会按错的除数换算。"""
    from gui_agent.models import COORD_SPACE_BY_MODEL_TYPE, DEFAULT_MODEL

    assert DEFAULT_MODEL == "Qwen/Qwen3.5-4B"
    assert COORD_SPACE_BY_MODEL_TYPE["qwen3_5"] == "rel1000"


# --- locate 的坐标空间 ------------------------------------------------------

MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1280 * 28 * 28


def _stub(cls, reply, **attrs):
    """不加载模型、不建连接，只装上 locate 用到的那几样。"""
    obj = cls.__new__(cls)
    obj.ask = lambda image, prompt, **kw: reply
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _smart_resize(h, w):
    from qwen_vl_utils.vision_process import smart_resize

    return smart_resize(h, w, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


def test_local_locate_normalizes_in_smart_resize_space():
    """Qwen 回的坐标在 smart_resize 之后的空间里，不是原图像素。"""
    h, w = 1080, 1920
    rh, rw = _smart_resize(h, w)
    assert (rw, rh) != (w, h), "尺寸一样的话这条测试区分不出对错"

    m = _stub(LocalQwenVL, '{"bbox_2d": [0, 0, %d, %d]}' % (rw, rh),
              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert m.locate(np.zeros((h, w, 3), dtype=np.uint8), "整块屏幕") == pytest.approx((0.5, 0.5))


def test_api_locate_defaults_to_image_size():
    """没说服务端怎么缩放时，按原图尺寸归一化。"""
    m = _stub(OpenAICompatVLM, '{"bbox_2d": [200, 100, 600, 300]}',
              min_pixels=None, max_pixels=None)
    assert m.locate(np.zeros((400, 800, 3), dtype=np.uint8), "中间") == pytest.approx((0.5, 0.5))


def test_api_locate_follows_smart_resize_when_configured():
    """服务端是 Qwen 系列时，构造时给 min/max_pixels 才能对上它的坐标空间。"""
    h, w = 1080, 1920
    rh, rw = _smart_resize(h, w)
    m = _stub(OpenAICompatVLM, '{"bbox_2d": [0, 0, %d, %d]}' % (rw, rh),
              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert m.locate(np.zeros((h, w, 3), dtype=np.uint8), "整块屏幕") == pytest.approx((0.5, 0.5))


def test_locate_returns_none_when_model_finds_nothing():
    m = _stub(OpenAICompatVLM, "屏幕上没有这个元素", min_pixels=None, max_pixels=None)
    assert m.locate(np.zeros((10, 10, 3), dtype=np.uint8), "不存在的按钮") is None


def test_locate_clamps_out_of_range_box():
    m = _stub(OpenAICompatVLM, '{"bbox_2d": [-50, -50, 5000, 5000]}',
              min_pixels=None, max_pixels=None)
    assert m.locate(np.zeros((100, 100, 3), dtype=np.uint8), "x") == (1.0, 1.0)


# --- 后端选择（本地 / API）--------------------------------------------------


def _args(**kw):
    import argparse

    from gui_agent.models import add_backend_args

    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    a = ap.parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_backend_args_cover_both_paths():
    a = _args()
    for name in ("model", "load_in_4bit", "api_base", "api_key", "api_qwen"):
        assert hasattr(a, name)
    assert a.api_base is None  # 默认本地


def test_api_backend_selected_when_base_url_given(monkeypatch):
    from gui_agent import models

    seen = {}
    monkeypatch.setattr(models, "OpenAICompatVLM",
                        lambda **kw: seen.update(kw) or "api")
    assert models.load_vlm(_args(api_base="http://x/v1", api_key="k")) == "api"
    assert seen["base_url"] == "http://x/v1" and seen["api_key"] == "k"
    assert seen["min_pixels"] is None  # 没加 --api-qwen 就按原图尺寸归一化


def test_api_qwen_passes_pixel_budget(monkeypatch):
    from gui_agent import models

    seen = {}
    monkeypatch.setattr(models, "OpenAICompatVLM", lambda **kw: seen.update(kw))
    models.load_vlm(_args(api_base="http://x/v1", api_key="k", api_qwen=True))
    assert seen["min_pixels"] == 256 * 28 * 28 and seen["max_pixels"] == 1280 * 28 * 28


def test_api_key_falls_back_to_environment(monkeypatch):
    from gui_agent import models

    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setattr(models, "OpenAICompatVLM", lambda **kw: seen.update(kw))
    models.load_vlm(_args(api_base="http://x/v1"))
    assert seen["api_key"] == "from-env"


def test_missing_api_key_exits_with_message(monkeypatch):
    from gui_agent import models

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        models.load_vlm(_args(api_base="http://x/v1"))


def test_local_backend_when_no_base_url(monkeypatch):
    from gui_agent import models

    seen = {}
    monkeypatch.setattr(models, "LocalQwenVL",
                        lambda mid, load_in_4bit=False, adapter=None: seen.update(
                            model=mid, q=load_in_4bit) or "local")
    assert models.load_vlm(_args(load_in_4bit=True)) == "local"
    assert seen["q"] is True and "Qwen" in seen["model"]


# --- LoRA 适配器 ------------------------------------------------------------


def test_adapter_flag_exists_and_defaults_to_none():
    a = _args()
    assert hasattr(a, "adapter") and a.adapter is None


def test_adapter_is_passed_to_local_backend(monkeypatch):
    from gui_agent import models

    seen = {}
    monkeypatch.setattr(models, "LocalQwenVL",
                        lambda mid, load_in_4bit=False, adapter=None: seen.update(
                            model=mid, adapter=adapter))
    models.load_vlm(_args(adapter="checkpoints/lora"))
    assert seen["adapter"] == "checkpoints/lora"


def test_no_adapter_by_default(monkeypatch):
    """微调前的基线必须是不挂适配器跑出来的。"""
    from gui_agent import models

    seen = {}
    monkeypatch.setattr(models, "LocalQwenVL",
                        lambda mid, load_in_4bit=False, adapter=None: seen.update(adapter=adapter))
    models.load_vlm(_args())
    assert seen["adapter"] is None


# --- 故障注入（大纲第 6 周第 2 项）------------------------------------------


class _CountingVLM:
    def __init__(self):
        self.calls = 0

    def ask(self, image, prompt, **kw):
        self.calls += 1
        return '{"action": {"type": "wait"}}'

    def resized_size(self, h, w):
        return (h, w)


def test_flaky_injects_about_the_given_rate():
    from gui_agent.models import FlakyVLM

    f = FlakyVLM(_CountingVLM(), rate=0.3, seed=0)
    for _ in range(1000):
        f.ask(None, "x")
    assert 250 <= f.injected <= 350, f"注入了 {f.injected}/1000，偏离 30% 太多"


def test_flaky_is_reproducible_across_runs():
    """同一个种子两次跑要一模一样，否则开关重试的对照不可比。"""
    from gui_agent.models import FlakyVLM

    def pattern():
        f = FlakyVLM(_CountingVLM(), rate=0.5, seed=7)
        return [f.ask(None, "x").startswith("（注入") for _ in range(50)]

    assert pattern() == pattern()


def test_flaky_rate_zero_never_injects():
    from gui_agent.models import FlakyVLM

    inner = _CountingVLM()
    f = FlakyVLM(inner, rate=0.0)
    for _ in range(20):
        assert f.ask(None, "x").startswith("{")
    assert f.injected == 0 and inner.calls == 20


def test_flaky_rate_one_always_injects():
    from gui_agent.models import FlakyVLM

    inner = _CountingVLM()
    f = FlakyVLM(inner, rate=1.0)
    for _ in range(20):
        assert not f.ask(None, "x").startswith("{")
    assert inner.calls == 0, "全注入时不该再去问真模型，白花推理时间"


def test_flaky_forwards_other_methods():
    from gui_agent.models import FlakyVLM

    assert FlakyVLM(_CountingVLM(), rate=1.0).resized_size(720, 1280) == (720, 1280)


def test_flaky_rejects_bad_rate():
    import pytest

    from gui_agent.models import FlakyVLM

    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            FlakyVLM(_CountingVLM(), rate=bad)


def test_injected_failure_is_recoverable_by_retry():
    """注入的故障必须是重试能救回来的那一类，否则测不出重试的价值。"""
    import numpy as np

    from gui_agent.agent import Agent
    from gui_agent.control import Controller, RecordingBackend
    from gui_agent.models import FlakyVLM
    from gui_agent.schema import Element, ScreenState

    screen = ScreenState(1280, 720, elements=[Element(id=0, bbox=(0.1, 0.1, 0.2, 0.2), text="x")])

    class P:
        def perceive(self, **kw):
            return screen, np.zeros((10, 10, 3), dtype=np.uint8)

    class Finisher:
        def ask(self, image, prompt, **kw):
            return '{"action": {"type": "finished"}}'

    flaky = FlakyVLM(Finisher(), rate=0.5, seed=3)
    t = Agent(P(), Controller(backend=RecordingBackend(), dry_run=True), flaky,
              retry_backoff=0).run("x")
    assert t.success is True and t.retries >= 1


# --- 归一化坐标（照 OS-Atlas / SeeClick 的做法）------------------------------


@pytest.mark.parametrize("text,want", [
    ('{"point": [0.42, 0.13]}', (0.42, 0.13)),
    ('[0.5, 0.5]', (0.5, 0.5)),
    ('  {"point":[0.0,1.0]}  ', (0.0, 1.0)),
])
def test_parse_norm_point_accepts_ratios(text, want):
    from gui_agent.models import parse_norm_point

    assert parse_norm_point(text) == want


@pytest.mark.parametrize("text", [
    '{"point": [512, 384]}',      # 像素值，不能当成比例
    '{"point": [1.5, 0.2]}',      # 越界
    '{"point": [-0.1, 0.2]}',
    '{"bbox_2d": [1, 2, 3, 4]}',  # 框不是点
    "找不到这个元素",
])
def test_parse_norm_point_rejects_non_ratios(text):
    """越界的值要判成解析失败，硬当比例会把点压到左上角。"""
    from gui_agent.models import parse_norm_point

    assert parse_norm_point(text) is None


def test_norm_prompt_asks_for_ratio_not_pixels():
    from gui_agent.models import GROUNDING_PROMPT, GROUNDING_PROMPT_NORM

    p = GROUNDING_PROMPT_NORM.format(instruction="保存按钮")
    assert "保存按钮" in p and "0 到 1" in p and "比例" in p
    assert "像素" not in p
    # 像素那套要留着：基座模型是按像素框预训练的，两边各用各的口径
    assert "像素" in GROUNDING_PROMPT.format(instruction="x")


def test_locate_uses_the_norm_path_when_enabled():
    """开了 norm_coords 就不再走像素换算，直接拿比例值。"""
    import numpy as np

    from gui_agent.models import LocalQwenVL

    class Fake(LocalQwenVL):
        def __init__(self, norm):
            self.norm_coords = norm
            self.asked = []

        def ask(self, image, prompt, **kw):
            self.asked.append(prompt)
            return '{"point": [0.25, 0.75]}' if self.norm_coords else '{"bbox_2d": [10, 20, 30, 40]}'

        def resized_size(self, h, w):
            return (h, w)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert Fake(True).locate(img, "保存") == (0.25, 0.75)
    # 像素那条路要除以尺寸：框中心 (20, 30) / (200, 100)
    assert Fake(False).locate(img, "保存") == pytest.approx((0.1, 0.3))


# --- 迁移到 Qwen3.5 之后加的 ------------------------------------------------


def test_resized_size_follows_the_patch_factor():
    """Qwen3.5 的切块系数是 32：1280x720 在 640 个 token 下送进去是 1056x576。"""
    m = _stub(LocalQwenVL, "", factor=32, min_pixels=256 * 32 * 32, max_pixels=640 * 32 * 32)
    assert m.resized_size(720, 1280) == (576, 1056)


def test_coord_size_uses_1000_for_relative_coordinates():
    m = _stub(LocalQwenVL, '{"bbox_2d": [0, 0, 1000, 1000]}', coord_space="rel1000",
              factor=32, min_pixels=256 * 32 * 32, max_pixels=640 * 32 * 32)
    assert m.coord_size(720, 1280) == (1000, 1000)
    assert m.locate(np.zeros((720, 1280, 3), dtype=np.uint8), "整块屏幕") == pytest.approx((0.5, 0.5))


def test_coord_size_defaults_to_resized_pixels():
    m = _stub(LocalQwenVL, "", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert m.coord_size(1080, 1920) == _smart_resize(1080, 1920)


def test_strip_thinking_keeps_only_the_answer():
    from gui_agent.models import strip_thinking

    assert strip_thinking('<think>先想想</think>\n{"a": 1}') == '{"a": 1}'
    assert strip_thinking('想了一堆</think>{"a": 1}') == '{"a": 1}'
    assert strip_thinking('{"a": 1}') == '{"a": 1}'
    assert strip_thinking("<think>一直没想完") == ""


def test_adapter_cannot_be_mounted_on_another_base(tmp_path):
    import json as _json

    from gui_agent.models import check_adapter_base

    (tmp_path / "adapter_config.json").write_text(
        _json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct"}), encoding="utf-8")
    check_adapter_base(str(tmp_path), "Qwen/Qwen2.5-VL-3B-Instruct")
    check_adapter_base(str(tmp_path), r"D:\models\Qwen2.5-VL-3B-Instruct")   # 本地路径，同名
    with pytest.raises(ValueError, match="Qwen3.5-4B"):
        check_adapter_base(str(tmp_path), "Qwen/Qwen3.5-4B")


def test_coord_space_registry_matches_the_probe():
    """口径是 scripts/probe_model.py 实测后登记的，改之前要重新量。"""
    from gui_agent.models import COORD_SPACE_BY_MODEL_TYPE, COORD_SPACES

    assert COORD_SPACE_BY_MODEL_TYPE["qwen2_5_vl"] == "pixel"
    assert COORD_SPACE_BY_MODEL_TYPE["qwen3_5"] == "rel1000"
    assert set(COORD_SPACE_BY_MODEL_TYPE.values()) <= set(COORD_SPACES)


def test_parse_box_reads_qwen35_list_output():
    """Qwen3.5 回的是代码块里的列表，每项带 label。"""
    raw = '```json\n[\n\t{"bbox_2d": [954, 150, 984, 200], "label": "close"}\n]\n```'
    assert parse_box(raw) == (954, 150, 984, 200)


def test_api_locate_divides_by_1000_for_rel1000_servers():
    """服务端是 Qwen3.5 这类 0~1000 口径时，框中心除以 1000，与图多大无关。"""
    m = _stub(OpenAICompatVLM, '{"bbox_2d": [0, 0, 1000, 1000]}',
              min_pixels=None, max_pixels=None, coord_space="rel1000")
    assert m.coord_size(400, 800) == (1000, 1000)
    assert m.locate(np.zeros((400, 800, 3), dtype=np.uint8), "整块屏幕") == pytest.approx((0.5, 0.5))


def test_api_without_coord_space_keeps_the_image_size():
    m = _stub(OpenAICompatVLM, "", min_pixels=None, max_pixels=None)
    assert m.coord_size(400, 800) == (400, 800)


def test_api_coord_space_flag_reaches_the_backend(monkeypatch):
    from gui_agent import models

    seen = {}
    monkeypatch.setattr(models, "OpenAICompatVLM", lambda **kw: seen.update(kw))
    models.load_vlm(_args(api_base="http://x/v1", api_key="k", api_coord_space="rel1000"))
    assert seen["coord_space"] == "rel1000" and seen["min_pixels"] is None
