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


def test_default_model_is_a_qwen_vl():
    from gui_agent.models import DEFAULT_MODEL

    assert "Qwen" in DEFAULT_MODEL and "VL" in DEFAULT_MODEL


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
