import numpy as np
import pytest

from gui_agent.perception import (
    DEFAULT_LONG_EDGE,
    DEFAULT_MAX_PIXELS,
    _quad_to_norm_bbox,
    annotate,
    imread,
    imwrite,
    resize_for_model,
    scale_factor,
)
from gui_agent.schema import Element


# --- 缩放 -------------------------------------------------------------------


def test_scale_factor_for_4k_screen():
    """3840x2160 = 8.29MP，总像素这一项比长边更紧，取 0.372，缩后 1430x804。"""
    s = scale_factor(3840, 2160)
    assert round(s, 4) == 0.3724
    assert (round(3840 * s), round(2160 * s)) == (1430, 804)


def test_scaled_size_within_limits_and_no_upscale():
    assert scale_factor(1024, 768) == 1.0
    for w, h in [(3840, 2160), (2560, 1440), (5120, 2880), (1366, 768)]:
        s = scale_factor(w, h)
        assert max(w * s, h * s) <= DEFAULT_LONG_EDGE + 1
        assert w * s * h * s <= DEFAULT_MAX_PIXELS + 1000


def test_limits_configurable_and_invalid_size_raises():
    assert scale_factor(3840, 2160, long_edge=2576, max_pixels=3_750_000) > scale_factor(3840, 2160)
    with pytest.raises(ValueError):
        scale_factor(0, 1080)


def test_resize_output_size_and_passthrough():
    small, s = resize_for_model(np.zeros((2160, 3840, 3), dtype=np.uint8))
    assert (small.shape[1], small.shape[0]) == (round(3840 * s), round(2160 * s))
    img = np.zeros((768, 1024, 3), dtype=np.uint8)
    assert resize_for_model(img) == (img, 1.0)


# --- 坐标归一化 -------------------------------------------------------------


def test_quad_to_norm_bbox_with_clamping():
    assert _quad_to_norm_bbox([[100, 50], [300, 50], [300, 90], [100, 90]], 1000, 500) == (
        0.1, 0.1, 0.3, 0.18
    )
    assert _quad_to_norm_bbox([[-20, -10], [1200, -10], [1200, 600], [-20, 600]], 1000, 500) == (
        0.0, 0.0, 1.0, 1.0
    )
    with pytest.raises(ValueError):
        _quad_to_norm_bbox([[0, 0], [1, 0], [1, 1], [0, 1]], 0, 100)


def test_norm_bbox_consistent_across_resolutions():
    """这是原图 OCR 的框能画到缩放图上的前提。"""
    a = _quad_to_norm_bbox([[384, 216], [768, 216], [768, 432], [384, 432]], 3840, 2160)
    b = _quad_to_norm_bbox([[143, 80], [286, 80], [286, 161], [143, 161]], 1430, 804)
    assert all(abs(u - v) < 0.005 for u, v in zip(a, b))


# --- 边界框绘制 -------------------------------------------------------------


def test_annotate_does_not_mutate_input():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = annotate(img, [Element(id=0, bbox=(0.1, 0.1, 0.3, 0.2), text="保存")])
    assert out.shape == img.shape and out.any() and not img.any()
    assert np.array_equal(annotate(img, []), img)


# --- 中文路径下的存读图 -----------------------------------------------------


def test_imwrite_imread_non_ascii_path(tmp_path):
    """cv2.imwrite 在中文路径下返回 False 但不抛异常，本项目目录名就是中文。"""
    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    target = tmp_path / "中文目录" / "带空格 的图.png"
    imwrite(str(target), img)  # 目录不存在也应自动建
    assert target.exists() and np.array_equal(imread(str(target)), img)


def test_imread_missing_file_raises(tmp_path):
    with pytest.raises((IOError, OSError)):
        imread(str(tmp_path / "不存在.png"))


# --- 缩放的更多情形 ---------------------------------------------------------


def test_portrait_orientation():
    """竖屏时长边是高，缩放系数按高算。"""
    s = scale_factor(1080, 1920)
    assert round(1080 * s) <= DEFAULT_LONG_EDGE and round(1920 * s) <= DEFAULT_LONG_EDGE


def test_square_screen():
    s = scale_factor(2000, 2000)
    assert 2000 * s * 2000 * s <= DEFAULT_MAX_PIXELS + 1000


def test_recommended_resolutions_need_no_scaling():
    """Claude 文档推荐的两档分辨率下不需要缩放，截图原样送模型。"""
    assert scale_factor(1024, 768) == 1.0
    assert scale_factor(1280, 720) == 1.0


def test_just_above_pixel_limit_gets_scaled():
    s = scale_factor(1600, 900)  # 1.44 MP，超过 1.15 MP 上限
    assert s < 1.0


def test_extreme_aspect_ratio_bound_by_long_edge():
    s = scale_factor(20000, 50)
    assert s == pytest.approx(DEFAULT_LONG_EDGE / 20000)


def test_resize_preserves_dtype_and_channels():
    img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    small, _ = resize_for_model(img)
    assert small.dtype == np.uint8 and small.shape[2] == 3


def test_resize_result_is_contiguous():
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    small, _ = resize_for_model(img)
    assert small.flags["C_CONTIGUOUS"]


# --- 坐标归一化的更多情形 ---------------------------------------------------


def test_quad_with_float_coords():
    quad = [[10.5, 20.5], [110.5, 20.5], [110.5, 40.5], [10.5, 40.5]]
    x1, y1, x2, y2 = _quad_to_norm_bbox(quad, 1000, 100)
    assert (round(x1, 4), round(x2, 4)) == (0.0105, 0.1105)


def test_degenerate_quad_gives_zero_area_bbox():
    quad = [[50, 50]] * 4
    x1, y1, x2, y2 = _quad_to_norm_bbox(quad, 100, 100)
    assert x1 == x2 == y1 == y2 == 0.5


def test_bbox_stays_within_unit_square():
    quad = [[-999, -999], [9999, -999], [9999, 9999], [-999, 9999]]
    for v in _quad_to_norm_bbox(quad, 640, 480):
        assert 0.0 <= v <= 1.0


# --- 画框 -------------------------------------------------------------------


def test_annotate_many_elements():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    elems = [Element(id=i, bbox=(i / 60, 0.1, i / 60 + 0.01, 0.15)) for i in range(50)]
    assert annotate(img, elems).shape == img.shape


def test_annotate_label_stays_inside_at_top_edge():
    """框贴着上边缘时，编号不能画到画布外面。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = annotate(img, [Element(id=1, bbox=(0.1, 0.0, 0.3, 0.05))])
    assert out.any()


# --- 图片读写 ---------------------------------------------------------------


def test_imwrite_jpg_extension(tmp_path):
    target = tmp_path / "图.jpg"
    imwrite(str(target), np.full((30, 30, 3), 200, dtype=np.uint8))
    assert target.exists() and imread(str(target)).shape == (30, 30, 3)


def test_imwrite_unsupported_extension_raises(tmp_path):
    with pytest.raises((IOError, OSError, Exception)):
        imwrite(str(tmp_path / "x.qqq"), np.zeros((5, 5, 3), dtype=np.uint8))


def test_imread_on_corrupt_file(tmp_path):
    bad = tmp_path / "坏图.png"
    bad.write_bytes(b"not a png")
    with pytest.raises((IOError, OSError)):
        imread(str(bad))


def test_default_limits_match_claude_doc():
    """对应 Claude 文档里较早模型那一档：长边 1568、总像素约 1.15 MP。"""
    assert DEFAULT_LONG_EDGE == 1568
    assert DEFAULT_MAX_PIXELS == 1_150_000
