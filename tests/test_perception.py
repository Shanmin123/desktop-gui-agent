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


def test_本机4k屏的缩放系数():
    """3840x2160 = 8.29MP，总像素这一项比长边更紧，取 0.372，缩后 1430x804。"""
    s = scale_factor(3840, 2160)
    assert round(s, 4) == 0.3724
    assert (round(3840 * s), round(2160 * s)) == (1430, 804)


def test_缩放后不超过上限且小屏不放大():
    assert scale_factor(1024, 768) == 1.0
    for w, h in [(3840, 2160), (2560, 1440), (5120, 2880), (1366, 768)]:
        s = scale_factor(w, h)
        assert max(w * s, h * s) <= DEFAULT_LONG_EDGE + 1
        assert w * s * h * s <= DEFAULT_MAX_PIXELS + 1000


def test_上限可调且非法分辨率报错():
    assert scale_factor(3840, 2160, long_edge=2576, max_pixels=3_750_000) > scale_factor(3840, 2160)
    with pytest.raises(ValueError):
        scale_factor(0, 1080)


def test_resize输出尺寸正确且小图原样返回():
    small, s = resize_for_model(np.zeros((2160, 3840, 3), dtype=np.uint8))
    assert (small.shape[1], small.shape[0]) == (round(3840 * s), round(2160 * s))
    img = np.zeros((768, 1024, 3), dtype=np.uint8)
    assert resize_for_model(img) == (img, 1.0)


# --- 坐标归一化 -------------------------------------------------------------


def test_四角点转归一化外接矩形并裁到边界():
    assert _quad_to_norm_bbox([[100, 50], [300, 50], [300, 90], [100, 90]], 1000, 500) == (
        0.1, 0.1, 0.3, 0.18
    )
    assert _quad_to_norm_bbox([[-20, -10], [1200, -10], [1200, 600], [-20, 600]], 1000, 500) == (
        0.0, 0.0, 1.0, 1.0
    )
    with pytest.raises(ValueError):
        _quad_to_norm_bbox([[0, 0], [1, 0], [1, 1], [0, 1]], 0, 100)


def test_同一元素在不同分辨率下归一化结果相同():
    """这是原图 OCR 的框能画到缩放图上的前提。"""
    a = _quad_to_norm_bbox([[384, 216], [768, 216], [768, 432], [384, 432]], 3840, 2160)
    b = _quad_to_norm_bbox([[143, 80], [286, 80], [286, 161], [143, 161]], 1430, 804)
    assert all(abs(u - v) < 0.005 for u, v in zip(a, b))


# --- 边界框绘制 -------------------------------------------------------------


def test_画框不改原图且无元素时原样返回():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = annotate(img, [Element(id=0, bbox=(0.1, 0.1, 0.3, 0.2), text="保存")])
    assert out.shape == img.shape and out.any() and not img.any()
    assert np.array_equal(annotate(img, []), img)


# --- 中文路径下的存读图 -----------------------------------------------------


def test_imwrite_imread_中文路径往返(tmp_path):
    """cv2.imwrite 在中文路径下返回 False 但不抛异常，本项目目录名就是中文。"""
    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    target = tmp_path / "中文目录" / "带空格 的图.png"
    imwrite(str(target), img)  # 目录不存在也应自动建
    assert target.exists() and np.array_equal(imread(str(target)), img)


def test_imread_文件不存在时报错(tmp_path):
    with pytest.raises((IOError, OSError)):
        imread(str(tmp_path / "不存在.png"))
