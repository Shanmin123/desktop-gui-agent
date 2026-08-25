import numpy as np
import pytest

from gui_agent.perception import (
    DEFAULT_LONG_EDGE,
    DEFAULT_MAX_PIXELS,
    _quad_to_norm_bbox,
    annotate,
    resize_for_model,
    scale_factor,
)
from gui_agent.schema import Element


# --- 缩放系数 ---------------------------------------------------------------


def test_本机4k屏的缩放系数():
    """3840x2160 = 8.29MP，总像素这一项比长边这一项更紧，取 0.372。"""
    s = scale_factor(3840, 2160)
    assert round(s, 4) == 0.3724
    assert round(3840 * s) == 1430 and round(2160 * s) == 804


def test_小屏不放大():
    assert scale_factor(1024, 768) == 1.0
    assert scale_factor(800, 600) == 1.0


def test_取三者最小值():
    """极端细长的图，长边这一项比总像素这一项更紧。"""
    s = scale_factor(10000, 100)
    assert s == pytest.approx(DEFAULT_LONG_EDGE / 10000)


def test_缩放后不超过两条上限():
    for w, h in [(3840, 2160), (2560, 1440), (1920, 1080), (5120, 2880), (1366, 768)]:
        s = scale_factor(w, h)
        nw, nh = w * s, h * s
        assert max(nw, nh) <= DEFAULT_LONG_EDGE + 1
        assert nw * nh <= DEFAULT_MAX_PIXELS + 1000


def test_上限可调():
    """换成输入上限更高的模型时，系数应随之变大。"""
    low = scale_factor(3840, 2160)
    high = scale_factor(3840, 2160, long_edge=2576, max_pixels=3_750_000)
    assert high > low


def test_非法分辨率报错():
    with pytest.raises(ValueError):
        scale_factor(0, 1080)
    with pytest.raises(ValueError):
        scale_factor(1920, -1)


# --- 实际缩放 ---------------------------------------------------------------


def test_resize输出尺寸与系数一致():
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    small, s = resize_for_model(img)
    assert (small.shape[1], small.shape[0]) == (round(3840 * s), round(2160 * s))
    assert small.shape[2] == 3


def test_小图原样返回():
    img = np.zeros((768, 1024, 3), dtype=np.uint8)
    small, s = resize_for_model(img)
    assert s == 1.0
    assert small is img


# --- 坐标归一化 -------------------------------------------------------------


def test_四角点转归一化外接矩形():
    quad = [[100, 50], [300, 50], [300, 90], [100, 90]]
    assert _quad_to_norm_bbox(quad, 1000, 500) == (0.1, 0.1, 0.3, 0.18)


def test_倾斜框取外接矩形():
    quad = [[100, 60], [300, 50], [305, 90], [105, 100]]
    x1, y1, x2, y2 = _quad_to_norm_bbox(quad, 1000, 500)
    assert (x1, y1) == (0.1, 0.1)
    assert (x2, y2) == (0.305, 0.2)


def test_越界坐标被裁到0和1():
    quad = [[-20, -10], [1200, -10], [1200, 600], [-20, 600]]
    assert _quad_to_norm_bbox(quad, 1000, 500) == (0.0, 0.0, 1.0, 1.0)


def test_同一元素在不同分辨率下归一化结果相同():
    """缩放前后跑 OCR，归一化坐标应当一致，这是可以在缩放图上跑 OCR 的前提。"""
    full = [[384, 216], [768, 216], [768, 432], [384, 432]]
    small = [[143, 80], [286, 80], [286, 161], [143, 161]]
    a = _quad_to_norm_bbox(full, 3840, 2160)
    b = _quad_to_norm_bbox(small, 1430, 804)
    for u, v in zip(a, b):
        assert abs(u - v) < 0.005


# --- 边界框绘制 -------------------------------------------------------------


def test_画框不改变图像尺寸且确实画了东西():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    elems = [
        Element(id=0, bbox=(0.1, 0.1, 0.3, 0.2), text="保存"),
        Element(id=1, bbox=(0.5, 0.5, 0.9, 0.7), text="cancel"),
    ]
    out = annotate(img, elems)
    assert out.shape == img.shape
    assert out.any(), "应当画上了框"
    assert not img.any(), "不应改动原图"


def test_没有元素时原样返回():
    img = np.full((100, 100, 3), 7, dtype=np.uint8)
    out = annotate(img, [])
    assert np.array_equal(out, img)


# --- 中文路径下的存读图 -----------------------------------------------------


def test_中文路径存图不能用cv2_imwrite(tmp_path):
    """记录 cv2 的行为，防止以后有人图省事换回去。

    本项目的上级目录名是中文，cv2.imwrite 在 Windows 上会静默失败。
    """
    import cv2

    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    p = tmp_path / "中文目录"
    p.mkdir()
    target = p / "图片.png"

    ok = cv2.imwrite(str(target), img)
    if not ok:
        assert not target.exists(), "cv2.imwrite 返回 False 时不应留下文件"


def test_imwrite_imread_中文路径往返(tmp_path):
    from gui_agent.perception import imread, imwrite

    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    target = tmp_path / "中文目录" / "带空格 的图.png"

    imwrite(str(target), img)
    assert target.exists() and target.stat().st_size > 0

    back = imread(str(target))
    assert back.shape == img.shape
    assert np.array_equal(back, img), "png 是无损的，应当逐像素相同"


def test_imwrite_自动建目录(tmp_path):
    from gui_agent.perception import imwrite

    target = tmp_path / "还没有" / "这两层" / "x.png"
    imwrite(str(target), np.zeros((5, 5, 3), dtype=np.uint8))
    assert target.exists()


def test_imread_文件不存在时报错(tmp_path):
    from gui_agent.perception import imread

    with pytest.raises((IOError, OSError, FileNotFoundError)):
        imread(str(tmp_path / "不存在.png"))
