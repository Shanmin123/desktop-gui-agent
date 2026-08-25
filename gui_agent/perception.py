"""桌面感知：截图、多分辨率适配、OCR、UI 元素识别与边界框绘制。

对应大纲第 2 周第 1、2、4 项。

多分辨率适配分两层：
1. 送模型前把截图缩小。缩放系数按 Claude Computer Use 文档给的算法算，取
   `1.0`、`长边上限/长边`、`sqrt(总像素上限/总像素)` 三者最小值。
2. 所有元素坐标一律归一化到 0~1（见 schema.py），下游不需要知道实际分辨率。

OCR 在缩小后的图上跑。归一化之后结果与在原图上跑是一致的，但快得多。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schema import Element, ScreenState

# 模型输入上限，对应 Claude 文档里较早模型那一档。
# 换用输入上限更高的模型时调这两个值即可。
DEFAULT_LONG_EDGE = 1568
DEFAULT_MAX_PIXELS = 1_150_000


def imwrite(path: str, img: np.ndarray) -> None:
    """存图。

    不能直接用 cv2.imwrite：在 Windows 上它按 ANSI 代码页处理路径，遇到非 ASCII
    路径会**返回 False 但不抛异常**，文件根本没写出来。本项目的上级目录名是中文，
    实测必然触发。改成先 imencode 再用 Python 的文件接口写二进制。
    """
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if not ok:
        raise IOError(f"图像编码失败：{path}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(buf.tobytes())


def imread(path: str) -> np.ndarray:
    """读图，同样绕开 cv2.imread 的非 ASCII 路径问题。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"图像读取失败：{path}")
    return img


def scale_factor(
    width: int,
    height: int,
    long_edge: int = DEFAULT_LONG_EDGE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> float:
    """算出把截图缩到模型能接受的尺寸所需的系数，永远不放大。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"分辨率非法：{width}x{height}")
    return min(
        1.0,
        long_edge / max(width, height),
        math.sqrt(max_pixels / (width * height)),
    )


def resize_for_model(
    img: np.ndarray,
    long_edge: int = DEFAULT_LONG_EDGE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Tuple[np.ndarray, float]:
    """把截图缩到模型输入尺寸，返回缩后的图和实际用的系数。"""
    h, w = img.shape[:2]
    s = scale_factor(w, h, long_edge, max_pixels)
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def _quad_to_norm_bbox(quad, w: int, h: int) -> Tuple[float, float, float, float]:
    """OCR 给的是四个角点，转成归一化的外接矩形。"""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (
        max(0.0, min(xs) / w),
        max(0.0, min(ys) / h),
        min(1.0, max(xs) / w),
        min(1.0, max(ys) / h),
    )


class Perception:
    """一次感知 = 截图 -> 缩放 -> OCR -> 归一化元素列表。

    OCR 模型加载慢，所以延迟到第一次用时才初始化，之后复用。
    """

    def __init__(
        self,
        monitor: int = 1,
        gpu: bool = True,
        langs: Tuple[str, ...] = ("ch_sim", "en"),
        long_edge: int = DEFAULT_LONG_EDGE,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        import mss  # 延迟导入，纯函数测试不需要它

        self.monitor = monitor
        self.gpu = gpu
        self.langs = list(langs)
        self.long_edge = long_edge
        self.max_pixels = max_pixels
        self._sct = mss.mss()
        self._reader = None

    # -- 截图 ---------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """抓一帧，返回 BGR 图（物理分辨率）。

        用 np.array 而不是 np.asarray：切片出来是非连续视图，底层 buffer 属于
        mss 的 ScreenShot 对象，cv2 对非连续数组的处理不可靠，这里直接拷一份。
        """
        shot = self._sct.grab(self._sct.monitors[self.monitor])
        return np.array(shot, copy=True)[:, :, :3].copy()  # BGRA -> BGR，连续

    # -- OCR ----------------------------------------------------------------

    @property
    def reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
        return self._reader

    def ocr(self, img: np.ndarray, min_confidence: float = 0.3) -> List[Element]:
        """在传入的图上跑 OCR，返回归一化坐标的元素列表。

        坐标按传入图的尺寸归一化，所以传原图还是缩放图，结果一致。

        传入的是 BGR（OpenCV 约定），但 easyocr 把三通道数组当 RGB 处理，内部走
        COLOR_RGB2GRAY，通道顺序不对会把红蓝的权重对调，彩色控件上会掉对比度，
        所以这里先转一次。
        """
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = []
        for quad, text, conf in self.reader.readtext(rgb):
            if conf < min_confidence:
                continue
            out.append(
                Element(
                    id=len(out),
                    bbox=_quad_to_norm_bbox(quad, w, h),
                    text=text,
                    source="ocr",
                    confidence=float(conf),
                )
            )
        return out

    # -- 完整一次感知 --------------------------------------------------------

    def perceive(
        self, run_ocr: bool = True, save_to: Optional[str] = None
    ) -> Tuple[ScreenState, np.ndarray]:
        """返回 (屏幕状态, 送模型用的缩放图)。

        OCR 跑在**原始分辨率**上，缩放图只用于送模型。实测（见 docs/环境配置文档.md）
        同一张 3840x2160 的截图，原图识别出 27 处文字，缩到 1430x804 只剩 3 处，
        而耗时只差 0.13 秒——easyocr 检测阶段内部本来就会把长边压到 2560，再往下缩
        就低于它的工作分辨率了。所以缩放只服务于模型的输入限制，不能顺带用来提速。

        两边坐标都归一化，所以元素框画在缩放图上依然对得上。
        """
        img = self.capture()
        h, w = img.shape[:2]

        elements = self.ocr(img) if run_ocr else []
        small, _ = resize_for_model(img, self.long_edge, self.max_pixels)

        path = ""
        if save_to:
            imwrite(save_to, small)
            path = save_to

        return ScreenState(width=w, height=h, image_path=path, elements=elements), small

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "Perception":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# -- 边界框绘制（大纲第 2 周第 4 项）-----------------------------------------


def annotate(img: np.ndarray, elements: List[Element], show_text: bool = False) -> np.ndarray:
    """在图上画出元素的边界框和编号，用于人工核对定位是否准确。

    只画编号不画识别出的文字，因为 OpenCV 的 putText 渲染不了中文。
    """
    out = img.copy()
    h, w = out.shape[:2]
    for e in elements:
        x1, y1, x2, y2 = e.bbox
        p1 = (round(x1 * w), round(y1 * h))
        p2 = (round(x2 * w), round(y2 * h))
        cv2.rectangle(out, p1, p2, (0, 200, 0), 1)
        label = str(e.id) if not show_text else f"{e.id}:{e.text[:10]}"
        cv2.putText(
            out, label, (p1[0], max(10, p1[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA,
        )
    return out


def benchmark(n: int = 5, monitor: int = 1) -> dict:
    """实测截图、缩放、OCR 各自的耗时，供第 6 周优化时对照。"""
    p = Perception(monitor=monitor)
    img = p.capture()
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    for _ in range(n):
        p.capture()
    cap_ms = (time.perf_counter() - t0) / n * 1000

    small, s = resize_for_model(img, p.long_edge, p.max_pixels)
    t0 = time.perf_counter()
    for _ in range(n):
        resize_for_model(img, p.long_edge, p.max_pixels)
    resize_ms = (time.perf_counter() - t0) / n * 1000

    p.reader  # 先把模型加载完，不计入耗时
    t0 = time.perf_counter()
    small_elems = p.ocr(small)
    ocr_small_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_elems = p.ocr(img)
    ocr_full_s = time.perf_counter() - t0

    p.close()
    return {
        "原始分辨率": f"{w}x{h}",
        "缩放后": f"{small.shape[1]}x{small.shape[0]}",
        "缩放系数": round(s, 4),
        "截图 ms": round(cap_ms, 1),
        "缩放 ms": round(resize_ms, 1),
        "OCR 全图 s": round(ocr_full_s, 2),
        "OCR 缩放图 s": round(ocr_small_s, 2),
        "全图识别数": len(full_elems),
        "缩放图识别数": len(small_elems),
    }
