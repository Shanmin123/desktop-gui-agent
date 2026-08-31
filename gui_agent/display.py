"""临时切换屏幕分辨率。

Claude Computer Use 文档推荐桌面分辨率取 1024×768 或 1280×720。

系统缩放决定了实际能截到多大。缩放不是 100% 时，Windows 按缩放比例建一个更大的
虚拟桌面：175% 下把分辨率设成 1280×720，截图拿到的是 2240×1260，仍然超出模型输入
限制。要真正跑在 1280×720，需要先把系统缩放改成 100%。

切换后用 captured_size() 确认实际尺寸，不要假定设了就是那个值。

用上下文管理器切换，退出时无论正常结束还是抛异常都还原。
"""

from __future__ import annotations

import ctypes
import warnings
from contextlib import contextmanager
from ctypes import wintypes
from typing import Iterator, Tuple

RECOMMENDED = (1280, 720)

_ENUM_CURRENT_SETTINGS = -1
_DM_PELSWIDTH = 0x00080000
_DM_PELSHEIGHT = 0x00100000
_CDS_OK = 0  # DISP_CHANGE_SUCCESSFUL


class _DEVMODE(ctypes.Structure):
    _fields_ = [
        ("dmDeviceName", wintypes.WCHAR * 32),
        ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD),
        ("dmPositionX", ctypes.c_long),
        ("dmPositionY", ctypes.c_long),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
        ("dmColor", wintypes.SHORT),
        ("dmDuplex", wintypes.SHORT),
        ("dmYResolution", wintypes.SHORT),
        ("dmTTOption", wintypes.SHORT),
        ("dmCollate", wintypes.SHORT),
        ("dmFormName", wintypes.WCHAR * 32),
        ("dmLogPixels", wintypes.WORD),
        ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD),
        ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD),
        ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD),
        ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD),
        ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD),
        ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD),
        ("dmPanningHeight", wintypes.DWORD),
    ]


def _current() -> _DEVMODE:
    dm = _DEVMODE()
    dm.dmSize = ctypes.sizeof(_DEVMODE)
    if not ctypes.windll.user32.EnumDisplaySettingsW(None, _ENUM_CURRENT_SETTINGS, ctypes.byref(dm)):
        raise OSError("读取当前显示设置失败")
    return dm


def current_resolution() -> Tuple[int, int]:
    dm = _current()
    return int(dm.dmPelsWidth), int(dm.dmPelsHeight)


def _ensure_dpi_aware() -> None:
    """声明本进程 DPI 感知。

    不声明的话，系统 API 返回的是按缩放比例虚拟化过的尺寸，和实际截到的图对不上。
    pyautogui 导入时会自己声明，于是同一份代码会因为导入顺序给出不同答案，这里显式
    声明一次消除这个不确定性。
    """
    try:
        # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def scaling_percent() -> int:
    """当前系统缩放比例，100 表示无缩放。"""
    return round(_current().dmLogPixels / 96 * 100)


def captured_size(monitor: int = 1) -> Tuple[int, int]:
    """截图实际会拿到的尺寸。系统缩放不是 100% 时与设定分辨率不同。

    问 mss 而不是 GetSystemMetrics：后者在 DPI 不感知的进程里返回虚拟化后的尺寸，
    与实际截到的图对不上。
    """
    import mss

    _ensure_dpi_aware()
    with mss.mss() as s:
        m = s.monitors[monitor]
        return m["width"], m["height"]


def set_resolution(width: int, height: int) -> None:
    dm = _current()
    dm.dmPelsWidth, dm.dmPelsHeight = width, height
    dm.dmFields = _DM_PELSWIDTH | _DM_PELSHEIGHT
    code = ctypes.windll.user32.ChangeDisplaySettingsW(ctypes.byref(dm), 0)
    if code != _CDS_OK:
        raise OSError(f"切换到 {width}x{height} 失败，返回码 {code}")


def restore() -> None:
    """还原到注册表里保存的默认分辨率。"""
    ctypes.windll.user32.ChangeDisplaySettingsW(None, 0)


@contextmanager
def resolution(width: int = RECOMMENDED[0], height: int = RECOMMENDED[1]) -> Iterator[Tuple[int, int]]:
    """临时切换分辨率，退出时还原。

    yield 出来的是截图实际会拿到的尺寸，系统缩放不是 100% 时与传入值不同。

        with resolution(1280, 720) as actual:
            ...
    """
    before = current_resolution()
    if before == (width, height):
        yield before
        return
    set_resolution(width, height)
    actual = captured_size()
    if actual != (width, height):
        warnings.warn(
            f"设定 {width}x{height}，实际截图尺寸 {actual[0]}x{actual[1]}。"
            f"系统缩放为 {scaling_percent()}%，改成 100% 才能真正跑在设定分辨率上。",
            stacklevel=2,
        )
    try:
        yield actual
    finally:
        restore()
