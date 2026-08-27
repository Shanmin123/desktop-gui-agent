import pytest

from gui_agent.control import (
    BLOCKED_HOTKEYS,
    Controller,
    RecordingBackend,
    normalize_hotkey,
)
from gui_agent.schema import Action


@pytest.fixture
def ctrl():
    """用假 backend，不碰真实鼠标键盘。"""
    return Controller(backend=RecordingBackend(3840, 2160))


# --- 坐标换算 ---------------------------------------------------------------


def test_归一化坐标换算到像素(ctrl):
    assert ctrl.to_pixel((0.5, 0.5)) == (1920, 1080)
    assert ctrl.to_pixel((0.0, 0.0)) == (0, 0)
    assert ctrl.to_pixel((1.0, 1.0)) == (3840, 2160)


def test_控制端分辨率与截图端不同也能点对():
    """归一化坐标免疫 DPI 缩放：控制端按自己的尺寸反归一化即可。

    假设截图是 3840x2160，而控制端只认 2194x1234（DPI 不感知的情况），
    同一个归一化坐标在两边都落在各自屏幕的同一相对位置上。
    """
    a = Controller(backend=RecordingBackend(3840, 2160))
    b = Controller(backend=RecordingBackend(2194, 1234))
    p = (0.25, 0.5)
    assert a.to_pixel(p) == (960, 1080)
    # 2194*0.25 = 548.5，Python 的 round 是四舍六入五取偶，落到 548
    assert b.to_pixel(p) == (548, 617)


# --- 十个动作 ---------------------------------------------------------------


def test_点击(ctrl):
    assert ctrl.execute(Action("click", point=(0.5, 0.5))).ok
    assert ctrl.backend.calls == [("click", 1920, 1080, "left", 1)]


def test_双击(ctrl):
    ctrl.execute(Action("left_double", point=(0.1, 0.2)))
    assert ctrl.backend.calls == [("click", 384, 432, "left", 2)]


def test_右键(ctrl):
    ctrl.execute(Action("right_single", point=(0.1, 0.2)))
    assert ctrl.backend.calls == [("click", 384, 432, "right", 1)]


def test_拖拽(ctrl):
    ctrl.execute(Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9)))
    assert ctrl.backend.calls == [("drag", 384, 216, 3456, 1944)]


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_四个方向的滚动(ctrl, direction):
    ctrl.execute(Action("scroll", point=(0.5, 0.5), direction=direction))
    assert ctrl.backend.calls == [("scroll", 1920, 1080, direction, 3)]


def test_英文输入走键盘(ctrl):
    ctrl.execute(Action("type", text="hello world"))
    assert ctrl.backend.calls == [("write", "hello world")]


def test_中文输入走剪贴板(ctrl):
    """pyautogui 的 KEYBOARD_KEYS 只有 ASCII，打不出中文，必须走剪贴板。"""
    ctrl.execute(Action("type", text="打开浏览器"))
    assert ctrl.backend.calls == [("paste", "打开浏览器")]


def test_混合中英文也走剪贴板(ctrl):
    ctrl.execute(Action("type", text="搜索 python"))
    assert ctrl.backend.calls[0][0] == "paste"


def test_组合键(ctrl):
    ctrl.execute(Action("hotkey", text="ctrl+s"))
    assert ctrl.backend.calls == [("hotkey", ("ctrl", "s"))]


def test_等待不产生桌面操作(ctrl):
    ctrl.dry_run = True  # 避免真的 sleep
    assert ctrl.execute(Action("wait")).ok
    assert ctrl.backend.calls == []


def test_终止动作不产生桌面操作(ctrl):
    assert ctrl.execute(Action("finished")).ok
    assert ctrl.execute(Action("call_user")).ok
    assert ctrl.backend.calls == []


# --- 组合键解析 -------------------------------------------------------------


def test_组合键大小写和空格都能解析():
    assert normalize_hotkey("Ctrl+S") == ["ctrl", "s"]
    assert normalize_hotkey("ctrl + shift + n") == ["ctrl", "shift", "n"]


def test_组合键别名归一():
    assert normalize_hotkey("control+c") == ["ctrl", "c"]
    assert normalize_hotkey("super+d") == ["win", "d"]
    assert normalize_hotkey("cmd+q") == ["win", "q"]
    assert normalize_hotkey("alt+Return") == ["alt", "enter"]
    assert normalize_hotkey("Escape") == ["esc"]


def test_空组合键报错():
    with pytest.raises(ValueError):
        normalize_hotkey("+++")


# --- 安全 -------------------------------------------------------------------


def test_锁屏类组合键被拒(ctrl):
    r = ctrl.execute(Action("hotkey", text="ctrl+alt+delete"))
    assert not r.ok and "禁用名单" in r.error
    assert ctrl.backend.calls == []


def test_win_l_被拒(ctrl):
    assert not ctrl.execute(Action("hotkey", text="Win+L")).ok


def test_关闭应用的alt_f4放行(ctrl):
    """大纲第 4 周的基础任务里有「关闭应用」，不能把 alt+f4 也禁掉。"""
    assert "alt+f4" not in BLOCKED_HOTKEYS
    assert ctrl.execute(Action("hotkey", text="alt+f4")).ok


@pytest.mark.parametrize(
    "bad",
    ["format c:", "rm -rf /", "shutdown -s -t 0", "del /f /s /q C:\\", "diskpart", "mkfs.ext4"],
)
def test_破坏性输入被拒(bad):
    c = Controller(backend=RecordingBackend())
    r = c.execute(Action("type", text=bad))
    assert not r.ok, f"{bad!r} 应当被拒"
    assert c.backend.calls == []


@pytest.mark.parametrize(
    "ok_text",
    [
        "format the paragraph",  # 最初用子串匹配时，这条被 "format " 误伤
        "format this cell as currency",
        "删除这一行",
        "hello",
        "how to shut down the app",  # 只拦 shutdown 开头，不拦这种
        "rm is a unix command",
    ],
)
def test_正常输入不被误伤(ok_text):
    c = Controller(backend=RecordingBackend())
    assert c.execute(Action("type", text=ok_text)).ok, f"{ok_text!r} 不该被拦"


def test_文本黑名单挡不住上下文相关的危险():
    """记录这道防线的已知局限，避免以后误以为它是可靠的安全保障。

    同一段文字打进文档无害、打进命令行危险，光看文本判断不出来。真正的保障是
    dry_run、FAILSAFE，以及把测试放在虚拟机里跑。
    """
    c = Controller(backend=RecordingBackend())
    # 这条在 cmd 里是删库，但形态上不像命令，黑名单放行了
    assert c.execute(Action("type", text="/f /s /q C:\\")).ok


# --- 失败处理与记录 ---------------------------------------------------------


def test_后端抛异常时返回失败而不是崩掉():
    class Broken(RecordingBackend):
        def click(self, *a, **k):
            raise RuntimeError("鼠标被占用")

    c = Controller(backend=Broken())
    r = c.execute(Action("click", point=(0.5, 0.5)))
    assert not r.ok
    assert "RuntimeError" in r.error and "鼠标被占用" in r.error


def test_每一步都记进history(ctrl):
    ctrl.execute(Action("click", point=(0.1, 0.1)))
    ctrl.execute(Action("hotkey", text="ctrl+alt+delete"))  # 会被拒
    assert len(ctrl.history) == 2
    assert ctrl.history[0][1].ok and not ctrl.history[1][1].ok


def test_耗时被记录(ctrl):
    r = ctrl.execute(Action("click", point=(0.5, 0.5)))
    assert r.elapsed >= 0.0


# --- 批量执行 ---------------------------------------------------------------


def test_run_遇终止动作停止(ctrl):
    results = ctrl.run([
        Action("click", point=(0.1, 0.1)),
        Action("finished"),
        Action("click", point=(0.9, 0.9)),  # 不该执行
    ])
    assert len(results) == 2
    assert len(ctrl.backend.calls) == 1


def test_run_遇失败停止(ctrl):
    results = ctrl.run([
        Action("click", point=(0.1, 0.1)),
        Action("hotkey", text="win+l"),  # 被拒
        Action("click", point=(0.9, 0.9)),  # 不该执行
    ])
    assert len(results) == 2
    assert not results[1].ok
    assert len(ctrl.backend.calls) == 1


# --- dry-run ----------------------------------------------------------------


def test_dry_run_默认用假后端():
    c = Controller(dry_run=True)
    assert isinstance(c.backend, RecordingBackend)
    assert c.execute(Action("click", point=(0.5, 0.5))).ok
