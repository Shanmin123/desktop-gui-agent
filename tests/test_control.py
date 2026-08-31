import pytest

from gui_agent.control import BLOCKED_HOTKEYS, Controller, RecordingBackend, normalize_hotkey
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
    """归一化坐标免疫 DPI 缩放：控制端按自己的尺寸反归一化即可。"""
    a = Controller(backend=RecordingBackend(3840, 2160))
    b = Controller(backend=RecordingBackend(2194, 1234))
    p = (0.25, 0.5)
    assert a.to_pixel(p) == (960, 1080)
    assert b.to_pixel(p) == (548, 617)  # 548.5 遇上 round 的五取偶


# --- 十个动作 ---------------------------------------------------------------


def test_三种点击(ctrl):
    ctrl.execute(Action("click", point=(0.5, 0.5)))
    ctrl.execute(Action("left_double", point=(0.1, 0.2)))
    ctrl.execute(Action("right_single", point=(0.1, 0.2)))
    assert ctrl.backend.calls == [
        ("click", 1920, 1080, "left", 1),
        ("click", 384, 432, "left", 2),
        ("click", 384, 432, "right", 1),
    ]


def test_拖拽(ctrl):
    ctrl.execute(Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9)))
    assert ctrl.backend.calls == [("drag", 384, 216, 3456, 1944)]


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_四个方向的滚动(ctrl, direction):
    ctrl.execute(Action("scroll", point=(0.5, 0.5), direction=direction))
    assert ctrl.backend.calls == [("scroll", 1920, 1080, direction, 3)]


def test_英文走键盘中文走剪贴板(ctrl):
    """pyautogui 的 KEYBOARD_KEYS 只有 ASCII，打不出中文。"""
    ctrl.execute(Action("type", text="hello"))
    ctrl.execute(Action("type", text="打开浏览器"))
    assert ctrl.backend.calls == [("write", "hello"), ("paste", "打开浏览器")]


def test_组合键(ctrl):
    ctrl.execute(Action("hotkey", text="ctrl+s"))
    assert ctrl.backend.calls == [("hotkey", ("ctrl", "s"))]


def test_等待与终止动作不产生桌面操作(ctrl):
    ctrl.dry_run = True  # 避免真的 sleep
    for a in [Action("wait"), Action("finished"), Action("call_user")]:
        assert ctrl.execute(a).ok
    assert ctrl.backend.calls == []


# --- 组合键解析 -------------------------------------------------------------


def test_组合键解析与别名归一():
    assert normalize_hotkey("Ctrl+S") == ["ctrl", "s"]
    assert normalize_hotkey("ctrl + shift + n") == ["ctrl", "shift", "n"]
    assert normalize_hotkey("super+d") == ["win", "d"]
    assert normalize_hotkey("alt+Return") == ["alt", "enter"]
    with pytest.raises(ValueError):
        normalize_hotkey("+++")


# --- 安全 -------------------------------------------------------------------


def test_锁屏类组合键被拒(ctrl):
    r = ctrl.execute(Action("hotkey", text="ctrl+alt+delete"))
    assert not r.ok and ctrl.backend.calls == []
    assert not ctrl.execute(Action("hotkey", text="Win+L")).ok


def test_关闭应用的alt_f4放行(ctrl):
    """大纲第 4 周的基础任务里有「关闭应用」，不能把 alt+f4 也禁掉。"""
    assert "alt+f4" not in BLOCKED_HOTKEYS
    assert ctrl.execute(Action("hotkey", text="alt+f4")).ok


def test_破坏性命令被拒而正常输入放行():
    """黑名单最初用子串匹配，把 "format the paragraph" 也拦了。"""
    c = Controller(backend=RecordingBackend())
    assert not c.execute(Action("type", text="format c:")).ok
    assert not c.execute(Action("type", text="rm -rf /")).ok
    assert c.execute(Action("type", text="format the paragraph")).ok
    assert c.execute(Action("type", text="删除这一行")).ok


def test_dry_run_对显式传入的后端也生效():
    """一度只在构造时挑 backend，传真实后端加 dry_run 会真的操作桌面。"""
    c = Controller(backend=RecordingBackend(), dry_run=True)
    assert c.execute(Action("click", point=(0.5, 0.5))).ok
    assert not c.execute(Action("hotkey", text="ctrl+alt+delete")).ok  # 拦截仍生效
    assert c.backend.calls == []
    assert len(c.dry_run_log) == 1


# --- 失败处理与批量执行 -----------------------------------------------------


def test_后端抛异常时返回失败而不是崩掉():
    class Broken(RecordingBackend):
        def click(self, *a, **k):
            raise RuntimeError("鼠标被占用")

    r = Controller(backend=Broken()).execute(Action("click", point=(0.5, 0.5)))
    assert not r.ok and "鼠标被占用" in r.error


def test_run_遇终止或失败即停(ctrl):
    assert len(ctrl.run([
        Action("click", point=(0.1, 0.1)), Action("finished"), Action("click", point=(0.9, 0.9)),
    ])) == 2
    c = Controller(backend=RecordingBackend())
    assert len(c.run([
        Action("click", point=(0.1, 0.1)), Action("hotkey", text="win+l"), Action("wait"),
    ])) == 2


def test_每一步都记进history(ctrl):
    ctrl.execute(Action("click", point=(0.1, 0.1)))
    ctrl.execute(Action("hotkey", text="ctrl+alt+delete"))
    assert [ok for _, r in ctrl.history for ok in [r.ok]] == [True, False]


def test_移动并读回位置(ctrl):
    want = ctrl.to_pixel((0.75, 0.25))
    ctrl.backend.move(*want)
    assert ctrl.backend.position() == want == (2880, 540)
