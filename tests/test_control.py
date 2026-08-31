import pytest

from gui_agent.control import BLOCKED_HOTKEYS, Controller, RecordingBackend, normalize_hotkey
from gui_agent.schema import Action


@pytest.fixture
def ctrl():
    """用假 backend，不碰真实鼠标键盘。"""
    return Controller(backend=RecordingBackend(3840, 2160))


# --- 坐标换算 ---------------------------------------------------------------


def test_norm_to_pixel(ctrl):
    assert ctrl.to_pixel((0.5, 0.5)) == (1920, 1080)
    assert ctrl.to_pixel((0.0, 0.0)) == (0, 0)
    assert ctrl.to_pixel((1.0, 1.0)) == (3840, 2160)


def test_click_correct_when_control_and_capture_sizes_differ():
    """归一化坐标免疫 DPI 缩放：控制端按自己的尺寸反归一化即可。"""
    a = Controller(backend=RecordingBackend(3840, 2160))
    b = Controller(backend=RecordingBackend(2194, 1234))
    p = (0.25, 0.5)
    assert a.to_pixel(p) == (960, 1080)
    assert b.to_pixel(p) == (548, 617)  # 548.5 遇上 round 的五取偶


# --- 十个动作 ---------------------------------------------------------------


def test_click_variants(ctrl):
    ctrl.execute(Action("click", point=(0.5, 0.5)))
    ctrl.execute(Action("left_double", point=(0.1, 0.2)))
    ctrl.execute(Action("right_single", point=(0.1, 0.2)))
    assert ctrl.backend.calls == [
        ("click", 1920, 1080, "left", 1),
        ("click", 384, 432, "left", 2),
        ("click", 384, 432, "right", 1),
    ]


def test_drag(ctrl):
    ctrl.execute(Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9)))
    assert ctrl.backend.calls == [("drag", 384, 216, 3456, 1944)]


@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
def test_scroll_directions(ctrl, direction):
    ctrl.execute(Action("scroll", point=(0.5, 0.5), direction=direction))
    assert ctrl.backend.calls == [("scroll", 1920, 1080, direction, 3)]


def test_ascii_typed_non_ascii_pasted(ctrl):
    """pyautogui 的 KEYBOARD_KEYS 只有 ASCII，打不出中文。"""
    ctrl.execute(Action("type", text="hello"))
    ctrl.execute(Action("type", text="打开浏览器"))
    assert ctrl.backend.calls == [("write", "hello"), ("paste", "打开浏览器")]


def test_hotkey(ctrl):
    ctrl.execute(Action("hotkey", text="ctrl+s"))
    assert ctrl.backend.calls == [("hotkey", ("ctrl", "s"))]


def test_wait_and_terminal_actions_are_noops(ctrl):
    ctrl.dry_run = True  # 避免真的 sleep
    for a in [Action("wait"), Action("finished"), Action("call_user")]:
        assert ctrl.execute(a).ok
    assert ctrl.backend.calls == []


# --- 组合键解析 -------------------------------------------------------------


def test_hotkey_parsing_and_aliases():
    assert normalize_hotkey("Ctrl+S") == ["ctrl", "s"]
    assert normalize_hotkey("ctrl + shift + n") == ["ctrl", "shift", "n"]
    assert normalize_hotkey("super+d") == ["win", "d"]
    assert normalize_hotkey("alt+Return") == ["alt", "enter"]
    with pytest.raises(ValueError):
        normalize_hotkey("+++")


# --- 安全 -------------------------------------------------------------------


def test_lock_screen_hotkeys_blocked(ctrl):
    r = ctrl.execute(Action("hotkey", text="ctrl+alt+delete"))
    assert not r.ok and ctrl.backend.calls == []
    assert not ctrl.execute(Action("hotkey", text="Win+L")).ok


def test_alt_f4_allowed(ctrl):
    """大纲第 4 周的基础任务里有「关闭应用」，不能把 alt+f4 也禁掉。"""
    assert "alt+f4" not in BLOCKED_HOTKEYS
    assert ctrl.execute(Action("hotkey", text="alt+f4")).ok


def test_destructive_commands_blocked_normal_text_allowed():
    """规则锚定在开头，正常英文里出现 format、rm 等词不应被拦。"""
    c = Controller(backend=RecordingBackend())
    assert not c.execute(Action("type", text="format c:")).ok
    assert not c.execute(Action("type", text="rm -rf /")).ok
    assert c.execute(Action("type", text="format the paragraph")).ok
    assert c.execute(Action("type", text="删除这一行")).ok


def test_dry_run_applies_to_explicit_backend():
    """dry_run 与用哪个 backend 无关，都不应产生实际操作。"""
    c = Controller(backend=RecordingBackend(), dry_run=True)
    assert c.execute(Action("click", point=(0.5, 0.5))).ok
    assert not c.execute(Action("hotkey", text="ctrl+alt+delete")).ok  # 拦截仍生效
    assert c.backend.calls == []
    assert len(c.dry_run_log) == 1


# --- 失败处理与批量执行 -----------------------------------------------------


def test_backend_exception_returns_failure():
    class Broken(RecordingBackend):
        def click(self, *a, **k):
            raise RuntimeError("鼠标被占用")

    r = Controller(backend=Broken()).execute(Action("click", point=(0.5, 0.5)))
    assert not r.ok and "鼠标被占用" in r.error


def test_run_stops_on_terminal_or_failure(ctrl):
    assert len(ctrl.run([
        Action("click", point=(0.1, 0.1)), Action("finished"), Action("click", point=(0.9, 0.9)),
    ])) == 2
    c = Controller(backend=RecordingBackend())
    assert len(c.run([
        Action("click", point=(0.1, 0.1)), Action("hotkey", text="win+l"), Action("wait"),
    ])) == 2


def test_history_records_every_step(ctrl):
    ctrl.execute(Action("click", point=(0.1, 0.1)))
    ctrl.execute(Action("hotkey", text="ctrl+alt+delete"))
    assert [ok for _, r in ctrl.history for ok in [r.ok]] == [True, False]


def test_move_and_read_position(ctrl):
    want = ctrl.to_pixel((0.75, 0.25))
    ctrl.backend.move(*want)
    assert ctrl.backend.position() == want == (2880, 540)


# --- 可配置项 ---------------------------------------------------------------


def test_custom_blocklists_can_be_injected():
    """黑名单是构造参数，不同场景可以换一套。"""
    c = Controller(backend=RecordingBackend(),
                   blocked_hotkeys=frozenset({"ctrl+s"}),
                   blocked_text=(r"^绝密",))
    assert not c.execute(Action("hotkey", text="ctrl+s")).ok
    assert not c.execute(Action("type", text="绝密文件")).ok
    assert c.execute(Action("hotkey", text="ctrl+alt+delete")).ok  # 不在新名单里
    assert c.execute(Action("type", text="format c:")).ok


def test_run_with_empty_list(ctrl):
    assert ctrl.run([]) == [] and ctrl.backend.calls == []


def test_wait_seconds_configurable():
    import time as _t

    c = Controller(backend=RecordingBackend(), wait_seconds=0.05)
    t0 = _t.perf_counter()
    c.execute(Action("wait"))
    assert 0.03 < _t.perf_counter() - t0 < 0.5
