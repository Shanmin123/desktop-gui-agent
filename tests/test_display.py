import os

import pytest

from gui_agent.display import RECOMMENDED, current_resolution

# 会真的切换屏幕分辨率的测试。默认不跑：普通一次 pytest 不该把用户的屏幕改来改去。
#   set GUI_AGENT_DISPLAY_TESTS=1 && pytest tests/test_display.py
needs_real_display = pytest.mark.skipif(
    not os.environ.get("GUI_AGENT_DISPLAY_TESTS"),
    reason="会切换真实分辨率，需设 GUI_AGENT_DISPLAY_TESTS=1 才跑",
)


def test_recommended_is_claude_doc_value():
    """Claude Computer Use 文档推荐 1024x768 或 1280x720。"""
    assert RECOMMENDED == (1280, 720)


def test_current_resolution_returns_positive_ints():
    w, h = current_resolution()
    assert isinstance(w, int) and isinstance(h, int)
    assert w > 0 and h > 0


def test_recommended_needs_no_scaling():
    from gui_agent.perception import scale_factor

    assert scale_factor(*RECOMMENDED) == 1.0


def test_scaling_percent_is_reasonable():
    from gui_agent.display import scaling_percent

    assert 100 <= scaling_percent() <= 400


def test_captured_size_matches_resolution_only_at_100_percent():
    """系统缩放不是 100% 时，截图尺寸与设定分辨率不一致。

    实测：175% 缩放下把分辨率设成 1280x720，截图拿到 2240x1260。
    """
    from gui_agent.display import captured_size, current_resolution, scaling_percent

    if scaling_percent() == 100:
        assert captured_size() == current_resolution()
    else:
        assert captured_size()[0] >= current_resolution()[0]


@needs_real_display
def test_resolution_warns_when_target_not_reached():
    """切换不到目标分辨率时必须发警告，不能静默失败。

    实际能不能切到取决于系统缩放，以及进程声明了哪一级 DPI 感知，所以这里断言的是
    「没达成就要警告」，而不是断言具体切成了多少。
    """
    import warnings as w

    from gui_agent.display import resolution

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        with resolution(1280, 720) as actual:
            pass
    warned = any("系统缩放" in str(c.message) for c in caught)
    assert (actual == (1280, 720)) != warned, "达成则不该警告，未达成则必须警告"


# --- 还原 -------------------------------------------------------------------


@pytest.fixture
def fake_display(monkeypatch):
    """把三个真正碰显示设置的函数换成记录调用，不动真实屏幕。"""
    from gui_agent import display

    calls = []
    monkeypatch.setattr(display, "current_resolution", lambda: (3840, 2160))
    monkeypatch.setattr(display, "set_resolution", lambda w, h: calls.append(("set", w, h)))
    monkeypatch.setattr(display, "captured_size", lambda monitor=1: (1280, 720))
    monkeypatch.setattr(display, "restore", lambda w=0, h=0: calls.append(("restore", w, h)))
    return display, calls


def test_resolution_restores_when_body_raises(fake_display):
    display, calls = fake_display
    with pytest.raises(RuntimeError):
        with display.resolution(1280, 720):
            raise RuntimeError("任务中途炸了")
    assert calls == [("set", 1280, 720), ("restore", 3840, 2160)]


def test_resolution_restores_when_size_check_fails(fake_display, monkeypatch):
    """改完分辨率之后、进入 with 之前出错，也必须还原。"""
    display, calls = fake_display

    def boom(monitor=1):
        raise OSError("读取截图尺寸失败")

    monkeypatch.setattr(display, "captured_size", boom)
    with pytest.raises(OSError):
        with display.resolution(1280, 720):
            pass
    assert calls == [("set", 1280, 720), ("restore", 3840, 2160)]


def test_restores_entry_resolution_not_registry_default(fake_display, monkeypatch):
    """必须切回进入时的尺寸。

    restore() 不带参数走的是注册表默认值，而用户当时用的分辨率未必就是默认值，
    那样「还原」会把屏幕改成第三个尺寸。
    """
    display, calls = fake_display
    monkeypatch.setattr(display, "current_resolution", lambda: (1920, 1080))
    with display.resolution(1280, 720):
        pass
    assert calls == [("set", 1280, 720), ("restore", 1920, 1080)]


def test_restore_without_args_falls_back_to_registry_default(monkeypatch):
    from gui_agent import display

    seen = []
    monkeypatch.setattr(display, "set_resolution", lambda w, h: seen.append((w, h)))
    monkeypatch.setattr(display.ctypes, "windll", type("W", (), {
        "user32": type("U", (), {"ChangeDisplaySettingsW": staticmethod(
            lambda *a: seen.append("registry"))})()
    })())
    display.restore()
    display.restore(1280, 720)
    assert seen == ["registry", (1280, 720)]


def test_resolution_is_a_noop_when_already_at_target(fake_display, monkeypatch):
    display, calls = fake_display
    monkeypatch.setattr(display, "current_resolution", lambda: (1280, 720))
    with display.resolution(1280, 720) as actual:
        assert actual == (1280, 720)
    assert calls == []


# --- 跨平台 -----------------------------------------------------------------


def test_windows_only_calls_raise_a_clear_message(monkeypatch):
    """大纲技术栈要求支持 Windows / macOS / Linux。切换分辨率是平台相关的，
    在别的平台上要给出能读懂的原因，而不是 AttributeError。"""
    from gui_agent import display

    monkeypatch.setattr(display, "IS_WINDOWS", False)
    for fn in (lambda: display.set_resolution(1280, 720), lambda: display.restore()):
        with pytest.raises(NotImplementedError, match="Windows"):
            fn()


def test_current_resolution_falls_back_to_mss_off_windows(monkeypatch):
    from gui_agent import display

    monkeypatch.setattr(display, "IS_WINDOWS", False)
    monkeypatch.setattr(display, "captured_size", lambda monitor=1: (1920, 1080))
    assert display.current_resolution() == (1920, 1080)


def test_scaling_is_one_hundred_off_windows(monkeypatch):
    from gui_agent import display

    monkeypatch.setattr(display, "IS_WINDOWS", False)
    assert display.scaling_percent() == 100


def test_dpi_call_is_a_noop_off_windows(monkeypatch):
    """非 Windows 上不能去碰 windll。"""
    from gui_agent import display

    monkeypatch.setattr(display, "IS_WINDOWS", False)
    monkeypatch.delattr(display.ctypes, "windll", raising=False)
    display._ensure_dpi_aware()  # 不该抛
