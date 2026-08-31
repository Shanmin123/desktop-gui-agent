import pytest

from gui_agent.display import RECOMMENDED, current_resolution


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
