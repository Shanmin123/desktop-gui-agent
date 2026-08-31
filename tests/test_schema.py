import json

import pytest

from gui_agent.schema import Action, Element, ScreenState, Step, Trajectory


# --- 坐标 -------------------------------------------------------------------


def test_norm_pixel_roundtrip_and_bounds():
    s = ScreenState(width=3840, height=2160)
    assert s.to_pixel((0.5, 0.5)) == (1920, 1080)
    assert s.to_pixel((1.0, 1.0)) == (3840, 2160)
    assert tuple(round(v, 4) for v in s.to_norm(1920, 1080)) == (0.5, 0.5)
    with pytest.raises(ValueError):
        s.to_pixel((1.2, 0.5))


def test_norm_point_scales_across_resolutions():
    """1430x804 是本机 3840x2160 按模型输入上限缩放后的尺寸。"""
    p = (0.25, 0.75)
    assert ScreenState(width=3840, height=2160).to_pixel(p) == (960, 1620)
    assert ScreenState(width=1430, height=804).to_pixel(p) == (358, 603)


def test_element_center():
    cx, cy = Element(id=1, bbox=(0.2, 0.4, 0.4, 0.6), text="保存").center()
    assert (cx, cy) == (pytest.approx(0.3), pytest.approx(0.5))


# --- 动作 -------------------------------------------------------------------


def test_all_ten_action_types_construct():
    Action("click", point=(0.5, 0.5))
    Action("left_double", point=(0.5, 0.5))
    Action("right_single", point=(0.5, 0.5))
    Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9))
    Action("scroll", point=(0.5, 0.5), direction="down")
    Action("type", text="hello")
    Action("hotkey", text="ctrl+s")
    Action("wait")
    Action("finished")
    Action("call_user")


def test_validated_at_construction():
    """模型输出的动作不可信，让它在构造时就炸，比带着坏坐标传到 pyautogui 好查。"""
    with pytest.raises(ValueError, match="缺少必需字段 point"):
        Action("click")
    with pytest.raises(ValueError, match="未知动作类型"):
        Action("swipe", point=(0.5, 0.5))
    with pytest.raises(ValueError, match="scroll 方向"):
        Action("scroll", point=(0.5, 0.5), direction="diagonal")
    with pytest.raises(ValueError, match="归一化"):
        Action("click", point=(1920, 1080))


def test_terminal_actions():
    assert Action("finished").is_terminal() and Action("call_user").is_terminal()
    assert not Action("click", point=(0.5, 0.5)).is_terminal()


# --- 序列化 -----------------------------------------------------------------


def test_action_dict_roundtrip_omits_empty():
    a = Action("drag", point=(0.1, 0.2), point2=(0.8, 0.9), thought="拖到回收站")
    b = Action.from_dict(a.to_dict())
    assert (b.point, b.point2, b.thought) == (a.point, a.point2, a.thought)
    assert Action("wait").to_dict() == {"type": "wait"}


def test_from_dict_accepts_list_coords_and_extra_keys():
    a = Action.from_dict({"type": "click", "point": [0.5, 0.5], "unknown": 1})
    assert a.point == (0.5, 0.5)


# --- 轨迹 -------------------------------------------------------------------


def test_trajectory_stats_and_json():
    screen = ScreenState(width=3840, height=2160, image_path="logs/1.png")
    t = Trajectory(task_id="open_browser", instruction="打开浏览器")
    t.steps.append(Step(screen, Action("click", point=(0.1, 0.9)), elapsed=1.5))
    t.steps.append(Step(screen, Action("finished"), elapsed=0.2))

    assert t.n_steps == 2 and abs(t.wall_time - 1.7) < 1e-9
    data = json.loads(t.to_json())
    assert data["n_steps"] == 2
    assert data["steps"][0]["action"]["type"] == "click"
    assert data["steps"][0]["screen"]["width"] == 3840
    assert data["success"] is None


def test_failed_step_records_error():
    step = Step(ScreenState(1920, 1080), Action("click", point=(0.5, 0.5)),
                ok=False, error="目标窗口不存在")
    data = json.loads(Trajectory("t1", "x", [step], success=False).to_json())
    assert data["steps"][0]["error"] == "目标窗口不存在" and data["success"] is False
