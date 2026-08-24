import json

import pytest

from gui_agent.schema import Action, Element, ScreenState, Step, Trajectory


# --- 坐标换算 ---------------------------------------------------------------


def test_归一化与像素互转():
    s = ScreenState(width=3840, height=2160)
    assert s.to_pixel((0.5, 0.5)) == (1920, 1080)
    assert s.to_pixel((0.0, 0.0)) == (0, 0)
    assert s.to_pixel((1.0, 1.0)) == (3840, 2160)
    x, y = s.to_norm(1920, 1080)
    assert (round(x, 4), round(y, 4)) == (0.5, 0.5)


def test_换分辨率后同一归一化坐标落在等比位置():
    a = ScreenState(width=3840, height=2160)
    b = ScreenState(width=1428, height=803)
    p = (0.25, 0.75)
    assert a.to_pixel(p) == (960, 1620)
    assert b.to_pixel(p) == (357, 602)


def test_越界坐标被拒绝():
    s = ScreenState(width=1920, height=1080)
    with pytest.raises(ValueError):
        s.to_pixel((1.2, 0.5))
    with pytest.raises(ValueError):
        s.to_pixel((-0.1, 0.5))


def test_元素中心点():
    e = Element(id=1, bbox=(0.2, 0.4, 0.4, 0.6), text="保存")
    assert e.center() == (0.30000000000000004, 0.5) or e.center() == (0.3, 0.5)


# --- 动作校验 ---------------------------------------------------------------


def test_十个动作类型都能构造():
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


def test_缺少必需字段时报错():
    with pytest.raises(ValueError, match="缺少必需字段 point"):
        Action("click")
    with pytest.raises(ValueError, match="缺少必需字段 point2"):
        Action("drag", point=(0.1, 0.1))
    with pytest.raises(ValueError, match="缺少必需字段 text"):
        Action("type")


def test_未知动作类型报错():
    with pytest.raises(ValueError, match="未知动作类型"):
        Action("swipe", point=(0.5, 0.5))


def test_滚动方向非法时报错():
    with pytest.raises(ValueError, match="scroll 方向"):
        Action("scroll", point=(0.5, 0.5), direction="diagonal")


def test_动作坐标必须归一化():
    with pytest.raises(ValueError, match="归一化"):
        Action("click", point=(1920, 1080))


def test_终止动作判定():
    assert Action("finished").is_terminal()
    assert Action("call_user").is_terminal()
    assert not Action("click", point=(0.5, 0.5)).is_terminal()


# --- 序列化 -----------------------------------------------------------------


def test_动作字典往返():
    a = Action("drag", point=(0.1, 0.2), point2=(0.8, 0.9), thought="把文件拖到回收站")
    d = a.to_dict()
    b = Action.from_dict(d)
    assert (b.type, b.point, b.point2, b.thought) == (a.type, a.point, a.point2, a.thought)


def test_to_dict_不输出空字段():
    d = Action("wait").to_dict()
    assert d == {"type": "wait"}


def test_from_dict_接受列表形式的坐标():
    """JSON 反序列化出来的坐标是 list，不是 tuple。"""
    a = Action.from_dict({"type": "click", "point": [0.5, 0.5]})
    assert a.point == (0.5, 0.5)


def test_from_dict_忽略多余字段():
    a = Action.from_dict({"type": "wait", "unknown_field": 123})
    assert a.type == "wait"


# --- 轨迹 -------------------------------------------------------------------


def test_轨迹统计与json输出():
    screen = ScreenState(width=3840, height=2160, image_path="logs/1.png")
    t = Trajectory(task_id="open_browser", instruction="打开浏览器")
    t.steps.append(Step(screen, Action("click", point=(0.1, 0.9)), elapsed=1.5))
    t.steps.append(Step(screen, Action("finished"), elapsed=0.2))

    assert t.n_steps == 2
    assert abs(t.wall_time - 1.7) < 1e-9

    data = json.loads(t.to_json())
    assert data["task_id"] == "open_browser"
    assert data["n_steps"] == 2
    assert data["steps"][0]["action"]["type"] == "click"
    assert data["steps"][0]["screen"]["width"] == 3840
    assert data["success"] is None


def test_失败步骤记录错误信息():
    screen = ScreenState(width=1920, height=1080)
    step = Step(screen, Action("click", point=(0.5, 0.5)), ok=False, error="目标窗口不存在")
    t = Trajectory(task_id="t1", instruction="x", steps=[step], success=False)
    data = json.loads(t.to_json())
    assert data["steps"][0]["ok"] is False
    assert data["steps"][0]["error"] == "目标窗口不存在"
    assert data["success"] is False
