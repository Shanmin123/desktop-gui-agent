"""ScreenAgent 原始动作 -> 本项目 Action 的转换。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_screenagent import SCROLL_POINT, to_action

W, H = 1024, 768


def conv(raw):
    return to_action(raw, W, H)


# --- 鼠标 -------------------------------------------------------------------


def test_left_click_maps_to_click_with_normalized_point():
    a, why = conv({"action_type": "MouseAction", "mouse_action_type": "click",
                   "mouse_button": "left", "mouse_position": {"width": 512, "height": 384}})
    assert why is None
    assert a.type == "click" and a.point == pytest.approx((0.5, 0.5))


def test_right_button_maps_to_right_single():
    a, _ = conv({"action_type": "MouseAction", "mouse_action_type": "click",
                 "mouse_button": "right", "mouse_position": {"width": 0, "height": 0}})
    assert a.type == "right_single" and a.point == (0.0, 0.0)


def test_double_click_maps_to_left_double():
    a, _ = conv({"action_type": "MouseAction", "mouse_action_type": "double_click",
                 "mouse_button": "left", "mouse_position": {"width": 256, "height": 192}})
    assert a.type == "left_double" and a.point == pytest.approx((0.25, 0.25))


@pytest.mark.parametrize("mt,direction", [("scroll_up", "up"), ("scroll_down", "down")])
def test_scroll_without_coordinates_falls_back_to_center(mt, direction):
    """原始记录里滚动没有坐标，用屏幕中心补上。"""
    a, why = conv({"action_type": "MouseAction", "mouse_action_type": mt})
    assert why is None
    assert a.type == "scroll" and a.direction == direction and a.point == SCROLL_POINT


@pytest.mark.parametrize("mt", ["move", "down", "up", "drag"])
def test_unmappable_mouse_primitives_are_skipped(mt):
    """drag 只记了落点没有起点，move/down/up 是拆开的原语，都凑不出 schema 的动作。"""
    a, why = conv({"action_type": "MouseAction", "mouse_action_type": mt,
                   "mouse_position": {"width": 1, "height": 1}})
    assert a is None and why.startswith("mouse:")


def test_click_without_position_is_skipped():
    a, why = conv({"action_type": "MouseAction", "mouse_action_type": "click"})
    assert a is None and "缺坐标" in why


def test_out_of_range_position_is_skipped():
    a, why = conv({"action_type": "MouseAction", "mouse_action_type": "click",
                   "mouse_position": {"width": 5000, "height": 384}})
    assert a is None and "越界" in why


# --- 键盘 -------------------------------------------------------------------


def test_text_input_maps_to_type():
    a, why = conv({"action_type": "KeyboardAction", "keyboard_action_type": "text",
                   "keyboard_text": "冯诺依曼"})
    assert why is None and a.type == "type" and a.text == "冯诺依曼"


def test_empty_text_is_kept():
    """清空输入框是合法动作，空串不能当没填丢掉。"""
    a, why = conv({"action_type": "KeyboardAction", "keyboard_action_type": "text",
                   "keyboard_text": ""})
    assert why is None and a.type == "type" and a.text == ""


def test_single_key_maps_to_hotkey():
    a, _ = conv({"action_type": "KeyboardAction", "keyboard_action_type": "press",
                 "keyboard_key": "Return"})
    assert a.type == "hotkey" and a.text == "Return"


def test_key_combo_list_is_joined():
    a, _ = conv({"action_type": "KeyboardAction", "keyboard_action_type": "press",
                 "keyboard_key": ["Control_L", "s"]})
    assert a.type == "hotkey" and a.text == "Control_L+s"


@pytest.mark.parametrize("key,expect", [
    ("Return", ["enter"]),
    ("Escape", ["esc"]),
    ("BackSpace", ["backspace"]),
    (["Control_L", "s"], ["ctrl", "s"]),
    (["Control_L", "Shift_L", "n"], ["ctrl", "shift", "n"]),
    (["Super_L", "r"], ["win", "r"]),
    (["Alt_L", "F4"], ["alt", "f4"]),
])
def test_converted_hotkeys_are_executable(key, expect):
    """X11 keysym 要能落到 pyautogui 的键名上，否则转出来的数据执行不了。"""
    from gui_agent.control import normalize_hotkey

    a, why = conv({"action_type": "KeyboardAction", "keyboard_action_type": "press",
                   "keyboard_key": key})
    assert why is None
    assert normalize_hotkey(a.text) == expect


def test_unparseable_key_is_skipped():
    a, why = conv({"action_type": "KeyboardAction", "keyboard_action_type": "press",
                   "keyboard_key": []})
    assert a is None and "缺键名" in why


# --- 其他 -------------------------------------------------------------------


def test_wait_action():
    a, why = conv({"action_type": "WaitAction", "wait_time": 1.0})
    assert why is None and a.type == "wait"


@pytest.mark.parametrize("kind", ["PlanAction", "EvaluateSubTaskAction"])
def test_language_stages_are_not_actions(kind):
    """计划和反思是自然语言，不对应 schema 里的 10 个动作，单独统计不落盘。"""
    a, why = conv({"action_type": kind, "element": "打开开始菜单"})
    assert a is None and why == f"其他:{kind}"


def test_lock_screen_combo_from_dataset_is_still_blocked():
    """数据集里若出现 Super_L+l，转换后仍要被控制模块拦住。"""
    from gui_agent.control import Controller, RecordingBackend

    a, _ = conv({"action_type": "KeyboardAction", "keyboard_action_type": "press",
                 "keyboard_key": ["Super_L", "l"]})
    c = Controller(backend=RecordingBackend())
    assert not c.execute(a).ok and c.backend.calls == []
