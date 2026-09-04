import numpy as np
import pytest

from gui_agent.agent import (
    Agent,
    _extract_json,
    build_prompt,
    format_elements,
    format_history,
    parse_step,
)
from gui_agent.control import Controller, RecordingBackend
from gui_agent.schema import Action, Element, ScreenState, Step


@pytest.fixture
def screen():
    return ScreenState(
        width=1920, height=1080,
        elements=[
            Element(id=0, bbox=(0.0, 0.0, 0.1, 0.05), text="文件"),
            Element(id=1, bbox=(0.2, 0.4, 0.4, 0.5), text="保存"),
            Element(id=2, bbox=(0.5, 0.5, 0.6, 0.6), text=""),  # 无文字，不进提示词
        ],
    )


class FakeVLM:
    """按脚本依次返回预设回复，不加载模型。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def ask(self, image, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else '{"action": {"type": "finished"}}'


class FakePerception:
    def __init__(self, state):
        self.state = state

    def perceive(self, **kw):
        return self.state, np.zeros((10, 10, 3), dtype=np.uint8)


# --- JSON 提取 --------------------------------------------------------------


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_code_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    assert _extract_json('我觉得应该点保存。\n{"a": 1}\n希望有用。') == {"a": 1}


def test_extract_nested_json():
    assert _extract_json('{"action": {"type": "click"}}') == {"action": {"type": "click"}}


def test_extract_ignores_braces_inside_strings():
    """字符串里的花括号不能被当成结构。"""
    assert _extract_json('{"thought": "点 {设置} 菜单", "n": 1}')["n"] == 1


def test_extract_skips_malformed_and_finds_next():
    assert _extract_json('{坏的} 然后 {"a": 2}') == {"a": 2}


def test_extract_returns_none_when_absent():
    assert _extract_json("完全没有 JSON") is None


# --- 解析成动作 -------------------------------------------------------------


def test_element_id_resolved_to_coordinates(screen):
    thought, action = parse_step('{"thought": "点保存", "action": {"type": "click", "element": 1}}', screen)
    assert thought == "点保存"
    assert action.type == "click"
    assert action.point == pytest.approx((0.3, 0.45))


def test_normalized_point_passes_through(screen):
    _, a = parse_step('{"action": {"type": "click", "point": [0.25, 0.75]}}', screen)
    assert a.point == (0.25, 0.75)


def test_unknown_element_id_raises(screen):
    with pytest.raises(ValueError, match="不在当前屏幕"):
        parse_step('{"action": {"type": "click", "element": 99}}', screen)


def test_unparseable_output_becomes_call_user(screen):
    thought, a = parse_step("我不知道该做什么", screen)
    assert a.type == "call_user" and "无法解析" in thought


def test_action_without_type_becomes_call_user(screen):
    _, a = parse_step('{"thought": "x"}', screen)
    assert a.type == "call_user"


def test_type_and_hotkey_actions(screen):
    _, a = parse_step('{"action": {"type": "type", "text": "打开浏览器"}}', screen)
    assert a.type == "type" and a.text == "打开浏览器"
    _, b = parse_step('{"action": {"type": "hotkey", "text": "ctrl+s"}}', screen)
    assert b.text == "ctrl+s"


def test_drag_with_two_points(screen):
    _, a = parse_step('{"action": {"type": "drag", "point": [0.1, 0.1], "point2": [0.9, 0.9]}}', screen)
    assert a.point == (0.1, 0.1) and a.point2 == (0.9, 0.9)


def test_scroll_with_element_and_direction(screen):
    _, a = parse_step('{"action": {"type": "scroll", "element": 0, "direction": "down"}}', screen)
    assert a.direction == "down" and a.point == pytest.approx((0.05, 0.025))


# --- 提示词 -----------------------------------------------------------------


def test_elements_listed_with_ids(screen):
    s = format_elements(screen)
    assert "[0] 文件" in s and "[1] 保存" in s


def test_elements_without_text_are_skipped(screen):
    assert "[2]" not in format_elements(screen)


def test_element_list_is_capped(screen):
    many = ScreenState(1920, 1080, elements=[
        Element(id=i, bbox=(0, 0, 0.1, 0.1), text=f"项{i}") for i in range(200)
    ])
    assert len(format_elements(many, limit=10).splitlines()) == 10


def test_empty_screen_says_so():
    assert "没有识别到" in format_elements(ScreenState(800, 600))


def test_history_shows_recent_steps(screen):
    steps = [Step(screen, Action("click", point=(0.1, 0.1))),
             Step(screen, Action("type", text="x"), ok=False, error="窗口没焦点")]
    h = format_history(steps)
    assert "第1步 click → 成功" in h and "窗口没焦点" in h


def test_history_empty_at_start():
    assert "第一步" in format_history([])


def test_prompt_contains_task_and_elements(screen):
    p = build_prompt("打开浏览器", screen, [])
    assert "打开浏览器" in p and "[1] 保存" in p and "finished" in p


# --- 完整循环 ---------------------------------------------------------------


def test_loop_stops_on_finished(screen):
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(1920, 1080)), vlm)
    t = a.run("保存文件")
    assert t.n_steps == 2 and t.success is True


def test_loop_stops_on_failed_action(screen):
    vlm = FakeVLM(['{"action": {"type": "hotkey", "text": "win+l"}}',  # 被安全拦截
                   '{"action": {"type": "finished"}}'])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm)
    t = a.run("锁屏")
    assert t.n_steps == 1 and t.success is False


def test_loop_stops_on_bad_element_id(screen):
    vlm = FakeVLM(['{"action": {"type": "click", "element": 99}}'])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm)
    t = a.run("点不存在的东西")
    assert t.n_steps == 1 and not t.steps[0].ok


def test_loop_respects_max_steps(screen):
    """动作各不相同，不会触发卡住检测，走到步数上限为止。"""
    vlm = FakeVLM([f'{{"action": {{"type": "click", "point": [0.{i}, 0.5]}}}}' for i in range(9)])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    t = Agent(FakePerception(screen), ctrl, vlm, max_steps=3).run("永远做不完")
    assert t.n_steps == 3 and t.success is False


def test_history_accumulates_across_steps(screen):
    vlm = FakeVLM(['{"action": {"type": "wait"}}', '{"action": {"type": "wait"}}',
                   '{"action": {"type": "finished"}}'])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    Agent(FakePerception(screen), ctrl, vlm).run("x")
    assert "（这是第一步）" in vlm.prompts[0]
    assert "第1步 wait" in vlm.prompts[1]
    assert "第2步 wait" in vlm.prompts[2]


def test_thought_is_recorded(screen):
    vlm = FakeVLM(['{"thought": "先点保存按钮", "action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.steps[0].action.thought == "先点保存按钮"


# --- 命令行的动作描述 -------------------------------------------------------


def test_describe_covers_all_action_types():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_agent import describe

    assert "0.500" in describe(Action("click", point=(0.5, 0.5)))
    assert "drag" in describe(Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9)))
    assert "down" in describe(Action("scroll", point=(0.5, 0.5), direction="down"))
    assert "'hello'" in describe(Action("type", text="hello"))
    assert "'ctrl+s'" in describe(Action("hotkey", text="ctrl+s"))
    assert describe(Action("finished")) == "finished"
    assert describe(Action("wait")) == "wait"


# --- 卡住检测 ---------------------------------------------------------------


def test_is_stuck_detects_repeated_action(screen):
    from gui_agent.agent import is_stuck

    same = [Step(screen, Action("click", point=(0.5, 0.5))) for _ in range(3)]
    assert is_stuck(same)


def test_is_stuck_ignores_different_actions(screen):
    from gui_agent.agent import is_stuck

    mixed = [Step(screen, Action("click", point=(0.5, 0.5))),
             Step(screen, Action("click", point=(0.6, 0.5))),
             Step(screen, Action("click", point=(0.5, 0.5)))]
    assert not is_stuck(mixed)


def test_is_stuck_needs_enough_steps(screen):
    from gui_agent.agent import is_stuck

    assert not is_stuck([Step(screen, Action("wait"))] * 2)


def test_is_stuck_distinguishes_by_text(screen):
    from gui_agent.agent import is_stuck

    steps = [Step(screen, Action("type", text=t)) for t in ("a", "a", "b")]
    assert not is_stuck(steps)


def test_loop_aborts_when_stuck(screen):
    """界面不变时模型会一直给同一个动作，不能任由它跑到步数上限。"""
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}'] * 10)
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(1920, 1080)), vlm,
              max_steps=10)
    t = a.run("点不动的东西")
    assert t.n_steps == 4  # 三次重复 + 一条终止记录
    assert t.steps[-1].action.type == "call_user"
    assert "重复" in t.steps[-1].error
    assert t.success is False
