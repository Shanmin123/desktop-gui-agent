"""LangChain 组装的提示词模板、输出解析与链。"""

import numpy as np
import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from gui_agent.chain import (
    GUI_PROMPT,
    TEMPLATE,
    ActionOutputParser,
    build_chain,
    render_prompt,
)
from gui_agent.schema import Element, ScreenState, Step
from gui_agent.schema import Action


@pytest.fixture
def screen():
    return ScreenState(
        width=1920, height=1080,
        elements=[
            Element(id=0, bbox=(0.0, 0.0, 0.1, 0.05), text="文件"),
            Element(id=1, bbox=(0.2, 0.4, 0.4, 0.5), text="保存"),
        ],
    )


class FakeVLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def ask(self, image, prompt, **kw):
        self.prompts.append(prompt)
        return self.reply


# --- 提示词模板 -------------------------------------------------------------


def test_prompt_is_a_langchain_template():
    assert isinstance(GUI_PROMPT, PromptTemplate)
    assert set(GUI_PROMPT.input_variables) == {"instruction", "history", "elements"}


def test_json_example_braces_survive_templating():
    """模板里 JSON 的花括号要写成双份，否则会被当成变量占位符。"""
    p = GUI_PROMPT.format(instruction="x", history="y", elements="z")
    assert '{"thought": "为什么这么做", "action": {"type": "click", "element": 12}}' in p
    assert "{{" not in p


def test_render_prompt_fills_task_and_elements(screen):
    p = render_prompt("打开浏览器", screen, [])
    assert "打开浏览器" in p
    assert "[1] 保存" in p
    assert "（这是第一步）" in p


def test_render_prompt_includes_history(screen):
    steps = [Step(screen, Action("click", point=(0.1, 0.1)))]
    assert "第1步 click" in render_prompt("x", screen, steps)


def test_agent_build_prompt_uses_the_same_template(screen):
    """agent.build_prompt 是这个模板的薄封装，不能各写一份提示词。"""
    from gui_agent.agent import build_prompt

    assert build_prompt("打开浏览器", screen, []) == render_prompt("打开浏览器", screen, [])


def test_system_prompt_is_derived_from_template():
    from gui_agent.agent import SYSTEM_PROMPT

    assert SYSTEM_PROMPT in TEMPLATE.replace("{{", "{").replace("}}", "}")
    assert "call_user" in SYSTEM_PROMPT and "{{" not in SYSTEM_PROMPT


# --- 输出解析器 -------------------------------------------------------------


def test_parser_resolves_element_id(screen):
    thought, action = ActionOutputParser(state=screen).parse(
        '{"thought": "点保存", "action": {"type": "click", "element": 1}}'
    )
    assert thought == "点保存"
    assert action.point == pytest.approx((0.3, 0.45))


def test_parser_converts_pixel_coords_with_model_size(screen):
    _, a = ActionOutputParser(state=screen, model_size=(1430, 804)).parse(
        '{"action": {"type": "click", "point": [715, 402]}}'
    )
    assert a.point == pytest.approx((0.5, 0.5), abs=0.01)


def test_parser_raises_on_garbage(screen):
    with pytest.raises(ValueError):
        ActionOutputParser(state=screen).parse("今天天气不错")


# --- 链 ---------------------------------------------------------------------


def test_chain_is_runnable_and_returns_thought_and_action(screen):
    vlm = FakeVLM('{"thought": "先点保存", "action": {"type": "click", "element": 1}}')
    chain = build_chain(vlm)
    assert isinstance(chain, Runnable)

    thought, action = chain.invoke({
        "instruction": "保存文件", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert thought == "先点保存" and action.type == "click"


def test_chain_feeds_the_rendered_prompt_to_the_model(screen):
    vlm = FakeVLM('{"action": {"type": "finished"}}')
    build_chain(vlm).invoke({
        "instruction": "关掉记事本", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert vlm.prompts == [render_prompt("关掉记事本", screen, [])]


def test_chain_applies_model_size_for_pixel_coords(screen):
    vlm = FakeVLM('{"action": {"type": "click", "point": [715, 402]}}')
    chain = build_chain(vlm, model_size_of=lambda img: (1430, 804))
    _, a = chain.invoke({
        "instruction": "x", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert a.point == pytest.approx((0.5, 0.5), abs=0.01)


def test_agent_rebuilds_chain_when_model_is_swapped(screen):
    """self.vlm 和链里的模型不能指向两个对象。"""
    from gui_agent.agent import Agent
    from gui_agent.control import Controller, RecordingBackend

    class FakePerception:
        def perceive(self, **kw):
            return screen, np.zeros((10, 10, 3), dtype=np.uint8)

    a = Agent(FakePerception(), Controller(backend=RecordingBackend()),
              FakeVLM('{"action": {"type": "finished"}}'))
    first = a.chain
    assert a.chain is first  # 同一个模型时复用

    a.vlm = FakeVLM('{"action": {"type": "wait"}}')
    assert a.chain is not first


# --- 两段式定位 -------------------------------------------------------------


class LocatingVLM(FakeVLM):
    """能定位的假模型：locate 返回预设点，记录被问过哪些描述。"""

    def __init__(self, reply, point=(0.4, 0.6)):
        super().__init__(reply)
        self.point = point
        self.located = []

    def locate(self, image, instruction):
        self.located.append(instruction)
        return self.point


def test_target_prompt_asks_for_a_control_not_coordinates():
    from gui_agent.chain import TARGET_TEMPLATE

    assert "target" in TARGET_TEMPLATE
    assert "不要写坐标" in TARGET_TEMPLATE and "不要写编号" in TARGET_TEMPLATE
    assert "{instruction}" in TARGET_TEMPLATE and "{history}" in TARGET_TEMPLATE


def test_target_prompt_has_no_element_list(screen):
    """两段式不给 OCR 元素清单，位置交给定位那一步。"""
    from gui_agent.chain import render_target_prompt

    p = render_target_prompt("打开浏览器", [])
    assert "打开浏览器" in p and "[1] 保存" not in p


def test_target_is_resolved_into_a_point(screen):
    vlm = LocatingVLM('{"thought": "关掉它", "action": {"type": "click", "target": "关闭按钮"}}',
                      point=(0.4, 0.6))
    thought, action = build_chain(vlm, locate_target=True).invoke({
        "instruction": "关闭窗口", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert thought == "关掉它"
    assert action.type == "click" and action.point == (0.4, 0.6)
    assert vlm.located == ["关闭按钮"]


def test_scroll_keeps_direction_through_two_stage(screen):
    vlm = LocatingVLM('{"action": {"type": "scroll", "target": "列表", "direction": "down"}}')
    _, a = build_chain(vlm, locate_target=True).invoke({
        "instruction": "往下翻", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert a.type == "scroll" and a.direction == "down" and a.point is not None


def test_actions_without_target_pass_through(screen):
    """type / hotkey / finished 不需要定位。"""
    for raw, kind in [('{"action": {"type": "type", "text": "你好"}}', "type"),
                      ('{"action": {"type": "hotkey", "text": "ctrl+s"}}', "hotkey"),
                      ('{"action": {"type": "finished"}}', "finished")]:
        vlm = LocatingVLM(raw)
        _, a = build_chain(vlm, locate_target=True).invoke({
            "instruction": "x", "state": screen, "steps": [],
            "image": np.zeros((10, 10, 3), dtype=np.uint8),
        })
        assert a.type == kind
        assert vlm.located == [], "不需要定位的动作不该调用 locate"


def test_falls_back_to_one_stage_when_model_gives_element(screen):
    """模型没按两段式回、直接给了 element 编号时，仍按一段式解析。"""
    vlm = LocatingVLM('{"action": {"type": "click", "element": 1}}')
    _, a = build_chain(vlm, locate_target=True).invoke({
        "instruction": "x", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert a.point == pytest.approx((0.3, 0.45))
    assert vlm.located == []


def test_unlocatable_target_raises(screen):
    class Blind(LocatingVLM):
        def locate(self, image, instruction):
            return None

    vlm = Blind('{"action": {"type": "click", "target": "并不存在的按钮"}}')
    with pytest.raises(ValueError, match="定位不到"):
        build_chain(vlm, locate_target=True).invoke({
            "instruction": "x", "state": screen, "steps": [],
            "image": np.zeros((10, 10, 3), dtype=np.uint8),
        })


def test_one_stage_is_unchanged_when_flag_is_off(screen):
    """默认仍是一段式，v1.0 基线描述的就是它。"""
    vlm = LocatingVLM('{"action": {"type": "click", "element": 1}}')
    _, a = build_chain(vlm).invoke({
        "instruction": "x", "state": screen, "steps": [],
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    })
    assert a.point == pytest.approx((0.3, 0.45)) and vlm.located == []


def test_agent_rebuilds_chain_when_locate_target_flips(screen):
    from gui_agent.agent import Agent
    from gui_agent.control import Controller, RecordingBackend

    class P:
        def perceive(self, **kw):
            return screen, np.zeros((10, 10, 3), dtype=np.uint8)

    a = Agent(P(), Controller(backend=RecordingBackend()), LocatingVLM("{}"),
              locate_target=True)
    assert a.locate_target is True
