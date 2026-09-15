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


# --- 提示词变体（大纲第 5 周第 4 项）----------------------------------------


def _state():
    return ScreenState(width=1280, height=720,
                       elements=[Element(id=0, bbox=(0.1, 0.1, 0.2, 0.2), text="文件")])


def test_every_variant_renders_without_leftover_placeholders():
    from gui_agent.chain import PROMPT_VARIANTS, render_prompt

    for name in PROMPT_VARIANTS:
        p = render_prompt("打开浏览器", _state(), [], variant=name)
        assert "{instruction}" not in p and "{elements}" not in p and "{history}" not in p
        assert "打开浏览器" in p and "[0] 文件" in p


def test_variants_actually_differ_from_base():
    from gui_agent.chain import PROMPT_VARIANTS

    others = {k: v for k, v in PROMPT_VARIANTS.items() if k != "base"}
    assert others, "至少要有一个对照变体"
    for name, tpl in others.items():
        assert tpl != PROMPT_VARIANTS["base"], f"{name} 和 base 一模一样，比不出东西"


def test_point_first_drops_the_element_first_wording():
    """这个变体的假设就是「别优先用编号」，措辞必须换掉。"""
    from gui_agent.chain import PROMPT_VARIANTS

    assert "优先用 element 编号" not in PROMPT_VARIANTS["point_first"]
    assert "必须给 point" in PROMPT_VARIANTS["point_first"]


def test_few_shot_examples_keep_valid_json():
    """例子里的 JSON 渲染后要还能解析，不然是在教模型输出错格式。"""
    import json
    import re

    from gui_agent.chain import render_prompt

    p = render_prompt("x", _state(), [], variant="few_shot")
    found = re.findall(r'\{"thought".*?\}\}', p)
    assert len(found) >= 2
    for s in found:
        assert json.loads(s)["action"]["type"] in ("hotkey", "click")


def test_unknown_variant_raises():
    import pytest

    from gui_agent.chain import render_prompt

    with pytest.raises(ValueError):
        render_prompt("x", _state(), [], variant="没有这个")


def test_default_variant_matches_the_plain_template():
    from gui_agent.chain import render_prompt

    assert render_prompt("x", _state(), []) == render_prompt("x", _state(), [], variant="base")


# --- 两段式第一问带元素清单 -------------------------------------------------


def test_target_prompt_can_carry_the_element_list(screen):
    """失败样例里错的主要是控件名本身不对，让模型照抄清单里的原文。"""
    from gui_agent.chain import render_target_prompt

    p = render_target_prompt("打开浏览器", [], screen)
    assert "照抄" in p and "[0]" in p
    assert "不要写坐标" in p and "不要写编号" in p


def test_target_prompt_without_state_is_the_old_one(screen):
    """不给 state 就还是原来那份，前面几轮的结果才比得下去。"""
    from gui_agent.chain import TARGET_TEMPLATE, render_target_prompt

    p = render_target_prompt("打开浏览器", [])
    assert "当前屏幕上的文字元素" not in p and "照抄" not in p
    # 模板里的花括号是双写的（LangChain 占位符转义），比对前还原成渲染后的样子
    head = TARGET_TEMPLATE.split("任务：")[0].replace("{{", "{").replace("}}", "}")
    assert p.startswith(head)


def test_element_limit_is_honoured_in_the_target_prompt():
    """清单要能裁短：提示词太长会连图片一起顶破训练的长度预算。"""
    from gui_agent.chain import render_target_prompt
    from gui_agent.schema import Element, ScreenState

    st = ScreenState(1024, 768, elements=[
        Element(id=i, bbox=(0.1, 0.01 * i, 0.2, 0.01 * i + 0.01), text=f"项目{i}")
        for i in range(50)])
    p = render_target_prompt("任务", [], st, 10)
    assert "[9]" in p and "[10]" not in p


def test_hotkey_content_written_into_target_is_used_as_text(screen):
    """键盘动作不需要位置：模型把按键写进 target 时，当 text 用，不去定位、不挂坐标。

    实测 lora_2sp 离线缺 text 的 6 条全是把内容写进了 target（Control_L+a、Tab）。
    """
    from gui_agent.chain import parse_with_target

    vlm = LocatingVLM("", point=(0.5, 0.5))
    _, act = parse_with_target(
        '{"thought": "全选", "action": {"type": "hotkey", "target": "Control_L+a"}}',
        vlm, None, screen)
    assert act.type == "hotkey" and act.text == "Control_L+a" and act.point is None
    assert vlm.located == []          # 根本不该去定位


def test_typed_content_written_into_target_is_used_as_text(screen):
    from gui_agent.chain import parse_with_target

    vlm = LocatingVLM("", point=(0.5, 0.5))
    _, act = parse_with_target('{"action": {"type": "type", "target": "迈腾"}}', vlm, None, screen)
    assert act.type == "type" and act.text == "迈腾" and act.point is None
    assert vlm.located == []


def test_keyboard_action_with_neither_text_nor_target_still_fails(screen):
    """两样都没有就没有内容可用，照样报缺 text，不能凭空补。"""
    from gui_agent.chain import parse_with_target

    vlm = LocatingVLM("", point=(0.5, 0.5))
    with pytest.raises(ValueError, match="text"):
        parse_with_target('{"action": {"type": "hotkey"}}', vlm, None, screen)


def test_keyboard_action_with_text_still_parses(screen):
    from gui_agent.chain import parse_with_target

    vlm = LocatingVLM("", point=(0.5, 0.5))
    _, act = parse_with_target(
        '{"thought": "存盘", "action": {"type": "hotkey", "text": "ctrl+s", "target": "保存"}}',
        vlm, None, screen)
    assert act.type == "hotkey" and act.text == "ctrl+s" and act.point is None
    assert vlm.located == []


# --- 两段式提示词变体（大纲第 5 周第 4 项）------------------------------------


def test_target_variants_keep_placeholders_and_differ_from_base():
    from gui_agent.chain import TARGET_VARIANTS

    assert {"base", "keyboard", "click_prior", "few_shot"} <= set(TARGET_VARIANTS)
    for name, tpl in TARGET_VARIANTS.items():
        assert "{instruction}" in tpl and "{history}" in tpl, name
        if name != "base":
            assert tpl != TARGET_VARIANTS["base"], name


def test_target_variants_render_and_base_stays_the_old_prompt():
    from gui_agent.chain import TARGET_VARIANTS, render_target_prompt

    base = render_target_prompt("打开浏览器", [])
    assert render_target_prompt("打开浏览器", [], variant="base") == base
    for name in TARGET_VARIANTS:
        p = render_target_prompt("打开浏览器", [], variant=name)
        assert "打开浏览器" in p and "{instruction}" not in p, name
    assert "ctrl+s" in render_target_prompt("打开浏览器", [], variant="keyboard")
    assert "left_double" in render_target_prompt("打开浏览器", [], variant="click_prior")


def test_unknown_target_variant_is_rejected():
    import pytest as _pytest

    from gui_agent.chain import render_target_prompt

    with _pytest.raises(ValueError):
        render_target_prompt("x", [], variant="nope")
