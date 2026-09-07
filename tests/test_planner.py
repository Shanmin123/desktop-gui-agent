"""任务拆解与子任务反思。"""

import numpy as np
import pytest

from gui_agent.agent import Agent
from gui_agent.control import Controller, RecordingBackend
from gui_agent.planner import (
    REFORMULATE,
    RETRY,
    SITUATIONS,
    SUCCESS,
    Planner,
    acting_instruction,
    parse_plan,
    parse_reflection,
)
from gui_agent.schema import Element, ScreenState


@pytest.fixture
def screen():
    return ScreenState(
        width=1024, height=768,
        elements=[Element(id=0, bbox=(0.0, 0.0, 0.1, 0.05), text="开始"),
                  Element(id=1, bbox=(0.2, 0.4, 0.4, 0.5), text="文件资源管理器")],
    )


class ScriptedVLM:
    """按脚本依次返回预设回复。"""

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


# --- 拆解结果的解析 ---------------------------------------------------------


def test_parse_plain_array():
    assert parse_plan('["打开开始菜单", "点击文件资源管理器图标"]') == [
        "打开开始菜单", "点击文件资源管理器图标"]


def test_parse_array_from_code_fence():
    assert parse_plan('```json\n["A", "B"]\n```') == ["A", "B"]


def test_parse_array_with_surrounding_prose():
    assert parse_plan('好的，计划如下：\n["A", "B"]\n希望有用。') == ["A", "B"]


def test_parse_array_wrapped_in_object():
    """有的模型会包一层，如 {"plan": [...]}。"""
    assert parse_plan('{"plan": ["A", "B"]}') == ["A", "B"]


def test_parse_drops_empty_entries():
    assert parse_plan('["A", "", "   ", "B"]') == ["A", "B"]


def test_parse_respects_limit():
    assert parse_plan('["1","2","3","4","5"]', limit=3) == ["1", "2", "3"]


def test_parse_returns_empty_when_absent():
    """拆不出来要返回空，让上层退回单步循环，而不是抛异常。"""
    assert parse_plan("我不知道怎么拆") == []
    assert parse_plan("") == []


def test_parse_object_array_as_returned_by_qwen():
    """实测 Qwen2.5-VL 的原样输出：不按提示词要求回字符串数组，回的是对象数组。"""
    raw = '''```json
[
    {"action": "打开开始菜单"},
    {"action": "点击搜索栏"}
]
```'''
    assert parse_plan(raw) == ["打开开始菜单", "点击搜索栏"]


def test_parse_screenagent_style_objects():
    """ScreenAgent 数据集里子任务的键叫 element。"""
    raw = ('[{"action_type": "PlanAction", "element": "打开开始菜单"}, '
           '{"action_type": "PlanAction", "element": "点击文件资源管理器图标"}]')
    assert parse_plan(raw) == ["打开开始菜单", "点击文件资源管理器图标"]


@pytest.mark.parametrize("key", ["subtask", "step", "description", "task", "content"])
def test_parse_accepts_other_common_keys(key):
    assert parse_plan('[{"%s": "打开浏览器"}]' % key) == ["打开浏览器"]


def test_parse_single_string_value_regardless_of_key():
    """键名没见过时，对象里只有一个字符串值就取它。"""
    assert parse_plan('[{"随便什么键": "打开浏览器"}]') == ["打开浏览器"]


def test_parse_mixes_strings_and_objects():
    assert parse_plan('["A", {"action": "B"}]') == ["A", "B"]


def test_parse_ignores_objects_without_usable_text():
    assert parse_plan('[{"n": 1, "ok": true}]') == []


def test_parse_skips_malformed_array_and_finds_next():
    assert parse_plan('[坏的] 然后 ["A"]') == ["A"]


# --- 反思结果的解析 ---------------------------------------------------------


@pytest.mark.parametrize("situation", SITUATIONS)
def test_parse_all_three_situations(situation):
    s, _ = parse_reflection('{"situation": "%s"}' % situation)
    assert s == situation


def test_parse_reflection_keeps_advice():
    s, advice = parse_reflection(
        '{"situation": "need_retry", "advice": "再点一次那个图标"}')
    assert s == RETRY and advice == "再点一次那个图标"


def test_bare_situation_word_is_accepted():
    assert parse_reflection("我判断是 sub_task_success")[0] == SUCCESS


def test_unknown_situation_falls_back_to_retry():
    """判定不出来时当作要重试，不能当成完成后往下推进。"""
    assert parse_reflection('{"situation": "不知道"}')[0] == RETRY
    assert parse_reflection("胡说八道")[0] == RETRY


# --- 执行阶段的任务描述 -----------------------------------------------------


def test_acting_instruction_without_subtask():
    assert acting_instruction("打开浏览器", None) == "打开浏览器"


def test_acting_instruction_keeps_overall_goal():
    """只给子任务会让模型忘了整体目标，两个都要带上。"""
    s = acting_instruction("上网查资料", "在搜索框输入关键词")
    assert "上网查资料" in s and "在搜索框输入关键词" in s


# --- Planner 推进 -----------------------------------------------------------


def test_plan_then_walk_through_subtasks(screen):
    p = Planner(ScriptedVLM(['["A", "B"]']))
    assert p.plan(None, "任务", screen) == ["A", "B"]
    assert p.current() == "A" and not p.done()
    p.advance()
    assert p.current() == "B" and not p.done()
    p.advance()
    assert p.current() is None and p.done()


def test_done_is_false_when_never_planned():
    """没拆出子任务时不能算完成，否则任务一开始就被判成功。"""
    p = Planner(ScriptedVLM(["拆不出来"]))
    p.plan(None, "任务", ScreenState(800, 600))
    assert p.subtasks == [] and not p.done() and p.current() is None


def test_reflect_returns_success(screen):
    p = Planner(ScriptedVLM(['["A"]', '{"situation": "sub_task_success"}']))
    p.plan(None, "任务", screen)
    assert p.reflect(None, "任务", [])[0] == SUCCESS


def test_reflect_without_current_subtask_is_success(screen):
    assert Planner(ScriptedVLM([])).reflect(None, "任务", [])[0] == SUCCESS


# --- 接进 Agent 循环 --------------------------------------------------------


def test_planning_is_off_by_default(screen):
    """v1.0 基线是在单步循环上测的，默认配置要保持一致。"""
    vlm = ScriptedVLM(['{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.subtasks == [] and t.reflections == []
    assert len(vlm.prompts) == 1  # 只有执行那一次调用


def test_finishing_all_subtasks_does_not_declare_task_success(screen):
    """实测反思会连续给 sub_task_success 而程序验收判定失败。子任务走完只是
    退回按整体任务继续，不能据此判定整条任务成功——那是提前终止。"""
    vlm = ScriptedVLM([
        '["A"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "sub_task_success"}',
        '{"action": {"type": "wait"}}',      # 子任务走完后继续按整体任务跑
    ])
    ctrl = Controller(backend=RecordingBackend(1024, 768), dry_run=True)
    t = Agent(FakePerception(screen), ctrl, vlm, plan=True, max_steps=2).run("x")
    assert t.n_steps == 2, "不该在子任务走完时就退出"
    assert t.success is False   # 走到步数上限，不是模型自称完成


def test_after_the_plan_is_exhausted_the_prompt_falls_back_to_the_task(screen):
    vlm = ScriptedVLM([
        '["A"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "sub_task_success"}',
        '{"action": {"type": "finished"}}',
    ])
    Agent(FakePerception(screen), Controller(backend=RecordingBackend(1024, 768)),
          vlm, plan=True, max_steps=3).run("打开浏览器")
    assert "当前子任务：A" in vlm.prompts[1]
    assert "当前子任务" not in vlm.prompts[3]   # 子任务用完，只剩整体任务


def test_empty_replan_keeps_the_existing_plan(screen):
    """重拆返回空不能抹掉原计划，否则规划中途消失。"""
    vlm = ScriptedVLM([
        '["A", "B"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "need_reformulate"}',
        '拆不出来',                                   # 重拆失败
        '{"action": {"type": "finished"}}',
    ])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend(1024, 768)),
              vlm, plan=True, max_steps=3).run("x")
    assert t.subtasks == ["A", "B"]


def test_plan_is_recorded_on_the_trajectory(screen):
    vlm = ScriptedVLM([
        '["打开开始菜单", "点开文件资源管理器"]',      # Planning
        '{"action": {"type": "click", "element": 0}}',  # Acting
        '{"situation": "sub_task_success"}',            # Reflecting
        '{"action": {"type": "click", "element": 1}}',
        '{"situation": "sub_task_success"}',
    ])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(1024, 768)),
              vlm, plan=True)
    t = a.run("打开文件资源管理器")
    assert t.subtasks == ["打开开始菜单", "点开文件资源管理器"]
    assert t.reflections == [SUCCESS, SUCCESS]


def test_subtask_appears_in_the_acting_prompt(screen):
    vlm = ScriptedVLM(['["打开开始菜单"]', '{"action": {"type": "finished"}}'])
    Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
          plan=True).run("打开文件资源管理器")
    assert "打开开始菜单" in vlm.prompts[1]


def test_retry_keeps_the_same_subtask(screen):
    vlm = ScriptedVLM([
        '["A", "B"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "need_retry"}',
        '{"action": {"type": "click", "element": 1}}',
        '{"situation": "need_retry"}',
    ])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(1024, 768)),
              vlm, plan=True, max_steps=2)
    t = a.run("x")
    assert t.reflections == [RETRY, RETRY]
    assert t.subtasks == ["A", "B"]  # 没有推进，也没有重拆


def test_reformulate_replans_within_limit(screen):
    vlm = ScriptedVLM([
        '["A"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "need_reformulate"}',
        '["B", "C"]',                                    # 重新拆解
        '{"action": {"type": "click", "element": 1}}',
        '{"situation": "sub_task_success"}',
    ])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(1024, 768)),
              vlm, plan=True, max_steps=2)
    t = a.run("x")
    assert t.subtasks == ["B", "C"]
    assert t.reflections == [REFORMULATE, SUCCESS]


def test_replan_uses_the_screen_after_the_action(screen):
    """need_reformulate 的意思是「看到现在的情况，原计划走不通」，
    重拆要基于执行后的屏幕，不是执行前的。"""
    before = ScreenState(width=1024, height=768,
                         elements=[Element(id=0, bbox=(0, 0, 0.1, 0.1), text="执行前")])
    after = ScreenState(width=1024, height=768,
                        elements=[Element(id=0, bbox=(0, 0, 0.1, 0.1), text="执行后")])

    class TwoScreens:
        def __init__(self):
            self.n = 0

        def perceive(self, **kw):
            self.n += 1
            state = before if self.n == 1 else after
            return state, np.zeros((10, 10, 3), dtype=np.uint8)

    vlm = ScriptedVLM([
        '["A"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "need_reformulate"}',
        '["B"]',
        '{"action": {"type": "finished"}}',
    ])
    Agent(TwoScreens(), Controller(backend=RecordingBackend(1024, 768)), vlm,
          plan=True, max_steps=2).run("x")
    replan_prompt = vlm.prompts[3]
    assert "执行后" in replan_prompt and "执行前" not in replan_prompt


def test_replan_failure_does_not_kill_the_trajectory(screen):
    """重拆调用本身出错时，保持原计划往下走。"""

    class FlakyPerception(FakePerception):
        def __init__(self, state):
            super().__init__(state)
            self.n = 0

        def perceive(self, run_ocr=True, **kw):
            self.n += 1
            if run_ocr and self.n > 1:   # 重拆时的那次带 OCR 的感知
                raise OSError("屏幕抓取失败")
            return super().perceive()

    vlm = ScriptedVLM([
        '["A", "B"]',
        '{"action": {"type": "click", "element": 0}}',
        '{"situation": "need_reformulate"}',
        '{"action": {"type": "finished"}}',
    ])
    t = Agent(FlakyPerception(screen), Controller(backend=RecordingBackend(1024, 768)),
              vlm, plan=True, max_steps=3).run("x")
    assert t.subtasks == ["A", "B"]      # 保持原计划
    assert t.reflections[0] == REFORMULATE
    assert t.n_steps >= 1                # 轨迹没被废掉


def test_failed_plan_falls_back_to_single_step_loop(screen):
    """拆解失败不该让任务跑不起来。"""
    vlm = ScriptedVLM(["模型今天不想拆解", '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()),
              vlm, plan=True).run("x")
    assert t.subtasks == [] and t.success is True


def test_reflect_every_reduces_reflection_calls(screen):
    vlm = ScriptedVLM([
        '["A"]',
        '{"action": {"type": "wait"}}',
        '{"action": {"type": "wait"}}',
        '{"situation": "need_retry"}',
    ])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    t = Agent(FakePerception(screen), ctrl, vlm, plan=True,
              reflect_every=2, max_steps=2).run("x")
    assert t.reflections == [RETRY]  # 两步只反思了一次


def test_trajectory_json_carries_plan_and_reflections(screen):
    import json

    vlm = ScriptedVLM(['["A"]', '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()),
              vlm, plan=True).run("x")
    d = json.loads(t.to_json())
    assert d["subtasks"] == ["A"] and "reflections" in d


# --- 提示词不能被照抄 -------------------------------------------------------


def test_plan_example_is_marked_as_a_different_task():
    """裸给一个格式示例时模型会原样抄回来。实测五个任务里四个回了示例里那两条，
    包括「关闭记事本」被拆成「点击文件资源管理器图标」。"""
    from gui_agent.planner import PLAN_TEMPLATE

    tpl = PLAN_TEMPLATE.template
    assert "别的" in tpl and "无关" in tpl, "要写明示例属于另一个任务"
    assert tpl.index("例子") < tpl.index("{instruction}"), "示例要在待拆任务之前"


def test_plan_example_does_not_use_the_desktop_targets_of_the_basic_tasks():
    """示例里不能出现基础任务会用到的控件，否则抄示例也能蒙对，看不出问题。"""
    from gui_agent.planner import PLAN_TEMPLATE

    example = PLAN_TEMPLATE.template.split("{instruction}")[0]
    for word in ("开始菜单", "文件资源管理器", "记事本"):
        assert word not in example, f"示例里不该出现「{word}」"


def test_reflect_example_does_not_name_a_real_situation():
    """示例里直接写 sub_task_success 会把模型往那个取值上带。"""
    from gui_agent.planner import REFLECT_TEMPLATE, SITUATIONS

    head = REFLECT_TEMPLATE.template.split("situation 三选一")[0]
    for s in SITUATIONS:
        assert s not in head, f"格式示例里不该出现具体取值 {s}"
