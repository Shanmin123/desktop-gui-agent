from pathlib import Path

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


def test_unparseable_output_raises(screen):
    with pytest.raises(ValueError, match="无法解析"):
        parse_step("我不知道该做什么", screen)


def test_action_without_type_raises(screen):
    with pytest.raises(ValueError, match="无法解析"):
        parse_step('{"thought": "x"}', screen)


def test_unknown_action_type_raises(screen):
    with pytest.raises(ValueError, match="未知动作类型"):
        parse_step('{"action": {"type": "swipe", "point": [0.5, 0.5]}}', screen)


@pytest.mark.parametrize("eid", ["[1, 2]", '{"a": 1}', '"保存"'])
def test_non_integer_element_id_raises(screen, eid):
    """模型可能把 element 写成列表或字典，int() 对它们抛的是 TypeError。"""
    with pytest.raises(ValueError, match="不是整数"):
        parse_step('{"action": {"type": "click", "element": %s}}' % eid, screen)


def test_string_element_id_still_works(screen):
    _, a = parse_step('{"action": {"type": "click", "element": "1"}}', screen)
    assert a.point == pytest.approx((0.3, 0.45))


@pytest.mark.parametrize("eid", ["1.9", "true"])
def test_non_integral_element_id_rejected(screen, eid):
    """int(1.9) 和 int(True) 都会悄悄变成 1，那就点到别的控件上了。"""
    with pytest.raises(ValueError, match="不是整数"):
        parse_step('{"action": {"type": "click", "element": %s}}' % eid, screen)


def test_empty_action_object_raises_value_error(screen):
    """action 是空对象时 Action() 抛的是 TypeError，得统一成 ValueError。"""
    with pytest.raises(ValueError):
        parse_step('{"action": {}}', screen)


def test_action_missing_type_key_raises_value_error(screen):
    with pytest.raises(ValueError):
        parse_step('{"action": {"point": [0.5, 0.5]}}', screen)


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
    """被安全策略拦下的动作重试到额度用光才结束。"""
    vlm = FakeVLM(['{"action": {"type": "hotkey", "text": "win+l"}}'] * 5)
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm)
    t = a.run("锁屏")
    assert t.n_steps == 3 and t.retries == 2  # 第一次 + 两次重试
    assert t.success is False and all(not s.ok for s in t.steps)


def test_loop_stops_on_bad_element_id(screen):
    vlm = FakeVLM(['{"action": {"type": "click", "element": 99}}'])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              retry_limit=0)
    t = a.run("点不存在的东西")
    assert t.n_steps == 1 and not t.steps[0].ok
    assert t.success is False  # 解析失败要判成失败，不能留 None 当没判过


def test_unparseable_output_is_a_failed_step(screen):
    """解析失败不能记成成功的 call_user，否则微调数据里就是错标。"""
    vlm = FakeVLM(["模型今天不想输出 JSON"])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              retry_limit=0).run("x")
    assert t.n_steps == 1 and not t.steps[0].ok and t.steps[0].error
    assert t.success is False


# --- 失败重试（大纲第 6 周第 2 项）------------------------------------------


def test_retry_limit_zero_stops_at_the_first_failure(screen):
    vlm = FakeVLM(['{"action": {"type": "hotkey", "text": "win+l"}}'] * 5)
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              retry_limit=0).run("锁屏")
    assert t.n_steps == 1 and t.success is False and t.retries == 0


def test_recovered_failure_does_not_end_the_task(screen):
    """第一次失败第二次成功，任务该正常走完，不能被那次失败带走。"""
    vlm = FakeVLM(['{"action": {"type": "hotkey", "text": "win+l"}}',
                   '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              retry_backoff=0).run("x")
    assert t.success is True and t.retries == 1 and t.n_steps == 2


def test_parse_failure_recovers_on_retry(screen):
    """模型偶尔吐不出合法 JSON，重问一次往往就好了。"""
    vlm = FakeVLM(["模型今天不想输出 JSON",
                   '{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              detect_change=False, retry_backoff=0).run("x")
    assert t.success is True and t.retries == 1
    assert [s.ok for s in t.steps] == [False, True, True]


def test_failure_counter_resets_after_a_good_step(screen):
    """重试额度不是整条任务累计的，中间成功过就该还回来。"""
    bad = '{"action": {"type": "hotkey", "text": "win+l"}}'
    ok = '{"action": {"type": "wait"}}'
    vlm = FakeVLM([bad, ok, bad, ok, bad, '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              detect_change=False, retry_backoff=0).run("x")
    assert t.success is True and t.retries == 3


def test_retry_burns_step_budget(screen):
    """重试要占步数额度，否则一直失败的任务永远跑不完。"""
    vlm = FakeVLM(['{"action": {"type": "hotkey", "text": "win+l"}}'] * 50)
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              max_steps=2, retry_limit=99, retry_backoff=0).run("x")
    assert t.n_steps == 2 and t.success is False


def test_call_user_is_not_retried(screen):
    """模型主动求助不是故障，重试没有意义。"""
    vlm = FakeVLM(['{"action": {"type": "call_user", "thought": "这一步我做不了"}}'] * 3)
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm,
              retry_backoff=0).run("x")
    assert t.n_steps == 1 and t.success is False and t.retries == 0


def test_perception_failure_keeps_earlier_steps(screen):
    """截图这类瞬时故障只废掉当前这步，前面跑出来的轨迹要留住。"""

    class FlakyPerception(FakePerception):
        def __init__(self, state):
            super().__init__(state)
            self.n = 0

        def perceive(self, **kw):
            self.n += 1
            if self.n == 3:
                raise OSError("屏幕抓取失败")
            return super().perceive(**kw)

    vlm = FakeVLM(['{"action": {"type": "wait"}}', '{"action": {"type": "scroll", '
                   '"point": [0.5, 0.5], "direction": "down"}}'])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    # 关掉变化检测：它每步多调一次 perceive，会让「第几次调用失败」这件事
    # 变得和实现细节绑定。这条测的是主感知失败时轨迹保不保得住。
    t = Agent(FlakyPerception(screen), ctrl, vlm, detect_change=False,
              retry_limit=0).run("x")
    assert t.n_steps == 3 and t.steps[0].ok and t.steps[1].ok
    assert not t.steps[2].ok and "屏幕抓取失败" in t.steps[2].error
    assert t.success is False


def test_transient_perception_failure_is_retried(screen):
    """截图偶发失败重来一次就好，不该废掉整条任务。"""

    class FlakyPerception(FakePerception):
        def __init__(self, state):
            super().__init__(state)
            self.n = 0

        def perceive(self, **kw):
            self.n += 1
            if self.n == 2:
                raise OSError("屏幕抓取失败")
            return super().perceive(**kw)

    vlm = FakeVLM(['{"action": {"type": "wait"}}', '{"action": {"type": "finished"}}'])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    t = Agent(FlakyPerception(screen), ctrl, vlm, detect_change=False,
              retry_backoff=0).run("x")
    assert t.success is True and t.retries == 1
    assert [s.ok for s in t.steps] == [True, False, True]


def test_model_failure_keeps_earlier_steps(screen):
    """显存不足这类不会自愈的故障，重试用光后要停下来。"""

    class BoomVLM(FakeVLM):
        def ask(self, image, prompt, **kw):
            if self.prompts:
                raise RuntimeError("显存不足")
            return super().ask(image, prompt, **kw)

    vlm = BoomVLM(['{"action": {"type": "wait"}}'])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    t = Agent(FakePerception(screen), ctrl, vlm, retry_backoff=0).run("x")
    assert t.n_steps == 4 and t.steps[0].ok and t.retries == 2
    assert all("显存不足" in s.error for s in t.steps[1:])
    assert t.success is False


# --- 截图落盘 ---------------------------------------------------------------


class SavingPerception(FakePerception):
    """把 save_to 转交给真正的写盘逻辑，验证 Agent 有把路径传下来。"""

    def perceive(self, run_ocr=True, save_to=None):
        state, img = super().perceive()
        if save_to:
            from gui_agent.perception import imwrite

            imwrite(save_to, img)
            state = ScreenState(state.width, state.height, image_path=save_to,
                                elements=state.elements)
        return state, img


def test_no_screenshots_saved_by_default(screen, tmp_path):
    vlm = FakeVLM(['{"action": {"type": "finished"}}'])
    t = Agent(SavingPerception(screen), Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.steps[0].screen.image_path == ""
    assert list(tmp_path.iterdir()) == []


def test_shot_dir_saves_one_image_per_step(screen, tmp_path):
    """轨迹要当微调样本用，每步得有配对的截图。"""
    vlm = FakeVLM(['{"action": {"type": "wait"}}', '{"action": {"type": "finished"}}'])
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    t = Agent(SavingPerception(screen), ctrl, vlm, shot_dir=str(tmp_path)).run(
        "保存/文件:测试", task_id="打开 记事本/1"
    )
    assert t.n_steps == 2
    assert len(list(tmp_path.glob("*.png"))) == 2
    for s in t.steps:
        assert s.screen.image_path and Path(s.screen.image_path).exists()


def test_repeated_runs_do_not_overwrite_each_others_shots(screen, tmp_path):
    """同一个任务跑两次，第二次不能把第一次的图盖掉。"""
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    a = Agent(SavingPerception(screen), ctrl, FakeVLM([]), shot_dir=str(tmp_path))
    paths = []
    for _ in range(2):
        a.vlm = FakeVLM(['{"action": {"type": "wait"}}', '{"action": {"type": "finished"}}'])
        paths += [s.screen.image_path for s in a.run("x", task_id="same_task").steps]
    assert len(set(paths)) == 4
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_failsafe_aborts_the_whole_run(screen):
    """急停要一路传到调用方，不能被循环记成一条失败步然后继续。"""
    import pyautogui

    class Panicking(RecordingBackend):
        def click(self, *a, **k):
            raise pyautogui.FailSafeException("鼠标甩到角落了")

    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}'])
    a = Agent(FakePerception(screen), Controller(backend=Panicking()), vlm)
    with pytest.raises(pyautogui.FailSafeException):
        a.run("x")


def test_shot_path_sanitizes_task_id(screen, tmp_path):
    """task_id 默认取自指令，里面可能有 / : 这类不能当文件名的字符。"""
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend()),
              FakeVLM([]), shot_dir=str(tmp_path))
    from gui_agent.schema import Trajectory

    p = Path(a._shot_path(Trajectory(task_id="打开 C:/临时*文件", instruction="x")))
    assert p.parent == tmp_path
    assert not (set(p.name) & set('\\/:*?"<>|'))


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
    assert "0.500" in describe(Action("left_double", point=(0.5, 0.5)))
    assert "0.500" in describe(Action("right_single", point=(0.5, 0.5)))
    assert "drag" in describe(Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9)))
    assert "down" in describe(Action("scroll", point=(0.5, 0.5), direction="down"))
    assert "'hello'" in describe(Action("type", text="hello"))
    assert "'ctrl+s'" in describe(Action("hotkey", text="ctrl+s"))
    assert describe(Action("finished")) == "finished"
    assert describe(Action("wait")) == "wait"
    assert describe(Action("call_user")) == "call_user"

    from gui_agent.schema import ACTION_TYPES

    covered = {"click", "left_double", "right_single", "drag", "scroll",
               "type", "hotkey", "finished", "wait", "call_user"}
    assert covered == set(ACTION_TYPES), "有动作类型没被这条测试覆盖"


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


# --- 像素坐标兜底 -----------------------------------------------------------


def test_pixel_coords_converted_to_normalized(screen):
    """提示词要求归一化，但模型实测会回像素值，如 (899, 84)。"""
    _, a = parse_step('{"action": {"type": "click", "point": [899, 84]}}', screen,
                      model_size=(1430, 804))
    assert a.point == pytest.approx((899 / 1430, 84 / 804))


def test_normalized_coords_left_alone(screen):
    _, a = parse_step('{"action": {"type": "click", "point": [0.6, 0.1]}}', screen,
                      model_size=(1430, 804))
    assert a.point == (0.6, 0.1)


def test_pixel_coords_clamped_to_one(screen):
    _, a = parse_step('{"action": {"type": "click", "point": [9999, 9999]}}', screen,
                      model_size=(1430, 804))
    assert a.point == (1.0, 1.0)


def test_drag_both_points_converted(screen):
    _, a = parse_step('{"action": {"type": "drag", "point": [100, 50], "point2": [700, 400]}}',
                      screen, model_size=(1430, 804))
    assert a.point[0] < 1 and a.point2[0] < 1


def test_pixel_coords_without_model_size_still_fail(screen):
    """没有模型尺寸就没法换算，让它照常报错而不是瞎猜。"""
    with pytest.raises(ValueError, match="归一化"):
        parse_step('{"action": {"type": "click", "point": [899, 84]}}', screen)


class FakeVLMWithSize(FakeVLM):
    def resized_size(self, h, w):
        return (804, 1430)  # (rh, rw)


class FakeVLMWithCoordSize(FakeVLMWithSize):
    def coord_size(self, h, w):
        return (1000, 1000)  # rel1000 口径


def test_agent_prefers_coord_size_over_resized_size(screen):
    vlm = FakeVLMWithCoordSize(['{"action": {"type": "click", "point": [500, 250]}}',
                                '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.steps[0].ok
    assert t.steps[0].action.point == pytest.approx((0.5, 0.25), abs=0.01)


def test_agent_passes_model_size(screen):
    vlm = FakeVLMWithSize(['{"action": {"type": "click", "point": [715, 402]}}',
                           '{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.steps[0].ok
    assert t.steps[0].action.point == pytest.approx((0.5, 0.5), abs=0.01)


# --- 错误检测：动作有没有产生效果 -------------------------------------------


class TwoFramePerception:
    """按脚本依次返回不同的画面，用来造出「界面变了/没变」两种情形。"""

    def __init__(self, state, frames):
        self.state = state
        self.frames = list(frames)
        self.calls = 0

    def perceive(self, **kw):
        self.calls += 1
        f = self.frames[min(self.calls - 1, len(self.frames) - 1)]
        return self.state, f


def _frame(value):
    return np.full((90, 160, 3), value, dtype=np.uint8)


def test_unchanged_screen_is_recorded_on_the_step(screen):
    """点击落在空白处也会 ok=True，只有比对屏幕才看得出没点中。"""
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    per = TwoFramePerception(screen, [_frame(10), _frame(10)])   # 前后一模一样
    t = Agent(per, Controller(backend=RecordingBackend(1920, 1080)), vlm).run("x")
    assert t.steps[0].ok is True and t.steps[0].changed is False


def test_changed_screen_is_recorded(screen):
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    per = TwoFramePerception(screen, [_frame(10), _frame(200)])  # 画面大变
    t = Agent(per, Controller(backend=RecordingBackend(1920, 1080)), vlm).run("x")
    assert t.steps[0].changed is True


def test_no_change_is_fed_back_into_the_prompt(screen):
    """没点中要明确告诉模型，它才有机会换个目标而不是原地重复。"""
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    per = TwoFramePerception(screen, [_frame(10), _frame(10)])
    Agent(per, Controller(backend=RecordingBackend(1920, 1080)), vlm).run("x")
    assert "界面没有变化" in vlm.prompts[1]


def test_detection_can_be_switched_off(screen):
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    per = TwoFramePerception(screen, [_frame(10), _frame(10)])
    t = Agent(per, Controller(backend=RecordingBackend(1920, 1080)), vlm,
              detect_change=False).run("x")
    assert t.steps[0].changed is None


def test_detection_failure_does_not_kill_the_step(screen):
    """检测本身出错时不做判断，不能把整步记成失败。"""

    class DetectBoom(TwoFramePerception):
        def perceive(self, run_ocr=True, **kw):
            if not run_ocr:            # 只有变化检测那次会传 run_ocr=False
                raise OSError("截图失败")
            return super().perceive(**kw)

    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    per = DetectBoom(screen, [_frame(10)])
    t = Agent(per, Controller(backend=RecordingBackend(1920, 1080)), vlm).run("x")
    assert t.steps[0].ok is True and t.steps[0].changed is None
    assert t.n_steps == 2 and t.success is True


def test_terminal_actions_are_not_checked(screen):
    """finished / call_user 没有对应的界面操作，不必比屏幕。"""
    vlm = FakeVLM(['{"action": {"type": "finished"}}'])
    per = TwoFramePerception(screen, [_frame(10)])
    t = Agent(per, Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.steps[0].changed is None and per.calls == 1


# --- 卡住判定用屏幕变化 -----------------------------------------------------


def test_repeated_action_that_changes_the_screen_is_not_stuck(screen):
    """同一位置的「下一步」按钮连点三页，界面每次都在变，不是卡住。"""
    from gui_agent.agent import is_stuck

    steps = [Step(screen, Action("click", point=(0.5, 0.5)), changed=True) for _ in range(3)]
    assert not is_stuck(steps)


def test_repeated_action_without_change_is_stuck(screen):
    from gui_agent.agent import is_stuck

    steps = [Step(screen, Action("click", point=(0.5, 0.5)), changed=False) for _ in range(3)]
    assert is_stuck(steps)


def test_one_change_among_repeats_is_enough(screen):
    from gui_agent.agent import is_stuck

    steps = [Step(screen, Action("click", point=(0.5, 0.5)), changed=False),
             Step(screen, Action("click", point=(0.5, 0.5)), changed=True),
             Step(screen, Action("click", point=(0.5, 0.5)), changed=False)]
    assert not is_stuck(steps)


def test_without_change_info_falls_back_to_action_repetition(screen):
    """没开检测时退回只看动作是否重复，行为与之前一致。"""
    from gui_agent.agent import is_stuck

    steps = [Step(screen, Action("click", point=(0.5, 0.5))) for _ in range(3)]
    assert is_stuck(steps)


# --- 图标元素进提示词（大纲第 6 周第 3 项）----------------------------------


def test_cv_elements_appear_in_the_element_list():
    """CV 补的图标框没有文字，但要有编号可指，否则补了等于没补。"""
    from gui_agent.agent import format_elements

    state = ScreenState(width=1280, height=720, elements=[
        Element(id=0, bbox=(0.1, 0.1, 0.2, 0.2), text="文件"),
        Element(id=1, bbox=(0.5, 0.5, 0.55, 0.55), text="", source="cv"),
    ])
    out = format_elements(state)
    assert "[0] 文件" in out and "[1]" in out and "图标" in out


def test_empty_ocr_elements_are_still_skipped():
    """OCR 偶尔给出空串，那是噪声，不该占编号。"""
    from gui_agent.agent import format_elements

    state = ScreenState(width=1280, height=720, elements=[
        Element(id=0, bbox=(0.1, 0.1, 0.2, 0.2), text="  ", source="ocr"),
        Element(id=1, bbox=(0.3, 0.3, 0.4, 0.4), text="保存", source="ocr"),
    ])
    out = format_elements(state)
    assert "[0]" not in out and "[1] 保存" in out


# --- 按配置决定要不要跑 OCR（大纲第 6 周第 3 项）----------------------------


class RecordingPerception:
    """记下每次 perceive 有没有要求跑 OCR。"""

    def __init__(self, state):
        self.state = state
        self.ocr_flags = []

    def perceive(self, run_ocr=True, save_to=None):
        self.ocr_flags.append(run_ocr)
        return self.state, np.zeros((10, 10, 3), dtype=np.uint8)


def test_one_stage_runs_ocr_every_step(screen):
    """一段式的提示词里有元素清单，每步都得跑。"""
    per = RecordingPerception(screen)
    vlm = FakeVLM(['{"action": {"type": "wait"}}', '{"action": {"type": "finished"}}'])
    Agent(per, Controller(backend=RecordingBackend(), dry_run=True), vlm,
          detect_change=False).run("x")
    assert per.ocr_flags == [True, True]


def test_two_stage_skips_ocr(screen):
    """两段式定位不看元素清单，OCR 白跑，1280x720 下每步 0.71s。"""
    per = RecordingPerception(screen)

    class Locating(FakeVLM):
        def locate(self, image, instruction):
            return (0.5, 0.5)

    vlm = Locating(['{"action": {"type": "click", "target": "保存按钮"}}',
                    '{"action": {"type": "finished"}}'])
    Agent(per, Controller(backend=RecordingBackend(), dry_run=True), vlm,
          locate_target=True, detect_change=False).run("x")
    assert per.ocr_flags == [False, False]


def test_two_stage_with_plan_runs_ocr_once(screen):
    """拆解子任务要看元素清单，只有那一次要跑。"""
    per = RecordingPerception(screen)

    class Planning(FakeVLM):
        def locate(self, image, instruction):
            return (0.5, 0.5)

    vlm = Planning(['["打开菜单", "点保存"]',                              # 拆解
                    '{"action": {"type": "click", "target": "菜单"}}',     # 第一步
                    '{"situation": "sub_task_success"}',                   # 反思
                    '{"action": {"type": "finished"}}'])
    Agent(per, Controller(backend=RecordingBackend(), dry_run=True), vlm,
          locate_target=True, plan=True, detect_change=False).run("x")
    assert per.ocr_flags[0] is True, "第一步要拿元素清单去拆解"
    assert not any(per.ocr_flags[1:]), "拆完之后就不需要了"


# --- 元素编号写成浮点 -------------------------------------------------------


@pytest.mark.parametrize("eid", ["1.0", "1.00", '"1.0"'])
def test_integral_float_element_id_accepted(screen, eid):
    """微调后的模型常把编号写成 14.0，那是合法的 14，不该判成解析失败。"""
    _, a = parse_step('{"action": {"type": "click", "element": %s}}' % eid, screen)
    assert a.point == pytest.approx((0.3, 0.45))


@pytest.mark.parametrize("eid", ["1.9", "0.5", "true", '"一"'])
def test_non_integral_element_id_still_rejected(screen, eid):
    """1.9 取整会变成 1，点到别的控件上，这类必须继续拒掉。"""
    with pytest.raises(ValueError, match="不是整数"):
        parse_step('{"action": {"type": "click", "element": %s}}' % eid, screen)


# --- 给图标候选框留名额（大纲第 6 周第 3 项）--------------------------------


def _crowd(n_text, n_icon):
    els = [Element(id=i, bbox=(0.1, i / 500, 0.2, i / 500 + 0.005), text=f"文字{i}")
           for i in range(n_text)]
    els += [Element(id=n_text + i, bbox=(0.5, i / 200, 0.52, i / 200 + 0.02),
                    text="", source="cv") for i in range(n_icon)]
    return ScreenState(1280, 720, elements=els)


def test_icons_get_reserved_slots_when_text_would_fill_them():
    """文字密集的界面上，OCR 不能把图标框全挤出去。"""
    from gui_agent.agent import RESERVED_FOR_UNNAMED, select_elements

    shown = select_elements(_crowd(70, 20), limit=60)
    assert len(shown) == 60
    assert sum(1 for e in shown if e.source == "cv") == RESERVED_FOR_UNNAMED


def test_text_keeps_the_rest_of_the_budget():
    from gui_agent.agent import select_elements

    shown = select_elements(_crowd(70, 20), limit=60)
    assert sum(1 for e in shown if e.source == "ocr") == 44


def test_no_icons_means_text_uses_the_whole_budget():
    from gui_agent.agent import select_elements

    assert len(select_elements(_crowd(70, 0), limit=60)) == 60


def test_few_elements_are_all_shown():
    from gui_agent.agent import select_elements

    assert len(select_elements(_crowd(3, 3), limit=60)) == 6


def test_icons_take_leftover_space_before_touching_the_reserve():
    """文字只有 10 个时，20 个图标框该进 20 个，不是只进 16 个。"""
    from gui_agent.agent import select_elements

    shown = select_elements(_crowd(10, 20), limit=60)
    assert sum(1 for e in shown if e.source == "cv") == 20


def test_reservation_never_exceeds_the_limit():
    from gui_agent.agent import select_elements

    for limit in (1, 5, 16, 17, 60):
        assert len(select_elements(_crowd(70, 20), limit=limit)) <= limit


def test_selected_ids_are_unchanged():
    """选择只决定显示哪些，不能改编号——编号是模型用来指元素的。"""
    from gui_agent.agent import select_elements

    state = _crowd(70, 20)
    shown = select_elements(state, limit=60)
    by_id = {e.id: e for e in state.elements}
    for e in shown:
        assert by_id[e.id].bbox == e.bbox


def test_elements_with_no_text_and_no_cv_source_are_skipped():
    from gui_agent.agent import select_elements

    state = ScreenState(1280, 720, elements=[
        Element(id=0, bbox=(0.1, 0.1, 0.2, 0.2), text="  ", source="ocr"),
        Element(id=1, bbox=(0.3, 0.3, 0.4, 0.4), text="保存", source="ocr"),
    ])
    assert [e.id for e in select_elements(state)] == [1]


def test_reserve_holds_even_when_text_almost_fills_the_budget():
    """名额是保证值：文字 59 个也要让出 16 个位置。

    写成「文字占满才腾位置」的话，59 个文字只会给图标框留 1 个位置，
    334 条上这一档就是 83.8% 和 83.5% 的差别。
    """
    from gui_agent.agent import RESERVED_FOR_UNNAMED, select_elements

    shown = select_elements(_crowd(59, 20), limit=60)
    assert sum(1 for e in shown if e.source == "cv") == RESERVED_FOR_UNNAMED
    assert len(shown) == 60


def test_reserve_shrinks_with_a_tiny_budget():
    """名额不超过一半，否则 limit 很小的时候文字会被挤干净。"""
    from gui_agent.agent import select_elements

    shown = select_elements(_crowd(70, 20), limit=6)
    assert len(shown) == 6
    assert sum(1 for e in shown if e.source == "cv") == 3
    assert sum(1 for e in shown if e.source == "ocr") == 3


# --- 停滞升级为重拆（大纲第 6 周第 1、2 项）--------------------------------


class PlanningVLM(FakeVLM):
    """按顺序回答：拆解 -> (动作, 反思) 循环。"""

    def locate(self, image, instruction):
        return (0.5, 0.5)


def test_repeated_need_retry_escalates_to_replan(tmp_path, screen):
    """反思连着说「没完成但方向对」，不能一直重试下去。

    实测复杂任务 copy_between_files 的 16 次反思里 11 次是 need_retry，
    反思看出来没推进，但没有东西把它升级，于是一直试到步数用光。
    """
    from gui_agent.monitor import Monitor, read_log

    log = tmp_path / "run.jsonl"
    plan = '["第一步", "第二步"]'
    act = '{"action": {"type": "wait"}}'
    retry = '{"situation": "need_retry"}'
    vlm = PlanningVLM([plan, act, retry, act, retry, act, retry, plan, act,
                       '{"situation": "sub_task_success"}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend(), dry_run=True),
              vlm, plan=True, detect_change=False, max_steps=5, retry_backoff=0,
              monitor=Monitor(str(log), quiet=True)).run("x")
    assert t.reflections[:3] == ["need_retry"] * 3
    escalated = [r for r in read_log(str(log))
                 if r["event"] == "reflect" and "升级为重拆" in r["situation"]]
    assert len(escalated) == 1, "第 3 次 need_retry 应该升级成重拆"


def test_no_escalation_below_the_limit(tmp_path, screen):
    """只连着两次就不该升级，免得刚起步就推翻计划。"""
    from gui_agent.monitor import Monitor, read_log

    log = tmp_path / "run.jsonl"
    vlm = PlanningVLM(['["a", "b"]', '{"action": {"type": "wait"}}',
                       '{"situation": "need_retry"}', '{"action": {"type": "wait"}}',
                       '{"situation": "need_retry"}', '{"action": {"type": "finished"}}'])
    Agent(FakePerception(screen), Controller(backend=RecordingBackend(), dry_run=True),
          vlm, plan=True, detect_change=False, max_steps=4, retry_backoff=0,
          monitor=Monitor(str(log), quiet=True)).run("x")
    assert not [r for r in read_log(str(log))
                if r["event"] == "reflect" and "升级为重拆" in r["situation"]]


def test_stall_counter_resets_when_a_subtask_succeeds(screen):
    """中间推进过就不该累计到升级。"""
    plan = '["a", "b"]'
    act = '{"action": {"type": "wait"}}'
    vlm = PlanningVLM([plan, act, '{"situation": "need_retry"}',
                       act, '{"situation": "sub_task_success"}',
                       act, '{"situation": "need_retry"}',
                       act, '{"situation": "need_retry"}'])
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(), dry_run=True),
              vlm, plan=True, detect_change=False, max_steps=4, retry_backoff=0)
    t = a.run("x")
    # 三次 need_retry 不连续，不该升级（只拆解了一次）
    assert t.reflections.count("need_retry") >= 2


def test_stall_limit_is_configurable(screen):
    plan = '["a"]'
    act = '{"action": {"type": "wait"}}'
    vlm = PlanningVLM([plan] + [act, '{"situation": "need_retry"}'] * 4)
    a = Agent(FakePerception(screen), Controller(backend=RecordingBackend(), dry_run=True),
              vlm, plan=True, detect_change=False, max_steps=4, stall_limit=99,
              retry_backoff=0)
    t = a.run("x")
    assert t.reflections == ["need_retry"] * len(t.reflections)
