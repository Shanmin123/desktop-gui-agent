"""执行状态的实时记录。

重点是「中途中断也留得下记录」——原来轨迹只在跑完时落盘，任务崩在半路就
什么都看不到。
"""

import json

import numpy as np
import pytest

from gui_agent.agent import Agent
from gui_agent.control import Controller, RecordingBackend
from gui_agent.monitor import Monitor, describe_action, read_log
from gui_agent.schema import Action, Element, ScreenState, Step, Trajectory


@pytest.fixture
def screen():
    return ScreenState(width=1920, height=1080,
                       elements=[Element(id=1, bbox=(0.2, 0.4, 0.4, 0.5), text="保存")])


# --- 动作描述 ---------------------------------------------------------------


@pytest.mark.parametrize("action,want", [
    (Action("click", point=(0.5, 0.5)), "click (0.500, 0.500)"),
    (Action("left_double", point=(0.1, 0.2)), "left_double (0.100, 0.200)"),
    (Action("scroll", point=(0.5, 0.5), direction="down"), "scroll (0.500, 0.500) down"),
    (Action("type", text="你好"), "type '你好'"),
    (Action("hotkey", text="ctrl+s"), "hotkey 'ctrl+s'"),
    (Action("wait"), "wait"),
    (Action("finished"), "finished"),
    (Action("call_user"), "call_user"),
])
def test_describe_covers_every_action_type(action, want):
    assert describe_action(action) == want


def test_describe_drag_shows_both_points():
    s = describe_action(Action("drag", point=(0.1, 0.1), point2=(0.9, 0.9)))
    assert "0.100" in s and "0.900" in s


def test_describe_covers_all_ten_types():
    from gui_agent.schema import ACTION_TYPES

    made = {"click", "left_double", "right_single", "drag", "scroll",
            "type", "hotkey", "wait", "finished", "call_user"}
    assert made == set(ACTION_TYPES)


# --- 落盘 -------------------------------------------------------------------


def test_each_step_is_flushed_immediately(tmp_path, screen):
    """跑到一半打开日志也该看得见前面的步骤。"""
    p = tmp_path / "run.jsonl"
    m = Monitor(str(p), quiet=True)
    m.task_start("打开浏览器")
    m.step(1, Step(screen, Action("click", point=(0.5, 0.5)), changed=True))
    # 还没 finish，文件里就该有两条
    rows = read_log(str(p))
    assert [r["event"] for r in rows] == ["start", "step"]


def test_finish_records_summary(tmp_path, screen):
    p = tmp_path / "run.jsonl"
    m = Monitor(str(p), quiet=True)
    traj = Trajectory("t", "x", [
        Step(screen, Action("click", point=(0.5, 0.5)), changed=True, elapsed=1.0),
        Step(screen, Action("click", point=(0.5, 0.5)), changed=False, elapsed=2.0),
    ], success=False)
    m.finish(traj)
    r = read_log(str(p))[-1]
    assert r["event"] == "finish" and r["success"] is False
    assert r["n_steps"] == 2 and r["changed_steps"] == 1 and r["unchanged_steps"] == 1


def test_plan_and_reflect_are_recorded(tmp_path):
    p = tmp_path / "run.jsonl"
    m = Monitor(str(p), quiet=True)
    m.plan(["打开开始菜单", "点图标"])
    m.reflect("sub_task_success", "打开开始菜单")
    rows = read_log(str(p))
    assert rows[0]["subtasks"] == ["打开开始菜单", "点图标"]
    assert rows[1]["situation"] == "sub_task_success"


def test_empty_plan_is_recorded_too(tmp_path):
    p = tmp_path / "run.jsonl"
    m = Monitor(str(p), quiet=True)
    m.plan([])
    assert read_log(str(p))[0]["subtasks"] == []


def test_previous_run_is_not_mixed_in(tmp_path, screen):
    p = tmp_path / "run.jsonl"
    Monitor(str(p), quiet=True).step(1, Step(screen, Action("wait")))
    m2 = Monitor(str(p), quiet=True)
    m2.step(1, Step(screen, Action("finished")))
    assert len(read_log(str(p))) == 1


def test_no_path_means_no_file(tmp_path, screen):
    m = Monitor(None, quiet=True)
    m.step(1, Step(screen, Action("wait")))     # 不该抛
    assert list(tmp_path.iterdir()) == []


# --- 读回 -------------------------------------------------------------------


def test_read_log_skips_truncated_last_line(tmp_path):
    """进程被杀时最后一行可能只写了一半，不能让整份日志读不出来。"""
    p = tmp_path / "run.jsonl"
    p.write_text('{"event": "start"}\n{"event": "step", "n": 1}\n{"event": "ste',
                 encoding="utf-8")
    rows = read_log(str(p))
    assert [r["event"] for r in rows] == ["start", "step"]


def test_read_missing_log_returns_empty(tmp_path):
    assert read_log(str(tmp_path / "没有.jsonl")) == []


def test_read_log_ignores_blank_lines(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text('{"event": "a"}\n\n\n{"event": "b"}\n', encoding="utf-8")
    assert len(read_log(str(p))) == 2


# --- 接进 Agent -------------------------------------------------------------


class FakeVLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def ask(self, image, prompt, **kw):
        return self.replies.pop(0) if self.replies else '{"action": {"type": "finished"}}'


class FakePerception:
    def __init__(self, state):
        self.state = state

    def perceive(self, **kw):
        return self.state, np.zeros((10, 10, 3), dtype=np.uint8)


def test_agent_writes_a_full_run_log(tmp_path, screen):
    p = tmp_path / "run.jsonl"
    vlm = FakeVLM(['{"action": {"type": "click", "element": 1}}',
                   '{"action": {"type": "finished"}}'])
    Agent(FakePerception(screen), Controller(backend=RecordingBackend(1920, 1080)), vlm,
          monitor=Monitor(str(p), quiet=True)).run("保存文件")
    events = [r["event"] for r in read_log(str(p))]
    assert events[0] == "start" and events[-1] == "finish"
    assert events.count("step") == 2


def test_run_log_survives_a_crash_midway(tmp_path, screen):
    """任务崩在半路，前面的步骤仍要在日志里。"""
    p = tmp_path / "run.jsonl"

    class Boom(FakePerception):
        def __init__(self, state):
            super().__init__(state)
            self.n = 0

        def perceive(self, run_ocr=True, **kw):
            self.n += 1
            if self.n >= 3:
                raise OSError("屏幕抓取失败")  # 不会自愈，重试完还是得停
            return super().perceive()

    vlm = FakeVLM(['{"action": {"type": "wait"}}'] * 4)
    ctrl = Controller(backend=RecordingBackend(), dry_run=True)
    Agent(Boom(screen), ctrl, vlm, detect_change=False, retry_backoff=0,
          monitor=Monitor(str(p), quiet=True)).run("x")
    rows = read_log(str(p))
    assert any(r["event"] == "step" for r in rows)
    assert [r["event"] for r in rows].count("retry") == 2
    assert rows[-1]["event"] == "finish" and rows[-1]["success"] is False


def test_retry_event_carries_the_reason(tmp_path):
    p = tmp_path / "run.jsonl"
    Monitor(str(p), quiet=True).retry(1, "坐标越界")
    r = read_log(str(p))[0]
    assert r["event"] == "retry" and r["n"] == 1 and "坐标越界" in r["reason"]


def test_agent_without_monitor_still_runs(screen):
    vlm = FakeVLM(['{"action": {"type": "finished"}}'])
    t = Agent(FakePerception(screen), Controller(backend=RecordingBackend()), vlm).run("x")
    assert t.success is True


def test_unchanged_step_shows_in_the_log(tmp_path, screen):
    p = tmp_path / "run.jsonl"
    m = Monitor(str(p), quiet=True)
    m.step(1, Step(screen, Action("click", point=(0.5, 0.5)), changed=False))
    assert read_log(str(p))[0]["changed"] is False
