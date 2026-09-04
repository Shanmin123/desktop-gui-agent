"""Agent 框架：任务拆解与规划、动作解析、结果反馈。

对应大纲第 3 周第 2、3 项，第 4 周。

循环结构参考 ScreenAgent 的 Planning-Acting-Reflecting：每一步先看屏幕，再让模型
给出 Thought 和 Action，执行后把结果写回历史供下一步参考。Thought 的写法对齐
UI-TARS，理由随执行记录一起存下来。

模型引用 OCR 元素编号来指定目标，由本模块查表换成归一化坐标。编号是我们自己编的，
不经过模型的坐标空间，比让模型直接输出像素坐标可靠。没有文字标签的图标用不了编号，
模型可以改用归一化坐标，或者由上层调用 models.locate。
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Tuple

from .control import Controller
from .schema import Action, ScreenState, Step, Trajectory

MAX_STEPS = 15
MAX_ELEMENTS = 60  # 送进提示词的元素上限，太多会挤占上下文
REPEAT_LIMIT = 3   # 同一个动作连续这么多次就停，避免在无效操作上空转

SYSTEM_PROMPT = """你在操作一台 Windows 电脑，目标是完成用户给的任务。

每一步会给你当前屏幕截图和屏幕上识别出的文字元素列表。你要输出下一步动作。

可用动作：
  click / left_double / right_single  点击，指定 element 或 point
  drag                                拖拽，需要 point 和 point2
  scroll                              滚动，指定 element 或 point，加 direction（up/down/left/right）
  type                                输入文本，需要 text
  hotkey                              组合键，如 "ctrl+s"，需要 text
  wait                                等待界面变化
  finished                            任务已完成
  call_user                           无法继续，需要人工介入

只返回一个 JSON 对象，不要有别的内容：
{"thought": "为什么这么做", "action": {"type": "click", "element": 12}}

指定位置时优先用 element 编号。列表里没有对应元素时，用 point 给归一化坐标，
形如 "point": [0.5, 0.5]，取值 0 到 1。"""


def format_elements(state: ScreenState, limit: int = MAX_ELEMENTS) -> str:
    """把识别出的元素列成编号清单。空文本的元素对模型没用，跳过。"""
    lines = []
    for e in state.elements:
        if not e.text.strip():
            continue
        cx, cy = e.center()
        lines.append(f"  [{e.id}] {e.text}  (位置 {cx:.2f}, {cy:.2f})")
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "  （没有识别到文字元素）"


def format_history(steps: List[Step], limit: int = 5) -> str:
    """最近几步做了什么、结果如何。太长会挤占上下文，只留末尾几步。"""
    if not steps:
        return "  （这是第一步）"
    out = []
    for i, s in enumerate(steps[-limit:], start=max(1, len(steps) - limit + 1)):
        status = "成功" if s.ok else f"失败：{s.error}"
        out.append(f"  第{i}步 {s.action.type} → {status}")
    return "\n".join(out)


def _signature(action: Action) -> tuple:
    """动作的可比较特征，用来判断是不是在重复同一件事。"""
    return (action.type, action.point, action.point2, action.text, action.direction)


def is_stuck(steps: List[Step], limit: int = REPEAT_LIMIT) -> bool:
    """末尾连续 limit 步是同一个动作，说明卡住了。

    动作没让界面产生变化时，模型看到的还是同一屏，会一直给同样的动作。实测在
    dry-run 下三步给出了完全相同的点击。
    """
    if len(steps) < limit:
        return False
    sigs = [_signature(s.action) for s in steps[-limit:]]
    return len(set(sigs)) == 1


def build_prompt(instruction: str, state: ScreenState, steps: List[Step]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"任务：{instruction}\n\n"
        f"已执行：\n{format_history(steps)}\n\n"
        f"当前屏幕上的文字元素：\n{format_elements(state)}\n\n"
        f"下一步动作："
    )


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里取出第一个完整的 JSON 对象。

    模型常在 JSON 前后加解释，或者用 ```json 包起来，所以按括号配对扫描，
    不能简单地取首尾大括号。
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_step(text: str, state: ScreenState) -> Tuple[str, Action]:
    """把模型输出解析成 (thought, Action)。

    element 编号在这里换成归一化坐标，编号不存在时抛 ValueError 由上层记为失败。
    解析不出动作时返回 call_user，让人接手而不是瞎点。
    """
    data = _extract_json(text)
    if not data or not isinstance(data.get("action"), dict):
        return (f"模型输出无法解析：{text[:80]}", Action("call_user"))

    thought = str(data.get("thought", ""))
    raw = dict(data["action"])

    eid = raw.pop("element", None)
    if eid is not None:
        match = next((e for e in state.elements if e.id == int(eid)), None)
        if match is None:
            raise ValueError(f"元素编号 {eid} 不在当前屏幕的识别结果里")
        raw["point"] = match.center()

    for k in ("point", "point2"):
        if isinstance(raw.get(k), list):
            raw[k] = tuple(raw[k])

    return thought, Action.from_dict(raw)


class Agent:
    """跑一条任务的完整循环。"""

    def __init__(
        self,
        perception,
        controller: Controller,
        vlm,
        max_steps: int = MAX_STEPS,
        repeat_limit: int = REPEAT_LIMIT,
    ) -> None:
        self.perception = perception
        self.controller = controller
        self.vlm = vlm
        self.max_steps = max_steps
        self.repeat_limit = repeat_limit

    def run(self, instruction: str, task_id: str = "") -> Trajectory:
        traj = Trajectory(task_id=task_id or instruction[:24], instruction=instruction)

        for _ in range(self.max_steps):
            state, model_img = self.perception.perceive()
            prompt = build_prompt(instruction, state, traj.steps)

            t0 = time.perf_counter()
            try:
                thought, action = parse_step(self.vlm.ask(model_img, prompt), state)
            except ValueError as e:
                traj.steps.append(
                    Step(state, Action("call_user", thought=str(e)),
                         ok=False, error=str(e), elapsed=time.perf_counter() - t0)
                )
                break

            action.thought = thought
            result = self.controller.execute(action)
            traj.steps.append(
                Step(state, action, ok=result.ok, error=result.error,
                     elapsed=time.perf_counter() - t0)
            )

            if action.type == "finished":
                traj.success = True
                break
            if action.type == "call_user" or not result.ok:
                traj.success = False
                break
            if is_stuck(traj.steps, self.repeat_limit):
                traj.steps.append(
                    Step(state, Action("call_user", thought="连续重复同一个动作，界面没有变化"),
                         ok=False, error=f"连续 {self.repeat_limit} 步重复同一动作")
                )
                traj.success = False
                break

        if traj.success is None and traj.n_steps >= self.max_steps:
            traj.success = False  # 走完步数上限还没结束，算失败
        return traj
