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
from pathlib import Path
from typing import List, Optional, Tuple

from .chain import TEMPLATE, build_chain
from .control import Controller, is_failsafe
from .planner import REFORMULATE, RETRY, SUCCESS, Planner, acting_instruction
from .schema import Action, ScreenState, Step, Trajectory

MAX_STEPS = 15
MAX_ELEMENTS = 60  # 送进提示词的元素上限，太多会挤占上下文
REPEAT_LIMIT = 3   # 同一个动作连续这么多次就停，避免在无效操作上空转

# 提示词模板在 chain.py，用 LangChain 的 PromptTemplate 管理，两边共用一份
SYSTEM_PROMPT = TEMPLATE.split("\n\n任务：")[0].replace("{{", "{").replace("}}", "}")


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
    """末尾连续 limit 步是同一个动作，就当卡住了。

    动作没让界面产生变化时，模型看到的还是同一屏，会一直给同样的动作。实测在
    dry-run 下三步给出了完全相同的点击。

    只看动作、不看屏幕，所以会误伤：同一位置的「下一步」按钮连点三页也算卡住。
    大纲第 6 周的鲁棒性优化里再把屏幕变化一起纳入判断。
    """
    if len(steps) < limit:
        return False
    sigs = [_signature(s.action) for s in steps[-limit:]]
    return len(set(sigs)) == 1


def build_prompt(instruction: str, state: ScreenState, steps: List[Step]) -> str:
    """套 chain.py 里的 LangChain 模板生成一步的提示词。"""
    from .chain import render_prompt

    return render_prompt(instruction, state, steps)


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


def _to_norm(p, model_size: Optional[Tuple[int, int]]):
    """坐标超出 0~1 时按像素处理，除以模型看到的尺寸换成归一化。

    提示词要求归一化坐标，但 Qwen2.5-VL 本来就是按像素训练的，实测会回
    (899, 84) 这样的值。与其让整步失败，不如按它的坐标空间换算。
    """
    if not (isinstance(p, (tuple, list)) and len(p) == 2):
        return p
    if max(p) <= 1.0 or model_size is None:
        return p
    w, h = model_size
    return (min(p[0] / w, 1.0), min(p[1] / h, 1.0))


def parse_step(
    text: str, state: ScreenState, model_size: Optional[Tuple[int, int]] = None
) -> Tuple[str, Action]:
    """把模型输出解析成 (thought, Action)。

    element 编号在这里换成归一化坐标。模型给的坐标超出 0~1 时按 model_size 换算。

    拿不到有效动作一律抛 ValueError，由上层记成失败步。不能返回 call_user 顶替：
    那是模型主动求助时的正常动作，和解析失败混在一起，日志里会把失败记成成功，
    第 3 周拿这些轨迹做微调数据就带进了错标。
    """
    data = _extract_json(text)
    if not data or not isinstance(data.get("action"), dict):
        raise ValueError(f"模型输出无法解析：{text[:80]}")

    thought = str(data.get("thought", ""))
    raw = dict(data["action"])

    eid = raw.pop("element", None)
    if eid is not None:
        # 不能直接 int()：int(1.9) 和 int(True) 都会悄悄变成 1，点到别的控件上
        if isinstance(eid, bool) or not isinstance(eid, int):
            try:
                eid = int(str(eid).strip())
            except (TypeError, ValueError):
                raise ValueError(f"元素编号不是整数：{eid!r}") from None
        match = next((e for e in state.elements if e.id == eid), None)
        if match is None:
            raise ValueError(f"元素编号 {eid} 不在当前屏幕的识别结果里")
        raw["point"] = match.center()

    for k in ("point", "point2"):
        if isinstance(raw.get(k), list):
            raw[k] = tuple(raw[k])
        if raw.get(k) is not None:
            raw[k] = _to_norm(raw[k], model_size)

    try:
        return thought, Action.from_dict(raw)
    except TypeError as e:
        # 缺 type 字段时 from_dict 抛的是 TypeError，统一成 ValueError
        raise ValueError(f"动作字段不完整：{raw!r}（{e}）") from None


class Agent:
    """跑一条任务的完整循环。"""

    def __init__(
        self,
        perception,
        controller: Controller,
        vlm,
        max_steps: int = MAX_STEPS,
        repeat_limit: int = REPEAT_LIMIT,
        shot_dir: Optional[str] = None,
        plan: bool = False,
        reflect_every: int = 1,
        max_replans: int = 2,
    ) -> None:
        self.perception = perception
        self.controller = controller
        self.vlm = vlm
        self.max_steps = max_steps
        self.repeat_limit = repeat_limit
        self.shot_dir = shot_dir  # 给了就每步存一张截图，微调样本要有配对的图
        # 先把任务拆成子任务再逐个执行。默认关：v1.0 的基线是在单步循环上测的，
        # 默认打开会让基线描述的不再是默认配置。规划的效果单独作为一组对照来测。
        self.plan = plan
        self.reflect_every = reflect_every  # 每几步判一次当前子任务的状态
        self.max_replans = max_replans      # 重新拆解的次数上限，防止来回打转
        self._chain = None
        self._chain_vlm = None

    @property
    def chain(self):
        """提示词 -> 模型 -> 解析 的 LangChain 链。

        按当前 self.vlm 组装并缓存。不在 __init__ 里定死：换模型时链要跟着换，
        否则 self.vlm 和链里的模型会指向两个对象。
        """
        if self._chain is None or self._chain_vlm is not self.vlm:
            self._chain = build_chain(self.vlm, model_size_of=self._model_size)
            self._chain_vlm = self.vlm
        return self._chain

    def _model_size(self, image):
        """模型实际看到的尺寸，用来把它回的像素坐标换算成归一化坐标。"""
        if not hasattr(self.vlm, "resized_size"):
            return None
        h, w = image.shape[:2]
        rh, rw = self.vlm.resized_size(h, w)
        return rw, rh

    def _shot_path(self, traj: Trajectory) -> Optional[str]:
        """本步截图存到哪。

        文件名带上这条轨迹的开始时间：同一个任务跑多次（`--repeat`）时，只用
        任务 id 加步号会让后一次把前一次的图覆盖掉，之前的轨迹就没有配对的图了。
        """
        if not self.shot_dir:
            return None
        stem = re.sub(r"[^\w.-]", "_", traj.task_id)[:32]  # task_id 来自指令，未必能当文件名
        run = f"{int(traj.started_at * 1000) % 10**9:09d}"
        return str(Path(self.shot_dir) / f"{stem}_{run}_{traj.n_steps:02d}.png")

    def run(self, instruction: str, task_id: str = "") -> Trajectory:
        traj = Trajectory(task_id=task_id or instruction[:24], instruction=instruction)
        last_state = ScreenState(width=0, height=0)
        planner = Planner(self.vlm) if self.plan else None
        planned = False  # 第一次感知拿到屏幕后才能拆解

        for _ in range(self.max_steps):
            t0 = time.perf_counter()
            try:
                state, model_img = self.perception.perceive(save_to=self._shot_path(traj))
                last_state = state

                if planner is not None and not planned:
                    # 拆不出子任务就退回单步循环，不让规划这一步卡死整条任务
                    planner.plan(model_img, instruction, state)
                    planned = True
                    traj.subtasks = list(planner.subtasks)

                thought, action = self.chain.invoke({
                    "instruction": acting_instruction(
                        instruction, planner.current() if planner else None),
                    "state": state,
                    "steps": traj.steps,
                    "image": model_img,
                })
            except Exception as e:
                # 截图、推理、解析任一环节出错都只废掉这一步，已经跑出来的轨迹要留住
                if is_failsafe(e):
                    raise
                msg = f"{type(e).__name__}: {e}"
                traj.steps.append(
                    Step(last_state, Action("call_user", thought=msg),
                         ok=False, error=msg, elapsed=time.perf_counter() - t0)
                )
                traj.success = False
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

            # Reflecting：看执行后的屏幕，判断当前子任务完成没有
            if planner is not None and planner.current() is not None \
                    and traj.n_steps % self.reflect_every == 0:
                try:
                    _, after_img = self.perception.perceive(run_ocr=False)
                    situation, _ = planner.reflect(after_img, instruction, traj.steps)
                except Exception as e:
                    if is_failsafe(e):
                        raise
                    situation = RETRY  # 反思失败不该中断任务，当作要重试
                traj.reflections.append(situation)

                if situation == SUCCESS:
                    # 只推进子任务，不据此判定整条任务成功。实测反思会连续给出
                    # sub_task_success 而程序验收判定失败，照它退出等于提前终止。
                    # 子任务走完后 current() 返回 None，退回按整体任务继续，
                    # 由模型自己给 finished 或走到步数上限。
                    planner.advance()
                elif situation == REFORMULATE and planner.replans < self.max_replans:
                    planner.replans += 1
                    try:
                        # 按执行后的屏幕重拆。need_reformulate 的意思就是「看到现在
                        # 的情况，原计划走不通」，拿动作执行前的屏幕重拆没有意义。
                        # 这里要带 OCR：拆解的提示词里有元素清单。
                        fresh, fresh_img = self.perception.perceive()
                        planner.plan(fresh_img, instruction, fresh)
                        traj.subtasks = list(planner.subtasks)
                    except Exception as e:
                        if is_failsafe(e):
                            raise
                        # 重拆失败就保持原计划往下走，不该废掉整条轨迹

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
