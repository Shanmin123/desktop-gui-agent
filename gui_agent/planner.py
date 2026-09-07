"""任务拆解与规划、子任务反思。

对应大纲第 3 周第 3 项，以及第 4 周闭环里的「任务规划」一环。

按 ScreenAgent 的三段结构（arXiv:2402.07945）：

  Planning    把用户任务拆成一串子任务
  Acting      针对当前子任务输出动作，由 agent.py 的循环执行
  Reflecting  看执行后的屏幕，判断当前子任务是完成了、要重试、还是得重新拆

反思的三种判定沿用 ScreenAgent 数据集里 EvaluateSubTaskAction 的取值：
`sub_task_success`、`need_retry`、`need_reformulate`，这样第 3 周的微调样本
和运行时的输出是同一套词。

规划和反思各多花一次模型调用。规划整条任务只做一次；反思默认每步一次，
可以调 `reflect_every` 降低开销。
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from .schema import ScreenState, Step

MAX_SUBTASKS = 8

SUCCESS, RETRY, REFORMULATE = "sub_task_success", "need_retry", "need_reformulate"
SITUATIONS = (SUCCESS, RETRY, REFORMULATE)

PLAN_TEMPLATE = PromptTemplate.from_template(
    """你在操作一台 Windows 电脑。把下面的任务拆成几个按顺序执行的子任务。

任务：{instruction}

当前屏幕上的文字元素：
{elements}

只返回一个 JSON 数组，不要有别的内容，每项是一句话：
["打开开始菜单", "点击文件资源管理器图标"]

拆到能一步步点出来为止，最多 {max_subtasks} 个。子任务描述要具体到界面上的控件，
不要写「完成任务」这种没有操作对应的句子。"""
)

REFLECT_TEMPLATE = PromptTemplate.from_template(
    """你在操作一台 Windows 电脑，正在完成任务「{instruction}」。

当前子任务：{subtask}

刚刚执行的动作：
{history}

这是执行后的屏幕。判断当前子任务的状态，只返回一个 JSON 对象：
{{"situation": "sub_task_success", "advice": "下一步的建议，可省略"}}

situation 三选一：
  sub_task_success   子任务已完成，可以进入下一个
  need_retry         没完成，但方向对，再试一次
  need_reformulate   这个拆法走不通，需要重新拆解"""
)


def parse_plan(text: str, limit: int = MAX_SUBTASKS) -> List[str]:
    """从模型输出里取出子任务清单，取不到返回空列表。"""
    from .agent import _extract_json

    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    out = [str(x).strip() for x in data if isinstance(x, (str, int, float))]
                    return [s for s in out if s][:limit]
        start = text.find("[", start + 1)

    # 有的模型会包一层对象，如 {"plan": [...]}
    data = _extract_json(text)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                out = [str(x).strip() for x in v if isinstance(x, (str, int, float))]
                return [s for s in out if s][:limit]
    return []


def parse_reflection(text: str) -> Tuple[str, str]:
    """解析反思结果，返回 (situation, advice)。解析不出就当作要重试。"""
    from .agent import _extract_json

    data = _extract_json(text)
    if isinstance(data, dict):
        s = str(data.get("situation", "")).strip()
        if s in SITUATIONS:
            return s, str(data.get("advice", ""))
    for s in SITUATIONS:  # 模型可能只回了个裸词
        if s in text:
            return s, ""
    return RETRY, ""


def acting_instruction(task: str, subtask: Optional[str]) -> str:
    """执行阶段送进提示词的任务描述。带上整体目标，避免只盯着子任务跑偏。"""
    return task if not subtask else f"{task}\n当前子任务：{subtask}"


class Planner:
    """按顺序推进一串子任务。"""

    def __init__(self, vlm, max_subtasks: int = MAX_SUBTASKS) -> None:
        self.vlm = vlm
        self.max_subtasks = max_subtasks
        self.subtasks: List[str] = []
        self.index = 0
        self.replans = 0

    # -- Planning ------------------------------------------------------------

    def plan(self, image, instruction: str, state: ScreenState) -> List[str]:
        """拆解任务。拆不出来返回空列表，上层退回单步循环。"""
        from .agent import format_elements

        prompt = PLAN_TEMPLATE.format(
            instruction=instruction,
            elements=format_elements(state),
            max_subtasks=self.max_subtasks,
        )
        self.subtasks = parse_plan(self.vlm.ask(image, prompt), self.max_subtasks)
        self.index = 0
        return self.subtasks

    # -- 当前进度 ------------------------------------------------------------

    def current(self) -> Optional[str]:
        if 0 <= self.index < len(self.subtasks):
            return self.subtasks[self.index]
        return None

    def done(self) -> bool:
        """有过拆解且已经走完全部子任务。"""
        return bool(self.subtasks) and self.index >= len(self.subtasks)

    def advance(self) -> None:
        self.index += 1

    # -- Reflecting ----------------------------------------------------------

    def reflect(self, image, instruction: str, steps: List[Step]) -> Tuple[str, str]:
        """判断当前子任务的状态。"""
        from .agent import format_history

        subtask = self.current()
        if subtask is None:
            return SUCCESS, ""
        prompt = REFLECT_TEMPLATE.format(
            instruction=instruction, subtask=subtask, history=format_history(steps, limit=3)
        )
        return parse_reflection(self.vlm.ask(image, prompt))
