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

# 例子必须写明是另一个任务的。实测直接给一个裸数组当格式示例时，模型会把示例
# 原样抄回来：五个任务里四个都回了示例里那两条，包括「关闭记事本」。
# ScreenAgent 的提示词也是这么区分的（先说「我的任务是搜索麦田怪圈」再给拆解）。
PLAN_TEMPLATE = PromptTemplate.from_template(
    """你在操作一台 Windows 电脑，要把用户的任务拆成几个按顺序执行的子任务。

先看一个**别的**任务的例子。如果任务是「上网查一下冯诺依曼」，拆解结果是：
["打开浏览器", "点击地址栏", "输入冯诺依曼并回车", "点开第一条搜索结果"]

下面才是你要拆的任务，和上面的例子无关。

任务：{instruction}

当前屏幕上的文字元素：
{elements}

针对上面这个任务，只返回一个 JSON 数组，不要有别的内容，每项一句话，
最多 {max_subtasks} 个。子任务描述要具体到界面上的控件，不要写「完成任务」
这种没有对应操作的句子。"""
)

REFLECT_TEMPLATE = PromptTemplate.from_template(
    """你在操作一台 Windows 电脑，正在完成任务「{instruction}」。

当前子任务：{subtask}

刚刚执行的动作：
{history}

这是执行后的屏幕。判断当前子任务的状态，只返回一个 JSON 对象：
{{"situation": "<下面三个之一>", "advice": "下一步的建议，可省略"}}

situation 三选一：
  sub_task_success   子任务已完成，可以进入下一个
  need_retry         没完成，但方向对，再试一次
  need_reformulate   这个拆法走不通，需要重新拆解"""
)


# 子任务可能被包在对象里。ScreenAgent 数据集用的是 element，实测 Qwen2.5-VL
# 回的是 action，两种都收。
_SUBTASK_KEYS = ("element", "action", "subtask", "step", "description", "task", "content")


def _subtask_text(item) -> str:
    """一项子任务可能是字符串，也可能是包着它的对象。"""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (int, float)):
        return str(item)
    if isinstance(item, dict):
        for k in _SUBTASK_KEYS:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # 只有一个字符串值时就是它，键叫什么无所谓
        vals = [v for v in item.values() if isinstance(v, str) and v.strip()]
        if len(vals) == 1:
            return vals[0].strip()
    return ""


def _from_list(data, limit: int) -> List[str]:
    out = [_subtask_text(x) for x in data]
    return [s for s in out if s][:limit]


def parse_plan(text: str, limit: int = MAX_SUBTASKS) -> List[str]:
    """从模型输出里取出子任务清单，取不到返回空列表。

    实测 Qwen2.5-VL 不按提示词要求回字符串数组，回的是
    `[{"action": "打开开始菜单"}, ...]`，所以对象和字符串两种都要认。
    """
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
                    got = _from_list(data, limit)
                    if got:
                        return got
                    break
        start = text.find("[", start + 1)

    # 有的模型会包一层对象，如 {"plan": [...]}
    data = _extract_json(text)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                got = _from_list(v, limit)
                if got:
                    return got
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
        """拆解任务。

        拆不出来时保留已有计划：重拆返回空列表就把原计划抹掉的话，后面既没有子任务
        可推进，也不会再反思，等于规划中途消失。首次拆解失败则本来就是空，上层退回
        单步循环。
        """
        from .agent import format_elements

        prompt = PLAN_TEMPLATE.format(
            instruction=instruction,
            elements=format_elements(state),
            max_subtasks=self.max_subtasks,
        )
        got = parse_plan(self.vlm.ask(image, prompt), self.max_subtasks)
        if got:
            self.subtasks = got
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
