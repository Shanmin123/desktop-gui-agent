"""用 LangChain 组装 Agent 的提示词与输出解析。

对应大纲第 3 周第 2 项「基于 LangChain 搭建基础多模态 Agent 框架」，
技术栈里的「提示词工程：LangChain Prompt Templates」也落在这里。

一步的链路是 提示词模板 -> 多模态模型 -> 动作解析：

    chain = build_chain(vlm)
    thought, action = chain.invoke({
        "instruction": "打开浏览器", "state": state, "steps": [], "image": img,
    })

模型这一环不是 LangChain 的 ChatModel：本项目的主干是本地 Qwen2.5-VL，每步要送
一张 numpy 截图，走 `vlm.ask(image, prompt)`，用 RunnableLambda 包成链上的一环。
这样提示词模板、解析器、链式组装都还是 LangChain 的，模型换成 API 后端也不用改链。

agent.py 的 build_prompt / parse_step 仍是这里的薄封装，两边共用同一个模板。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .schema import Action, ScreenState, Step

# 模板里有 JSON 例子，花括号要写成双份，否则会被当成变量占位符
TEMPLATE = """你在操作一台 Windows 电脑，目标是完成用户给的任务。

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
{{"thought": "为什么这么做", "action": {{"type": "click", "element": 12}}}}

指定位置时优先用 element 编号。列表里没有对应元素时，用 point 给归一化坐标，
形如 "point": [0.5, 0.5]，取值 0 到 1。

任务：{instruction}

已执行：
{history}

当前屏幕上的文字元素：
{elements}

下一步动作："""

GUI_PROMPT = PromptTemplate.from_template(TEMPLATE)


class ActionOutputParser(BaseOutputParser):
    """把模型输出解析成 (thought, Action)。

    element 编号要查当前屏幕的识别结果才能换成坐标，所以解析器带屏幕状态；
    model_size 用于把模型回的像素坐标换算成归一化坐标。
    """

    state: ScreenState
    model_size: Optional[Tuple[int, int]] = None

    def parse(self, text: str) -> Tuple[str, Action]:
        from .agent import parse_step  # 延迟导入，避免和 agent.py 循环引用

        return parse_step(text, self.state, self.model_size)

    @property
    def _type(self) -> str:
        return "gui_action"


def render_prompt(instruction: str, state: ScreenState, steps: List[Step]) -> str:
    """套用模板生成一步的提示词。"""
    from .agent import format_elements, format_history

    return GUI_PROMPT.format(
        instruction=instruction,
        history=format_history(steps),
        elements=format_elements(state),
    )


def build_chain(vlm, model_size_of=None) -> Runnable:
    """组装 提示词 -> 模型 -> 解析 的链。

    `model_size_of(image) -> (w, h)` 给出模型实际看到的尺寸，交给解析器换算坐标；
    不给就按提示词要求的归一化坐标处理。
    """

    def render(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **inputs,
            "prompt": render_prompt(inputs["instruction"], inputs["state"], inputs["steps"]),
        }

    def ask(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {**inputs, "text": vlm.ask(inputs["image"], inputs["prompt"])}

    def parse(inputs: Dict[str, Any]) -> Tuple[str, Action]:
        size = model_size_of(inputs["image"]) if model_size_of else None
        return ActionOutputParser(state=inputs["state"], model_size=size).parse(inputs["text"])

    return RunnableLambda(render) | RunnableLambda(ask) | RunnableLambda(parse)
