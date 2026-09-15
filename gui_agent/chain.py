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


# --- 提示词变体（大纲第 5 周第 4 项）----------------------------------------
#
# 第 2 周的失败归因：模型 111/120 次用 element 编号指位置，而 OCR 只认文字，
# 图标没有编号可指，于是点在错的地方。下面每个变体针对一个假设，用
# scripts/tune_prompt.py 在 ScreenAgent 测试划分上比，不靠感觉挑。

_POSITION_RULE = """指定位置时优先用 element 编号。列表里没有对应元素时，用 point 给归一化坐标，
形如 "point": [0.5, 0.5]，取值 0 到 1。"""

_POINT_FIRST_RULE = """指定位置用 point 给归一化坐标，形如 "point": [0.5, 0.5]，取值 0 到 1，
以截图左上角为 (0, 0)、右下角为 (1, 1)。
只有当你要操作的正好是列表里那段文字本身时，才用 element 编号。
图标、按钮、输入框这些没有文字，列表里不会有，必须给 point。"""

_KEYBOARD_RULE = _POSITION_RULE + """

能用快捷键完成的就别去点菜单：保存 ctrl+s、全选 ctrl+a、复制粘贴 ctrl+c / ctrl+v、
新建 ctrl+n、关闭 ctrl+w。要输入文字用 type，不要一个字一个字点。"""

_CLICK_RULE = _POSITION_RULE + """

绝大多数控件是单击：按钮、菜单、标签页、输入框、工具栏图标、链接，都用 click。
只有打开文件、文件夹、桌面图标才用 left_double。不确定就用 click。"""

_EXAMPLES = """
两个例子（只是示范格式，和当前任务无关）：
任务「保存文件」，屏幕上没有可见的保存按钮 ->
{{"thought": "记事本用 ctrl+s 保存最快", "action": {{"type": "hotkey", "text": "ctrl+s"}}}}
任务「点左上角的返回箭头」，箭头是图标、元素列表里没有 ->
{{"thought": "箭头在左上角，估计在 (0.03, 0.06)", "action": {{"type": "click", "point": [0.03, 0.06]}}}}
"""


def _variant(old: str, new: str) -> str:
    out = TEMPLATE.replace(old, new)
    if out == TEMPLATE:
        raise AssertionError("提示词变体没替换成功，模板改过了就要同步改这里")
    return out


PROMPT_VARIANTS = {
    "base": TEMPLATE,
    "point_first": _variant(_POSITION_RULE, _POINT_FIRST_RULE),
    "keyboard": _variant(_POSITION_RULE, _KEYBOARD_RULE),
    # 第 3 周复杂任务实测：4 条任务 56 步里 47 步是 left_double，模型一律双击。
    # 离线也有：ScreenAgent 353 条里 click 被答成 left_double 33 次。
    "click_prior": _variant(_POSITION_RULE, _CLICK_RULE),
    "few_shot": _variant("\n任务：{instruction}", _EXAMPLES + "\n任务：{instruction}"),
}


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


def render_prompt(instruction: str, state: ScreenState, steps: List[Step],
                  variant: str = "base", elements_limit: Optional[int] = None) -> str:
    """套用模板生成一步的提示词。

    variant 选提示词变体，见 PROMPT_VARIANTS。elements_limit 限制元素清单的条数，
    不给就用 agent.MAX_ELEMENTS；构建微调样本时用它把超长样本压进长度预算。
    """
    from .agent import MAX_ELEMENTS, format_elements, format_history

    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"没有 {variant!r} 这个提示词变体，可选 {list(PROMPT_VARIANTS)}")
    prompt = GUI_PROMPT if variant == "base" else \
        PromptTemplate.from_template(PROMPT_VARIANTS[variant])
    return prompt.format(
        instruction=instruction,
        history=format_history(steps),
        elements=format_elements(state, MAX_ELEMENTS if elements_limit is None
                                else elements_limit),
    )


TARGET_TEMPLATE = """你在操作一台 Windows 电脑，目标是完成用户给的任务。

每一步会给你当前屏幕截图。你要输出下一步动作。

可用动作：
  click / left_double / right_single  点击，需要 target
  scroll                              滚动，需要 target 和 direction（up/down/left/right）
  type                                输入文本，需要 text
  hotkey                              组合键，如 "ctrl+s"，需要 text
  wait                                等待界面变化
  finished                            任务已完成
  call_user                           无法继续，需要人工介入

只返回一个 JSON 对象，不要有别的内容：
{{"thought": "为什么这么做", "action": {{"type": "click", "target": "要点的控件"}}}}

任务：{instruction}

已执行：
{history}

target 写界面上那个控件本身，比如「保存按钮」「地址栏」「左上角的关闭图标」。
不要写坐标，也不要写编号，位置由另一步解析。"""

TARGET_PROMPT = PromptTemplate.from_template(TARGET_TEMPLATE)

# --- 两段式的提示词变体（大纲第 5 周第 4 项；交付路径是两段式）------------------
#
# 对着一段式那几个变体的同样假设：键盘动作用得太少、单击被答成双击、给例子能不能稳住
# 格式。都只改「只问点什么」的第一问，定位提示词不变。

_TARGET_RULE = """target 写界面上那个控件本身，比如「保存按钮」「地址栏」「左上角的关闭图标」。
不要写坐标，也不要写编号，位置由另一步解析。"""

_TARGET_KEYBOARD_RULE = _TARGET_RULE + """

能用快捷键完成的就别去点菜单：保存 ctrl+s、全选 ctrl+a、复制粘贴 ctrl+c / ctrl+v、
新建 ctrl+n、关闭 ctrl+w。要输入文字用 type，不要一个字一个字点。"""

_TARGET_CLICK_RULE = _TARGET_RULE + """

绝大多数控件是单击：按钮、菜单、标签页、输入框、工具栏图标、链接，都用 click。
只有打开文件、文件夹、桌面图标才用 left_double。不确定就用 click。"""

_TARGET_EXAMPLES = """
两个例子（只是示范格式，和当前任务无关）：
任务「保存文件」，屏幕上没有可见的保存按钮 ->
{{"thought": "记事本用 ctrl+s 保存最快", "action": {{"type": "hotkey", "text": "ctrl+s"}}}}
任务「关掉这个窗口」 ->
{{"thought": "窗口右上角有关闭按钮", "action": {{"type": "click", "target": "窗口右上角的关闭按钮"}}}}
"""


def _target_variant(old: str, new: str) -> str:
    out = TARGET_TEMPLATE.replace(old, new)
    if out == TARGET_TEMPLATE:
        raise AssertionError("两段式提示词变体没替换成功，模板改过了就要同步改这里")
    return out


TARGET_VARIANTS = {
    "base": TARGET_TEMPLATE,
    "keyboard": _target_variant(_TARGET_RULE, _TARGET_KEYBOARD_RULE),
    "click_prior": _target_variant(_TARGET_RULE, _TARGET_CLICK_RULE),
    "few_shot": _target_variant("任务：{instruction}", _TARGET_EXAMPLES + "任务：{instruction}"),
}

# 带元素清单的变体。两段式的失败样例里，错的主要不是定位，是模型报出来的控件名
# 本身不对——它凭印象写，写出 "Firefox W..." 这种截断的、屏幕上并不存在的字符串，
# 第二段自然定位不到。把 OCR 认出来的文字列出来让它照抄，比让它自己想要稳。
# 编号只是为了让清单可读，回答里仍然写文字本身，不写编号（编号解析不了图标）。
_ELEMENTS_RULE = """target 优先照抄下面清单里的原文，一个字都不要改。
清单里没有的（图标、没有文字的按钮）再自己描述，比如「左上角的关闭图标」。
不要写坐标，也不要写编号，位置由另一步解析。

当前屏幕上的文字元素：
{elements}"""

TARGET_WITH_ELEMENTS_TEMPLATE = TARGET_TEMPLATE.replace(
    """target 写界面上那个控件本身，比如「保存按钮」「地址栏」「左上角的关闭图标」。
不要写坐标，也不要写编号，位置由另一步解析。""", _ELEMENTS_RULE)
assert TARGET_WITH_ELEMENTS_TEMPLATE != TARGET_TEMPLATE, "模板改过了就要同步改这里"

TARGET_WITH_ELEMENTS_PROMPT = PromptTemplate.from_template(TARGET_WITH_ELEMENTS_TEMPLATE)

NEEDS_TARGET = ("click", "left_double", "right_single", "scroll")


def render_target_prompt(instruction: str, steps: List[Step],
                         state: Optional[ScreenState] = None,
                         elements_limit: Optional[int] = None,
                         variant: str = "base") -> str:
    """两段式里第一段的提示词：只问点什么，不问点哪。

    给了 state 就把 OCR 元素清单一并列出来，让模型照抄原文当 target；不给就是
    原来那份，模型全凭截图自己写名字。
    """
    from .agent import MAX_ELEMENTS, format_elements, format_history

    if variant not in TARGET_VARIANTS:
        raise ValueError(f"没有 {variant!r} 这个两段式提示词变体，可选 {list(TARGET_VARIANTS)}")
    if state is None:
        prompt = (TARGET_PROMPT if variant == "base"
                  else PromptTemplate.from_template(TARGET_VARIANTS[variant]))
        return prompt.format(instruction=instruction, history=format_history(steps))
    return TARGET_WITH_ELEMENTS_PROMPT.format(
        instruction=instruction, history=format_history(steps),
        elements=format_elements(state, MAX_ELEMENTS if elements_limit is None
                                 else elements_limit))


def build_chain(vlm, model_size_of=None, locate_target: bool = False) -> Runnable:
    """组装 提示词 -> 模型 -> 解析 的链。

    `model_size_of(image) -> (w, h)` 给出模型实际看到的尺寸，交给解析器换算坐标；
    不给就按提示词要求的归一化坐标处理。

    locate_target=True 走两段式：第一段只让模型说要操作哪个控件（target 描述），
    第二段用 `vlm.locate` 的定位提示词把描述解析成坐标。

    实测 120 条 ScreenSpot 样本：当前一段式命中 40.8%，两段式 62.5%，直接问
    坐标（已知目标控件）70.0%。一段式吃亏在模型 111/120 次都用 OCR 元素编号
    指位置，而 OCR 只认文字，没有文字的图标就没有编号可指。
    """

    def render(inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (render_target_prompt(inputs["instruction"], inputs["steps"])
                  if locate_target else
                  render_prompt(inputs["instruction"], inputs["state"], inputs["steps"]))
        return {**inputs, "prompt": prompt}

    def ask(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {**inputs, "text": vlm.ask(inputs["image"], inputs["prompt"])}

    def parse(inputs: Dict[str, Any]) -> Tuple[str, Action]:
        size = model_size_of(inputs["image"]) if model_size_of else None
        if not locate_target:
            return ActionOutputParser(state=inputs["state"], model_size=size).parse(inputs["text"])
        return parse_with_target(inputs["text"], vlm, inputs["image"], inputs["state"], size)

    return RunnableLambda(render) | RunnableLambda(ask) | RunnableLambda(parse)


def parse_with_target(text: str, vlm, image, state: ScreenState,
                      model_size=None) -> Tuple[str, Action]:
    """解析两段式的输出，把 target 描述换成坐标。"""
    from .agent import _extract_json, parse_step
    from .schema import ACTION_ALIAS

    data = _extract_json(text)
    raw = data.get("action") if isinstance(data, dict) else None
    target = raw.get("target") if isinstance(raw, dict) else None

    # 先按别名归一再判断要不要定位：Qwen2.5 基座写的是 left_click，不归一的话带着 target
    # 也会走进下面「不需要位置」的分支、报缺 point（09-16 重跑基座两段式，353 条里 111 条这样错）
    kind = str(raw.get("type", "click")) if isinstance(raw, dict) else "click"
    kind = ACTION_ALIAS.get(kind, kind)
    if (kind in ("type", "hotkey") and isinstance(target, str) and target.strip()
            and not str(raw.get("text") or "").strip()):
        # 键盘动作把内容写进了 target：lora_2sp 离线 353 条里缺 text 的 6 条全是这样
        # （Control_L+a、Tab、迈腾……），端到端 5 个任务里有 9 步因此失败、重试还连着犯。
        # 两段式样本 45% 带 target，把键名带偏了；训练数据里 160 条键盘动作一条都没带
        # target。内容本身是对的，就当 text 用——仍然不去定位、不挂坐标。
        rest = {k: v for k, v in raw.items() if k not in ("target", "element", "point", "point2")}
        rest["text"] = target.strip()
        return str(data.get("thought", "")), Action.from_dict({**rest, "type": kind})
    if not (isinstance(target, str) and target.strip()) or kind not in NEEDS_TARGET:
        # 没给 target，或者给了但这个动作根本不需要位置（type / hotkey 这些），
        # 就按一段式那套再解析一次。后一种是实测出来的：第一问里列了元素清单之后，
        # 模型会把 hotkey 也写成 {"type": "hotkey", "target": "..."}，
        # 再往上套一个 point 只会把错误盖住，让它按原样报缺 text 更清楚。
        return parse_step(text, state, model_size)

    point = vlm.locate(image, target.strip())
    if point is None:
        raise ValueError(f"定位不到「{target.strip()}」")

    rest = {k: v for k, v in raw.items() if k not in ("target", "element", "point")}
    rest["point"] = point
    return str(data.get("thought", "")), Action.from_dict({**rest, "type": kind})
