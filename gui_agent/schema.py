"""三个核心数据格式：屏幕识别结果、动作、执行记录。

动作空间沿用 UI-TARS 的桌面子集（arXiv:2501.12326 Table 1），坐标同样用按屏幕
尺寸归一化的相对值。这样第 3 周用公开数据微调时不必再做一层格式转换。

样本的组织方式参考 ScreenAgent：每条记录都带上截图当时的分辨率，模型原始输出
和人工修正后的输出分开存，后者才用于训练。

后续感知、控制、Agent、微调数据、评估日志都依赖这里的定义，改动要谨慎。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 归一化坐标，取值 0~1，与屏幕分辨率无关
Point = Tuple[float, float]

# 动作类型，对应 UI-TARS Table 1 的桌面部分
ACTION_TYPES = (
    "click",
    "left_double",
    "right_single",
    "drag",
    "scroll",
    "type",
    "hotkey",
    "wait",
    "finished",
    "call_user",
)

# 每种动作必须提供哪些字段
_REQUIRED = {
    "click": ("point",),
    "left_double": ("point",),
    "right_single": ("point",),
    "drag": ("point", "point2"),
    "scroll": ("point", "direction"),
    "type": ("text",),
    "hotkey": ("text",),
    "wait": (),
    "finished": (),
    "call_user": (),
}

SCROLL_DIRECTIONS = ("up", "down", "left", "right")

# 模型常用别的命名给出动作，实测 Qwen2.5-VL 输出的是 left_click 而不是 click。
# 这些写法语义明确，直接归一，不必让整条任务因为叫法不同而失败。
ACTION_ALIAS = {
    "left_click": "click",
    "single_click": "click",
    "tap": "click",
    "double_click": "left_double",
    "left_double_click": "left_double",
    "right_click": "right_single",
    "type_text": "type",
    "input": "type",
    "key": "hotkey",
    "hotkeys": "hotkey",
    "press": "hotkey",
    "done": "finished",
    "finish": "finished",
    "complete": "finished",
    "ask_user": "call_user",
}


def _check_point(name: str, p: Any) -> None:
    if not (isinstance(p, (tuple, list)) and len(p) == 2):
        raise ValueError(f"{name} 应为 (x, y)，实际为 {p!r}")
    for v in p:
        if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} 的坐标应是 0~1 的归一化值，实际为 {p!r}")


# --------------------------------------------------------------------------
# 1. 屏幕识别结果
# --------------------------------------------------------------------------


@dataclass
class Element:
    """屏幕上的一个可交互元素或一段文字。"""

    id: int
    bbox: Tuple[float, float, float, float]  # 归一化 (x1, y1, x2, y2)
    text: str = ""
    source: str = "ocr"  # ocr | cv
    confidence: float = 1.0

    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class ScreenState:
    """一次屏幕感知的结果。

    width / height 记的是截图时的物理分辨率。元素坐标已归一化，所以换分辨率
    不影响下游，但保留原始尺寸便于回放和排查。
    """

    width: int
    height: int
    image_path: str = ""
    elements: List[Element] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_pixel(self, p: Point) -> Tuple[int, int]:
        """归一化坐标 -> 物理像素坐标。"""
        _check_point("point", p)
        return round(p[0] * self.width), round(p[1] * self.height)

    def to_norm(self, x: int, y: int) -> Point:
        """物理像素坐标 -> 归一化坐标。"""
        return x / self.width, y / self.height


# --------------------------------------------------------------------------
# 2. 动作
# --------------------------------------------------------------------------


@dataclass
class Action:
    """Agent 输出的一个动作。

    thought 是模型给出的理由，UI-TARS 每步同时输出 Thought 和 Action，这里保持
    一致，后面做失败归类和微调数据都要用。
    """

    type: str
    point: Optional[Point] = None  # click / left_double / right_single / scroll / drag 起点
    point2: Optional[Point] = None  # drag 终点
    text: Optional[str] = None  # type 的内容，或 hotkey 的组合键
    direction: Optional[str] = None  # scroll 方向
    thought: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.type = ACTION_ALIAS.get(self.type, self.type)
        if self.type not in ACTION_TYPES:
            raise ValueError(f"未知动作类型 {self.type!r}，可选：{ACTION_TYPES}")
        for name in _REQUIRED[self.type]:
            if getattr(self, name) is None:
                raise ValueError(f"动作 {self.type} 缺少必需字段 {name}")
        for name in ("point", "point2"):
            p = getattr(self, name)
            if p is not None:
                _check_point(name, p)
                setattr(self, name, (float(p[0]), float(p[1])))
        if self.type == "scroll" and self.direction not in SCROLL_DIRECTIONS:
            raise ValueError(f"scroll 方向应为 {SCROLL_DIRECTIONS}，实际为 {self.direction!r}")

    def to_dict(self) -> Dict[str, Any]:
        """只保留有值的字段，日志和训练数据都更干净。"""
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        allowed = {"type", "point", "point2", "text", "direction", "thought"}
        kw = {k: v for k, v in d.items() if k in allowed}
        for name in ("point", "point2"):
            if isinstance(kw.get(name), list):
                kw[name] = tuple(kw[name])
        return cls(**kw)

    def is_terminal(self) -> bool:
        """任务是否到此结束。"""
        return self.type in ("finished", "call_user")


# --------------------------------------------------------------------------
# 3. 执行记录
# --------------------------------------------------------------------------


@dataclass
class Step:
    """一步 = 看到什么 + 做了什么 + 结果如何。"""

    screen: ScreenState
    action: Action
    ok: bool = True
    error: str = ""
    elapsed: float = 0.0


@dataclass
class Trajectory:
    """一次任务的完整执行记录。

    既是运行日志，也是第 3 周的微调样本、第 4 周的评估记录。
    """

    task_id: str
    instruction: str
    steps: List[Step] = field(default_factory=list)
    success: Optional[bool] = None  # None 表示还没判定
    started_at: float = field(default_factory=time.time)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def wall_time(self) -> float:
        return sum(s.elapsed for s in self.steps)

    def to_json(self, indent: int = 2) -> str:
        data = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "success": self.success,
            "n_steps": self.n_steps,
            "wall_time": round(self.wall_time, 3),
            "started_at": self.started_at,
            "steps": [
                {
                    "screen": {
                        "width": s.screen.width,
                        "height": s.screen.height,
                        "image_path": s.screen.image_path,
                        "n_elements": len(s.screen.elements),
                        "timestamp": s.screen.timestamp,
                    },
                    "action": s.action.to_dict(),
                    "ok": s.ok,
                    "error": s.error,
                    "elapsed": round(s.elapsed, 3),
                }
                for s in self.steps
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=indent)
