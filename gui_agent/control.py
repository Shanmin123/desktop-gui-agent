"""桌面控制：把 Action 变成真实的鼠标键盘操作。

对应大纲第 2 周第 3 项。支持 schema.py 里定义的 10 个动作。

坐标一律是归一化的 0~1，落到像素由本模块用 `backend.size()` 换算。这一点顺带
免疫了 DPI 缩放问题：不管截图端报告多大、控制端报告多大，两边各自按自己的尺寸
做归一化和反归一化，点击位置都是对的。

真正操作桌面的部分收在 PyAutoGUIBackend 里，Controller 只做换算、校验和记录。
测试注入一个假 backend，就能在不碰真实鼠标的前提下验证全部逻辑。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .schema import Action

# 会锁屏或打断会话的组合键，执行会直接被拒。alt+f4 不在其中，因为大纲第 4 周的
# 基础任务里就有「关闭应用」。
BLOCKED_HOTKEYS = frozenset({"ctrl+alt+delete", "ctrl+alt+del", "win+l", "meta+l"})

# 明显是破坏性命令的输入内容，命中即拒。对应大纲「合规与落地说明」第 2 条。
#
# 必须锚定在开头并且要求命令的形态，不能用子串匹配：最初写成 "format " 时，
# "format the paragraph" 这种正常英文也被拦了，单元测试当场抓到。
#
# 这道防线本身是不可靠的，因为同一段文字打进 Word 无害、打进 cmd 危险，光看
# 文本判断不出上下文。它只能挡住最明显的情况，真正的保障是 dry_run、FAILSAFE，
# 以及把测试放在虚拟机里跑。
BLOCKED_TEXT = (
    r"^format\s+[a-z]:",       # format c:
    r"^rm\s+-[rf]{1,2}\b",     # rm -rf
    r"^del\s+/[fsq]\b",        # del /f /s /q
    r"^rd\s+/s\b",             # rd /s
    r"^shutdown\b",
    r"^diskpart\b",
    r"^mkfs\b",
)

# pyautogui 认识的键名和常见写法的对应
_KEY_ALIAS = {
    "control": "ctrl",
    "cmd": "win",
    "command": "win",
    "super": "win",
    "meta": "win",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "pgup": "pageup",
    "pgdn": "pagedown",
}


@dataclass
class ActionResult:
    ok: bool
    error: str = ""
    elapsed: float = 0.0


def normalize_hotkey(text: str) -> List[str]:
    """把 'Ctrl+S' 这类写法拆成 pyautogui 认识的键名列表。"""
    keys = [k.strip().lower() for k in text.replace(" ", "").split("+") if k.strip()]
    if not keys:
        raise ValueError(f"组合键为空：{text!r}")
    return [_KEY_ALIAS.get(k, k) for k in keys]


class PyAutoGUIBackend:
    """真正操作桌面的那一层，别的地方不要直接 import pyautogui。"""

    def __init__(self, move_duration: float = 0.1, type_interval: float = 0.01) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True  # 鼠标甩到左上角可强制中断
        self._pg = pyautogui
        self.move_duration = move_duration
        self.type_interval = type_interval

    def size(self) -> Tuple[int, int]:
        return tuple(self._pg.size())

    def move(self, x: int, y: int) -> None:
        """只移动不点击。不是 Action，是联调和校准用的原语。"""
        self._pg.moveTo(x, y, duration=self.move_duration)

    def position(self) -> Tuple[int, int]:
        """读回光标当前位置，用来验证移动是否落到了预期像素。"""
        return tuple(self._pg.position())

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self._pg.click(x, y, button=button, clicks=clicks, duration=self.move_duration)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float) -> None:
        self._pg.moveTo(x1, y1, duration=self.move_duration)
        self._pg.dragTo(x2, y2, duration=duration, button="left")

    def scroll(self, x: int, y: int, direction: str, clicks: int) -> None:
        self._pg.moveTo(x, y, duration=self.move_duration)
        if direction == "up":
            self._pg.scroll(clicks)
        elif direction == "down":
            self._pg.scroll(-clicks)
        elif direction == "right":
            self._pg.hscroll(clicks)
        else:  # left
            self._pg.hscroll(-clicks)

    def write(self, text: str) -> None:
        self._pg.write(text, interval=self.type_interval)

    def paste(self, text: str) -> None:
        """非 ASCII 走剪贴板。

        pyautogui 的 KEYBOARD_KEYS 里只有 ASCII，write() 打不出中文，实测确认。
        ScreenAgent 也是靠剪贴板解决这个问题的。
        """
        import pyperclip

        pyperclip.copy(text)
        self._pg.hotkey("ctrl", "v")

    def hotkey(self, keys: Sequence[str]) -> None:
        self._pg.hotkey(*keys)


class RecordingBackend:
    """假 backend，只记录调用，不碰真实桌面。给测试和 dry-run 用。"""

    def __init__(self, width: int = 3840, height: int = 2160) -> None:
        self._size = (width, height)
        self.calls: List[tuple] = []

    def size(self) -> Tuple[int, int]:
        return self._size

    def move(self, x, y):
        self.calls.append(("move", x, y))
        self._pos = (x, y)

    def position(self) -> Tuple[int, int]:
        return getattr(self, "_pos", (0, 0))

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y, button, clicks))

    def drag(self, x1, y1, x2, y2, duration):
        self.calls.append(("drag", x1, y1, x2, y2))

    def scroll(self, x, y, direction, clicks):
        self.calls.append(("scroll", x, y, direction, clicks))

    def write(self, text):
        self.calls.append(("write", text))

    def paste(self, text):
        self.calls.append(("paste", text))

    def hotkey(self, keys):
        self.calls.append(("hotkey", tuple(keys)))


class Controller:
    """执行 Action。

    dry_run=True 时只记录不执行，用于调试和演示，对应大纲「合规与落地说明」。
    """

    def __init__(
        self,
        backend=None,
        dry_run: bool = False,
        scroll_clicks: int = 3,
        drag_duration: float = 0.3,
        wait_seconds: float = 0.5,
        blocked_hotkeys: frozenset = BLOCKED_HOTKEYS,
        blocked_text: Sequence[str] = BLOCKED_TEXT,
    ) -> None:
        self.dry_run = dry_run
        self.backend = backend if backend is not None else (
            RecordingBackend() if dry_run else PyAutoGUIBackend()
        )
        self.scroll_clicks = scroll_clicks
        self.drag_duration = drag_duration
        self.wait_seconds = wait_seconds
        self.blocked_hotkeys = blocked_hotkeys
        self.blocked_text = tuple(blocked_text)
        self.history: List[Tuple[Action, ActionResult]] = []

    # -- 坐标换算 -----------------------------------------------------------

    def to_pixel(self, point) -> Tuple[int, int]:
        w, h = self.backend.size()
        return round(point[0] * w), round(point[1] * h)

    # -- 安全检查 -----------------------------------------------------------

    def _reject(self, action: Action) -> Optional[str]:
        """返回拒绝理由，None 表示放行。"""
        if action.type == "hotkey":
            combo = "+".join(normalize_hotkey(action.text))
            if combo in self.blocked_hotkeys:
                return f"组合键 {combo} 在禁用名单里"
        if action.type == "type" and action.text:
            low = action.text.strip().lower()
            for pat in self.blocked_text:
                if re.search(pat, low):
                    return f"输入内容像是破坏性命令，命中规则 {pat!r}"
        return None

    # -- 执行 ---------------------------------------------------------------

    def execute(self, action: Action) -> ActionResult:
        t0 = time.perf_counter()
        try:
            reason = self._reject(action)
            if reason:
                result = ActionResult(False, reason, time.perf_counter() - t0)
            else:
                self._dispatch(action)
                result = ActionResult(True, "", time.perf_counter() - t0)
        except Exception as e:  # 单步失败不该让整条任务崩掉，交给上层决定重试
            result = ActionResult(False, f"{type(e).__name__}: {e}", time.perf_counter() - t0)

        self.history.append((action, result))
        return result

    def _dispatch(self, action: Action) -> None:
        t = action.type

        if t in ("finished", "call_user"):
            return  # 终止信号，没有对应的桌面操作

        if t == "wait":
            if not self.dry_run:
                time.sleep(self.wait_seconds)
            return

        if t == "click":
            self.backend.click(*self.to_pixel(action.point))
        elif t == "left_double":
            self.backend.click(*self.to_pixel(action.point), clicks=2)
        elif t == "right_single":
            self.backend.click(*self.to_pixel(action.point), button="right")
        elif t == "drag":
            x1, y1 = self.to_pixel(action.point)
            x2, y2 = self.to_pixel(action.point2)
            self.backend.drag(x1, y1, x2, y2, self.drag_duration)
        elif t == "scroll":
            x, y = self.to_pixel(action.point)
            self.backend.scroll(x, y, action.direction, self.scroll_clicks)
        elif t == "type":
            if action.text.isascii():
                self.backend.write(action.text)
            else:
                self.backend.paste(action.text)
        elif t == "hotkey":
            self.backend.hotkey(normalize_hotkey(action.text))
        else:
            raise ValueError(f"没有为动作 {t} 实现执行逻辑")

    def run(self, actions) -> List[ActionResult]:
        """按顺序执行一串动作，遇到终止动作或失败就停。"""
        out = []
        for a in actions:
            r = self.execute(a)
            out.append(r)
            if not r.ok or a.is_terminal():
                break
        return out
