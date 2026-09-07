"""桌面控制：把 Action 变成真实的鼠标键盘操作。

对应大纲第 2 周第 3 项。支持 schema.py 里定义的 10 个动作。

坐标为归一化的 0~1，由本模块用 `backend.size()` 换算成像素，因此不受 DPI 缩放影响。

真正操作桌面的部分收在 PyAutoGUIBackend 里，Controller 只做换算、校验和记录，
测试时注入 RecordingBackend 替换。
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .schema import Action

# 会锁屏或打断会话的组合键，执行会直接被拒。alt+f4 不在其中，因为大纲第 4 周的
# 基础任务里就有「关闭应用」。
#
# 匹配前先按 _KEY_ALIAS 归一并当作无序集合比较，所以 "ctrl+alt+del"、
# "alt+ctrl+delete" 这些写法都会命中，不必在这里逐一列出。
BLOCKED_HOTKEYS = frozenset({"ctrl+alt+delete", "win+l"})

# 形似破坏性命令的输入，命中即拒。规则锚定在命令开头，换行和 ; && || | & 之后
# 都算一个新命令的开头，逐段检查。
#
# 文本本身判断不出上下文（同一段话打进文档无害、打进命令行危险），这里只挡最
# 明显的情况，也挡不住把命令拆成几次 type 输入。主要的保障是 dry_run 和 FAILSAFE。
BLOCKED_TEXT = (
    r"^format\s+[a-z]:",       # format c:
    r"^rm\s+-[rf]{1,2}\b",     # rm -rf
    r"^del\s+/[fsq]\b",        # del /f /s /q
    r"^rd\s+/s\b",             # rd /s
    r"^shutdown\b",
    r"^diskpart\b",
    r"^mkfs\b",
)

# pyautogui 认识的键名和常见写法的对应。
#
# 左右分开的写法（winleft、ctrlright 等）也归到不带方位的那个：pyautogui 认它们，
# 不归一的话 winleft+l 一样能锁屏，却匹配不上黑名单里的 win+l。
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
    "winleft": "win",
    "winright": "win",
    "ctrlleft": "ctrl",
    "ctrlright": "ctrl",
    "altleft": "alt",
    "altright": "alt",
    "shiftleft": "shift",
    "shiftright": "shift",
    # X11 keysym 写法，ScreenAgent 数据集用的就是这套（它按 VNC 协议记录）
    "control_l": "ctrl",
    "control_r": "ctrl",
    "shift_l": "shift",
    "shift_r": "shift",
    "alt_l": "alt",
    "alt_r": "alt",
    "super_l": "win",
    "super_r": "win",
    "prior": "pageup",
    "next": "pagedown",
    "print": "printscreen",
}


@dataclass
class ActionResult:
    ok: bool
    error: str = ""
    elapsed: float = 0.0


def is_failsafe(e: BaseException) -> bool:
    """是不是 pyautogui 的 FAILSAFE 中断。

    FAILSAFE 是用户把鼠标甩到屏幕角落主动喊停，属于中断信号而不是一次失败的动作，
    被 `except Exception` 吞掉就等于没有急停。只查已经导入的模块：没导入过
    pyautogui，异常就不可能是它抛的，也就不必为此把它导进来。
    """
    mod = sys.modules.get("pyautogui")
    return mod is not None and isinstance(e, mod.FailSafeException)


def command_segments(text: str) -> List[str]:
    """把一段输入切成若干「命令开头」，换行和 shell 分隔符都算断点。

    只按行首匹配的话，`echo ok; rm -rf /` 这种拼接写法会整段漏过去。
    """
    return [s.strip() for s in re.split(r"[\n\r;&|]+", text) if s.strip()]


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

        pyautogui 的 KEYBOARD_KEYS 只含 ASCII，write() 打不出中文。
        """
        import pyperclip

        try:
            previous = pyperclip.paste()  # 用完还回去，别把用户的剪贴板覆盖了
        except Exception:
            previous = None

        pyperclip.copy(text)
        try:
            self._pg.hotkey("ctrl", "v")
        finally:
            # 放 finally：粘贴这一步失败（含 FAILSAFE 急停）也得把剪贴板还回去，
            # 否则用户原来复制的东西就被我们顶掉了
            if previous is not None:
                time.sleep(0.1)  # 等粘贴真正完成，否则还原会把内容抢回去
                try:
                    pyperclip.copy(previous)
                except Exception:
                    pass

    def hotkey(self, keys: Sequence[str]) -> None:
        self._pg.hotkey(*keys)


class RecordingBackend:
    """假 backend，只记录调用，不碰真实桌面。给测试和 dry-run 用。"""

    def __init__(self, width: int = 3840, height: int = 2160) -> None:
        self._size = (width, height)
        self.calls: List[tuple] = []
        self._pos = (0, 0)

    def size(self) -> Tuple[int, int]:
        return self._size

    def move(self, x, y):
        self.calls.append(("move", x, y))
        self._pos = (x, y)

    def position(self) -> Tuple[int, int]:
        return self._pos

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
        self._blocked_key_sets = frozenset(
            frozenset(normalize_hotkey(h)) for h in blocked_hotkeys
        )
        self.blocked_text = tuple(re.compile(p) for p in blocked_text)
        self.history: List[Tuple[Action, ActionResult]] = []
        self.dry_run_log: List[Action] = []  # dry_run 下本该执行的动作

    # -- 坐标换算 -----------------------------------------------------------

    def to_pixel(self, point) -> Tuple[int, int]:
        """归一化坐标 -> 可点击的像素坐标。

        右下角要减 1：1920 宽的屏幕像素编号到 1919，归一化 1.0 直接乘出来是 1920，
        点在屏幕外。
        """
        w, h = self.backend.size()
        x = min(max(round(point[0] * w), 0), w - 1)
        y = min(max(round(point[1] * h), 0), h - 1)
        return x, y

    # -- 安全检查 -----------------------------------------------------------

    def _reject(self, action: Action) -> Optional[str]:
        """返回拒绝理由，None 表示放行。"""
        if action.type == "hotkey":
            keys = normalize_hotkey(action.text)
            # 按无序集合比较：模型给 alt+ctrl+delete 和 ctrl+alt+delete 是一回事
            if frozenset(keys) in self._blocked_key_sets:
                return f"组合键 {'+'.join(keys)} 在禁用名单里"
        if action.type == "type" and action.text:
            for seg in command_segments(action.text.lower()):
                for pat in self.blocked_text:
                    if pat.search(seg):
                        return f"输入内容像是破坏性命令，命中规则 {pat.pattern!r}"
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
            if is_failsafe(e):
                raise  # 急停要一路往上传，不能记成一次普通失败
            result = ActionResult(False, f"{type(e).__name__}: {e}", time.perf_counter() - t0)

        self.history.append((action, result))
        return result

    def _dispatch(self, action: Action) -> None:
        t = action.type

        if t in ("finished", "call_user"):
            return  # 终止信号，没有对应的桌面操作

        # dry_run 在这里拦，与用哪个 backend 无关
        if self.dry_run:
            self.dry_run_log.append(action)
            return

        if t == "wait":
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
