"""基础桌面任务与程序化验收条件。

对应大纲第 4 周第 4 项。每个任务配一个能用程序判断的成功条件，不靠人工看结果，
这样第 7 周统计任务成功率时数字才有依据。

安全约定：文件操作一律限制在 scratch 目录内，不碰真实文档。大纲列的「发送消息」
改成在记事本里输入并保存到 scratch，避免向真人发出不可逆的消息。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

SCRATCH = Path(__file__).resolve().parents[1] / "logs" / "scratch"


# --- 验收用的探针 -----------------------------------------------------------


def process_running(name: str) -> bool:
    """进程是否在运行，name 形如 notepad.exe。"""
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"],
        capture_output=True, text=True, errors="replace",
    ).stdout
    return name.lower() in out.lower()


def window_title_contains(keyword: str) -> bool:
    """有没有哪个窗口标题含指定关键词。"""
    import pygetwindow as gw

    return any(keyword.lower() in (t or "").lower() for t in gw.getAllTitles())


def file_contains(path: Path, keyword: str) -> bool:
    if not path.exists():
        return False
    try:
        return keyword in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


# --- 任务定义 ---------------------------------------------------------------


@dataclass
class Task:
    id: str
    instruction: str
    check: Callable[[], bool]
    setup: Callable[[], None] = lambda: None
    teardown: Callable[[], None] = lambda: None
    note: str = ""


def _prepare_scratch() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)


def _write_sample() -> None:
    _prepare_scratch()
    (SCRATCH / "sample.txt").write_text("这是用于测试的示例文件。\n", encoding="utf-8")


def _clear_output() -> None:
    _prepare_scratch()
    (SCRATCH / "output.txt").unlink(missing_ok=True)


def _close_notepad() -> None:
    subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"],
                   capture_output=True, text=True)


def _open_notepad() -> None:
    _prepare_scratch()
    subprocess.Popen(["notepad.exe"])


def basic_tasks() -> List[Task]:
    """大纲第 4 周要调试的 5 个基础任务。"""
    return [
        Task(
            id="open_browser",
            instruction="打开浏览器",
            check=lambda: process_running("msedge.exe") or process_running("chrome.exe"),
        ),
        Task(
            id="search_content",
            instruction="在浏览器里搜索 python",
            check=lambda: window_title_contains("python"),
        ),
        Task(
            id="open_file",
            instruction=f"用记事本打开文件 {SCRATCH / 'sample.txt'}",
            setup=_write_sample,
            check=lambda: window_title_contains("sample"),
            teardown=_close_notepad,
        ),
        Task(
            id="type_and_save",
            instruction=f"在记事本里输入「你好」，保存到 {SCRATCH / 'output.txt'}",
            setup=lambda: (_clear_output(), _open_notepad()),
            check=lambda: file_contains(SCRATCH / "output.txt", "你好"),
            teardown=_close_notepad,
            note="替代大纲的「发送消息」：真发消息不可逆且对外，改为写入本地文件",
        ),
        Task(
            id="close_app",
            instruction="关闭记事本",
            setup=_open_notepad,
            check=lambda: not process_running("notepad.exe"),
            teardown=_close_notepad,
        ),
    ]
