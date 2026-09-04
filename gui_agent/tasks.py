"""基础桌面任务与程序化验收条件。

对应大纲第 4 周第 4 项。每个任务配一个能用程序判断的成功条件，不靠人工看结果，
这样第 7 周统计任务成功率时数字才有依据。

验收一律对比执行前后的状态差异，不能只看当前状态：dry-run 时「浏览器进程存在」
判定为通过，但那是用户本来就开着浏览器，Agent 什么都没做。这种验收会让成功率虚高。

安全约定：文件操作限制在 scratch 目录内，不碰真实文档。大纲列的「发送消息」改成
在记事本里输入并保存到 scratch，避免向真人发出不可逆的消息。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Set

SCRATCH = Path(__file__).resolve().parents[1] / "logs" / "scratch"
BROWSERS = ("msedge.exe", "chrome.exe", "firefox.exe")


# --- 状态探针 ---------------------------------------------------------------


def pids_of(name: str) -> Set[str]:
    """指定进程名当前的所有 PID。"""
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH", "/FO", "CSV"],
        capture_output=True, text=True, errors="replace",
    ).stdout
    pids = set()
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[0].lower() == name.lower():
            pids.add(parts[1])
    return pids


def process_running(name: str) -> bool:
    return bool(pids_of(name))


def browser_pids() -> Set[str]:
    return {f"{b}:{p}" for b in BROWSERS for p in pids_of(b)}


def window_titles() -> Set[str]:
    import pygetwindow as gw

    return {t for t in gw.getAllTitles() if t}


def new_title_contains(before: Set[str], keyword: str) -> bool:
    """执行后新出现的窗口标题里有没有包含关键词的。

    只看新增，不看全部：屏幕上本来就有的窗口不算数。
    """
    k = keyword.lower()
    return any(k in t.lower() for t in window_titles() - before)


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
    """一个任务。setup 返回执行前的状态快照，check 拿它和执行后对比。"""

    id: str
    instruction: str
    check: Callable[[Dict], bool]
    setup: Callable[[], Dict] = lambda: {}
    teardown: Callable[[], None] = lambda: None
    note: str = ""


def _scratch() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH


def _close_notepad() -> None:
    subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"], capture_output=True, text=True)


def _setup_open_file() -> Dict:
    (_scratch() / "sample.txt").write_text("这是用于测试的示例文件。\n", encoding="utf-8")
    _close_notepad()
    return {"titles": window_titles()}


def _setup_type_and_save() -> Dict:
    (_scratch() / "output.txt").unlink(missing_ok=True)
    _close_notepad()
    subprocess.Popen(["notepad.exe"])
    return {}


def _setup_close_app() -> Dict:
    subprocess.Popen(["notepad.exe"])
    import time

    time.sleep(1.5)  # 等窗口起来，否则「关闭」无从谈起
    return {"was_running": process_running("notepad.exe")}


def _check_close_app(before: Dict) -> bool:
    """必须确认执行前记事本确实开着，否则「已关闭」没有意义。"""
    return before.get("was_running", False) and not process_running("notepad.exe")


def basic_tasks() -> List[Task]:
    """大纲第 4 周要调试的 5 个基础任务。"""
    return [
        Task(
            id="open_browser",
            instruction="打开浏览器",
            setup=lambda: {"pids": browser_pids()},
            check=lambda before: bool(browser_pids() - before["pids"]),
            note="验收看是否出现新的浏览器进程，不看浏览器是否在运行",
        ),
        Task(
            id="search_content",
            instruction="在浏览器里搜索 python",
            setup=lambda: {"titles": window_titles()},
            check=lambda before: new_title_contains(before["titles"], "python"),
            note="只认新出现的窗口标题，屏幕上本来就有的不算",
        ),
        Task(
            id="open_file",
            instruction=f"用记事本打开文件 {SCRATCH / 'sample.txt'}",
            setup=_setup_open_file,
            check=lambda before: new_title_contains(before["titles"], "sample"),
            teardown=_close_notepad,
        ),
        Task(
            id="type_and_save",
            instruction=f"在记事本里输入「你好」，保存到 {SCRATCH / 'output.txt'}",
            setup=_setup_type_and_save,
            check=lambda before: file_contains(SCRATCH / "output.txt", "你好"),
            teardown=_close_notepad,
            note="替代大纲的「发送消息」：真发消息不可逆且对外，改为写入本地文件",
        ),
        Task(
            id="close_app",
            instruction="关闭记事本",
            setup=_setup_close_app,
            check=_check_close_app,
            teardown=_close_notepad,
        ),
    ]
