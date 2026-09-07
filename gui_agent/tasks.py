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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Set

import psutil

SCRATCH = Path(__file__).resolve().parents[1] / "logs" / "scratch"

# 进程名按平台不同。大纲技术栈要求支持 Windows / macOS / Linux，进程查询和
# 结束都走 psutil，不用 tasklist / taskkill 这类 Windows 专有命令。
if sys.platform == "win32":
    BROWSERS = ("msedge.exe", "chrome.exe", "firefox.exe")
    EDITOR = ("notepad.exe",)
    EDITOR_NAME = "notepad.exe"
elif sys.platform == "darwin":
    BROWSERS = ("Safari", "Google Chrome", "firefox")
    EDITOR = ("open", "-a", "TextEdit")
    EDITOR_NAME = "TextEdit"
else:
    BROWSERS = ("chrome", "chromium", "firefox")
    EDITOR = ("gedit",)
    EDITOR_NAME = "gedit"


# --- 状态探针 ---------------------------------------------------------------


def pids_of(name: str) -> Set[str]:
    """指定进程名当前的所有 PID。名字不区分大小写。"""
    want = name.lower()
    pids = set()
    for proc in psutil.process_iter(["name"]):
        try:
            got = (proc.info["name"] or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if got == want:
            pids.add(str(proc.pid))
    return pids


def process_running(name: str) -> bool:
    return bool(pids_of(name))


def browser_pids() -> Set[str]:
    return {f"{b}:{p}" for b in BROWSERS for p in pids_of(b)}


def window_titles() -> Set[str]:
    """当前所有窗口标题。

    pygetwindow 只在 Windows 上完整可用，macOS 部分可用，Linux 不支持。取不到时
    返回空集合：依赖它的验收条件比对的是「新增的标题」，空集合下不会误判为通过。
    """
    try:
        import pygetwindow as gw

        return {t for t in gw.getAllTitles() if t}
    except Exception:  # 各平台抛的异常类型不一，统一按取不到处理
        return set()


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


# 本模块自己启动的记事本进程，只关这些
_OWN_NOTEPADS: Set[int] = set()


def _open_notepad() -> int:
    pid = subprocess.Popen(list(EDITOR)).pid
    _OWN_NOTEPADS.add(pid)
    return pid


def _close_notepad() -> None:
    """关掉本模块启动的文本编辑器。

    只按 PID 关自己启动的那些：按进程名一律结束会连用户开着的、可能有未保存内容的
    窗口一起杀掉。任务准备和收尾都会调到这里，误伤代价太大。
    """
    for pid in list(_OWN_NOTEPADS):
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=3)
        except psutil.TimeoutExpired:
            try:
                proc.kill()
            except psutil.Error:
                pass
        except psutil.Error:
            pass  # 已经退出了
        _OWN_NOTEPADS.discard(pid)


def _notepad_alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        # 退出后没被回收的僵尸进程还查得到，不算还开着
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _setup_open_file() -> Dict:
    (_scratch() / "sample.txt").write_text("这是用于测试的示例文件。\n", encoding="utf-8")
    _close_notepad()
    return {"titles": window_titles()}


def _setup_type_and_save() -> Dict:
    (_scratch() / "output.txt").unlink(missing_ok=True)
    _close_notepad()
    _open_notepad()
    return {}


def _setup_close_app() -> Dict:
    pid = _open_notepad()
    import time

    time.sleep(1.5)  # 等窗口起来，否则「关闭」无从谈起
    return {"pid": pid, "was_running": _notepad_alive(pid)}


def _check_open_browser(before: Dict) -> bool:
    """要求既出现新的浏览器进程，也出现新窗口。

    只看进程会误判：浏览器持续起后台辅助进程（更新程序、渲染器），任何一个新 PID
    都会让验收通过。实测 Agent 只输出了一个 call_user、一次点击都没有，按进程判定
    却算通过。真正打开浏览器一定会有新的顶层窗口，两个条件都要满足。
    """
    return bool(browser_pids() - before["pids"]) and bool(window_titles() - before["titles"])


def _check_close_app(before: Dict) -> bool:
    """必须确认执行前那个记事本确实开着，否则「已关闭」没有意义。

    只看我们启动的那个进程：用户自己另开着一个记事本时，按全局进程名判断会让这
    个任务永远不通过。
    """
    pid = before.get("pid")
    return bool(before.get("was_running")) and pid is not None and not _notepad_alive(pid)


def basic_tasks() -> List[Task]:
    """大纲第 4 周要调试的 5 个基础任务。"""
    return [
        Task(
            id="open_browser",
            instruction="打开浏览器",
            setup=lambda: {"pids": browser_pids(), "titles": window_titles()},
            check=_check_open_browser,
            note="要新进程加新窗口两个条件：浏览器的后台辅助进程会让只看进程的判定误通过",
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
