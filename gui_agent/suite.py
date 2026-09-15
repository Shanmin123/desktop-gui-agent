"""第 7 周评测集：25 个桌面任务，分三档难度。

对应大纲第 7 周「设计 20 个不同难度的桌面任务测试集」。只有 5 个基础任务、4 个复杂
任务时，一个任务就是 20 个点，2/5 的 95% 置信区间是 12%~77%，比不出两套配置的差别。

分档
  T1  单个程序，1~3 步
  T2  单个程序，多步或要过对话框
  T3  跨程序，或要完成多个子目标

验收沿用 tasks.py 的约定：setup 记下执行前的状态，check 只看执行后的变化；文件只在
scratch 目录里读写；teardown 只关自己启动的记事本。验收方式参照 WebArena 的三类
（docs/数据集说明.md）：
  string_match  文件内容、窗口标题里有没有指定文字
  url_match     浏览器停在哪个页面：用 scratch 里的本地网页，看页面标题，离线、结果固定
  program       文件系统状态：文件夹建没建、改没改名、挪没挪、图片是不是 PNG

不收改系统设置、发消息、命令行这几类，在真机上跑风险太大；除原有的 search_content
外也不依赖外网内容。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from gui_agent import tasks as T
from gui_agent.tasks import Task, new_title_contains

CALCULATORS = ("CalculatorApp.exe", "Calculator.exe", "win32calc.exe")
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
ORDER_NUMBER = "A7X-2931"
MEETING_TIME = "14:30"

DRAFT = "这是一份草稿。\n草稿第二段写的是计划。\n请审阅这份草稿。\n"
APP_LOG = "INFO 启动完成\nDEBUG 连接池参数 size=8\nINFO 收到第一个请求\n"
MEETING = "项目周会纪要\n日期：9 月 18 日\n时间：14:30—15:30\n地点：3 号会议室\n议题：第 4 周评测进度\n"

SITE_INDEX = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>任务首页</title></head>
<body>
<h1>任务首页</h1>
<p>订单号：<b>{order}</b></p>
<p><a href="product.html">产品说明</a></p>
<form action="search.html" method="get">
  <input name="q" placeholder="搜索商品"> <button type="submit">搜索</button>
</form>
</body></html>
"""
SITE_PRODUCT = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>产品说明页</title></head>
<body><h1>产品说明</h1><p>任务用的本地网页。</p></body></html>
"""
SITE_SEARCH = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>搜索结果</title></head>
<body><h1>搜索结果</h1><p id="q"></p>
<script>
  const q = new URLSearchParams(location.search).get("q") || "";
  document.title = "搜索结果：" + q;
  document.getElementById("q").textContent = "你搜索的是：" + q;
</script>
</body></html>
"""

# 现有 9 个任务在评测集里的档位
LEVELS = {
    "open_browser": "T1", "search_content": "T1", "open_file": "T1", "close_app": "T1",
    "type_and_save": "T2",
    "open_two_apps": "T3", "append_and_save_as": "T3", "copy_between_files": "T3",
    "write_two_files": "T3",
}

ORDER = [
    # T1
    "open_browser", "search_content", "open_file", "close_app",
    "open_explorer_folder", "create_folder", "open_calculator", "maximize_window",
    # T2
    "type_and_save", "append_line_in_place", "rename_file", "replace_all_text",
    "write_three_lines", "save_blank_image", "follow_local_link", "delete_log_line",
    "move_file_into_folder",
    # T3
    "open_two_apps", "append_and_save_as", "copy_between_files", "write_two_files",
    "calculate_to_file", "copy_order_number", "submit_local_search", "extract_meeting_time",
]


# --- 工具 -------------------------------------------------------------------


def _wait(seconds: float) -> None:
    time.sleep(seconds)


def _write(relpath: str, text: str) -> Path:
    path = T._scratch() / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _remove(relpath: str) -> None:
    """删掉上一轮留下的产物，只删 scratch 之内的路径。"""
    path = T.SCRATCH / relpath
    if T.SCRATCH.resolve() not in path.resolve().parents:
        raise ValueError(f"{path} 不在 scratch 里")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _read(path: Path) -> str:
    """读文本，去掉记事本可能写入的 BOM。文件不在返回空串。"""
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _calculator_pids() -> set:
    return {f"{name}:{pid}" for name in CALCULATORS for pid in T.pids_of(name)}


def _windows_titled(keyword: str) -> list:
    """标题含关键词的窗口。pygetwindow 用不了时返回空列表，验收按不通过算。"""
    try:
        import pygetwindow as gw

        k = keyword.lower()
        return [w for w in gw.getAllWindows() if k in (w.title or "").lower()]
    except Exception:  # 各平台抛的异常类型不一，统一按取不到处理
        return []


def _window_maximized(keyword: str) -> bool:
    return any(getattr(w, "isMaximized", False) for w in _windows_titled(keyword))


def _unmaximize(keyword: str) -> None:
    for w in _windows_titled(keyword):
        try:
            if w.isMaximized:
                w.restore()
        except Exception:
            pass


def _titles() -> Dict:
    T._scratch()
    return {"titles": T.window_titles()}


def _write_site() -> Dict:
    _write("site/index.html", SITE_INDEX.format(order=ORDER_NUMBER))
    _write("site/product.html", SITE_PRODUCT)
    _write("site/search.html", SITE_SEARCH)
    return {"titles": T.window_titles()}


def _close_notepad() -> None:
    T._close_notepad()


# --- 准备与验收 ---------------------------------------------------------------


def _setup_create_folder() -> Dict:
    T._scratch()
    _remove("新项目")
    return {}


def _setup_calculator() -> Dict:
    return {"pids": _calculator_pids(), "titles": T.window_titles()}


def _check_open_calculator(before: Dict) -> bool:
    """新窗口标题是计算器就算；标题取决于系统语言，取不到名字时要求新进程加新窗口。

    UWP 计算器关掉后进程可能还挂在后台，只看进程会漏判重新打开的情况。
    """
    new_titles = T.window_titles() - before["titles"]
    named = any("计算器" in t or "calculator" in t.lower() for t in new_titles)
    new_process = bool(_calculator_pids() - before["pids"])
    return named or (new_process and bool(new_titles))


def _setup_maximize() -> Dict:
    path = _write("maximize_me.txt", "把这个窗口最大化。\n")
    T._close_notepad()
    T._open_notepad(path)
    _wait(1.5)
    # 新版记事本会记住上次的窗口状态，先还原，否则任务一开始就已经完成
    _unmaximize("maximize_me")
    return {"was_maximized": _window_maximized("maximize_me")}


def _check_maximize(before: Dict) -> bool:
    return not before.get("was_maximized", True) and _window_maximized("maximize_me")


def _setup_todo() -> Dict:
    _write("todo.txt", "买面包\n")
    T._close_notepad()
    return {}


def _check_todo(before: Dict) -> bool:
    text = _read(T.SCRATCH / "todo.txt")
    return "买面包" in text and "买牛奶" in text


def _setup_rename() -> Dict:
    _write("report_v1.txt", "季度报告初稿\n")
    _remove("report_final.txt")
    return {}


def _check_rename(before: Dict) -> bool:
    """旧文件还在就不算：复制一份新名字不是重命名。"""
    return (not (T.SCRATCH / "report_v1.txt").exists()
            and "季度报告初稿" in _read(T.SCRATCH / "report_final.txt"))


def _setup_replace() -> Dict:
    _write("draft.txt", DRAFT)
    T._close_notepad()
    return {"count": DRAFT.count("草稿")}


def _check_replace(before: Dict) -> bool:
    text = _read(T.SCRATCH / "draft.txt")
    return "草稿" not in text and text.count("终稿") == before.get("count", DRAFT.count("草稿"))


def _setup_fruits() -> Dict:
    T._scratch()
    _remove("fruits.txt")
    T._close_notepad()
    return {}


def _check_fruits(before: Dict) -> bool:
    lines = [line.strip() for line in _read(T.SCRATCH / "fruits.txt").splitlines() if line.strip()]
    return lines == ["苹果", "香蕉", "橙子"]


def _setup_png() -> Dict:
    T._scratch()
    _remove("blank.png")
    return {"since": time.time() - 1}


def _check_png(before: Dict) -> bool:
    """文件要在任务开始之后写出来，而且真是 PNG：只改个扩展名的文本文件不算。"""
    path = T.SCRATCH / "blank.png"
    try:
        fresh = path.stat().st_mtime >= before.get("since", float("inf"))
        return fresh and path.read_bytes()[:8] == PNG_MAGIC
    except OSError:
        return False


def _setup_log() -> Dict:
    _write("app.log", APP_LOG)
    T._close_notepad()
    return {}


def _check_log(before: Dict) -> bool:
    """DEBUG 那行要没了，另外两行要都还在：整个文件清空不算完成。"""
    text = _read(T.SCRATCH / "app.log")
    return "DEBUG" not in text and "INFO 启动完成" in text and "INFO 收到第一个请求" in text


def _setup_move() -> Dict:
    T._scratch()
    _remove("inbox")
    _remove("归档")
    _write("inbox/a.txt", "待归档的文件\n")
    (T.SCRATCH / "归档").mkdir(parents=True, exist_ok=True)
    return {}


def _check_move(before: Dict) -> bool:
    return (not (T.SCRATCH / "inbox" / "a.txt").exists()
            and "待归档的文件" in _read(T.SCRATCH / "归档" / "a.txt"))


def _setup_output(name: str) -> Dict:
    T._scratch()
    _remove(name)
    T._close_notepad()
    return {}


def _setup_order() -> Dict:
    _setup_output("order.txt")
    return _write_site()


def _setup_meeting() -> Dict:
    _write("meeting.txt", MEETING)
    return _setup_output("answer.txt")


# --- 任务 -------------------------------------------------------------------


def _new_tasks() -> List[Task]:
    s = T.SCRATCH
    site = s / "site" / "index.html"
    return [
        Task(
            id="open_explorer_folder", level="T1",
            instruction=f"打开文件资源管理器，进入文件夹 {s}",
            setup=_titles,
            check=lambda before: new_title_contains(before["titles"], T.SCRATCH.name),
            note="资源管理器窗口标题是当前文件夹名",
        ),
        Task(
            id="create_folder", level="T1",
            instruction=f"在文件资源管理器里，在 {s} 下新建一个名为「新项目」的文件夹",
            setup=_setup_create_folder,
            check=lambda before: (T.SCRATCH / "新项目").is_dir(),
        ),
        Task(
            id="open_calculator", level="T1",
            instruction="打开计算器",
            setup=_setup_calculator,
            check=_check_open_calculator,
        ),
        Task(
            id="maximize_window", level="T1",
            instruction="把标题里有 maximize_me 的记事本窗口最大化",
            setup=_setup_maximize,
            check=_check_maximize,
            teardown=_close_notepad,
        ),
        Task(
            id="append_line_in_place", level="T2",
            instruction=f"用记事本打开 {s / 'todo.txt'}，在末尾另起一行写「买牛奶」，直接保存",
            setup=_setup_todo,
            check=_check_todo,
            teardown=_close_notepad,
        ),
        Task(
            id="rename_file", level="T2",
            instruction=f"在文件资源管理器里把 {s / 'report_v1.txt'} 重命名为 report_final.txt",
            setup=_setup_rename,
            check=_check_rename,
            note="资源管理器默认隐藏扩展名，照原样输入 report_final.txt 会得到 report_final.txt.txt",
        ),
        Task(
            id="replace_all_text", level="T2",
            instruction=f"用记事本把 {s / 'draft.txt'} 里的「草稿」全部替换成「终稿」，然后保存",
            setup=_setup_replace,
            check=_check_replace,
            teardown=_close_notepad,
            note="一共 3 处，漏一处不算",
        ),
        Task(
            id="write_three_lines", level="T2",
            instruction=f"用记事本新建文件，写三行：苹果、香蕉、橙子，每行一个，保存为 {s / 'fruits.txt'}",
            setup=_setup_fruits,
            check=_check_fruits,
            teardown=_close_notepad,
        ),
        Task(
            id="save_blank_image", level="T2",
            instruction=f"打开画图，新建一张空白图片，保存为 {s / 'blank.png'}",
            setup=_setup_png,
            check=_check_png,
        ),
        Task(
            id="follow_local_link", level="T2",
            instruction=f"用浏览器打开 {site}，点击页面上的「产品说明」链接",
            setup=_write_site,
            check=lambda before: new_title_contains(before["titles"], "产品说明页"),
            note="本地网页，看点完之后的页面标题",
        ),
        Task(
            id="delete_log_line", level="T2",
            instruction=f"用记事本打开 {s / 'app.log'}，删掉含有 DEBUG 的那一行，保存",
            setup=_setup_log,
            check=_check_log,
            teardown=_close_notepad,
        ),
        Task(
            id="move_file_into_folder", level="T2",
            instruction=f"在文件资源管理器里把 {s / 'inbox' / 'a.txt'} 移动到 {s / '归档'} 文件夹里",
            setup=_setup_move,
            check=_check_move,
            note="原位置的文件要没了，复制过去不算",
        ),
        Task(
            id="calculate_to_file", level="T3",
            instruction=f"用计算器算出 128 × 256 的结果，把结果写进 {s / 'calc.txt'} 并保存",
            setup=lambda: _setup_output("calc.txt"),
            check=lambda before: "32768" in _read(T.SCRATCH / "calc.txt"),
            teardown=_close_notepad,
            note="跨计算器和记事本；验收只看结果文件",
        ),
        Task(
            id="copy_order_number", level="T3",
            instruction=f"用浏览器打开 {site}，把页面上的订单号写进 {s / 'order.txt'} 并保存",
            setup=_setup_order,
            check=lambda before: ORDER_NUMBER in _read(T.SCRATCH / "order.txt"),
            teardown=_close_notepad,
            note="跨浏览器和记事本",
        ),
        Task(
            id="submit_local_search", level="T3",
            instruction=f"用浏览器打开 {site}，在页面的搜索框里输入「显卡」并提交",
            setup=_write_site,
            check=lambda before: new_title_contains(before["titles"], "搜索结果：显卡"),
            note="结果页把查询词写进标题，标题里有「显卡」才算提交成功",
        ),
        Task(
            id="extract_meeting_time", level="T3",
            instruction=f"打开 {s / 'meeting.txt'}，找到会议开始的时间，把这个时间写进 {s / 'answer.txt'} 并保存",
            setup=_setup_meeting,
            check=lambda before: MEETING_TIME in _read(T.SCRATCH / "answer.txt"),
            teardown=_close_notepad,
            note="要读懂内容再写到另一个文件",
        ),
    ]


def suite_tasks() -> List[Task]:
    """25 个任务，按 ORDER 排列：现有 9 个补上档位，加新写的 16 个。"""
    old = {t.id: replace(t, level=LEVELS[t.id]) for t in T.basic_tasks() + T.complex_tasks()}
    new = {t.id: t for t in _new_tasks()}
    pool = {**old, **new}
    return [pool[task_id] for task_id in ORDER]
