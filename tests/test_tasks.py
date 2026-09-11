from gui_agent.tasks import (
    SCRATCH,
    Task,
    basic_tasks,
    file_contains,
    new_title_contains,
    pids_of,
    process_running,
    window_titles,
)


def test_five_basic_tasks():
    """大纲第 4 周第 4 项要求调试 5 个基础任务。"""
    tasks = basic_tasks()
    assert len(tasks) == 5
    assert [t.id for t in tasks] == [
        "open_browser", "search_content", "open_file", "type_and_save", "close_app"
    ]


def test_instructions_are_non_empty():
    for t in basic_tasks():
        assert t.instruction.strip()


def test_file_paths_in_instructions_stay_in_scratch():
    from gui_agent.tasks import SCRATCH

    for t in basic_tasks():
        if "\\" in t.instruction or "/" in t.instruction:
            assert str(SCRATCH) in t.instruction


def test_setup_only_touches_scratch(tmp_path, monkeypatch):
    """任务准备阶段实际落盘的位置要在 scratch 里，光看指令字符串说明不了问题。"""
    import gui_agent.tasks as T

    monkeypatch.setattr(T, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(T, "_open_notepad", lambda: -1)
    monkeypatch.setattr(T, "_close_notepad", lambda: None)
    monkeypatch.setattr(T, "window_titles", lambda: set())
    monkeypatch.setattr(T, "_notepad_alive", lambda pid: True)

    before = {p for p in tmp_path.rglob("*")}
    for t in T.basic_tasks():
        t.setup()
    written = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert written, "至少 open_file 会写 sample.txt"
    for p in written:
        assert (tmp_path / "scratch") in p.parents


def test_message_task_is_substituted():
    """大纲的「发送消息」改成写本地文件，理由记在 note 里。"""
    t = next(x for x in basic_tasks() if x.id == "type_and_save")
    assert "不可逆" in t.note


# --- 验收必须看状态变化，不能看当前状态 -------------------------------------


def test_open_browser_check_needs_a_new_process():
    """浏览器本来就开着时不能算通过，这是 dry-run 里暴露过的问题。"""
    t = next(x for x in basic_tasks() if x.id == "open_browser")
    from gui_agent.tasks import browser_pids

    baseline = {"pids": browser_pids()}   # 快照就是当前状态
    assert t.check(baseline) is False     # 没有新进程，必须判不通过


def test_search_check_ignores_preexisting_titles():
    """屏幕上本来就有含关键词的窗口时不能算通过。"""
    t = next(x for x in basic_tasks() if x.id == "search_content")
    assert t.check({"titles": window_titles()}) is False


def test_close_app_requires_it_was_running_first():
    """执行前记事本没开的话，「已关闭」不成立。"""
    t = next(x for x in basic_tasks() if x.id == "close_app")
    assert t.check({"was_running": False}) is False


def test_new_title_contains_only_counts_new_windows():
    before = window_titles()
    assert new_title_contains(before, "explorer") is False
    assert new_title_contains(set(), "") is (len(window_titles()) > 0)


# --- 探针 -------------------------------------------------------------------


def test_pids_of_returns_ids_for_running_process():
    pids = pids_of("explorer.exe")
    assert pids and all(p.isdigit() for p in pids)


def test_pids_of_empty_for_nonexistent():
    assert pids_of("绝对不存在xyz.exe") == set()


def test_process_running():
    assert process_running("explorer.exe")
    assert not process_running("绝对不存在xyz.exe")


def test_window_titles_non_empty():
    assert window_titles()


def test_file_contains(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("包含关键词你好在里面", encoding="utf-8")
    assert file_contains(p, "你好")
    assert not file_contains(p, "再见")
    assert not file_contains(tmp_path / "没有.txt", "x")


def test_task_defaults():
    t = Task(id="x", instruction="y", check=lambda before: True)
    assert t.setup() == {} and t.teardown() is None


# --- 跨平台 -----------------------------------------------------------------


def test_process_lookup_uses_psutil_not_platform_commands():
    """进程查询不能用 tasklist / taskkill 这类 Windows 专有命令。"""
    from pathlib import Path as _P

    src = _P(__file__).resolve().parents[1] / "gui_agent" / "tasks.py"
    text = src.read_text(encoding="utf-8")
    for cmd in ('"tasklist"', '"taskkill"'):
        assert cmd not in text, f"{cmd} 是 Windows 专有的"
    assert "import psutil" in text


def test_editor_and_browsers_are_chosen_per_platform():
    import gui_agent.tasks as T

    assert T.EDITOR and T.BROWSERS
    assert isinstance(T.EDITOR, tuple) and isinstance(T.BROWSERS, tuple)


def test_window_titles_returns_empty_set_when_unavailable(monkeypatch):
    """pygetwindow 在 Linux 上不可用，取不到标题时返回空集合，
    验收比对的是新增标题，空集合不会误判为通过。"""
    import builtins

    import gui_agent.tasks as T

    real = builtins.__import__

    def boom(name, *a, **kw):
        if name == "pygetwindow":
            raise ImportError("Linux 不支持")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert T.window_titles() == set()
    assert not T.new_title_contains(set(), "python")


def test_notepad_alive_is_false_for_a_dead_pid():
    import gui_agent.tasks as T

    assert T._notepad_alive(999_999_999) is False


# --- 验收条件的可靠性 -------------------------------------------------------


def test_open_browser_needs_both_a_new_process_and_a_new_window(monkeypatch):
    """只看进程会误判：浏览器的后台辅助进程不断变化，实测 Agent 只输出了一个
    call_user、什么都没做，按进程判定却算通过。"""
    import gui_agent.tasks as T

    t = next(x for x in T.basic_tasks() if x.id == "open_browser")
    before = {"pids": {"chrome.exe:1"}, "titles": {"记事本"}}

    # 只多了后台进程，没有新窗口 —— 不通过
    monkeypatch.setattr(T, "browser_pids", lambda: {"chrome.exe:1", "chrome.exe:2"})
    monkeypatch.setattr(T, "window_titles", lambda: {"记事本"})
    assert t.check(before) is False

    # 只有新窗口、没有新进程 —— 也不通过
    monkeypatch.setattr(T, "browser_pids", lambda: {"chrome.exe:1"})
    monkeypatch.setattr(T, "window_titles", lambda: {"记事本", "新标签页"})
    assert t.check(before) is False

    # 两个都有 —— 通过
    monkeypatch.setattr(T, "browser_pids", lambda: {"chrome.exe:1", "chrome.exe:2"})
    monkeypatch.setattr(T, "window_titles", lambda: {"记事本", "新标签页"})
    assert t.check(before) is True


def test_open_browser_setup_records_both_baselines():
    import gui_agent.tasks as T

    t = next(x for x in T.basic_tasks() if x.id == "open_browser")
    snap = t.setup()
    assert "pids" in snap and "titles" in snap


# --- 复杂任务（大纲第 6 周第 1 项）------------------------------------------


def test_complex_tasks_have_unique_ids_and_do_not_clash_with_basic():
    from gui_agent.tasks import basic_tasks, complex_tasks

    ids = [t.id for t in complex_tasks()]
    assert len(ids) == len(set(ids))
    assert not (set(ids) & {t.id for t in basic_tasks()})


def test_complex_tasks_only_touch_scratch():
    """指令里出现的路径必须都在 scratch 下，不能碰真实文档。"""
    import re

    from gui_agent.tasks import SCRATCH, complex_tasks

    for t in complex_tasks():
        for path in re.findall(r"[A-Za-z]:\[^\s，、]+|/[^\s，、]+\.txt", t.instruction):
            assert str(SCRATCH) in path, f"{t.id} 指令里有 scratch 之外的路径：{path}"


def test_complex_checks_fail_before_anything_runs(tmp_path, monkeypatch):
    """跑之前验收必须全不通过，否则成功率就是白送的。"""
    import gui_agent.tasks as T

    monkeypatch.setattr(T, "SCRATCH", tmp_path)
    for t in T.complex_tasks():
        if t.id == "open_two_apps":
            continue  # 这条要真进程，另有测试
        assert t.check({}) is False, f"{t.id} 在什么都没做时就判通过了"


def test_append_task_needs_both_old_and_new_content(tmp_path, monkeypatch):
    import gui_agent.tasks as T

    monkeypatch.setattr(T, "SCRATCH", tmp_path)
    target = tmp_path / "reviewed.txt"
    target.write_text("已审阅\n", encoding="utf-8")          # 只有新内容
    assert T._check_append_and_save_as({}) is False
    target.write_text("原始内容\n已审阅\n", encoding="utf-8")  # 两样都有
    assert T._check_append_and_save_as({}) is True


def test_two_files_task_needs_both_files(tmp_path, monkeypatch):
    import gui_agent.tasks as T

    monkeypatch.setattr(T, "SCRATCH", tmp_path)
    (tmp_path / "first.txt").write_text("第一份\n", encoding="utf-8")
    assert T._check_write_two_files({}) is False  # 只写了一个不算完成
    (tmp_path / "second.txt").write_text("第二份\n", encoding="utf-8")
    assert T._check_write_two_files({}) is True


def test_open_two_apps_needs_both_programs(monkeypatch):
    import gui_agent.tasks as T

    before = {"browsers": {"1"}, "editors": {"2"}, "titles": {"旧窗口"}}
    monkeypatch.setattr(T, "window_titles", lambda: {"旧窗口", "新窗口"})

    monkeypatch.setattr(T, "browser_pids", lambda: {"1", "3"})
    monkeypatch.setattr(T, "pids_of", lambda name: {"2"})          # 编辑器没新起
    assert T._check_open_two_apps(before) is False

    monkeypatch.setattr(T, "browser_pids", lambda: {"1"})          # 浏览器没新起
    monkeypatch.setattr(T, "pids_of", lambda name: {"2", "4"})
    assert T._check_open_two_apps(before) is False

    monkeypatch.setattr(T, "browser_pids", lambda: {"1", "3"})
    assert T._check_open_two_apps(before) is True


def test_open_two_apps_needs_a_new_window(monkeypatch):
    """只多出后台进程不算，和 open_browser 一个道理。"""
    import gui_agent.tasks as T

    before = {"browsers": {"1"}, "editors": {"2"}, "titles": {"旧窗口"}}
    monkeypatch.setattr(T, "browser_pids", lambda: {"1", "3"})
    monkeypatch.setattr(T, "pids_of", lambda name: {"2", "4"})
    monkeypatch.setattr(T, "window_titles", lambda: {"旧窗口"})
    assert T._check_open_two_apps(before) is False
