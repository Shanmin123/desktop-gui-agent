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


def test_file_tasks_stay_in_scratch():
    for t in basic_tasks():
        if "\\" in t.instruction or "/" in t.instruction:
            assert "scratch" in t.instruction


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
