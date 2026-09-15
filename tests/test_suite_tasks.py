"""第 7 周评测集：任务数量、档位、准备只写 scratch、验收在做完之前不通过、做完之后通过。"""

import os
from collections import Counter

import pytest

import gui_agent.tasks as T
from gui_agent import suite as S


def _task(task_id):
    return next(t for t in S.suite_tasks() if t.id == task_id)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """不开程序、不看真实窗口，scratch 指到临时目录。"""
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(T, "SCRATCH", scratch)
    monkeypatch.setattr(T, "_open_notepad", lambda *a, **k: -1)
    monkeypatch.setattr(T, "_close_notepad", lambda: None)
    monkeypatch.setattr(T, "_notepad_alive", lambda pid: False)
    monkeypatch.setattr(T, "window_titles", lambda: set())
    monkeypatch.setattr(T, "pids_of", lambda name: set())
    monkeypatch.setattr(T, "browser_pids", lambda: set())
    monkeypatch.setattr(S, "_wait", lambda seconds: None)
    monkeypatch.setattr(S, "_windows_titled", lambda keyword: [])
    return scratch


# --- 规模与档位 ---------------------------------------------------------------


def test_suite_has_at_least_twenty_unique_tasks():
    ids = [t.id for t in S.suite_tasks()]
    assert len(ids) >= 20
    assert len(ids) == len(set(ids))


def test_every_task_has_one_of_three_levels_and_each_level_has_several():
    counts = Counter(t.level for t in S.suite_tasks())
    assert set(counts) == {"T1", "T2", "T3"}
    assert min(counts.values()) >= 5


def test_suite_keeps_the_nine_existing_tasks():
    existing = {t.id for t in T.basic_tasks() + T.complex_tasks()}
    assert existing <= {t.id for t in S.suite_tasks()}


def test_existing_task_lists_are_unchanged():
    """basic / complex 两个集合和之前的实验保持一致，只是评测集里多了档位。"""
    assert len(T.basic_tasks()) == 5 and len(T.complex_tasks()) == 4
    assert all(t.level == "" for t in T.basic_tasks() + T.complex_tasks())


def test_instructions_with_paths_point_into_scratch():
    for t in S.suite_tasks():
        assert t.instruction.strip()
        if ":\\" in t.instruction or "/" in t.instruction:
            assert str(T.SCRATCH) in t.instruction, t.id


# --- 准备与验收 ---------------------------------------------------------------


def test_setups_write_only_inside_scratch(tmp_path, sandbox):
    for t in S.suite_tasks():
        t.setup()
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written
    assert all(sandbox in p.parents for p in written)


def test_every_check_fails_right_after_setup(sandbox):
    """准备好之后什么都没做，验收必须全部不通过，否则成功率是白送的。"""
    for t in S.suite_tasks():
        before = t.setup()
        assert t.check(before) is False, t.id


def test_setup_clears_output_left_by_a_previous_run(sandbox):
    t = _task("calculate_to_file")
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "calc.txt").write_text("32768", encoding="utf-8")
    before = t.setup()
    assert t.check(before) is False


GOAL = {
    "create_folder": lambda s: (s / "新项目").mkdir(),
    "append_line_in_place": lambda s: (s / "todo.txt").write_text("买面包\n买牛奶\n", encoding="utf-8"),
    "rename_file": lambda s: (s / "report_v1.txt").rename(s / "report_final.txt"),
    "replace_all_text": lambda s: (s / "draft.txt").write_text(
        S.DRAFT.replace("草稿", "终稿"), encoding="utf-8"),
    "write_three_lines": lambda s: (s / "fruits.txt").write_text("苹果\n香蕉\n橙子\n", encoding="utf-8-sig"),
    "save_blank_image": lambda s: (s / "blank.png").write_bytes(S.PNG_MAGIC + bytes(16)),
    "delete_log_line": lambda s: (s / "app.log").write_text(
        "INFO 启动完成\nINFO 收到第一个请求\n", encoding="utf-8"),
    "move_file_into_folder": lambda s: (s / "inbox" / "a.txt").rename(s / "归档" / "a.txt"),
    "calculate_to_file": lambda s: (s / "calc.txt").write_text("32768\n", encoding="utf-8"),
    "copy_order_number": lambda s: (s / "order.txt").write_text("订单号 A7X-2931\n", encoding="utf-8"),
    "extract_meeting_time": lambda s: (s / "answer.txt").write_text("14:30\n", encoding="utf-8"),
}


@pytest.mark.parametrize("task_id", sorted(GOAL))
def test_file_checks_pass_once_the_goal_state_exists(sandbox, task_id):
    t = _task(task_id)
    before = t.setup()
    assert t.check(before) is False
    GOAL[task_id](sandbox)
    assert t.check(before) is True


def test_replace_check_rejects_a_partial_replacement(sandbox):
    t = _task("replace_all_text")
    before = t.setup()
    (sandbox / "draft.txt").write_text(S.DRAFT.replace("草稿", "终稿", 1), encoding="utf-8")
    assert t.check(before) is False


def test_delete_line_check_rejects_deleting_too_much(sandbox):
    t = _task("delete_log_line")
    before = t.setup()
    (sandbox / "app.log").write_text("INFO 启动完成\n", encoding="utf-8")
    assert t.check(before) is False


def test_rename_check_rejects_a_copy_with_the_new_name(sandbox):
    t = _task("rename_file")
    before = t.setup()
    (sandbox / "report_final.txt").write_text("季度报告初稿\n", encoding="utf-8")
    assert t.check(before) is False


def test_rename_check_rejects_a_doubled_extension(sandbox):
    """隐藏扩展名时照打 report_final.txt 会变成 report_final.txt.txt，不算完成。"""
    t = _task("rename_file")
    before = t.setup()
    (sandbox / "report_v1.txt").rename(sandbox / "report_final.txt.txt")
    assert t.check(before) is False


def test_move_check_rejects_a_copy(sandbox):
    t = _task("move_file_into_folder")
    before = t.setup()
    (sandbox / "归档" / "a.txt").write_text("待归档的文件\n", encoding="utf-8")
    assert t.check(before) is False


def test_three_lines_check_needs_three_separate_lines(sandbox):
    t = _task("write_three_lines")
    before = t.setup()
    (sandbox / "fruits.txt").write_text("苹果 香蕉 橙子\n", encoding="utf-8")
    assert t.check(before) is False


def test_png_check_rejects_a_text_file_named_png(sandbox):
    t = _task("save_blank_image")
    before = t.setup()
    (sandbox / "blank.png").write_text("not a png", encoding="utf-8")
    assert t.check(before) is False


def test_png_check_rejects_a_file_older_than_the_task(sandbox):
    t = _task("save_blank_image")
    before = t.setup()
    path = sandbox / "blank.png"
    path.write_bytes(S.PNG_MAGIC + bytes(16))
    old = before["since"] - 3600
    os.utime(path, (old, old))
    assert t.check(before) is False


# --- 窗口与进程类 -------------------------------------------------------------


@pytest.mark.parametrize("task_id,title", [
    ("follow_local_link", "产品说明页 - 个人 - Microsoft Edge"),
    ("submit_local_search", "搜索结果：显卡 - 个人 - Microsoft Edge"),
    ("open_explorer_folder", "scratch - 文件资源管理器"),
])
def test_title_checks_pass_on_a_new_window_with_the_expected_title(sandbox, monkeypatch, task_id, title):
    t = _task(task_id)
    before = t.setup()
    monkeypatch.setattr(T, "window_titles", lambda: {title})
    assert t.check(before) is True


def test_search_check_needs_the_query_in_the_title(sandbox, monkeypatch):
    t = _task("submit_local_search")
    before = t.setup()
    monkeypatch.setattr(T, "window_titles", lambda: {"搜索结果： - 个人 - Microsoft Edge"})
    assert t.check(before) is False


def test_title_checks_ignore_windows_that_were_already_open(sandbox, monkeypatch):
    monkeypatch.setattr(T, "window_titles", lambda: {"产品说明页 - 个人 - Microsoft Edge"})
    t = _task("follow_local_link")
    before = t.setup()
    assert t.check(before) is False


def test_calculator_counts_a_new_calculator_window(sandbox, monkeypatch):
    t = _task("open_calculator")
    before = t.setup()
    monkeypatch.setattr(T, "window_titles", lambda: {"计算器"})
    assert t.check(before) is True


def test_calculator_ignores_an_unrelated_new_window(sandbox, monkeypatch):
    t = _task("open_calculator")
    before = t.setup()
    monkeypatch.setattr(T, "window_titles", lambda: {"无标题 - 记事本"})
    assert t.check(before) is False


def test_maximize_needs_a_maximized_window_that_was_not_maximized_before(sandbox, monkeypatch):
    class FakeWindow:
        title = "maximize_me.txt - 记事本"
        isMaximized = True

        def restore(self):
            pass

    t = _task("maximize_window")
    before = t.setup()
    monkeypatch.setattr(S, "_windows_titled", lambda keyword: [FakeWindow()])
    assert t.check(before) is True
    assert t.check({"was_maximized": True}) is False


def test_remove_refuses_paths_outside_scratch(sandbox):
    sandbox.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        S._remove("../outside.txt")
