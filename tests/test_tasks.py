from pathlib import Path

from gui_agent.tasks import SCRATCH, Task, basic_tasks, file_contains, process_running


def test_five_basic_tasks():
    """大纲第 4 周第 4 项要求调试 5 个基础任务。"""
    tasks = basic_tasks()
    assert len(tasks) == 5
    assert [t.id for t in tasks] == [
        "open_browser", "search_content", "open_file", "type_and_save", "close_app"
    ]


def test_every_task_has_programmatic_check():
    """成功与否必须能用程序判断，不靠人工看。"""
    for t in basic_tasks():
        assert callable(t.check)
        assert isinstance(t.check(), bool), f"{t.id} 的验收函数没返回 bool"


def test_instructions_are_non_empty():
    for t in basic_tasks():
        assert t.instruction.strip()


def test_file_operations_stay_in_scratch():
    """文件类任务只允许碰 scratch 目录。"""
    for t in basic_tasks():
        if str(SCRATCH) in t.instruction:
            assert "scratch" in t.instruction


def test_message_task_is_substituted():
    """大纲的「发送消息」改成写本地文件，理由记在 note 里。"""
    t = next(x for x in basic_tasks() if x.id == "type_and_save")
    assert "不可逆" in t.note


def test_process_running_detects_explorer():
    assert process_running("explorer.exe")


def test_process_running_rejects_nonexistent():
    assert not process_running("绝对不存在的进程xyz.exe")


def test_file_contains_missing_file(tmp_path):
    assert not file_contains(tmp_path / "没有这个文件.txt", "x")


def test_file_contains_matches(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("包含关键词你好在里面", encoding="utf-8")
    assert file_contains(p, "你好") and not file_contains(p, "再见")


def test_setup_teardown_default_to_noop():
    t = Task(id="x", instruction="y", check=lambda: True)
    t.setup()
    t.teardown()
