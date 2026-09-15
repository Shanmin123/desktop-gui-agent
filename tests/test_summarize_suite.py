"""评测集汇总：Wilson 区间，成功率 / 耗时 / 错误率的口径，dry-run 日志不计入。"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("summarize_suite", ROOT / "scripts" / "summarize_suite.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _run(task, level, passed, oks, wall=10.0, retries=0):
    return {"task": task, "level": level, "run": 1, "passed": passed, "steps": len(oks),
            "wall_time": wall, "retries": retries,
            "trajectory": {"steps": [{"ok": ok} for ok in oks]}}


def test_wilson_interval_matches_the_textbook_values():
    lo, hi = S.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert S.wilson(0, 10)[0] == 0.0
    assert S.wilson(10, 10)[1] == 1.0
    assert S.wilson(0, 0) == (0.0, 0.0)


def test_summarize_counts_error_steps_and_keeps_aborted_runs_in_the_denominator():
    records = [
        _run("a", "T1", True, [True, True], wall=8.0),
        _run("b", "T2", False, [False, True, False], wall=20.0, retries=1),
        {"task": "c", "level": "T3", "run": 1, "passed": False, "steps": 0, "wall_time": 0.0,
         "actions": [], "error": "setup 失败：OSError"},
    ]
    o = S.summarize(records)
    assert (o["runs"], o["passed"], o["aborted"]) == (3, 1, 1)
    assert o["success_rate"] == pytest.approx(1 / 3)
    assert (o["error_steps"], o["steps"]) == (2, 5)
    assert o["error_rate"] == pytest.approx(0.4)
    assert o["avg_wall_time"] == pytest.approx(14.0), "没跑起来的那次没有耗时，不该拉低均值"
    assert o["avg_wall_time_passed"] == pytest.approx(8.0)
    assert o["retries"] == 1


def test_by_level_and_by_task_group_repeated_runs():
    records = [_run("a", "T1", True, [True]), _run("a", "T1", False, [False]),
               _run("b", "T3", True, [True])]
    levels = S.by_level(records)
    assert set(levels) == {"T1", "T3"}
    assert (levels["T1"]["passed"], levels["T1"]["runs"]) == (1, 2)
    assert S.by_task(records)["a"] == {"level": "T1", "runs": 2, "passed": 1}


def test_main_skips_dry_run_logs_and_writes_a_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LOGS", tmp_path)
    (tmp_path / "tasks_vm_q35_2sp_suite_suite.json").write_text(json.dumps({
        "live": True, "model": "Qwen/Qwen3.5-4B", "resolution": "1280x720",
        "records": [_run("a", "T1", True, [True])]}), encoding="utf-8")
    (tmp_path / "tasks_x_suite_dryrun.json").write_text(json.dumps({
        "live": False, "records": [_run("a", "T1", False, [True])]}), encoding="utf-8")

    configs = S.main([])
    assert list(configs) == ["vm_q35_2sp_suite_suite"]
    saved = json.loads((tmp_path / "suite_summary.json").read_text(encoding="utf-8"))
    assert saved["vm_q35_2sp_suite_suite"]["overall"]["passed"] == 1
