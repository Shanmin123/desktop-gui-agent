"""报告图表：缺日志的图跳过，有日志就出图。不碰真实的 logs/ 和 docs/figures/。"""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_charts", ROOT / "scripts" / "make_charts.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    logs, out = tmp_path / "logs", tmp_path / "figures"
    logs.mkdir()
    out.mkdir()
    monkeypatch.setattr(C, "LOGS", logs)
    monkeypatch.setattr(C, "OUT", out)
    return logs, out


def test_every_chart_is_skipped_when_its_logs_are_missing(dirs):
    plt = C._plt()
    for fn in (C.chart_screenspot, C.chart_tokens, C.chart_screenagent, C.chart_prompts,
               C.chart_training, C.chart_suite):
        assert fn(plt) is False, fn.__name__
    assert list(dirs[1].iterdir()) == []


def test_prompt_chart_is_drawn_from_tune_prompt_logs(dirs):
    logs, out = dirs
    (logs / "prompt_q35_2s.json").write_text(json.dumps({
        "n": 60, "mode": "two_stage",
        "results": {"base": {"type_accuracy": 0.5, "joint_accuracy": 0.3},
                    "keyboard": {"type_accuracy": 0.6, "joint_accuracy": 0.35}},
    }), encoding="utf-8")
    assert C.chart_prompts(C._plt()) is True
    assert (out / "prompt_variants.png").stat().st_size > 0


def test_suite_chart_groups_records_by_level(dirs):
    logs, out = dirs
    records = [{"task": "a", "level": "T1", "passed": True}, {"task": "b", "level": "T3", "passed": False}]
    (logs / "tasks_vm_q35_2sp_suite_suite.json").write_text(json.dumps({"records": records}),
                                                             encoding="utf-8")
    assert C.chart_suite(C._plt()) is True
    assert (out / "suite_by_level.png").exists()


def test_joint_accuracy_is_recomputed_from_cases_for_old_logs():
    d = {"n": 3, "cases": [{"gt": "click", "pred": "click", "dist": 0.05},
                           {"gt": "click", "pred": "click", "dist": 0.2},
                           {"gt": "type", "pred": "type", "dist": None}]}
    assert C.joint_accuracy(d) == pytest.approx(2 / 3)


def test_suite_labels_are_readable():
    """图例名不要直接用日志文件名。"""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_charts", root / "scripts" / "make_charts.py")
    charts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(charts)

    assert charts.suite_label("tasks_live_q25_proj_suite") == "对齐层全量 + 语言模型 LoRA"
    assert charts.suite_label("tasks_vm_q25_base_suite") == "基座"
    assert charts.suite_label("tasks_something_else_suite") == "something_else"


SA_CASES = [
    {"i": 0, "gt": "click", "pred": "click", "dist": 0.05, "gt_text": None, "pred_text": None},
    {"i": 1, "gt": "click", "pred": "click", "dist": 0.12, "gt_text": None, "pred_text": None},
    {"i": 2, "gt": "type", "pred": "click", "dist": None, "gt_text": "abc", "pred_text": None},
    {"i": 3, "gt": "hotkey", "pred": "hotkey", "dist": None, "gt_text": "ctrl+s", "pred_text": "ctrl+s"},
]


def test_screenagent_chart_uses_the_community_metric_names(dirs):
    """图上的指标要和报告、score_actions.py 一致：Op.F1 与 Step SR，不是自己起的名字。"""
    logs, out = dirs
    (logs / "screenagent_q25_proj_2s.json").write_text(
        json.dumps({"n": len(SA_CASES), "cases": SA_CASES}), encoding="utf-8")
    assert C.chart_screenagent(C._plt()) is True
    assert (out / "screenagent_metrics.png").stat().st_size > 0
    assert C.SCREENAGENT_METRICS == ["Op.F1 macro", "Op.F1 micro", "Step SR ≤0.10", "Step SR ≤0.14"]


def test_screenagent_metrics_match_the_eval_scripts_own_functions():
    """图里的数必须和 eval_screenagent.py 算出来的一样，不能各算一套。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_screenagent", ROOT / "scripts" / "eval_screenagent.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    f1 = ev.op_f1(SA_CASES)
    got = C.action_metrics({"n": len(SA_CASES), "cases": SA_CASES})
    assert got == pytest.approx([f1["macro_f1"], f1["micro_f1"],
                                 ev.step_success(SA_CASES, 0.10), ev.step_success(SA_CASES, 0.14)])
