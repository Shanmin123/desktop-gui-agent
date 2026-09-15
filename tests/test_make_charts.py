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
