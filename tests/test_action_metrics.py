"""动作评测的社区口径指标：Op.F1（Mind2Web / SeeClick 那套）和 Step SR（含 AITW 的 0.14 阈值）。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_screenagent", ROOT / "scripts" / "eval_screenagent.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def _case(gt, pred, dist=None, gt_text=None, pred_text=None):
    return {"gt": gt, "pred": pred, "dist": dist, "gt_text": gt_text, "pred_text": pred_text}


def test_micro_f1_equals_type_accuracy_when_no_text_involved():
    cases = [_case("click", "click", 0.02), _case("click", "type"), _case("scroll", "scroll", 0.01)]
    assert E.op_f1(cases)["micro_f1"] == 2 / 3


def test_macro_f1_gives_small_classes_the_same_weight():
    """点击占多数时 micro 会被拉高，macro 才看得出少数类没学会。"""
    cases = [_case("click", "click", 0.01)] * 9 + [_case("scroll", "click", 0.01)]
    m = E.op_f1(cases)
    assert m["micro_f1"] == 0.9
    assert m["macro_f1"] < 0.6          # scroll 一条没对，macro 立刻掉下来
    assert m["per_type"]["scroll"]["f1"] == 0.0


def test_keyboard_actions_need_the_same_text():
    """Mind2Web 对 TYPE 是连输入内容一起比的，只对上类型不算对。"""
    same = [_case("type", "type", None, "你好", "你好")]
    diff = [_case("type", "type", None, "你好", "再见")]
    assert E.op_f1(same)["micro_f1"] == 1.0
    assert E.op_f1(diff)["micro_f1"] == 0.0
    assert E.step_success(same, 0.10) == 1.0
    assert E.step_success(diff, 0.10) == 0.0


def test_missing_ground_truth_text_does_not_penalize():
    """真值没写内容（老日志就没存），这一项不卡。"""
    assert E.op_f1([_case("hotkey", "hotkey")])["micro_f1"] == 1.0


def test_step_success_uses_the_distance_threshold():
    cases = [_case("click", "click", 0.12)]
    assert E.step_success(cases, 0.10) == 0.0
    assert E.step_success(cases, 0.14) == 1.0      # AITW 的 14% 口径


def test_parse_failures_count_against_every_metric():
    cases = [_case("click", "click", 0.01), _case("click", "解析失败")]
    assert E.op_f1(cases)["micro_f1"] == 0.5
    assert E.step_success(cases, 0.14) == 0.5
