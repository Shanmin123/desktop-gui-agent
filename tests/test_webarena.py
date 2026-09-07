"""WebArena 任务规格 -> 本项目格式。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_webarena import to_record

TASK = {
    "sites": ["shopping_admin"],
    "task_id": 0,
    "require_login": True,
    "intent_template": "What is the top-{{n}} best-selling product in {{year}}",
    "instantiation_dict": {"n": 1, "year": 2022},
    "intent": "What is the top-1 best-selling product in 2022",
    "intent_template_id": 279,
    "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "x"}},
}


def test_keeps_instruction_and_template():
    r = to_record(TASK)
    assert r["instruction"] == "What is the top-1 best-selling product in 2022"
    assert r["template_id"] == 279 and "{{n}}" in r["template"]
    assert r["params"] == {"n": 1, "year": 2022}


def test_keeps_check_types():
    """验收方式是本项目设计验收条件的参照，要留下。"""
    assert to_record(TASK)["check_types"] == ["string_match"]


def test_marks_source():
    assert to_record(TASK)["source"] == "webarena"


def test_missing_fields_do_not_raise():
    r = to_record({})
    assert r["instruction"] == "" and r["sites"] == [] and r["check_types"] == []
    assert r["requires_login"] is False


def test_eval_none_is_tolerated():
    r = to_record({"task_id": 5, "eval": None})
    assert r["check_types"] == [] and r["task_id"] == 5


@pytest.mark.parametrize("types", [
    ["string_match"], ["url_match"], ["program_html"],
    ["string_match", "program_html"],
])
def test_all_three_check_types_pass_through(types):
    assert to_record({"eval": {"eval_types": types}})["check_types"] == types
