"""Mind2Web 的 operation 字段解析。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_mind2web import COLUMNS, parse_op


def test_click_operation():
    op, value = parse_op('{"original_op": "CLICK", "value": "", "op": "CLICK"}')
    assert op == "CLICK" and value == ""


def test_type_operation_keeps_value():
    op, value = parse_op('{"original_op": "TYPE", "value": "Brooklyn Central", "op": "TYPE"}')
    assert op == "TYPE" and value == "Brooklyn Central"


def test_select_operation():
    op, _ = parse_op('{"original_op": "SELECT", "value": "Economy", "op": "SELECT"}')
    assert op == "SELECT"


def test_falls_back_to_op_when_original_missing():
    assert parse_op('{"op": "HOVER"}')[0] == "HOVER"


def test_already_parsed_dict_is_accepted():
    assert parse_op({"original_op": "CLICK", "value": "x"}) == ("CLICK", "x")


@pytest.mark.parametrize("bad", ["", "不是 JSON", "{坏的", None])
def test_unparseable_operation_returns_empty(bad):
    """脏数据不能让整个分片的处理中断。"""
    assert parse_op(bad) == ("", "")


def test_null_value_becomes_empty_string():
    assert parse_op('{"original_op": "CLICK", "value": null}') == ("CLICK", "")


def test_columns_exclude_the_heavy_ones():
    """整份 13.6 GB 都在截图和 HTML 上，读元数据不能把它们带上。"""
    for heavy in ("screenshot", "raw_html", "cleaned_html", "pos_candidates", "neg_candidates"):
        assert heavy not in COLUMNS
    for needed in ("annotation_id", "confirmed_task", "operation", "action_reprs"):
        assert needed in COLUMNS
