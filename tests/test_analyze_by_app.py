"""ScreenAgent test 步骤按应用归类：关键词规则和逐类统计。"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_by_app", ROOT / "scripts" / "analyze_by_app.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


@pytest.mark.parametrize("instruction,app", [
    ("Find the top 5 most profitable days in the financial_data table", "表格数据"),
    ("Insert a 5x6 table", "办公文档"),
    ("Draw a triangle on an image using GIMP", "图像编辑"),
    ("Insert a triangle", "办公文档"),
    ("Install xeyes in the command line", "终端与代码"),
    ("Find the wrong line of code and fix it", "终端与代码"),
    ("Please play the Find the Difference game on the screen", "游戏"),
    ("Open Calculator and calculate the result of 50 minus 8", "系统工具"),
    ("Rename a target txt document", "系统工具"),
    ("Jump the PDF to page 5", "办公文档"),
    ("Download a paper on object detection from the TPAMI journal using Bing search engine", "浏览器与网页"),
    ("Book a hotel in Sanya online", "浏览器与网页"),
])
def test_app_of_follows_the_keyword_rules(instruction, app):
    assert A.app_of(instruction) == app


def test_every_test_step_gets_one_known_app():
    apps = A.apps_of_test_steps()
    assert len(apps) == 353
    assert set(apps) <= set(A.APPS)


def test_by_app_counts_type_and_joint_accuracy():
    apps = ["浏览器与网页", "浏览器与网页", "游戏"]
    cases = [
        {"i": 0, "gt": "click", "pred": "click", "dist": 0.05},
        {"i": 1, "gt": "click", "pred": "click", "dist": 0.30},
        {"i": 2, "gt": "type", "pred": "click", "dist": None},
    ]
    stats = A.by_app(cases, apps)
    assert stats["浏览器与网页"] == {"n": 2, "type_accuracy": 1.0, "joint_accuracy": 0.5}
    assert stats["游戏"] == {"n": 1, "type_accuracy": 0.0, "joint_accuracy": 0.0}


def test_every_test_step_belongs_to_one_of_70_sessions():
    sessions = A.sessions_of_test_steps()
    assert len(sessions) == 353
    assert len(set(sessions)) == 70


def test_overall_counts_a_session_only_when_every_step_is_right():
    sessions = ["s1", "s1", "s2", "s3"]
    cases = [
        {"i": 0, "gt": "click", "pred": "click", "dist": 0.05},
        {"i": 1, "gt": "type", "pred": "type", "dist": None},       # s1 两步都对
        {"i": 2, "gt": "click", "pred": "click", "dist": 0.30},     # s2 类型对，点偏了
        {"i": 3, "gt": "click", "pred": "解析失败", "dist": None},   # s3 拿不出可执行的动作
    ]
    o = A.overall(cases, sessions)
    assert (o["sessions"], o["sessions_done"]) == (3, 1)
    assert o["session_success"] == pytest.approx(1 / 3)
    assert o["type_accuracy"] == pytest.approx(3 / 4)
    assert o["joint_accuracy"] == pytest.approx(2 / 4)
    assert o["unexecutable_rate"] == pytest.approx(1 / 4)
