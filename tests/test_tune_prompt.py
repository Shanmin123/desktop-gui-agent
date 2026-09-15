"""提示词变体对比脚本：一段式、两段式两条路径用假模型走通，结果文件字段齐全。不加载真模型、不跑 OCR。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from gui_agent.perception import imwrite

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tune_prompt", ROOT / "scripts" / "tune_prompt.py")
TP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TP)


class FakeVLM:
    prompts = []

    def __init__(self, model, adapter=None):
        FakeVLM.prompts = []

    def coord_size(self, height, width):
        return 1000, 1000

    def ask(self, image, prompt, max_new_tokens=256):
        FakeVLM.prompts.append(prompt)
        if "不要写坐标" in prompt:  # 两段式第一问
            return '{"thought": "点保存", "action": {"type": "click", "target": "保存按钮"}}'
        return '{"thought": "点保存", "action": {"type": "click", "point": [500, 520]}}'

    def locate(self, image, target):
        return 0.5, 0.52


class FakePerception:
    def ocr(self, img):
        return []

    def close(self):
        pass


def _records(tmp_path):
    img = tmp_path / "shot.png"
    imwrite(str(img), np.zeros((768, 1024, 3), dtype=np.uint8))
    return [
        {"session_id": "s1", "image": str(img), "instruction": "save", "instruction_zh": "保存文件",
         "action": {"type": "click", "point": [0.5, 0.5]}},
        {"session_id": "s1", "image": str(img), "instruction": "type", "instruction_zh": "输入你好",
         "action": {"type": "type", "text": "你好"}},
    ]


def _run(tmp_path, monkeypatch, *argv):
    monkeypatch.setattr(TP, "LocalQwenVL", FakeVLM)
    monkeypatch.setattr(TP, "Perception", FakePerception)
    monkeypatch.setattr(TP, "ROOT", tmp_path)
    records = _records(tmp_path)
    monkeypatch.setattr(TP, "load", lambda limit=None: records[:limit] if limit else records)
    monkeypatch.setattr(sys, "argv", ["tune_prompt.py", *argv])
    TP.main()


def test_two_stage_mode_runs_every_target_variant(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "--mode", "two_stage", "--tag", "t2")
    out = json.loads((tmp_path / "logs" / "prompt_t2.json").read_text(encoding="utf-8"))
    assert out["mode"] == "two_stage"
    assert set(out["results"]) == set(TP.TARGET_VARIANTS)
    base = out["results"]["base"]
    assert base["type_accuracy"] == 0.5            # 第一条点击对，第二条真值是打字
    assert base["joint_accuracy"] == 0.5           # 定位点距真值 0.02，算点准
    assert base["parse_failures"] == 0
    assert any("ctrl+s" in p for p in FakeVLM.prompts), "keyboard 变体的提示词要真的送进模型"


def test_one_stage_mode_still_runs_with_the_element_list_configs(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "--variants", "base,no_elements", "--limit", "1", "--tag", "t1")
    out = json.loads((tmp_path / "logs" / "prompt_t1.json").read_text(encoding="utf-8"))
    assert out["mode"] == "one_stage" and out["n"] == 1
    assert out["results"]["no_elements"]["with_elements"] is False
    assert out["results"]["base"]["type_accuracy"] == 1.0
