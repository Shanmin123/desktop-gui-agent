"""LoRA 训练脚本里不依赖模型权重的那部分：样本编码与 loss 屏蔽。

屏蔽错了训练就白跑：提示词那段不置为 -100 的话，损失里混进大量提示词 token，
模型学的是复述提示词而不是给出动作。这个错误不会报异常，只会让效果上不去，
所以单独测。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_lora
from train_lora import IGNORE, encode, load_rows


class FakeTensor:
    """够用的假张量：只要 shape、clone、切片赋值。"""

    def __init__(self, data):
        self.data = list(data)

    @property
    def shape(self):
        return (1, len(self.data))

    def clone(self):
        return FakeTensor(self.data)

    def __setitem__(self, key, value):
        _, sl = key
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else len(self.data)
        for i in range(start, stop):
            self.data[i] = value


class FakeProcessor:
    """按字符数当 token 数，够验证屏蔽长度的逻辑。"""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        text = msgs[0]["content"][1]["text"]
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text=None, images=None, return_tensors=None):
        self.calls.append(text[0])
        return {"input_ids": FakeTensor(list(range(len(text[0]))))}


@pytest.fixture
def row(tmp_path):
    from PIL import Image

    p = tmp_path / "shot.png"
    Image.new("RGB", (64, 48)).save(p)
    return {"kind": "action", "image": str(p), "prompt": "任务：打开浏览器",
            "response": '{"action": {"type": "click", "point": [0.5, 0.5]}}'}


# --- loss 屏蔽 --------------------------------------------------------------


def test_prompt_tokens_are_masked(row):
    proc = FakeProcessor()
    b = encode(proc, row, max_len=10_000)
    labels, ids = b["labels"].data, b["input_ids"].data

    head = proc.apply_chat_template(
        [{"role": "user", "content": [None, {"text": row["prompt"]}]}])
    head_len = len(head)

    assert labels[:head_len] == [IGNORE] * head_len, "提示词那段必须屏蔽"
    assert IGNORE not in labels[head_len:], "回答那段不能屏蔽"
    assert labels[head_len:] == ids[head_len:]


def test_answer_is_not_empty_after_masking(row):
    b = encode(FakeProcessor(), row, max_len=10_000)
    assert sum(1 for v in b["labels"].data if v != IGNORE) > 0


def test_end_token_is_part_of_the_answer(row):
    """回答末尾要带 <|im_end|>，否则模型学不会在哪停。"""
    proc = FakeProcessor()
    encode(proc, row, max_len=10_000)
    assert proc.calls[0].endswith("<|im_end|>")
    assert row["response"] in proc.calls[0]


def test_longer_answer_leaves_more_unmasked(row):
    short = encode(FakeProcessor(), row, max_len=10_000)
    long_row = dict(row, response=row["response"] * 3)
    long = encode(FakeProcessor(), long_row, max_len=10_000)
    n = lambda b: sum(1 for v in b["labels"].data if v != IGNORE)
    assert n(long) > n(short)


def test_longer_prompt_masks_more(row):
    a = encode(FakeProcessor(), row, max_len=10_000)
    b = encode(FakeProcessor(), dict(row, prompt=row["prompt"] * 5), max_len=10_000)
    m = lambda x: sum(1 for v in x["labels"].data if v == IGNORE)
    assert m(b) > m(a)


# --- 超长样本 ---------------------------------------------------------------


def test_too_long_sample_is_skipped(row):
    assert encode(FakeProcessor(), row, max_len=5) is None


def test_sample_within_limit_is_kept(row):
    assert encode(FakeProcessor(), row, max_len=10_000) is not None


# --- 数据加载 ---------------------------------------------------------------


def test_missing_dataset_exits_with_hint(monkeypatch, tmp_path):
    import train_lora

    monkeypatch.setattr(train_lora, "DATA", tmp_path)
    with pytest.raises(SystemExit, match="build_finetune_data"):
        load_rows("train")


def test_rows_with_missing_images_are_dropped(monkeypatch, tmp_path):
    import train_lora

    from PIL import Image

    good = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(good)
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"kind": "action", "image": str(good), "prompt": "p", "response": "r"}) + "\n"
        + json.dumps({"kind": "action", "image": str(tmp_path / "missing.png"),
                      "prompt": "p", "response": "r"}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(train_lora, "DATA", tmp_path)
    rows = load_rows("train")
    assert len(rows) == 1 and rows[0]["image"] == str(good)


def test_limit_is_respected(monkeypatch, tmp_path):
    import train_lora

    from PIL import Image

    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p)
    lines = [json.dumps({"kind": "action", "image": str(p), "prompt": "p", "response": "r"})] * 5
    (tmp_path / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(train_lora, "DATA", tmp_path)
    assert len(load_rows("train", limit=2)) == 2


# --- 按长度分桶 -------------------------------------------------------------


def _rows(lengths):
    return [{"prompt": "x" * n, "response": "", "kind": "action"} for n in lengths]


def test_sortish_batches_keeps_every_sample():
    import random

    from train_lora import sortish_batches

    rows = _rows([10, 500, 30, 2000, 70, 1200, 90, 300])
    out = sortish_batches(rows, 4, random.Random(0))
    assert len(out) == len(rows)
    assert sorted(len(r["prompt"]) for r in out) == sorted(len(r["prompt"]) for r in rows)


def test_sortish_batches_makes_each_batch_length_homogeneous():
    """一批里长度要接近，否则显存碎片攒得快。"""
    import random

    from train_lora import sortish_batches

    rows = _rows(list(range(0, 3200, 100)))   # 32 条，长度 0~3100
    out = sortish_batches(rows, 4, random.Random(1))
    spreads = []
    for i in range(0, len(out), 4):
        ls = [len(r["prompt"]) for r in out[i:i + 4]]
        spreads.append(max(ls) - min(ls))
    # 每批内部跨度不超过 3 个刻度（300），而整体跨度是 3100
    assert max(spreads) <= 300, spreads


def test_sortish_batches_shuffles_batch_order():
    """批之间要乱序，不然就是先练短样本再练长样本。"""
    import random

    from train_lora import sortish_batches

    rows = _rows(list(range(0, 800, 10)))
    a = [len(r["prompt"]) for r in sortish_batches(rows, 4, random.Random(0))]
    b = [len(r["prompt"]) for r in sortish_batches(rows, 4, random.Random(7))]
    assert a != b, "两个不同种子应给出不同的批顺序"
    assert a != sorted(a), "不能是整体升序"


# --- 超参开关（照 SeeClick / ShowUI 的做法加的）-----------------------------


def _parser_args(argv):
    """跑一遍 main 的参数解析，不真的训练。"""
    import argparse
    import inspect
    import re

    src = inspect.getsource(train_lora.main)
    body = src.split("args = ap.parse_args()")[0]
    body = body.split("ap = argparse.ArgumentParser()")[1]
    ap = argparse.ArgumentParser()
    ns = {"ap": ap, "argparse": argparse}
    exec(re.sub(r"^    ", "", body, flags=re.M), ns)
    return ap.parse_args(argv)


def test_lora_alpha_defaults_to_twice_the_rank():
    a = _parser_args(["--rank", "32"])
    assert a.lora_alpha is None      # 空着就在建 LoraConfig 时取 2 倍
    assert _parser_args(["--lora-alpha", "16"]).lora_alpha == 16


def test_target_modules_switch_covers_the_mlp():
    """ShowUI 训 Qwen2-VL 时 LoRA 是挂满整个语言模型的，不只注意力。"""
    assert _parser_args([]).lora_targets == "attn"
    assert _parser_args(["--lora-targets", "all"]).lora_targets == "all"


def test_optimizer_knobs_have_the_old_defaults():
    """默认值必须和加开关之前一致，否则前面几轮的结果就不能比了。"""
    a = _parser_args([])
    assert a.weight_decay == 0.01 and a.adam_beta2 == 0.999
    assert a.scheduler == "onecycle" and a.warmup_ratio == 0.05
