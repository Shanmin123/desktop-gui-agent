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
        self.template_kwargs = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.template_kwargs.append(kw)
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


# --- 迁移到 Qwen3.5 之后加的 ------------------------------------------------


def test_encode_turns_thinking_off(row):
    """训练和推理都要关思考模式，否则 Qwen3.5 的提示词两边长得不一样。"""
    proc = FakeProcessor()
    encode(proc, row, max_len=10_000)
    assert proc.template_kwargs and all(kw.get("enable_thinking") is False for kw in proc.template_kwargs)


QWEN25_NAMES = [
    "model.visual.blocks.0.attn.qkv", "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.gate_proj", "model.visual.blocks.0.mlp.up_proj",
    "model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj", "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj", "lm_head",
    "model.visual.merger.mlp.0", "model.visual.merger.mlp.2",
]
QWEN35_NAMES = [
    "model.visual.blocks.0.attn.qkv", "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.linear_fc1", "model.visual.blocks.0.mlp.linear_fc2",
    "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_z",
    "model.language_model.layers.0.linear_attn.in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.3.mlp.gate_proj", "lm_head",
    "model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2",
]


def _matched(regex, names):
    import re as _re

    return {n for n in names if _re.fullmatch(regex, n)}


def test_lora_attn_targets_on_qwen25_are_exactly_qkvo():
    """Qwen2.5 上只挂注意力的结果要和迁移前的 q/k/v/o 列表完全一样，老结果才可复现。"""
    got = _matched(train_lora.lora_target_regex(QWEN25_NAMES, "attn"), QWEN25_NAMES)
    assert got == {n for n in QWEN25_NAMES if n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")}


def test_lora_attn_targets_on_qwen35_include_linear_attention():
    got = _matched(train_lora.lora_target_regex(QWEN35_NAMES, "attn"), QWEN35_NAMES)
    suffixes = {n.rsplit(".", 1)[-1] for n in got}
    assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "o_proj"} <= suffixes
    assert not any("visual" in n for n in got) and "lm_head" not in got
    assert not any(n.endswith("gate_proj") for n in got)


def test_lora_all_targets_never_touch_the_vision_tower():
    """Qwen2.5-VL 视觉塔的 MLP 也叫 gate/up_proj，按名字列表会被一起挂上。"""
    got = _matched(train_lora.lora_target_regex(QWEN25_NAMES, "all"), QWEN25_NAMES)
    assert "model.language_model.layers.0.mlp.gate_proj" in got
    assert not any("visual" in n for n in got)


def test_lora_targets_fail_loudly_when_nothing_matches():
    with pytest.raises(ValueError):
        train_lora.lora_target_regex(["model.visual.blocks.0.attn.qkv"], "attn")


def test_est_tokens_counts_image_by_patch_factor(tmp_path):
    from PIL import Image

    p = tmp_path / "x.png"
    Image.new("RGB", (320, 320)).save(p)
    row = {"image": str(p), "prompt": "", "response": ""}
    tok = lambda s: {"input_ids": []}
    assert train_lora.est_tokens(row, tok, 10_000, factor=32) == 100
    assert train_lora.est_tokens(row, tok, 10_000, factor=28) == (320 * 320) // (28 * 28)
    assert train_lora.est_tokens(row, tok, 50, factor=32) == 50


def test_lora_qkvo_targets_skip_linear_attention_on_qwen35():
    got = _matched(train_lora.lora_target_regex(QWEN35_NAMES, "qkvo"), QWEN35_NAMES)
    assert got == {"model.language_model.layers.3.self_attn.q_proj",
                   "model.language_model.layers.3.self_attn.o_proj"}


def test_lora_qkvo_equals_attn_on_qwen25():
    """Qwen2.5 没有线性注意力，两种写法挂到的层一样。"""
    assert (_matched(train_lora.lora_target_regex(QWEN25_NAMES, "qkvo"), QWEN25_NAMES)
            == _matched(train_lora.lora_target_regex(QWEN25_NAMES, "attn"), QWEN25_NAMES))


def test_answer_only_loss_matches_full_sequence():
    """只留回答段的 logits 算出来的 loss，要和整段算的一样。"""
    import torch
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.manual_seed(0)
    L, V, head = 12, 50, 7
    logits = torch.randn(1, L, V)
    labels = torch.randint(0, V, (1, L))
    labels[:, :head] = train_lora.IGNORE
    b = train_lora.answer_only({"input_ids": torch.zeros(1, L, dtype=torch.long), "labels": labels})
    k = b["logits_to_keep"]
    assert k == L - head + 1 and b["labels"].shape[1] == k
    assert torch.allclose(ForCausalLMLoss(logits, labels, V),
                          ForCausalLMLoss(logits[:, -k:], b["labels"], V))


def test_answer_only_leaves_the_inputs_whole():
    import torch

    ids = torch.arange(5).unsqueeze(0)
    labels = torch.tensor([[train_lora.IGNORE] * 3 + [3, 4]])
    b = train_lora.answer_only({"input_ids": ids, "labels": labels})
    assert torch.equal(b["input_ids"], ids) and b["logits_to_keep"] == 3


def test_lora_nogate_targets_skip_the_per_head_gate_projections():
    """in_proj_a / in_proj_b 每个头只输出一个标量（衰减门和写入强度），不是注意力投影。"""
    got = _matched(train_lora.lora_target_regex(QWEN35_NAMES, "attn_nogate"), QWEN35_NAMES)
    suffixes = {n.rsplit(".", 1)[-1] for n in got}
    assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "o_proj"} <= suffixes
    assert not ({"in_proj_a", "in_proj_b"} & suffixes)


def test_lora_nogate_equals_attn_on_qwen25():
    """Qwen2.5 没有门控投影，两种写法挂到的层一样，老结果仍可复现。"""
    assert (_matched(train_lora.lora_target_regex(QWEN25_NAMES, "attn_nogate"), QWEN25_NAMES)
            == _matched(train_lora.lora_target_regex(QWEN25_NAMES, "attn"), QWEN25_NAMES))


def test_lora_targets_can_add_the_vision_merger():
    """对齐层（merger）默认不挂，加上之后语言模型那部分不变，视觉塔的 block 仍然不碰。"""
    plain = _matched(train_lora.lora_target_regex(QWEN35_NAMES, "attn"), QWEN35_NAMES)
    got = _matched(train_lora.lora_target_regex(QWEN35_NAMES, "attn", with_merger=True), QWEN35_NAMES)
    assert got - plain == {"model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"}
    assert not any(".blocks." in n for n in got)


def test_lora_merger_targets_cover_both_generations_naming():
    """Qwen2.5-VL 的对齐层叫 merger.mlp.0/2，Qwen3.5 叫 merger.linear_fc1/2。"""
    got = _matched(train_lora.lora_target_regex(QWEN25_NAMES, "attn", with_merger=True), QWEN25_NAMES)
    assert {"model.visual.merger.mlp.0", "model.visual.merger.mlp.2"} <= got
    assert not any(".blocks." in n for n in got)


def test_lora_merger_fails_loudly_when_the_model_has_none():
    with pytest.raises(ValueError):
        train_lora.lora_target_regex(["model.language_model.layers.0.self_attn.q_proj"],
                                     "attn", with_merger=True)


def test_merger_modules_for_full_training_found_in_both_generations():
    """全量训练对齐层时交给 PEFT 的 modules_to_save，两代都定位到 visual.merger。"""
    assert train_lora.merger_modules(QWEN25_NAMES) == ["model.visual.merger"]
    assert train_lora.merger_modules(QWEN35_NAMES) == ["model.visual.merger"]


def test_merger_modules_fail_loudly_when_the_model_has_none():
    with pytest.raises(ValueError):
        train_lora.merger_modules(["model.language_model.layers.0.self_attn.q_proj"])


def test_best_step_picks_the_lowest_validation_loss():
    """579 条数据训 6 轮会过拟合，交付的应该是验证 loss 最低那一步的权重。"""
    steps = [{"step": 50, "loss": 1.5}, {"step": 100, "loss": 1.2, "val_loss": 1.11},
             {"step": 150, "loss": 0.9, "val_loss": 1.23}, {"step": 200, "loss": 0.5, "val_loss": 1.40}]
    assert train_lora.best_step(steps) == (100, 1.11)


def test_best_step_returns_nothing_when_never_validated():
    assert train_lora.best_step([{"step": 50, "loss": 1.5}]) == (None, None)


def test_merger_skip_list_keeps_lm_head_unquantized():
    """显式传 llm_int8_skip_modules 会覆盖默认列表，漏掉 lm_head 会让 4-bit 加载在前向时断言失败。"""
    assert "lm_head" in train_lora.MERGER_LINEARS
    assert {"mlp.0", "mlp.2"} <= set(train_lora.MERGER_LINEARS)
