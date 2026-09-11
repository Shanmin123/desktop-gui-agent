"""用 PEFT 对 Qwen2.5-VL 做 LoRA 微调。

对应大纲第 5 周第 2 项。

只训练回答部分：提示词那段的 label 置为 -100。不屏蔽的话损失里混进大量提示词
token，模型学的是复述提示词而不是给出动作。

显存实测（3B、r=16、q/k/v/o、开梯度检查点、batch 1、1024×768 输入）峰值
9.44 GB，本机 12 GB 够用，见 logs/lora_probe.json。

用法：
    python scripts/train_lora.py --limit 40 --epochs 1   # 小样本跑通
    python scripts/train_lora.py                         # 正式训练
    python scripts/train_lora.py --load-in-4bit          # 显存不够时
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "finetune"
OUTPUT = ROOT / "checkpoints"

IGNORE = -100  # 交叉熵忽略的 label


def load_rows(split: str, limit=None) -> list:
    p = DATA / f"{split}.jsonl"
    if not p.is_file():
        raise SystemExit(f"没有 {p}，先跑 scripts/build_finetune_data.py")
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    rows = [r for r in rows if Path(r["image"]).is_file()]
    return rows[:limit] if limit else rows


def encode(processor, row: dict, max_len: int):
    """一条样本 -> (inputs, labels)，labels 只在回答部分有效。"""
    from PIL import Image

    img = Image.open(row["image"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": row["prompt"]}]}]
    head = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = row["response"] + "<|im_end|>"

    full = processor(text=[head + answer], images=[img], return_tensors="pt")
    # 只编码提示词那段，用它的长度定出要屏蔽多少个 token
    head_len = processor(text=[head], images=[img], return_tensors="pt")["input_ids"].shape[1]

    ids = full["input_ids"]
    if ids.shape[1] > max_len:
        return None
    labels = ids.clone()
    labels[:, :head_len] = IGNORE
    full["labels"] = labels
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8, help="梯度累积步数，等效批大小")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条，调试用")
    ap.add_argument("--max-len", type=int, default=2048, help="超长样本直接跳过")
    ap.add_argument("--eval-every", type=int, default=200, help="每多少步在验证集上看一次")
    ap.add_argument("--eval-samples", type=int, default=40)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tag", default="lora")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = load_rows("train", args.limit)
    val = load_rows("val", args.eval_samples)
    kinds = {}
    for r in train:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"训练 {len(train)} 条 {kinds}，验证 {len(val)} 条")

    print(f"加载 {args.model} ……")
    t0 = time.perf_counter()
    kwargs = {"device_map": "cuda"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    else:
        kwargs["dtype"] = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s，"
          f"显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {trainable / 1e6:.1f} M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = math.ceil(len(train) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 1),
                                                pct_start=0.05)

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for r in val:
                b = encode(processor, r, args.max_len)
                if b is None:
                    continue
                losses.append(model(**b.to("cuda")).loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    OUTPUT.mkdir(exist_ok=True)
    log = {"args": vars(args), "train_size": len(train), "kinds": kinds, "steps": []}
    print(f"\n开始训练：{args.epochs} 轮，累积 {args.accum} 步，共约 {total} 次更新")
    model.train()
    seen = skipped = 0
    running = []
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        random.shuffle(train)
        for r in train:
            b = encode(processor, r, args.max_len)
            if b is None:
                skipped += 1
                continue
            loss = model(**b.to("cuda")).loss / args.accum
            loss.backward()
            running.append(loss.item() * args.accum)
            seen += 1

            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step = seen // args.accum
                if step % 10 == 0 or step == 1:
                    avg = sum(running[-args.accum * 10:]) / len(running[-args.accum * 10:])
                    print(f"  轮 {epoch + 1} 更新 {step}/{total}  loss {avg:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  "
                          f"峰值 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                    log["steps"].append({"step": step, "epoch": epoch + 1, "loss": round(avg, 4)})
                if args.eval_every and step % args.eval_every == 0:
                    v = evaluate()
                    print(f"    验证 loss {v:.4f}")
                    log["steps"][-1]["val_loss"] = round(v, 4)

    log["skipped_too_long"] = skipped
    log["final_val_loss"] = round(evaluate(), 4)
    log["train_minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    log["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    print(f"\n训练完成：{log['train_minutes']} 分钟，验证 loss {log['final_val_loss']}，"
          f"峰值 {log['peak_vram_gb']} GB，跳过超长样本 {skipped} 条")

    out = OUTPUT / args.tag
    model.save_pretrained(out)
    (ROOT / "logs" / f"train_{args.tag}.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"权重存到 {out}")


if __name__ == "__main__":
    main()
