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
import re
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


# 注意力里的投影层。全注意力是 q/k/v/o；Qwen3.5 另有 24 层线性注意力（Gated DeltaNet），
# 投影叫 in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj——只挂 q/k/v/o 的话
# 32 层里只碰得到 8 层。
ATTENTION_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
# Gated DeltaNet 的状态门：in_proj_b 出 beta = sigmoid(b)（写入强度），in_proj_a 出
# g = -exp(A_log) * softplus(a + dt_bias)（遗忘率），每个头只输出一个标量（2560 -> 32）。
# r=16 挂上去秩已到满秩一半，旁路参数是原矩阵的 51%，占全部 LoRA 参数 13.8%，
# 而这两层的权重只占被挂层的 0.30%。attn_nogate 就是把它们摘掉的对照。
GATE_PROJ = ("in_proj_a", "in_proj_b")


def lora_target_regex(linear_names, which: str = "attn", with_merger: bool = False) -> str:
    """语言模型里实际存在的投影层，拼成 PEFT 的 target_modules 正则。

    视觉塔一律不挂：Qwen2.5-VL 视觉塔的 MLP 也叫 gate_proj / up_proj / down_proj，
    按名字列表匹配会连它一起挂上。
    """
    if which == "qkvo":
        # 只挂全注意力。Qwen3.5 里 32 层只有 8 层是全注意力，用来对照「线性注意力要不要挂」
        want = ("q_proj", "k_proj", "v_proj", "o_proj")
    elif which == "attn_nogate":
        want = tuple(n for n in ATTENTION_PROJ if n not in GATE_PROJ)
    else:
        want = ATTENTION_PROJ + (MLP_PROJ if which == "all" else ())
    found = sorted({n.rsplit(".", 1)[-1] for n in linear_names
                    if "visual" not in n and n.rsplit(".", 1)[-1] in want})
    if not found:
        raise ValueError(f"模型里找不到要挂 LoRA 的投影层（{which}）")
    regex = r"^(?!.*visual).*\.(" + "|".join(found) + r")$"
    if with_merger:
        # 视觉塔的对齐层（projection）。两代叫法不同：Qwen2.5-VL 是 merger.mlp.0 / mlp.2
        # （nn.Sequential），Qwen3.5 是 merger.linear_fc1 / linear_fc2，所以按模型里的实际名字拼。
        names = sorted({n.split(".merger.", 1)[1] for n in linear_names if ".merger." in n})
        if not names:
            raise ValueError("模型里找不到视觉塔的对齐层（merger）")
        regex += r"|^.*\.merger\.(" + "|".join(re.escape(x) for x in names) + r")$"
    return regex


# 全量训练对齐层时要让它不被量化。transformers 的跳过规则是按线性层自己的名字匹配，
# 写模块路径（"visual"）或父模块名（"merger"）都不生效，实测要写到叶子这一层。
# 两代的叫法都列上：Qwen2.5-VL 是 mlp.0 / mlp.2，Qwen3.5 是 linear_fc1 / linear_fc2。
# lm_head 必须一起列上：显式传 llm_int8_skip_modules 会覆盖 transformers 的默认跳过列表，
# 而默认列表里本来就有 lm_head。它和词嵌入共享权重，被量化后 bitsandbytes 会断言失败。
MERGER_LINEARS = ["lm_head", "mlp.0", "mlp.2", "linear_fc1", "linear_fc2"]


def best_step(steps: list) -> tuple:
    """验证 loss 最低的那次更新，返回 (步数, loss)；没验证过就返回 (None, None)。

    579 条数据配几千万可训练参数，实测两轮左右就到最优，再训验证 loss 会回升
    （150 步 1.11 -> 300 步 1.23）。交付的应该是这一步的权重，不是最后一步的。
    """
    evals = [(s["step"], s["val_loss"]) for s in steps if s.get("val_loss") is not None]
    return min(evals, key=lambda x: x[1]) if evals else (None, None)


def merger_modules(linear_names) -> list:
    """全量训练对齐层时交给 PEFT 的 modules_to_save。

    两代的对齐层路径都是 visual.merger，只是里面的线性层叫法不同（Qwen2.5-VL 是
    mlp.0 / mlp.2，Qwen3.5 是 linear_fc1 / linear_fc2），所以按整个 merger 模块来存。
    """
    names = {n.split(".merger.")[0] + ".merger" for n in linear_names if ".merger." in n}
    if not names:
        raise ValueError("模型里找不到视觉塔的对齐层（merger）")
    return sorted(names)


def est_tokens(row: dict, tokenizer, max_pixels_blocks: int, factor: int = 28) -> int:
    """估一条样本的 token 数：文本实算，图片按 factor×factor 的块数算（不超过上限）。

    只读图片头拿尺寸，不解码，1000 多条不到一秒。
    """
    from PIL import Image

    with Image.open(row["image"]) as im:
        w, h = im.size
    return (len(tokenizer(row["prompt"] + row["response"])["input_ids"])
            + min(max_pixels_blocks, (w * h) // (factor * factor)))


def sortish_batches(rows: list, accum: int, rng) -> list:
    """按长度排好再切成一批批，然后打乱批的顺序。

    样本长度差一倍多（定位约 1300 token、动作到 2800），一次更新里混着长短样本时
    缓存分配器的碎片攒得很快，整卡占满之后要么变慢要么停住。同一批里长度接近，
    显存块能反复复用。批之间的顺序仍然是随机的，不会退化成「先练定位再练动作」。
    """
    order = sorted(rows, key=lambda r: len(r["prompt"]) + len(r["response"]))
    batches = [order[i:i + accum] for i in range(0, len(order), accum)]
    rng.shuffle(batches)
    return [r for b in batches for r in b]


def encode(processor, row: dict, max_len: int):
    """一条样本 -> (inputs, labels)，labels 只在回答部分有效。"""
    from PIL import Image

    img = Image.open(row["image"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": row["prompt"]}]}]
    # 和推理时一样关掉思考模式：Qwen3.5 的生成提示因此以空的 <think></think> 结尾，
    # 回答紧接其后，屏蔽长度才和推理时的输入对得上
    head = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
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


def answer_only(batch) -> dict:
    """只在回答那段算 logits，loss 与整段计算相同。

    labels 前面是屏蔽掉的提示词，那些位置的 logits 不进 loss。Qwen3.5 词表 24.8 万，
    一条 1000 token 的样本整段算 logits 要 1 GB，loss 里升到 float32、再加上梯度又是两份；
    回答只有几十到一百多 token。位置 p 的 logits 预测第 p+1 个 token，所以从第一个回答
    token 的前一位开始留，labels 截掉同样长的前缀，loss 内部的错位对齐不变。
    """
    labels = batch["labels"]
    valid = (labels[0] != IGNORE).nonzero()
    if len(valid) == 0:
        return dict(batch)
    keep = labels.shape[1] - int(valid[0]) + 1
    return {**batch, "labels": labels[:, -keep:], "logits_to_keep": keep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B",
                    help="基座；Qwen2.5-VL-3B-Instruct 作对照时显式指定")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="默认 2 倍的 rank。SeeClick 用的是固定 16")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="attn", choices=["attn", "attn_nogate", "qkvo", "all"],
                    help="attn 挂语言模型里全部注意力投影（Qwen2.5-VL 是 q/k/v/o，Qwen3.5 另含"
                         "线性注意力的 in_proj_*/out_proj）；attn_nogate 再摘掉 Gated DeltaNet 的"
                         "两个状态门 in_proj_a/in_proj_b；qkvo 只挂全注意力的 q/k/v/o；"
                         "all 再加 MLP，ShowUI 训 Qwen2-VL 用的就是 all")
    ap.add_argument("--keep-best", action="store_true",
                    help="只在验证 loss 创新低时存权重，交付最优那一步而不是最后一步。"
                         "小数据容易过拟合，配合调小 --eval-every 用")
    ap.add_argument("--train-merger", default="no", choices=["no", "lora", "full"],
                    help="视觉塔对齐层（projection）怎么训。no 冻结，只调语言模型，适合只改回答方式；"
                         "lora 给对齐层挂 LoRA；full 全量训练对齐层，是「图能看懂但领域对齐不好」时的"
                         "推荐做法，这时视觉塔不做 4-bit 量化")
    ap.add_argument("--merger-lr", type=float, default=None,
                    help="对齐层单独的学习率，默认取 --lr 的两倍：这部分参数少，收敛要更快一点")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW 的默认值是 0.01，SeeClick 用 0.1")
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="SeeClick 用 0.95")
    ap.add_argument("--scheduler", default="onecycle", choices=["onecycle", "cosine"],
                    help="cosine 是带预热的余弦退火，SeeClick / Aguvis 用的那套")
    ap.add_argument("--warmup-ratio", type=float, default=0.05,
                    help="预热占总步数的比例。SeeClick 0.01，Aguvis 0.03")
    ap.add_argument("--accum", type=int, default=8, help="梯度累积步数，等效批大小")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条，调试用")
    ap.add_argument("--kinds", default=None,
                    help="只用这些类型的样本，逗号分隔，如 action,plan。调配比用")
    ap.add_argument("--max-len", type=int, default=2048, help="超长样本直接跳过")
    ap.add_argument("--max-pixels", type=int, default=1280,
                    help="送进模型的图片上限，单位是视觉 token 数（Qwen2.5-VL 一个 token 是"
                         "28×28 像素，Qwen3.5 是 32×32）。图片 token 是样本长度的大头")
    ap.add_argument("--mem-fraction", type=float, default=0.8,
                    help="限制本进程能用的显存比例。Windows 的 WDDM 在显存超额时会"
                         "静默换页到主机内存，不报 OOM 只是慢几百倍——限住之后直接"
                         "抛 OOM，问题当场可见")
    ap.add_argument("--eval-every", type=int, default=200, help="每多少步在验证集上看一次")
    ap.add_argument("--eval-samples", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=20,
                    help="每多少次更新存一次适配器权重。只在最后存的话，训练中途卡死"
                         "就什么都拿不到——第一次跑到 160/175 时 backward 挂住，"
                         "几个小时的训练全丢了。0 表示只在结束时存")
    ap.add_argument("--init-adapter", default=None,
                    help="在已有适配器的基础上接着训，用于顺序课程：先训定位、"
                         "再在它上面训动作。OS-Atlas 就是这个路子——先做定位预训练，"
                         "再做动作微调，而不是把两类样本混在一轮里")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tag", default="lora")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None,
                    help="从哪个目录读 train.jsonl / val.jsonl，默认 data/finetune")
    args = ap.parse_args()

    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)


    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    train = load_rows("train")
    val = load_rows("val")
    if args.kinds:
        want = {k.strip() for k in args.kinds.split(",") if k.strip()}
        train = [r for r in train if r.get("kind") in want]
        val = [r for r in val if r.get("kind") in want]
    if args.limit:
        train = train[:args.limit]
    val = val[:args.eval_samples]
    kinds = {}
    for r in train:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"训练 {len(train)} 条 {kinds}，验证 {len(val)} 条")

    print(f"加载 {args.model} ……")
    t0 = time.perf_counter()
    kwargs = {"device_map": "cuda"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        # 全量训练对齐层时它不能量化：4-bit 的权重没法直接更新。transformers 的跳过规则按
        # 模块名精确匹配，不是路径子串，所以写 "visual" 不管用，得写 "merger"；这样也只有
        # 对齐层留在 bf16，视觉塔其余 block 照样 4-bit。
        skip = MERGER_LINEARS if args.train_merger == "full" else None
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=skip)
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    ip = AutoProcessor.from_pretrained(args.model).image_processor
    factor = int(getattr(ip, "patch_size", 14) or 14) * int(getattr(ip, "merge_size", 2) or 2)
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * factor * factor, max_pixels=args.max_pixels * factor * factor)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s，"
          f"显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"接着 {args.init_adapter} 的权重训")
    else:
        linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        targets = lora_target_regex(linear_names, args.lora_targets,
                                    with_merger=args.train_merger == "lora")
        keep = merger_modules(linear_names) if args.train_merger == "full" else None
        print(f"LoRA 目标：{targets}")
        if keep:
            print(f"全量训练：{keep}")
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha or args.rank * 2,
            lora_dropout=args.lora_dropout, bias="none", modules_to_save=keep,
            task_type="CAUSAL_LM", target_modules=targets))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {trainable / 1e6:.1f} M")

    # 先按长度筛一遍，再算总步数。不筛的话超长样本在训练时被 encode 丢掉，
    # 实际更新次数比 total 少（上一轮 105 次对 156 次），OneCycleLR 的学习率
    # 就退不到底。
    too_long = [r for r in train if est_tokens(r, processor.tokenizer, args.max_pixels, factor)
                > args.max_len]
    if too_long:
        keep = {id(r) for r in train} - {id(r) for r in too_long}
        train = [r for r in train if id(r) in keep]
        print(f"按长度预筛掉 {len(too_long)} 条超长样本，剩 {len(train)} 条")

    merger_lr = args.merger_lr or args.lr * 2
    on_merger = [p for n, p in model.named_parameters() if p.requires_grad and ".merger." in n]
    on_lm = [p for n, p in model.named_parameters() if p.requires_grad and ".merger." not in n]
    groups = [{"params": on_lm, "lr": args.lr}]
    max_lr = args.lr
    if on_merger:
        groups.append({"params": on_merger, "lr": merger_lr})
        max_lr = [args.lr, merger_lr]
        print(f"对齐层单独一组：{sum(p.numel() for p in on_merger) / 1e6:.2f} M，学习率 {merger_lr:.1e}")
    opt = torch.optim.AdamW(groups, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, args.adam_beta2))
    total = math.ceil(len(train) * args.epochs / args.accum)
    if args.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=round(total * args.warmup_ratio),
            num_training_steps=max(total, 1))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, total_steps=max(total, 1), pct_start=args.warmup_ratio)

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for r in val:
                b = encode(processor, r, args.max_len)
                if b is None:
                    continue
                losses.append(model(**answer_only(b.to("cuda"))).loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    OUTPUT.mkdir(exist_ok=True)
    best = (0, float("inf"))
    n_lora = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A"))
    print(f"挂了 LoRA 的层 {n_lora} 个，切块系数 {factor}")
    log = {"args": vars(args), "train_size": len(train), "kinds": kinds, "steps": [],
           "base_model_type": getattr(model.config, "model_type", None),
           "patch_factor": factor, "lora_layers": n_lora}

    def save(step: int, log: dict, t_start: float, skipped: int, tag_suffix=None) -> None:
        """存一次权重，顺手把「这份权重练了多少步」写清楚。

        只有最后存一次的话，中途卡死就什么都拿不到。适配器只有 7.4 M 参数，
        存一次不到一秒，多存几次不心疼。
        """
        out = OUTPUT / (args.tag if tag_suffix is None else args.tag + tag_suffix)
        model.save_pretrained(out)
        (ROOT / "logs" / f"train_{args.tag}.json").write_text(json.dumps(
            {**log, "saved_at_step": step, "planned_steps": total,
             "skipped_too_long": skipped,
             "train_minutes": round((time.perf_counter() - t_start) / 60, 1),
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)},
            ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n开始训练：{args.epochs} 轮，累积 {args.accum} 步，共约 {total} 次更新")
    model.train()
    seen = skipped = 0
    running = []
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        for r in sortish_batches(train, args.accum, random):
            b = encode(processor, r, args.max_len)
            if b is None:
                skipped += 1
                continue
            loss = model(**answer_only(b.to("cuda"))).loss / args.accum
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
                    log["steps"][-1]["val_loss"] = round(v, 4)
                    if args.keep_best and v < best[1]:
                        best = (step, v)
                        log["best_step"], log["best_val_loss"] = step, round(v, 4)
                        save(step, log, t_start, skipped)
                        print(f"    验证 loss {v:.4f}（新低，已存）")
                    else:
                        print(f"    验证 loss {v:.4f}"
                              + (f"（最优仍是第 {best[0]} 步 {best[1]:.4f}）" if args.keep_best else ""))
                if args.save_every and step % args.save_every == 0:
                    save(step, log, t_start, skipped, "_last" if args.keep_best else None)
                    # 顺手把分配器占着不用的块还回去：样本长度差一倍多，碎片攒得快，
                    # 整卡占满之后会明显变慢。不要放到每次更新——4-bit 每次前向都要
                    # 反量化权重，每步清缓存会把那些临时缓冲区反复重分配，实测每次
                    # 更新从 36 s 涨到 100 s 以上。
                    torch.cuda.empty_cache()

    log["final_val_loss"] = round(evaluate(), 4)
    log["train_minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    log["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    print(f"\n训练完成：{log['train_minutes']} 分钟，验证 loss {log['final_val_loss']}，"
          f"峰值 {log['peak_vram_gb']} GB，跳过超长样本 {skipped} 条")

    save(seen // args.accum, log, t_start, skipped)
    print(f"权重存到 {OUTPUT / args.tag}")


if __name__ == "__main__":
    main()
