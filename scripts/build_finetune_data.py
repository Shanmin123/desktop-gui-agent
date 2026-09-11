"""构建 LoRA 微调的训练集与验证集。

对应大纲第 5 周第 1 项。

第 2 周测出来瓶颈在定位：模型在「直接问坐标」的提示词下命中 70%，在 Agent 的
动作提示词下只有 40.8%，因为它 111/120 次都在用 OCR 元素编号指位置，而 OCR 只
认文字，图标没有编号可指。所以训练样本按两段分别构造：

  定位（grounding）  截图 + 元素描述 -> {"bbox_2d": [...]}，来自 Mind2Web
                     用的提示词和 models.GROUNDING_PROMPT 完全一致
  动作（action）     截图 + 任务 -> {"thought":..., "action":{...}}，来自 ScreenAgent
                     让模型直接给归一化坐标，不再依赖元素编号

训练用的提示词必须和推理时一字不差，否则学到的东西迁移不过去。

Mind2Web 的截图是整页长图（实测 1280×5429，6.9 MP），远超模型的 1.0 MP 预算，
整张送进去元素会被压得看不见。这里裁出包含目标的 1280×720 窗口——正好是本项目
的运行分辨率，不需要缩放。

用法：
    python scripts/build_finetune_data.py --limit-mind2web 200   # 先小样本确认
    python scripts/build_finetune_data.py                        # 全量
"""

import argparse
import json
import random
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.models import GROUNDING_PROMPT
from gui_agent.schema import Action

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finetune"
CROPS = OUT / "crops"

VIEW_W, VIEW_H = 1280, 720          # 裁剪窗口，等于本项目的运行分辨率
MIND2WEB_REPO = "osunlp/Multimodal-Mind2Web"

# 动作阶段的提示词。不给 OCR 元素清单：训练目标就是让模型直接给坐标，
# 给了清单它又会去用编号。
ACTION_TEMPLATE = """你在操作一台 Windows 电脑，目标是完成用户给的任务。

任务：{instruction}

看这张截图，输出下一步动作。只返回一个 JSON 对象：
{{"thought": "为什么这么做", "action": {{"type": "click", "point": [0.5, 0.5]}}}}

可用动作：click / left_double / right_single 需要 point；scroll 需要 point 和
direction；type 和 hotkey 需要 text；wait / finished / call_user 不需要参数。
point 是归一化到 0~1 的坐标。"""


# --------------------------------------------------------------------------
# 动作样本：ScreenAgent
# --------------------------------------------------------------------------


def action_samples(split: str) -> list:
    """ScreenAgent 的可执行动作 -> 训练样本。"""
    src = ROOT / "data" / "screenagent" / f"{split}.jsonl"
    if not src.is_file():
        raise SystemExit(f"没有 {src}，先跑 scripts/prepare_screenagent.py")

    out = []
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        if not Path(r["image"]).is_file():
            continue
        act = dict(r["action"])
        # 坐标保留四位，够精确又不让回答变长
        if act.get("point"):
            act["point"] = [round(v, 4) for v in act["point"]]
        out.append({
            "kind": "action",
            "source": "screenagent",
            "image": r["image"],
            "prompt": ACTION_TEMPLATE.format(instruction=r["instruction_zh"] or r["instruction"]),
            "response": json.dumps({"thought": "", "action": act}, ensure_ascii=False),
        })
    return out


# --------------------------------------------------------------------------
# 定位样本：Mind2Web
# --------------------------------------------------------------------------


def describe(repr_str: str, attrs: dict) -> str:
    """从 target_action_reprs 取出元素描述。

    格式是 `[tag]  文字 -> 操作` 或 `[tag]  文字 -> TYPE: 值`，要的是中间那段文字。
    文字为空时退回 aria_label。
    """
    s = (repr_str or "").strip()
    if s.startswith("["):
        s = s.split("]", 1)[-1]
    s = s.split(" -> ", 1)[0].strip()
    if not s:
        s = (attrs.get("aria_label") or attrs.get("title") or attrs.get("alt") or "").strip()
    return s


def crop_box(bx: float, by: float, bw: float, bh: float, W: int, H: int) -> tuple:
    """裁一个 VIEW_W×VIEW_H 的窗口，尽量把目标放在中间且不越界。"""
    cx, cy = bx + bw / 2, by + bh / 2
    x0 = int(min(max(cx - VIEW_W / 2, 0), max(W - VIEW_W, 0)))
    y0 = int(min(max(cy - VIEW_H / 2, 0), max(H - VIEW_H, 0)))
    return x0, y0, min(x0 + VIEW_W, W), min(y0 + VIEW_H, H)


def grounding_samples(split: str, limit=None, seed: int = 42) -> tuple:
    """Mind2Web 的（截图，元素描述）-> bbox。返回 (样本, 跳过原因统计)。"""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from PIL import Image

    snap = Path(snapshot_download(repo_id=MIND2WEB_REPO, repo_type="dataset", max_workers=8))
    files = sorted((snap / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"没有 {split} 分片")

    CROPS.mkdir(parents=True, exist_ok=True)
    cols = ["annotation_id", "action_uid", "target_action_reprs", "pos_candidates", "screenshot"]
    out, skip = [], Counter()

    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=cols):
            for r in batch.to_pylist():
                if limit and len(out) >= limit:
                    return out, skip
                pc = r.get("pos_candidates") or []
                sc = r.get("screenshot") or {}
                if not pc or not sc.get("bytes"):
                    skip["缺候选元素或截图"] += 1
                    continue
                try:
                    cand = json.loads(pc[0])
                    attrs = json.loads(cand["attributes"])
                    nums = [float(v) for v in attrs["bounding_box_rect"].split(",")]
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    skip["bbox 解析失败"] += 1
                    continue
                bx, by, bw, bh = nums[:4]
                if bw <= 1 or bh <= 1:
                    skip["目标框太小"] += 1
                    continue

                desc = describe(r.get("target_action_reprs"), attrs)
                if not desc:
                    skip["没有元素描述"] += 1
                    continue

                img = Image.open(BytesIO(sc["bytes"])).convert("RGB")
                W, H = img.size
                if not (0 <= bx < W and 0 <= by < H):
                    skip["目标框在图外"] += 1
                    continue

                x0, y0, x1, y1 = crop_box(bx, by, bw, bh, W, H)
                crop = img.crop((x0, y0, x1, y1))
                # 裁剪后的坐标
                cb = [bx - x0, by - y0, bx - x0 + bw, by - y0 + bh]
                cw, ch = crop.size
                if not (0 <= cb[0] < cw and 0 <= cb[1] < ch):
                    skip["裁剪后目标不在窗口内"] += 1
                    continue
                cb = [max(0.0, min(cb[0], cw)), max(0.0, min(cb[1], ch)),
                      max(0.0, min(cb[2], cw)), max(0.0, min(cb[3], ch))]

                name = f"{r['annotation_id']}_{r['action_uid']}.jpg"
                path = CROPS / name
                crop.save(path, "JPEG", quality=88)
                out.append({
                    "kind": "grounding",
                    "source": "mind2web",
                    "image": str(path.resolve()),
                    "prompt": GROUNDING_PROMPT.format(instruction=desc),
                    "response": json.dumps(
                        {"bbox_2d": [round(v, 1) for v in cb]}, ensure_ascii=False),
                })
    return out, skip


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-mind2web", type=int, default=3000,
                    help="定位样本上限，裁剪图要占硬盘")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-grounding", action="store_true", help="只出动作样本")
    args = ap.parse_args()
    random.seed(args.seed)

    train = action_samples("train")
    val = action_samples("val")
    print(f"动作样本（ScreenAgent）：训练 {len(train)}，验证 {len(val)}")

    if not args.no_grounding:
        g, skip = grounding_samples("train", limit=args.limit_mind2web)
        print(f"定位样本（Mind2Web）：{len(g)}")
        if skip:
            print("  跳过：", dict(skip))
        random.shuffle(g)
        n_val = max(1, len(g) // 10)
        val += g[:n_val]
        train += g[n_val:]

    random.shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val)]:
        p = OUT / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
        kinds = Counter(r["kind"] for r in rows)
        print(f"\n{p}  {len(rows)} 条  {dict(kinds)}")


if __name__ == "__main__":
    main()
