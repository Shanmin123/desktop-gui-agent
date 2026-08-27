"""在 ScreenSpot 桌面分片上评测 UI 元素定位精度。

指标沿用 ScreenSpot 的口径：模型给出的点落在真值框内算命中。

这个脚本有两个用途：
1. 验证坐标约定。如果模型输出的坐标空间理解错了，命中率会接近 0，一眼看得出来。
2. 给出第 3 周 LoRA 微调前的基线，微调后跑同一个脚本对比。

用法：
    python scripts/eval_grounding.py --limit 20        # 先小样本确认跑得通
    python scripts/eval_grounding.py                   # 全量 334 条
    python scripts/eval_grounding.py --save-fail 10    # 另存前 10 个失败样本
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gui_agent.models import DEFAULT_MODEL, LocalQwenVL
from gui_agent.perception import annotate, imwrite
from gui_agent.schema import Element

ROOT = Path(__file__).resolve().parents[1]
TEST_JSONL = ROOT / "data" / "grounding" / "test.jsonl"


def load_records(limit=None):
    with TEST_JSONL.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return recs[:limit] if limit else recs


def hit(point, bbox) -> bool:
    """点落在框内算命中，ScreenSpot 的标准判定。"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--save-fail", type=int, default=0, help="另存前 N 个失败样本用于排查")
    ap.add_argument("--tag", default="base", help="结果文件名后缀，用于区分微调前后")
    args = ap.parse_args()

    from datasets import load_dataset

    print("加载评测集 ……")
    ds = load_dataset("rootsautomation/ScreenSpot", split="test")
    recs = load_records(args.limit)
    print(f"  {len(recs)} 条")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, load_in_4bit=args.load_in_4bit)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s")

    import torch

    if torch.cuda.is_available():
        print(f"  显存占用 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    stats = defaultdict(lambda: [0, 0])  # key -> [命中, 总数]
    latencies, failures, n_unparsed = [], [], 0

    for i, r in enumerate(recs):
        img = np.array(ds[r["index"]]["image"].convert("RGB"))[:, :, ::-1]  # RGB->BGR
        t = time.perf_counter()
        pred = vlm.locate(img, r["instruction"])
        latencies.append(time.perf_counter() - t)

        ok = pred is not None and hit(pred, r["bbox"])
        if pred is None:
            n_unparsed += 1

        for key in ("总体", r["platform"], r["element_type"]):
            stats[key][1] += 1
            stats[key][0] += ok

        if not ok and len(failures) < args.save_fail:
            failures.append((r, pred, img))

        if (i + 1) % 20 == 0:
            c, n = stats["总体"]
            print(f"  {i+1}/{len(recs)}  命中 {c}/{n} = {c/n:.1%}")

    print(f"\n{'分组':<12}{'命中':<8}{'总数':<8}准确率")
    for key in sorted(stats, key=lambda k: (k != "总体", k)):
        c, n = stats[key]
        print(f"{key:<12}{c:<8}{n:<8}{c/n:.1%}")

    print(f"\n平均单条耗时 {sum(latencies)/len(latencies):.2f}s，未能解析出坐标 {n_unparsed} 条")

    out = ROOT / "logs" / f"grounding_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": len(recs),
                "accuracy": {k: {"hit": v[0], "total": v[1]} for k, v in stats.items()},
                "avg_latency_s": sum(latencies) / len(latencies),
                "unparsed": n_unparsed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已存到 {out}")

    for r, pred, img in failures:
        elems = [Element(id=0, bbox=tuple(r["bbox"]), text="真值")]
        if pred is not None:
            d = 0.01
            elems.append(
                Element(id=1, bbox=(pred[0] - d, pred[1] - d, pred[0] + d, pred[1] + d), text="预测")
            )
        name = f"fail_{r['index']}_{r['instruction'][:20]}".replace(":", "_").replace("/", "_")
        imwrite(str(ROOT / "logs" / "fails" / f"{name}.png"), annotate(img, elems))
    if failures:
        print(f"失败样本已存到 {ROOT / 'logs' / 'fails'}")


if __name__ == "__main__":
    main()
