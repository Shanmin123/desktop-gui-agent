"""在 ScreenAgent 的测试划分上评测动作生成。

对应大纲第 5 周第 3 项「对比微调前后模型在 GUI 任务理解与动作生成上的效果」。

评测集是 ScreenAgent 官方的 test 划分，353 条、70 个 session，整份没有参与训练。

给模型一张截图和任务描述，让它输出下一步动作，与人工修正后的动作比：

  动作类型准确率  预测类型与真值一致的比例
  键盘动作召回    真值是 type / hotkey 时模型是否也给键盘动作。端到端测试里
                  模型 38 步 0 次用键盘，这一项是主要观察对象
  点击距离        两边都是坐标类动作时，预测点与真值点的归一化距离

单步评测，不喂前序历史。同一 session 的各步有先后依赖，带历史评测要按时间戳
重建顺序，留到整条轨迹的评测里做。

用法：
    python scripts/eval_screenagent.py --limit 20   # 先小样本确认跑得通
    python scripts/eval_screenagent.py              # 全量
    python scripts/eval_screenagent.py --tag lora   # 微调后换 tag 再跑一遍
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.agent import parse_step
from gui_agent.chain import render_prompt
from gui_agent.models import DEFAULT_MODEL, LocalQwenVL
from gui_agent.perception import Perception, imread, resize_for_model
from gui_agent.schema import ScreenState

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "screenagent" / "test.jsonl"

KEYBOARD = {"type", "hotkey"}
POINTED = {"click", "left_double", "right_single", "scroll"}
HIT_THRESHOLDS = (0.05, 0.10)


def load(limit=None) -> list:
    if not TEST.is_file():
        raise SystemExit(f"没有 {TEST}，先跑 scripts/prepare_screenagent.py")
    recs = [json.loads(l) for l in TEST.open(encoding="utf-8")]
    return recs[:limit] if limit else recs


def distance(a, b) -> float:
    return math.dist(a, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--no-ocr", action="store_true",
                    help="不跑 OCR，提示词里不带元素清单。默认带，与实际循环一致")
    args = ap.parse_args()

    recs = load(args.limit)
    print(f"评测集 {len(recs)} 条，{len({r['session_id'] for r in recs})} 个 session")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s")

    perception = None if args.no_ocr else Perception()

    n_type_ok = n_parse_fail = 0
    kb_total = kb_hit = 0
    dists, latencies = [], []
    confusion = Counter()
    gt_types = Counter()

    try:
        for i, r in enumerate(recs):
            gt = r["action"]
            gt_types[gt["type"]] += 1

            img = imread(r["image"])
            h, w = img.shape[:2]
            elements = perception.ocr(img) if perception else []
            state = ScreenState(width=w, height=h, elements=elements)
            model_img, _ = resize_for_model(img)

            prompt = render_prompt(r["instruction_zh"] or r["instruction"], state, [])
            rh, rw = vlm.resized_size(*model_img.shape[:2])

            t = time.perf_counter()
            raw = vlm.ask(model_img, prompt)
            latencies.append(time.perf_counter() - t)

            try:
                _, pred = parse_step(raw, state, model_size=(rw, rh))
            except ValueError:
                n_parse_fail += 1
                confusion[(gt["type"], "解析失败")] += 1
                continue

            confusion[(gt["type"], pred.type)] += 1
            n_type_ok += pred.type == gt["type"]

            if gt["type"] in KEYBOARD:
                kb_total += 1
                kb_hit += pred.type in KEYBOARD

            if gt["type"] in POINTED and pred.type in POINTED and gt.get("point") and pred.point:
                dists.append(distance(pred.point, tuple(gt["point"])))

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(recs)}  类型准确 {n_type_ok}/{i+1} = {n_type_ok/(i+1):.1%}")
    finally:
        if perception:
            perception.close()

    n = len(recs)
    print(f"\n{'='*52}")
    print(f"动作类型准确率   {n_type_ok}/{n} = {n_type_ok/n:.1%}")
    print(f"解析失败         {n_parse_fail}")
    if kb_total:
        print(f"键盘动作召回     {kb_hit}/{kb_total} = {kb_hit/kb_total:.1%}")
    if dists:
        dists.sort()
        print(f"点击距离         中位 {dists[len(dists)//2]:.3f}，"
              f"均值 {sum(dists)/len(dists):.3f}（归一化，{len(dists)} 对）")
        for th in HIT_THRESHOLDS:
            k = sum(d <= th for d in dists)
            print(f"  距离 ≤ {th:.2f}      {k}/{len(dists)} = {k/len(dists):.1%}")
    print(f"平均单条耗时     {sum(latencies)/len(latencies):.2f}s")

    print("\n真值类型 -> 预测类型（前 15）：")
    for (g, p), c in confusion.most_common(15):
        print(f"  {g:<14} -> {p:<14} {c}")

    out = ROOT / "logs" / f"screenagent_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "n": n,
        "with_ocr": not args.no_ocr,
        "type_accuracy": n_type_ok / n,
        "parse_failures": n_parse_fail,
        "keyboard_recall": (kb_hit / kb_total) if kb_total else None,
        "keyboard_total": kb_total,
        "point_pairs": len(dists),
        "point_distance_mean": (sum(dists) / len(dists)) if dists else None,
        "point_hit_rate": {str(th): sum(d <= th for d in dists) / len(dists)
                           for th in HIT_THRESHOLDS} if dists else {},
        "avg_latency_s": sum(latencies) / len(latencies),
        "ground_truth_types": dict(gt_types),
        "confusion": {f"{g}->{p}": c for (g, p), c in confusion.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
