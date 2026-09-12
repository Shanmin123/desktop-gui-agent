"""在留出的拆解样本上量任务拆解质量。

对应大纲第 6 周第 1 项。复杂任务的成功率只有 0/4，而且卡在执行层，用它衡量不出
拆解本身的好坏。这里只看拆解：给任务和截图，让模型拆，和人工修正过的参考拆解比。

评测集是 ScreenAgent 的 PlanAction，按 session 划出的验证部分，没参与训练。

指标：
  覆盖率    每条参考子任务，在预测里找最像的那条，取字符级 F1，再平均。
            衡量「该做的步骤有没有被拆出来」。
  多余步数  预测条数减参考条数，正数就是拆多了。凭空补「新建文件夹」这类步骤
            会体现在这里。
  拆不出来  返回空数组的条数。

字符级 F1 不需要额外的评判模型，中文按字比较，词序不敏感，够用来做前后对比。

用法：
    python scripts/eval_plan.py                                  # 基座模型
    python scripts/eval_plan.py --adapter checkpoints/lora_v3    # 微调后
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.models import DEFAULT_MODEL, LocalQwenVL
from gui_agent.perception import imread, resize_for_model
from gui_agent.planner import MAX_SUBTASKS, PLAN_TEMPLATE, parse_plan

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "data" / "screenagent" / "plans.jsonl"


def char_f1(a: str, b: str) -> float:
    """两句话的字符级 F1。中文按字比，词序不敏感。"""
    ca, cb = Counter(a.replace(" ", "")), Counter(b.replace(" ", ""))
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p, r = common / sum(cb.values()), common / sum(ca.values())
    return 2 * p * r / (p + r)


def coverage(reference: list, predicted: list) -> float:
    """每条参考子任务在预测里的最佳匹配分，取平均。"""
    if not reference:
        return 0.0
    if not predicted:
        return 0.0
    return statistics.mean(max(char_f1(ref, got) for got in predicted) for ref in reference)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="挂上 LoRA 权重评测微调后的模型")
    ap.add_argument("--max-pixels", type=int, default=1280,
                    help="图片上限，单位 28x28 的块。微调时降过这个值的话，"
                         "用同一个值评测才能看出权重本身的效果")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    if not PLANS.is_file():
        raise SystemExit(f"没有 {PLANS}，先跑 scripts/prepare_screenagent.py")
    recs = [json.loads(l) for l in PLANS.open(encoding="utf-8")]
    recs = [r for r in recs if r.get("split") == args.split and Path(r["image"]).is_file()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"拆解评测集（{args.split}）{len(recs)} 条")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, adapter=args.adapter,
                      max_pixels=args.max_pixels * 28 * 28)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s")

    rows, covs, extras, empty = [], [], [], 0
    for i, r in enumerate(recs):
        img, _ = resize_for_model(imread(r["image"]))
        prompt = PLAN_TEMPLATE.format(
            instruction=r["instruction_zh"] or r["instruction"],
            elements="  （这一步不看元素清单）", max_subtasks=MAX_SUBTASKS)
        got = parse_plan(vlm.ask(img, prompt), MAX_SUBTASKS)
        ref = r["subtasks"][:MAX_SUBTASKS]
        c = coverage(ref, got)
        covs.append(c)
        extras.append(len(got) - len(ref))
        empty += not got
        rows.append({"instruction": r["instruction_zh"], "reference": ref,
                     "predicted": got, "coverage": round(c, 3)})
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(recs)}  覆盖率均值 {statistics.mean(covs):.3f}")

    print(f"\n{'=' * 46}")
    print(f"覆盖率      均值 {statistics.mean(covs):.3f}  中位 {statistics.median(covs):.3f}")
    print(f"多余步数    均值 {statistics.mean(extras):+.2f}  "
          f"拆多了的 {sum(1 for x in extras if x > 0)}/{len(extras)} 条")
    print(f"拆不出来    {empty}/{len(recs)} 条")

    out = ROOT / "logs" / f"plan_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "adapter": args.adapter,
        "max_pixels": args.max_pixels, "split": args.split, "n": len(recs),
        "coverage_mean": statistics.mean(covs), "coverage_median": statistics.median(covs),
        "extra_steps_mean": statistics.mean(extras), "empty": empty, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
