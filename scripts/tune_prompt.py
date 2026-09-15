"""比较提示词变体，挑一个动作生成更准的。

对应大纲第 5 周第 4 项。

评测集和 eval_screenagent.py 同一套：ScreenAgent 官方 test 划分，353 条、
70 个 session，整份没有参与训练。同一批样本、同一个模型，只换提示词。

两种模式：
  one_stage  一段式变体（gui_agent/chain.py 的 PROMPT_VARIANTS），提示词带 OCR 元素清单；
             另有 no_elements 配置，用 base 模板但不给清单。第 3 周那组对照用的这个
  two_stage  两段式变体（TARGET_VARIANTS）：第一问只问点什么，第二问定位，不跑 OCR。
             交付的执行路径是两段式，这组对比才对系统有指导意义

指标：
  动作类型准确率  预测类型与真值一致的比例
  键盘动作召回    真值是 type / hotkey 时模型是否也给键盘动作
  点击距离中位数  两边都是坐标类动作时，预测点与真值点的归一化距离
  类型对且点准    类型对；坐标类动作还要求距离 ≤ 0.10
  解析失败        模型没吐出能解析的 JSON 的条数（两段式里还包括定位不到）

用法：
    python scripts/tune_prompt.py --limit 60                                            # 一段式
    python scripts/tune_prompt.py --mode two_stage --model Qwen/Qwen3.5-4B --limit 60    # 两段式
    python scripts/tune_prompt.py --variants base,point_first                           # 只比两个
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.agent import parse_step
from gui_agent.chain import PROMPT_VARIANTS, TARGET_VARIANTS, parse_with_target, render_prompt, \
    render_target_prompt
from gui_agent.models import DEFAULT_MODEL, MAX_NEW_TOKENS, LocalQwenVL
from gui_agent.perception import Perception, imread, resize_for_model
from gui_agent.schema import ScreenState

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "screenagent" / "test.jsonl"

KEYBOARD = {"type", "hotkey"}
POINTED = {"click", "left_double", "right_single", "scroll"}

# 一段式：变体名 -> (模板变体, 是否给元素清单)
ONE_STAGE = {name: (name, True) for name in PROMPT_VARIANTS}
ONE_STAGE["no_elements"] = ("base", False)
TWO_STAGE = list(TARGET_VARIANTS)


def load(limit=None) -> list:
    with TEST.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return recs[:limit] if limit else recs


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="挂上 LoRA 权重再比")
    ap.add_argument("--mode", default="one_stage", choices=["one_stage", "two_stage"])
    ap.add_argument("--variants", default=None, help="逗号分隔；不给就跑这个模式的全部变体")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    available = list(ONE_STAGE) if args.mode == "one_stage" else TWO_STAGE
    names = [v.strip() for v in (args.variants or ",".join(available)).split(",") if v.strip()]
    unknown = [v for v in names if v not in available]
    if unknown:
        raise SystemExit(f"{args.mode} 没有这些变体：{unknown}，可选 {available}")

    recs = load(args.limit)
    print(f"评测集 {len(recs)} 条，{len({r['session_id'] for r in recs})} 个 session，模式 {args.mode}")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, adapter=args.adapter)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s")

    # 截图预处理与提示词无关，一张图只做一次。一段式要 OCR 清单，353 条 OCR 比推理还慢，
    # 各变体共用；两段式不跑 OCR
    perception = Perception() if args.mode == "one_stage" else None
    if perception:
        print("预跑 OCR ……")
    cache = {}
    for i, r in enumerate(recs):
        img = imread(r["image"])
        h, w = img.shape[:2]
        model_img, _ = resize_for_model(img)
        elements = perception.ocr(img) if perception else []
        cache[i] = (ScreenState(width=w, height=h, elements=elements), model_img)
        if perception and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)}")
    if perception:
        perception.close()

    results = {}
    for name in names:
        n_ok = n_fail = n_joint = kb_total = kb_hit = 0
        dists, latencies = [], []
        # 只看总准确率看不出变体把预测往哪一类推了，混淆和预测分布都记下来
        confusion, pred_types = Counter(), Counter()

        for i, r in enumerate(recs):
            gt = r["action"]
            state, model_img = cache[i]
            instruction = r["instruction_zh"] or r["instruction"]
            rh, rw = vlm.coord_size(*model_img.shape[:2])

            t = time.perf_counter()
            try:
                if args.mode == "one_stage":
                    variant, with_elements = ONE_STAGE[name]
                    st = state if with_elements else ScreenState(width=state.width, height=state.height)
                    raw = vlm.ask(model_img, render_prompt(instruction, st, [], variant=variant),
                                  max_new_tokens=args.max_new_tokens)
                    _, pred = parse_step(raw, st, model_size=(rw, rh))
                else:
                    raw = vlm.ask(model_img, render_target_prompt(instruction, [], variant=name),
                                  max_new_tokens=args.max_new_tokens)
                    _, pred = parse_with_target(raw, vlm, model_img, state, (rw, rh))
            except ValueError:
                latencies.append(time.perf_counter() - t)
                n_fail += 1
                continue
            latencies.append(time.perf_counter() - t)

            confusion[f'{gt["type"]}->{pred.type}'] += 1
            pred_types[pred.type] += 1
            type_ok = pred.type == gt["type"]
            n_ok += type_ok
            if gt["type"] in KEYBOARD:
                kb_total += 1
                kb_hit += pred.type in KEYBOARD
            d = None
            if gt["type"] in POINTED and pred.type in POINTED and gt.get("point") and pred.point:
                d = distance(pred.point, tuple(gt["point"]))
                dists.append(d)
            n_joint += type_ok and (d is None or d <= 0.10)

            if (i + 1) % 20 == 0:
                print(f"  [{name}] {i+1}/{len(recs)}  类型准确 {n_ok}/{i+1} = {n_ok/(i+1):.1%}")

        n = len(recs)
        results[name] = {
            "mode": args.mode,
            "type_accuracy": n_ok / n,
            "joint_accuracy": n_joint / n,
            "keyboard_recall": (kb_hit / kb_total) if kb_total else None,
            "keyboard_total": kb_total,  # 分母只算解析成功的样本，变体之间不一样
            "median_distance": statistics.median(dists) if dists else None,
            "n_pointed_pairs": len(dists),
            "parse_failures": n_fail,
            "sec_per_sample": sum(latencies) / len(latencies),
            "pred_types": dict(pred_types.most_common()),
            "gt_types": dict(Counter(r["action"]["type"] for r in recs).most_common()),
            "confusion": dict(confusion.most_common()),
        }
        if args.mode == "one_stage":
            results[name].update({"variant": ONE_STAGE[name][0], "with_elements": ONE_STAGE[name][1]})
        r = results[name]
        print(f"[{name}] 类型 {r['type_accuracy']:.1%}  类型对且点准 {r['joint_accuracy']:.1%}  "
              f"键盘 {r['keyboard_recall'] or 0:.1%}  解析失败 {n_fail}\n")

    print(f"\n{'变体':<14}{'类型准确':<10}{'类型对且点准':<12}{'键盘召回':<10}{'点击距离':<10}{'解析失败':<9}耗时")
    for name in names:
        r = results[name]
        kb = f"{r['keyboard_recall']:.1%}" if r["keyboard_recall"] is not None else "—"
        d = f"{r['median_distance']:.3f}" if r["median_distance"] is not None else "—"
        print(f"{name:<14}{r['type_accuracy']:<10.1%}{r['joint_accuracy']:<12.1%}{kb:<10}{d:<10}"
              f"{r['parse_failures']:<9}{r['sec_per_sample']:.2f}s")

    out = ROOT / "logs" / f"prompt_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": len(recs), "model": args.model, "adapter": args.adapter,
                               "mode": args.mode, "max_new_tokens": args.max_new_tokens,
                               "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
