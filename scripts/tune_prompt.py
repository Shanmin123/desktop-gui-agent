"""比较提示词变体，挑一个动作生成更准的。

对应大纲第 5 周第 4 项。

评测集和 eval_screenagent.py 同一套：ScreenAgent 官方 test 划分，353 条、
70 个 session，整份没有参与训练。同一批样本、同一个模型，只换提示词。

变体见 gui_agent/chain.py 的 PROMPT_VARIANTS，另有一个 no_elements 配置，
用 base 模板但不给元素清单，用来看 OCR 清单到底帮了还是拖了后腿。

指标：
  动作类型准确率  预测类型与真值一致的比例
  键盘动作召回    真值是 type / hotkey 时模型是否也给键盘动作
  点击距离中位数  两边都是坐标类动作时，预测点与真值点的归一化距离
  解析失败        模型没吐出能解析的 JSON 的条数

用法：
    python scripts/tune_prompt.py --limit 40                    # 先小样本
    python scripts/tune_prompt.py                               # 全量 353
    python scripts/tune_prompt.py --variants base,point_first   # 只比两个
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.agent import parse_step
from gui_agent.chain import PROMPT_VARIANTS, render_prompt
from gui_agent.models import DEFAULT_MODEL, LocalQwenVL
from gui_agent.perception import Perception, imread, resize_for_model
from gui_agent.schema import ScreenState

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "screenagent" / "test.jsonl"

KEYBOARD = {"type", "hotkey"}
POINTED = {"click", "left_double", "right_single", "scroll"}

# 变体名 -> (模板变体, 是否给元素清单)
CONFIGS = {name: (name, True) for name in PROMPT_VARIANTS}
CONFIGS["no_elements"] = ("base", False)


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
    ap.add_argument("--variants", default=",".join(CONFIGS))
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in CONFIGS]
    if unknown:
        raise SystemExit(f"没有这些变体：{unknown}，可选 {list(CONFIGS)}")

    recs = load(args.limit)
    print(f"评测集 {len(recs)} 条，{len({r['session_id'] for r in recs})} 个 session")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, adapter=args.adapter)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s")

    # OCR 结果与提示词无关，一张图只跑一次，给所有变体共用。353 条 OCR 比推理还慢，
    # 每个变体重跑一遍纯属浪费。
    print("预跑 OCR ……")
    perception = Perception()
    cache = {}
    for i, r in enumerate(recs):
        img = imread(r["image"])
        h, w = img.shape[:2]
        model_img, _ = resize_for_model(img)
        cache[i] = (ScreenState(width=w, height=h, elements=perception.ocr(img)), model_img)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)}")
    perception.close()

    results = {}
    for name in names:
        variant, with_elements = CONFIGS[name]
        n_ok = n_fail = kb_total = kb_hit = 0
        dists, latencies = [], []

        for i, r in enumerate(recs):
            gt = r["action"]
            state, model_img = cache[i]
            if not with_elements:
                state = ScreenState(width=state.width, height=state.height)

            prompt = render_prompt(r["instruction_zh"] or r["instruction"], state, [],
                                   variant=variant)
            rh, rw = vlm.resized_size(*model_img.shape[:2])

            t = time.perf_counter()
            raw = vlm.ask(model_img, prompt)
            latencies.append(time.perf_counter() - t)

            try:
                _, pred = parse_step(raw, state, model_size=(rw, rh))
            except ValueError:
                n_fail += 1
                continue

            n_ok += pred.type == gt["type"]
            if gt["type"] in KEYBOARD:
                kb_total += 1
                kb_hit += pred.type in KEYBOARD
            if gt["type"] in POINTED and pred.type in POINTED \
                    and gt.get("point") and pred.point:
                dists.append(distance(pred.point, tuple(gt["point"])))

            if (i + 1) % 50 == 0:
                print(f"  [{name}] {i+1}/{len(recs)}  类型准确 {n_ok}/{i+1} = {n_ok/(i+1):.1%}")

        n = len(recs)
        results[name] = {
            "variant": variant,
            "with_elements": with_elements,
            "type_accuracy": n_ok / n,
            "keyboard_recall": (kb_hit / kb_total) if kb_total else None,
            "keyboard_total": kb_total,  # 分母只算解析成功的样本，变体之间不一样
            "median_distance": statistics.median(dists) if dists else None,
            "n_pointed_pairs": len(dists),
            "parse_failures": n_fail,
            "sec_per_sample": sum(latencies) / len(latencies),
        }
        r = results[name]
        print(f"[{name}] 类型 {r['type_accuracy']:.1%}  "
              f"键盘 {r['keyboard_recall'] or 0:.1%}  "
              f"距离 {r['median_distance'] if r['median_distance'] is None else round(r['median_distance'], 3)}  "
              f"解析失败 {n_fail}\n")

    print(f"\n{'变体':<14}{'类型准确':<11}{'键盘召回':<11}{'点击距离':<11}{'解析失败':<10}耗时")
    for name in names:
        r = results[name]
        kb = f"{r['keyboard_recall']:.1%}" if r["keyboard_recall"] is not None else "—"
        d = f"{r['median_distance']:.3f}" if r["median_distance"] is not None else "—"
        print(f"{name:<14}{r['type_accuracy']:<11.1%}{kb:<11}{d:<11}"
              f"{r['parse_failures']:<10}{r['sec_per_sample']:.2f}s")

    out = ROOT / "logs" / f"prompt_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": len(recs), "model": args.model,
                               "adapter": args.adapter, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
