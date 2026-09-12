"""量感知模块本身的准确率与速度。

对应大纲第 6 周第 3 项。

评测集用 ScreenSpot 桌面分片（334 条，194 条文字目标 + 140 条图标目标），
每条给一句指令和目标控件的真值框。这里不问模型，只看感知模块：

  命中    识别出的元素里，有没有哪个的中心落在真值框内
          —— 有，模型就能按编号点到它；没有，模型再聪明也指不了

  清单内  同上，但只算真正会写进提示词的那批（select_elements 选出来的，上限 60）
          识别得多不等于用得上，进不了清单的元素模型根本看不到

  元素数  平均识别出多少个元素，反映提示词被占掉多少
  耗时    每张图的感知秒数

按 text / icon 分开看：OCR 只认文字，图标那一半是短板，这正是要优化的地方。

用法：
    python scripts/eval_perception.py --limit 30          # 先小样本确认跑得通
    python scripts/eval_perception.py                     # 全量 334 条
    python scripts/eval_perception.py --configs base,cv   # 只跑指定配置
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gui_agent.agent import MAX_ELEMENTS, select_elements
from gui_agent.perception import Perception, detect_cv_elements, merge_elements
from gui_agent.schema import ScreenState

ROOT = Path(__file__).resolve().parents[1]
TEST_JSONL = ROOT / "data" / "grounding" / "test.jsonl"

# 每个配置是一组感知参数。base 是现在线上的设置，其余是候选。
CONFIGS = {
    "base":      dict(min_confidence=0.3, scale=1.0, mag_ratio=1.0, cv=False),
    "conf0.1":   dict(min_confidence=0.1, scale=1.0, mag_ratio=1.0, cv=False),
    "mag1.5":    dict(min_confidence=0.3, scale=1.0, mag_ratio=1.5, cv=False),
    "half":      dict(min_confidence=0.3, scale=0.5, mag_ratio=1.0, cv=False),
    "cv":        dict(min_confidence=0.3, scale=1.0, mag_ratio=1.0, cv=True),
}


def covers(elements, bbox) -> bool:
    """元素中心落在真值框内就算命中，和 eval_grounding 的判定口径一致。"""
    x1, y1, x2, y2 = bbox
    for e in elements:
        cx, cy = e.center()
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def perceive_one(per, img, cfg):
    """按配置在一张图上跑一次感知，返回 (元素列表, 耗时)。"""
    import cv2

    t0 = time.perf_counter()
    src = img
    if cfg["scale"] != 1.0:
        src = cv2.resize(img, None, fx=cfg["scale"], fy=cfg["scale"],
                         interpolation=cv2.INTER_AREA)
    elements = per.ocr(src, min_confidence=cfg["min_confidence"],
                       mag_ratio=cfg["mag_ratio"])
    if cfg["cv"]:
        elements = merge_elements(elements, detect_cv_elements(src))
    return elements, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--tag", default="v2.0")
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"没有这些配置：{unknown}，可选 {list(CONFIGS)}")

    from datasets import load_dataset

    print("加载评测集 ……")
    ds = load_dataset("rootsautomation/ScreenSpot", split="test")
    with TEST_JSONL.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    if args.limit:
        recs = recs[: args.limit]
    print(f"  {len(recs)} 条")

    per = Perception(gpu=not args.no_gpu)
    per.reader  # 先把 OCR 模型加载好，不然第一条的耗时把加载时间也算进去

    results = {}
    for name in names:
        cfg = CONFIGS[name]
        stats = defaultdict(lambda: [0, 0, 0])  # 分组 -> [命中, 前60命中, 总数]
        times, counts = [], []

        for i, r in enumerate(recs):
            img = np.array(ds[r["index"]]["image"].convert("RGB"))[:, :, ::-1]
            elements, dt = perceive_one(per, img, cfg)
            times.append(dt)
            counts.append(len(elements))

            ok = covers(elements, r["bbox"])
            # 量的是提示词真正会显示的那批，不是简单截前 60 个：
            # select_elements 会给图标候选框留名额
            shown = select_elements(ScreenState(width=1, height=1, elements=elements),
                                   MAX_ELEMENTS)
            ok60 = covers(shown, r["bbox"])
            for key in ("总体", r["element_type"]):
                s = stats[key]
                s[0] += ok
                s[1] += ok60
                s[2] += 1

            if (i + 1) % 50 == 0:
                c, _, n = stats["总体"]
                print(f"  [{name}] {i+1}/{len(recs)}  命中 {c}/{n} = {c/n:.1%}")

        results[name] = {
            "config": cfg,
            "groups": {k: {"hit": v[0], "hit_top60": v[1], "n": v[2]} for k, v in stats.items()},
            "sec_per_image": sum(times) / len(times),
            "elements_per_image": sum(counts) / len(counts),
        }
        g = results[name]["groups"]["总体"]
        print(f"[{name}] 命中 {g['hit']/g['n']:.1%}  清单内 {g['hit_top60']/g['n']:.1%}  "
              f"{results[name]['sec_per_image']:.2f}s/张  "
              f"{results[name]['elements_per_image']:.0f} 元素/张\n")

    per.close()

    print(f"\n{'配置':<10}{'总体':<9}{'文字':<9}{'图标':<9}{'清单内':<9}{'元素数':<8}耗时")
    for name in names:
        r = results[name]
        g = r["groups"]

        def pct(k, field="hit"):
            v = g.get(k)
            return f"{v[field]/v['n']:.1%}" if v else "—"

        print(f"{name:<10}{pct('总体'):<9}{pct('text'):<9}{pct('icon'):<9}"
              f"{pct('总体','hit_top60'):<9}{r['elements_per_image']:<8.0f}"
              f"{r['sec_per_image']:.2f}s")

    out = ROOT / "logs" / f"perception_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": len(recs), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
