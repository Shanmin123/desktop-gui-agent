"""换基座之前先量两件事：定位坐标的口径，动作输出的格式。

坐标口径由预训练定死，提示词改不动（第 3 周实测），猜错了命中率接近 0。
这里在 ScreenSpot 上抽一批样本，同一个框中心按三种假设各换算一遍：

  pixel     除以缩放后的尺寸（Qwen2.5-VL 是这样）
  rel1000   除以 1000
  original  除以原图尺寸

对的那个命中率会远高于另外两个。量出来之后登记到 models.COORD_SPACE_BY_MODEL_TYPE。

动作格式在 ScreenAgent 测试集上看：一段式、两段式提示词各问几条，存原始输出，
统计能不能解析、有没有思考段、输出多长（生成上限 256 够不够）。

用法：
    python scripts/probe_model.py --model Qwen/Qwen3.5-4B --n 40 --screenagent 10
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gui_agent.agent import _extract_json
from gui_agent.chain import render_prompt, render_target_prompt
from gui_agent.models import DEFAULT_MODEL, GROUNDING_PROMPT, LocalQwenVL, box_center, parse_box
from gui_agent.perception import imread, resize_for_model
from gui_agent.schema import ScreenState

ROOT = Path(__file__).resolve().parents[1]
GROUNDING_TEST = ROOT / "data" / "grounding" / "test.jsonl"
SCREENAGENT_TEST = ROOT / "data" / "screenagent" / "test.jsonl"

# 现用的提示词写明了「像素值」。另外两个不提单位，看模型自己默认用哪套
GROUNDING_PROMPTS = {
    "zh_pixel": GROUNDING_PROMPT,
    "zh_plain": ("请在截图中找到「{instruction}」对应的界面元素，"
                 "只返回一个 JSON 对象，格式为 {{\"bbox_2d\": [x1, y1, x2, y2]}}。"
                 "不要输出任何其他内容。"),
    "en_plain": ('Locate "{instruction}" in the screenshot and output its bounding box '
                 'as JSON: {{"bbox_2d": [x1, y1, x2, y2]}}.'),
}
SPACES = ("pixel", "rel1000", "original")


def spread(rows: list, n: int) -> list:
    """等间隔抽 n 条，平台和元素类型都能覆盖到。"""
    if n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def to_norm(center, space: str, size, resized):
    (w, h), (rw, rh) = size, resized
    cx, cy = center
    div = {"pixel": (rw, rh), "rel1000": (1000, 1000), "original": (w, h)}[space]
    return cx / div[0], cy / div[1]


def inside(point, bbox) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def probe_grounding(vlm, n: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset("rootsautomation/ScreenSpot", split="test")
    recs = spread([json.loads(l) for l in GROUNDING_TEST.open(encoding="utf-8")], n)
    hits = {p: Counter() for p in GROUNDING_PROMPTS}
    parsed = Counter()
    beyond = {p: Counter() for p in GROUNDING_PROMPTS}  # 框超出缩放尺寸 / 超出 1000 的条数
    rows = []
    for i, r in enumerate(recs):
        img = np.array(ds[r["index"]]["image"].convert("RGB"))[:, :, ::-1]
        h, w = img.shape[:2]
        rh, rw = vlm.resized_size(h, w)
        row = {"index": r["index"], "size": [w, h], "resized": [rw, rh],
               "element_type": r["element_type"], "platform": r["platform"],
               "instruction": r["instruction"], "bbox": r["bbox"], "outputs": {}}
        for name, tmpl in GROUNDING_PROMPTS.items():
            raw = vlm.ask(img, tmpl.format(instruction=r["instruction"]), max_new_tokens=64)
            box = parse_box(raw)
            out = {"raw": raw, "box": list(box) if box else None, "hit": {}}
            if box is not None:
                parsed[name] += 1
                c = box_center(box)
                for s in SPACES:
                    ok = inside(to_norm(c, s, (w, h), (rw, rh)), r["bbox"])
                    out["hit"][s] = ok
                    hits[name][s] += ok
                beyond[name]["x>缩放宽或y>缩放高"] += box[2] > rw or box[3] > rh
                beyond[name]["坐标>1000"] += max(box) > 1000
            row["outputs"][name] = out
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  定位 {i+1}/{len(recs)}")

    print(f"\n定位口径（{len(recs)} 条，命中 = 框中心按该口径换算后落在真值框里）")
    print(f"{'提示词':<10}{'解析':<6}" + "".join(f"{s:<10}" for s in SPACES) + "越界统计")
    for name in GROUNDING_PROMPTS:
        print(f"{name:<10}{parsed[name]:<6}" + "".join(f"{hits[name][s]:<10}" for s in SPACES)
              + json.dumps(dict(beyond[name]), ensure_ascii=False))
    return {"n": len(recs), "parsed": dict(parsed),
            "hits": {p: dict(c) for p, c in hits.items()},
            "beyond": {p: dict(c) for p, c in beyond.items()}, "rows": rows}


def probe_actions(vlm, n: int) -> dict:
    recs = spread([json.loads(l) for l in SCREENAGENT_TEST.open(encoding="utf-8")], n)
    tok = vlm.processor.tokenizer
    stats = {"one_stage": Counter(), "two_stage": Counter()}
    lengths = {"one_stage": [], "two_stage": []}
    rows = []
    for r in recs:
        img = imread(r["image"])
        h, w = img.shape[:2]
        model_img, _ = resize_for_model(img)
        instruction = r["instruction_zh"] or r["instruction"]
        state = ScreenState(width=w, height=h, elements=[])
        row = {"image": r["image"], "gt": r["action"], "outputs": {}}
        for kind, prompt in (("one_stage", render_prompt(instruction, state, [])),
                             ("two_stage", render_target_prompt(instruction, []))):
            t = time.perf_counter()
            raw = vlm.ask(model_img, prompt, max_new_tokens=256)
            sec = time.perf_counter() - t
            n_tok = len(tok(raw)["input_ids"])
            lengths[kind].append(n_tok)
            s = stats[kind]
            s["条数"] += 1
            s["带代码块"] += "```" in raw
            s["带思考段"] += "<think>" in raw or "</think>" in raw
            s["顶到 256"] += n_tok >= 250
            s["thought 含中文"] += bool(re.search(r'"thought"\s*:\s*"[^"]*[一-鿿]', raw))
            try:
                data = _extract_json(raw)
                act = data.get("action") if isinstance(data, dict) else None
                s["能解析"] += 1
                s["有 action"] += isinstance(act, dict)
                pred = act.get("type") if isinstance(act, dict) else None
                s["类型对"] += pred == r["action"]["type"]
            except Exception:
                pred = None
            row["outputs"][kind] = {"raw": raw, "tokens": n_tok, "seconds": round(sec, 2),
                                    "pred_type": pred}
        rows.append(row)

    print(f"\n动作输出格式（ScreenAgent {len(recs)} 条，不带元素清单）")
    for kind, s in stats.items():
        ls = sorted(lengths[kind])
        print(f"  {kind:<10} {json.dumps(dict(s), ensure_ascii=False)}  "
              f"输出 token 中位 {ls[len(ls)//2]}，最长 {ls[-1]}")
    return {"n": len(recs), "stats": {k: dict(v) for k, v in stats.items()},
            "lengths": lengths, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=40, help="定位抽多少条")
    ap.add_argument("--screenagent", type=int, default=10, help="动作格式抽多少条，0 不看")
    ap.add_argument("--max-pixels", type=int, default=1280, help="图片上限，单位视觉 token 数")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, max_tokens=args.max_pixels, coord_space="pixel")
    print(f"加载 {args.model}：{time.perf_counter() - t0:.1f}s，"
          f"model_type={vlm.model.config.model_type}，切块系数 {vlm.factor}")

    result = {"model": args.model, "model_type": vlm.model.config.model_type,
              "factor": vlm.factor, "max_pixels": args.max_pixels}
    if args.n:
        result["grounding"] = probe_grounding(vlm, args.n)
    if args.screenagent:
        result["actions"] = probe_actions(vlm, args.screenagent)

    tag = args.tag or args.model.rstrip("/").split("/")[-1]
    out = ROOT / "logs" / f"probe_{tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始输出已存到 {out}")


if __name__ == "__main__":
    main()
