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
    python scripts/eval_screenagent.py --adapter checkpoints/lora_v2 --tag lora
    python scripts/eval_screenagent.py --locate-target --tag base_2s   # 两段式
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
from gui_agent.chain import parse_with_target, render_prompt, render_target_prompt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_finetune_data import TARGET_ELEMENTS_LIMIT
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


def named_target(raw: str):
    """模型这一步说要操作哪个控件，取不出来就返回 None。"""
    from gui_agent.agent import _extract_json

    try:
        data = _extract_json(raw)
    except Exception:
        return None
    act = data.get("action") if isinstance(data, dict) else None
    t = act.get("target") if isinstance(act, dict) else None
    return t.strip() if isinstance(t, str) and t.strip() else None


# 指标按 GUI 智能体论文里常用的那套来，自造的名字没人能对照：
#   Op.F1        动作类型的 F1（Mind2Web、SeeClick、OS-Atlas 都报这个）。macro 把少数类和多数类
#                同权，micro 等于类型准确率。键盘动作还要求文本一致才算对，和 Mind2Web 对 TYPE
#                比较输入内容的做法一致
#   Step SR      这一步整体算不算做对：类型对，坐标类动作还要点得够准
#   距离阈值     AITW 用「预测点与真值点的距离 ≤ 屏幕尺寸的 14%」判对，这里两档都报：
#                0.14 与 AITW 对齐，0.10 是更严的自定档
KEYBOARD_TYPES = ("type", "hotkey")


def _text_ok(case: dict) -> bool:
    """键盘动作的内容对不对。真值没写内容就不卡这一项。"""
    gt_text = (case.get("gt_text") or "").strip()
    if not gt_text:
        return True
    return (case.get("pred_text") or "").strip().lower() == gt_text.lower()


def op_f1(cases: list) -> dict:
    """按动作类型算 P / R / F1。键盘动作要内容也对。"""
    types = sorted({c["gt"] for c in cases} | {c["pred"] for c in cases if c["pred"] != "解析失败"})
    per, f1s = {}, []
    for t in types:
        pred_t = [c for c in cases if c["pred"] == t]
        gt_t = [c for c in cases if c["gt"] == t]
        hit = sum(1 for c in pred_t
                  if c["gt"] == t and (t not in KEYBOARD_TYPES or _text_ok(c)))
        p = hit / len(pred_t) if pred_t else 0.0
        r = hit / len(gt_t) if gt_t else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[t] = {"precision": p, "recall": r, "f1": f, "gt": len(gt_t), "pred": len(pred_t)}
        if gt_t:
            f1s.append(f)
    micro = sum(1 for c in cases
                if c["gt"] == c["pred"] and (c["gt"] not in KEYBOARD_TYPES or _text_ok(c))) / len(cases)
    return {"macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "micro_f1": micro, "per_type": per}


def step_success(cases: list, threshold: float) -> float:
    """一步算做对：类型对（键盘动作连内容一起对），坐标类动作的距离在阈值内。"""
    ok = 0
    for c in cases:
        if c["gt"] != c["pred"]:
            continue
        if c["gt"] in KEYBOARD_TYPES and not _text_ok(c):
            continue
        if c.get("dist") is not None and c["dist"] > threshold:
            continue
        ok += 1
    return ok / len(cases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--adapter", default=None, help="挂上 LoRA 权重评测微调后的模型")
    ap.add_argument("--max-pixels", type=int, default=1280,
                    help="图片上限，单位是视觉 token 数（Qwen2.5-VL 一个 token 是 28×28 像素，Qwen3.5 是 32×32）。微调时降过这个值的话，"
                         "用同一个值评测才能看出权重本身的效果")
    ap.add_argument("--coord-space", default=None, choices=["pixel", "rel1000"],
                    help="模型回的定位坐标是什么口径。不给就按模型登记的口径")
    ap.add_argument("--no-ocr", action="store_true",
                    help="不跑 OCR，提示词里不带元素清单。默认带，与实际循环一致")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="生成长度上限。之前所有对照都是 128 跑的，默认不动它；"
                         "要看放宽之后的效果就显式传 256")
    ap.add_argument("--ocr-cpu", action="store_true",
                    help="OCR 放到 CPU 上跑。带元素清单评测时显卡上同时有模型和 OCR，"
                         "12 G 的卡会被挤满，然后 WDDM 静默换页——看着 100% 占用，"
                         "实际一步都不走")
    ap.add_argument("--target-elements", action="store_true",
                    help="两段式第一问里带 OCR 元素清单（要配同样训练出来的权重）")
    ap.add_argument("--locate-target", action="store_true",
                    help="两段式：第一段只问要操作哪个控件，第二段用定位提示词换成坐标。"
                         "这是 --locate-target 上线时跑的那条路径，"
                         "用它评测才是在量真正部署的配置")
    args = ap.parse_args()

    recs = load(args.limit)
    print(f"评测集 {len(recs)} 条，{len({r['session_id'] for r in recs})} 个 session"
          f"{'，两段式' if args.locate_target else ''}")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, adapter=args.adapter, max_tokens=args.max_pixels,
                      coord_space=args.coord_space)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s")

    # 两段式的提示词里本来就没有元素清单，agent.py 在这条路径上也不跑 OCR，
    # 评测跟着一起关掉才是同一个配置
    # 两段式默认不跑 OCR（提示词里没有清单，agent.py 也是这么做的）；
    # --target-elements 要用清单，就得跑
    perception = (None if args.no_ocr or (args.locate_target and not args.target_elements)
                  else Perception(gpu=not args.ocr_cpu))

    n_type_ok = n_parse_fail = 0
    kb_total = kb_hit = 0
    dists, latencies, cases = [], [], []
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

            instruction = r["instruction_zh"] or r["instruction"]
            prompt = (render_target_prompt(instruction, [],
                                           state if args.target_elements else None,
                                           TARGET_ELEMENTS_LIMIT)
                      if args.locate_target else render_prompt(instruction, state, []))
            rh, rw = vlm.coord_size(*model_img.shape[:2])

            t = time.perf_counter()
            raw = vlm.ask(model_img, prompt, max_new_tokens=args.max_new_tokens)

            try:
                if args.locate_target:
                    # 第二段的定位调用也算在这一步的耗时里，两段式本来就要两次前向
                    _, pred = parse_with_target(raw, vlm, model_img, state, (rw, rh))
                else:
                    _, pred = parse_step(raw, state, model_size=(rw, rh))
            except ValueError as e:
                latencies.append(time.perf_counter() - t)
                n_parse_fail += 1
                confusion[(gt["type"], "解析失败")] += 1
                cases.append({"i": i, "gt": gt["type"], "pred": "解析失败",
                              "target": named_target(raw), "why": str(e)[:80],
                              "gt_text": gt.get("text")})
                continue
            latencies.append(time.perf_counter() - t)

            confusion[(gt["type"], pred.type)] += 1
            n_type_ok += pred.type == gt["type"]

            if gt["type"] in KEYBOARD:
                kb_total += 1
                kb_hit += pred.type in KEYBOARD

            d = None
            if gt["type"] in POINTED and pred.type in POINTED and gt.get("point") and pred.point:
                d = distance(pred.point, tuple(gt["point"]))
                dists.append(d)
            # 逐条留痕：类型对不对、点得准不准、两段式报的控件名是什么。
            # 只有混淆矩阵的话，「类型对了但点偏了」这种查不出来。
            cases.append({"i": i, "gt": gt["type"], "pred": pred.type,
                          "target": named_target(raw),
                          "dist": None if d is None else round(d, 3),
                          "gt_text": gt.get("text"), "pred_text": pred.text})

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
    # 联合指标：一步要算成功，类型得对，坐标类动作还得点得够准。
    # 单看类型准确率会高估，单看点击距离又漏掉类型错的那些。
    joint = sum(1 for c in cases if c["gt"] == c["pred"]
                and (c.get("dist") is None or c["dist"] <= 0.10))
    print(f"类型对且点得准   {joint}/{n} = {joint/n:.1%}（坐标类动作要求距离 ≤ 0.10）")
    f1 = op_f1(cases)
    print(f"Op.F1            macro {f1['macro_f1']:.1%}，micro {f1['micro_f1']:.1%}（键盘动作要内容也对）")
    print(f"Step SR          距离 ≤ 0.10 {step_success(cases, 0.10):.1%}，"
          f"≤ 0.14（AITW 口径）{step_success(cases, 0.14):.1%}")
    print(f"平均单条耗时     {sum(latencies)/len(latencies):.2f}s")

    print("\n真值类型 -> 预测类型（前 15）：")
    for (g, p), c in confusion.most_common(15):
        print(f"  {g:<14} -> {p:<14} {c}")

    out = ROOT / "logs" / f"screenagent_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "adapter": args.adapter,
        "max_pixels": args.max_pixels,
        "coord_space": vlm.coord_space,
        "n": n,
        "with_ocr": perception is not None,
        "locate_target": args.locate_target,
        "target_elements": args.target_elements,
        "max_new_tokens": args.max_new_tokens,
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
        "joint_accuracy": joint / n,
        "op_f1": op_f1(cases),
        "step_success_rate": {"0.10": step_success(cases, 0.10), "0.14": step_success(cases, 0.14)},
        "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
