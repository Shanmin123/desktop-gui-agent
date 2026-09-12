"""构建 LoRA 微调的训练集与验证集。

对应大纲第 5 周第 1 项。

第 2 周测出来瓶颈在定位：模型在「直接问坐标」的提示词下命中 70%，在 Agent 的
动作提示词下只有 40.8%，因为它 111/120 次都在用 OCR 元素编号指位置，而 OCR 只
认文字，图标没有编号可指。所以训练样本按两段分别构造：

  定位（grounding）  截图 + 元素描述 -> {"bbox_2d": [...]}，来自 Mind2Web
                     提示词和 models.GROUNDING_PROMPT 完全一致
  动作（action）     截图 + 任务 -> {"thought":..., "action":{...}}，来自 ScreenAgent
                     提示词用生产模板 chain.render_prompt，带元素清单和历史
  拆解（plan）       截图 + 任务 -> 子任务数组，来自 ScreenAgent 的 PlanAction
                     提示词和 planner.PLAN_TEMPLATE 完全一致

训练用的提示词必须和推理时一字不差，否则学到的东西迁移不过去。第一版三处不一致
（动作用了简化模板、thought 存成空串、定位样本占 70%），动作类型准确率从 42.2%
掉到 30.6%，原因分析见 docs/第3周实验报告.md。

反思样本（EvaluateSubTaskAction）有 897 条但没有收：其中 875 条标签都是
sub_task_success，拿它训练只会强化「反思一律报成功」这个已经存在的毛病。

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

from gui_agent.agent import select_elements
from gui_agent.chain import render_prompt
from gui_agent.models import GROUNDING_PROMPT
from gui_agent.perception import imread
from gui_agent.planner import MAX_SUBTASKS, PLAN_TEMPLATE
from gui_agent.schema import Action, Element, ScreenState, Step

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finetune"
CROPS = OUT / "crops"

VIEW_W, VIEW_H = 1280, 720          # 裁剪窗口，等于本项目的运行分辨率
MIND2WEB_REPO = "osunlp/Multimodal-Mind2Web"

# --------------------------------------------------------------------------
# 动作样本：ScreenAgent
# --------------------------------------------------------------------------


def action_samples(split: str, ocr=None) -> list:
    """ScreenAgent 的可执行动作 -> 训练样本。

    三处和第一版不同，都是照着第一版掉分的原因改的：

    1. 提示词用生产模板（`chain.render_prompt`），带 OCR 元素清单和已执行历史。
       第一版用的是一个简化模板，训练时没有元素清单，推理时却有，还被要求「优先用
       element 编号」——模型没学过怎么用编号，于是写出 `14.0` 这种东西。
    2. `thought` 填人工修正过的说明文字，不再是空串。第一版教模型别写理由，
       动作类型准确率从 42.2% 掉到 30.6%。
    3. 一份回复里的多个动作共用同一张截图，按顺序把前面的动作写进历史。
       不这样做就是同一个输入配几个不同的目标动作，等于教一个矛盾的映射。

    真值点落在某个 OCR 元素里就用 element 编号，否则给 point——提示词要求的就是
    这个取舍，让模型学会什么时候该用编号。
    """
    src = ROOT / "data" / "screenagent" / f"{split}.jsonl"
    if not src.is_file():
        raise SystemExit(f"没有 {src}，先跑 scripts/prepare_screenagent.py")

    rows = [json.loads(line) for line in src.open(encoding="utf-8")]
    out, groups = [], {}
    for r in rows:
        if Path(r["image"]).is_file():
            groups.setdefault((r["session_id"], r["image"]), []).append(r)

    for (_, image), steps_raw in groups.items():
        img = imread(image)
        h, w = img.shape[:2]
        elements = ocr(img, image) if ocr else []
        state = ScreenState(width=w, height=h, elements=elements)
        # 只在提示词真的会列出来的那些元素里挑编号。清单有上限，拿一个没显示的
        # 编号当目标就是在教模型输出它看不到的东西。
        shown = select_elements(state)
        history = []
        for r in steps_raw:
            act = dict(r["action"])
            if act.get("point"):
                act["point"] = [round(v, 4) for v in act["point"]]
                eid = element_at(shown, act["point"])
                if eid is not None:
                    act = {k: v for k, v in act.items() if k != "point"}
                    act["element"] = eid
            # 不带子任务：评测脚本和默认配置（plan=False）都只有整体任务，
            # 训练时喂子任务就又是一处训练/推理不一致。同一张图上的多个动作靠
            # 历史区分，不靠子任务。
            instruction = r["instruction_zh"] or r["instruction"]
            thought = r.get("thought", "")
            if not thought:
                continue  # 没有说明文字的不收，免得和有说明文字的样本教法不一致
            out.append({
                "kind": "action",
                "source": "screenagent",
                "image": image,
                "prompt": render_prompt(instruction, state, list(history)),
                "response": json.dumps({"thought": thought, "action": act},
                                       ensure_ascii=False),
            })
            history.append(Step(state, Action.from_dict(r["action"]), ok=True, changed=True))
    return out


OCR_CACHE = ROOT / "data" / "screenagent" / "_ocr_cache.json"


def cached_ocr(ocr):
    """把 OCR 结果按图片路径缓存到磁盘。

    调配比要重建几次数据，每次为一千来张图重跑 OCR 是 8 分钟白等。
    """
    if ocr is None:
        return None, None
    cache = json.loads(OCR_CACHE.read_text(encoding="utf-8")) if OCR_CACHE.is_file() else {}
    stats = {"命中": 0, "新算": 0}

    def run(img, key):
        if key in cache:
            stats["命中"] += 1
            return [Element(**e) for e in cache[key]]
        els = ocr(img)
        cache[key] = [{"id": e.id, "bbox": list(e.bbox), "text": e.text,
                       "source": e.source, "confidence": e.confidence} for e in els]
        stats["新算"] += 1
        return els

    def save():
        OCR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        OCR_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"  OCR 缓存：{stats}，共 {len(cache)} 张")

    return run, save


def element_at(elements, point):
    """真值点落在哪个元素里，返回最小的那个的编号，没有就返回 None。"""
    x, y = point
    hit = [e for e in elements
           if e.bbox[0] <= x <= e.bbox[2] and e.bbox[1] <= y <= e.bbox[3] and e.text.strip()]
    if not hit:
        return None
    return min(hit, key=lambda e: (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])).id


def plan_samples(split: str) -> list:
    """ScreenAgent 的 PlanAction 列表 -> 拆解样本，对应 planner.PLAN_TEMPLATE。

    第一版完全没用这批数据。复杂任务实测里拆解会凭空补步骤、也会拆错方向，
    这是能直接拿数据教的。
    """
    src = ROOT / "data" / "screenagent" / "plans.jsonl"
    if not src.is_file():
        return []
    out = []
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") != split or not Path(r["image"]).is_file():
            continue
        img = imread(r["image"])
        h, w = img.shape[:2]
        out.append({
            "kind": "plan",
            "source": "screenagent",
            "image": r["image"],
            "prompt": PLAN_TEMPLATE.format(
                instruction=r["instruction_zh"] or r["instruction"],
                elements="  （这一步不看元素清单）",
                max_subtasks=MAX_SUBTASKS),
            "response": json.dumps(r["subtasks"][:MAX_SUBTASKS], ensure_ascii=False),
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
    ap.add_argument("--limit-grounding", type=int, default=None,
                    help="定位样本只留这么多。第一版定位占了 70%%，动作生成被挤掉")
    ap.add_argument("--no-plan", action="store_true", help="不出拆解样本")
    ap.add_argument("--no-ocr", action="store_true",
                    help="动作样本的提示词里不带元素清单。带清单要先跑一遍 OCR")
    args = ap.parse_args()
    random.seed(args.seed)

    ocr = None
    per = None
    if not args.no_ocr:
        from gui_agent.perception import Perception

        print("动作样本的提示词要带元素清单，先跑一遍 OCR ……")
        per = Perception()
        per.reader
        ocr, save_cache = cached_ocr(per.ocr)
    try:
        train = action_samples("train", ocr=ocr)
        val = action_samples("val", ocr=ocr)
    finally:
        if per is not None:
            save_cache()
            per.close()
    print(f"动作样本（ScreenAgent）：训练 {len(train)}，验证 {len(val)}")

    if not args.no_plan:
        pt, pv = plan_samples("train"), plan_samples("val")
        print(f"拆解样本（ScreenAgent）：训练 {len(pt)}，验证 {len(pv)}")
        train += pt
        val += pv

    if not args.no_grounding:
        g, skip = grounding_samples("train", limit=args.limit_mind2web)
        if args.limit_grounding is not None:
            random.shuffle(g)
            g = g[:args.limit_grounding]
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
