"""把 ScreenAgent 数据集转成本项目的数据格式。

对应大纲第 3 周第 1 项。

数据随 ScreenAgent 仓库一起发布，不在 HuggingFace 上：

    git clone --depth 1 https://github.com/niuzaisheng/ScreenAgent.git
    python scripts/prepare_screenagent.py --src <仓库>/data/ScreenAgent

原始记录是一次会话里的一步：截图 + 提示词 + 模型输出 + 人工修正后的动作。
按 ScreenAgent 的约定，训练用的是修正版 `actions`，不是模型原始输出。

动作、计划、反思三类分开落盘，三类都能当微调样本：

  actions      鼠标、键盘、等待，对应 schema.py 的 10 个动作
  plans        PlanAction 列表，一条任务拆成几个子任务
  reflections  EvaluateSubTaskAction，判定当前子任务的状态

每条动作还带上 `thought`：`LLM_response_editer` 里 JSON 之前那段人工修正过的说明
文字。第一版微调把它丢了、`thought` 一律存成空串，等于教模型别写理由，动作类型
准确率从 42.2% 掉到 30.6%。

文件名带 `_neg_plan` / `_neg_eval` 的是故意写错的负样本，计划和反思都不能收。

输出：
    data/screenagent/train.jsonl / val.jsonl / test.jsonl
    data/screenagent/plans.jsonl
    data/screenagent/reflections.jsonl
    data/screenagent/samples/*.png   抽样画框，人工核对坐标
"""

import argparse
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.control import normalize_hotkey
from gui_agent.planner import SITUATIONS
from gui_agent.perception import imread, imwrite
from gui_agent.schema import Action, Element
from gui_agent.perception import annotate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "screenagent"

# 鼠标动作 -> schema.py 的动作类型
_MOUSE = {
    "click": "click",          # 右键的情况看 mouse_button
    "double_click": "left_double",
    "scroll_up": "scroll",
    "scroll_down": "scroll",
}

# 滚动记录里没有坐标，用屏幕中心。滚动作用于当前焦点区域，落在哪一点不影响语义。
SCROLL_POINT = (0.5, 0.5)


def to_action(raw: dict, w: int, h: int):
    """一条原始动作 -> Action，转不了的返回 None 和原因。"""
    kind = raw.get("action_type")

    if kind == "WaitAction":
        return Action("wait"), None

    if kind == "MouseAction":
        mt = raw.get("mouse_action_type")
        if mt not in _MOUSE:
            # move / down / up 是分解开的原语，drag 只记了一个落点、没有起点，
            # 都凑不出 schema 里的动作
            return None, f"mouse:{mt}"
        pos = raw.get("mouse_position")
        if mt.startswith("scroll"):
            return Action("scroll", point=SCROLL_POINT,
                          direction="up" if mt == "scroll_up" else "down"), None
        if not pos:
            return None, "mouse:缺坐标"
        point = (pos["width"] / w, pos["height"] / h)
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            return None, "mouse:坐标越界"
        t = _MOUSE[mt]
        if t == "click" and raw.get("mouse_button") == "right":
            t = "right_single"
        return Action(t, point=point), None

    if kind == "KeyboardAction":
        kt = raw.get("keyboard_action_type")
        if kt == "text":
            text = raw.get("keyboard_text")
            if text is None:
                return None, "keyboard:缺文本"
            return Action("type", text=text), None
        if kt == "press":
            key = raw.get("keyboard_key")
            keys = key if isinstance(key, list) else [key]
            combo = "+".join(str(k) for k in keys if k)
            if not combo:
                return None, "keyboard:缺键名"
            try:
                normalize_hotkey(combo)  # 确认键名能落到 pyautogui 上
            except ValueError:
                return None, f"keyboard:键名无法解析 {combo!r}"
            return Action("hotkey", text=combo), None
        return None, f"keyboard:{kt}"

    return None, f"其他:{kind}"


NEG_MARKERS = ("_neg_plan", "_neg_eval")  # 故意写错的负样本，不能当训练目标


def is_negative(path: Path) -> bool:
    return any(m in path.name for m in NEG_MARKERS)


def subtask_of(d: dict) -> str:
    """原始记录里没有单独的子任务字段，它写在送给模型的提示词里。"""
    for key in ("send_prompt_zh", "send_prompt"):
        m = re.search(r'现在的子任务是\s*[“"]?(.+?)[”"]?\s*[。\n]', str(d.get(key) or ""))
        if m:
            return m.group(1).strip()
    return ""


def response_prose(text) -> str:
    """取 `LLM_response_editer` 里 JSON 之前那段说明文字，当作这一步的 thought。

    人工修正过的原文形如「根据现有屏幕图像的状态，我们可以看到一个搜索框。……
    因此，下一步操作应该是：```json [...] ```」，前半段就是理由。
    """
    if not isinstance(text, str):
        return ""
    m = re.search(r"```", text)
    cut = m.start() if m else text.find("[")
    prose = (text[:cut] if cut > 0 else "").strip()
    # 结尾那句「下一步操作应该是：」只是在引出 JSON，去掉
    return re.sub(r"[，。:：]?\s*(因此[，,]?)?下一步(的)?操作应该是[：:]?\s*$", "", prose).strip()


def build(src: Path, split: str) -> tuple:
    """遍历一个划分，返回 (记录列表, 跳过原因统计, 阶段统计)。"""
    records, skipped, stages = [], Counter(), Counter()
    plans, reflections = [], []
    for jf in sorted((src / split).rglob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            skipped[f"读文件失败:{type(e).__name__}"] += 1
            continue

        w, h = d.get("video_width"), d.get("video_height")
        img = jf.parent / "images" / (d.get("saved_image_name") or "")
        if not (w and h and img.is_file()):
            skipped["缺截图或分辨率"] += 1
            continue

        negative = is_negative(jf)
        thought = "" if negative else response_prose(d.get("LLM_response_editer"))
        task_zh = d.get("task_prompt_zh") or d.get("task_prompt") or ""
        task_en = d.get("task_prompt_en") or d.get("task_prompt") or ""
        subtask = subtask_of(d)

        # 一份回复里的 PlanAction 合起来就是一条拆解计划
        steps = [str(a.get("element", "")).strip() for a in d.get("actions") or []
                 if isinstance(a, dict) and a.get("action_type") == "PlanAction"
                 and str(a.get("element", "")).strip()]
        if steps and not negative:
            plans.append({
                "source": "screenagent", "session_id": d.get("session_id", jf.parent.name),
                "image": str(img.resolve()), "width": w, "height": h,
                "instruction": task_en, "instruction_zh": task_zh, "subtasks": steps,
            })

        for raw in d.get("actions") or []:
            if not isinstance(raw, dict):
                skipped["动作不是对象"] += 1
                continue
            stages[raw.get("action_type")] += 1
            if raw.get("action_type") == "EvaluateSubTaskAction" and not negative:
                situation = str(raw.get("situation", "")).strip()
                if situation in SITUATIONS:
                    reflections.append({
                        "source": "screenagent",
                        "session_id": d.get("session_id", jf.parent.name),
                        "image": str(img.resolve()), "width": w, "height": h,
                        "instruction": task_en, "instruction_zh": task_zh,
                        "subtask": subtask, "situation": situation,
                        "advice": str(raw.get("advice", "")).strip(),
                    })
            action, why = to_action(raw, w, h)
            if action is None:
                skipped[why] += 1
                continue
            records.append({
                "source": "screenagent",
                "session_id": d.get("session_id", jf.parent.name),
                "image": str(img.resolve()),
                "width": w,
                "height": h,
                "instruction": d.get("task_prompt_en") or d.get("task_prompt") or "",
                "instruction_zh": task_zh,
                "action": action.to_dict(),
                "thought": thought,
                "subtask": subtask,
            })
    return records, skipped, stages, plans, reflections


def dump_samples(records, out_dir: Path, n: int = 6) -> None:
    """抽几条把动作点画到截图上，人工确认坐标换算没搞反。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = [r for r in records if r["action"].get("point")]
    for i, r in enumerate(random.sample(pool, min(n, len(pool)))):
        x, y = r["action"]["point"]
        d = 0.012
        marker = Element(id=i, bbox=(x - d, y - d, x + d, y + d), text=r["action"]["type"])
        imwrite(str(out_dir / f"screenagent_{i}_{r['action']['type']}.png"),
                annotate(imread(r["image"]), [marker]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ScreenAgent 仓库里的 data/ScreenAgent 目录")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    src = Path(args.src)
    if not (src / "train").is_dir():
        raise SystemExit(f"{src} 下没有 train 目录，--src 要指到仓库的 data/ScreenAgent")

    # 官方把测试集打成了 zip，解开后一并处理
    zipped = src / "test.zip"
    if zipped.is_file() and not (src / "test").is_dir():
        print(f"解压 {zipped.name} ……")
        with zipfile.ZipFile(zipped) as z:
            z.extractall(src)

    # 官方已经分好 train / test，测试集整份留作评测，不参与训练也不参与验证。
    # 合起来重新划分会把测试样本混进训练池，微调后拿它评测出来的数字就是污染的。
    pool, all_skipped, all_stages = [], Counter(), Counter()
    held_out = []
    plans_pool, reflect_pool = [], []
    for split in ("train", "test"):
        if not (src / split).is_dir():
            continue
        recs, skipped, stages, plans, reflections = build(src, split)
        print(f"{split}: {len(recs)} 条可执行动作，{len(plans)} 条拆解计划，"
              f"{len(reflections)} 条反思")
        (held_out if split == "test" else pool).extend(recs)
        all_skipped += skipped
        all_stages += stages
        # 计划和反思只从训练划分取，测试划分整份留作评测
        if split == "train":
            plans_pool.extend(plans)
            reflect_pool.extend(reflections)

    print("\n原始动作类型:")
    for k, v in all_stages.most_common():
        print(f"  {str(k):<24} {v}")
    print("\n未转换:")
    for k, v in all_skipped.most_common():
        print(f"  {str(k):<24} {v}")

    print("\n转换后的动作分布:")
    for k, v in Counter(r["action"]["type"] for r in pool + held_out).most_common():
        print(f"  {k:<14} {v}")

    # 训练集内部再按 session 切出验证集，同一次会话不跨两边，
    # 避免同一屏的画面在训练和验证里都出现
    sessions = sorted({r["session_id"] for r in pool})
    random.shuffle(sessions)
    n_val = max(1, int(len(sessions) * args.val_ratio))
    val_ids = set(sessions[:n_val])
    train = [r for r in pool if r["session_id"] not in val_ids]
    val = [r for r in pool if r["session_id"] in val_ids]

    leaked = {r["session_id"] for r in held_out} & {r["session_id"] for r in pool}
    if leaked:
        raise SystemExit(f"训练集和测试集出现了相同的 session：{sorted(leaked)[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, recs in [("train", train), ("val", val), ("test", held_out)]:
        p = OUT / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                     encoding="utf-8")
        print(f"\n{p}  {len(recs)} 条，{len({r['session_id'] for r in recs})} 个 session")

    # 计划和反思按 session 跟着动作的划分走，验证集那些 session 的也归验证集，
    # 免得同一次会话的画面两边都出现
    for name, recs in [("plans", plans_pool), ("reflections", reflect_pool)]:
        p = OUT / f"{name}.jsonl"
        for r in recs:
            r["split"] = "val" if r["session_id"] in val_ids else "train"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                     encoding="utf-8")
        n_val = sum(1 for r in recs if r["split"] == "val")
        print(f"{p}  {len(recs)} 条（训练 {len(recs) - n_val} / 验证 {n_val}）")

    n_thought = sum(1 for r in pool if r.get("thought"))
    print(f"\n带人工修正说明文字的动作：{n_thought}/{len(pool)}")

    dump_samples(pool, OUT / "samples")
    print(f"抽样画框已存到 {OUT / 'samples'}，请人工确认标记落在动作描述的位置上。")


if __name__ == "__main__":
    main()
