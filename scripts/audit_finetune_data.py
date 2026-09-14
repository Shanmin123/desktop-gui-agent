"""微调数据集审计：找静默的错标、重复、泄漏。

对应大纲第 5 周第 1 项。这类问题不会让训练报错，只让结果变差又看不出原因——
第一版微调掉 11.6 个点，就是 thought 存成空串加提示词不一致，全是静默的。

十项检查：
  1  空的 prompt 或 response
  2  完全重复的样本
  3  同一张图同一提示词配了多个不同回答（矛盾映射）
  4  训练集和验证集用了同一张截图
  5  评测用的 test 截图泄漏进训练或验证集
  6  动作样本的 element 编号不在提示词清单里 / 动作解析不出来
  7  thought 里直接写了坐标（模型会学着抄）
  8  动作样本的 thought 是空串
  9  定位样本的坐标超出范围（像素框出图 / 归一化点不在 0~1）
  10 拆解样本为空或有重复子任务

用法：
    python scripts/audit_finetune_data.py                  # data/finetune
    python scripts/audit_finetune_data.py data/finetune_2sb
"""

import json, re, sys
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui_agent.schema import Action

def load(split):
    d = sys.argv[1] if len(sys.argv) > 1 else "data/finetune"
    return [json.loads(l) for l in (ROOT/d/f"{split}.jsonl").open(encoding="utf-8")]

train, val = load("train"), load("val")
problems = []

def issue(msg, n=None):
    problems.append(f"{msg}" + (f"（{n} 条）" if n is not None else ""))

# 1 空字段
for name, rows in (("train", train), ("val", val)):
    n = sum(1 for r in rows if not r["prompt"].strip() or not r["response"].strip())
    if n: issue(f"{name} 有空的 prompt 或 response", n)

# 2 重复样本
for name, rows in (("train", train), ("val", val)):
    keys = Counter((r["prompt"], r["response"]) for r in rows)
    dup = sum(c - 1 for c in keys.values() if c > 1)
    if dup: issue(f"{name} 有完全重复的样本", dup)

# 3 同一输入配不同目标
for name, rows in (("train", train), ("val", val)):
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault((r["image"], r["prompt"]), set()).add(r["response"])
    conflict = sum(1 for v in by_prompt.values() if len(v) > 1)
    if conflict: issue(f"{name} 同一张图同一提示词配了多个不同回答", conflict)

# 4 训练/验证的图片重叠
overlap = {r["image"] for r in train} & {r["image"] for r in val}
if overlap: issue("训练集和验证集用了同一张截图", len(overlap))

# 5 test 划分泄漏进训练
test_imgs = {json.loads(l)["image"] for l in (ROOT/"data"/"screenagent"/"test.jsonl").open(encoding="utf-8")}
leak = ({r["image"] for r in train} | {r["image"] for r in val}) & test_imgs
if leak: issue("评测用的 test 截图出现在训练或验证集里", len(leak))

# 6 动作样本：编号必须在清单里，动作必须能解析
# 两段式样本的位置写成 target 控件名，坐标是推理时第二段解析出来的，schema 里
# 没有这一项，不能按解析失败算——但控件名必须是非空字符串，而且不能再带坐标。
bad_eid = bad_act = bad_target = 0
for r in train + val:
    if r["kind"] != "action": continue
    act = json.loads(r["response"])["action"]
    if "target" in act:
        t = act["target"]
        if not (isinstance(t, str) and t.strip()) or {"point", "element"} & set(act):
            bad_target += 1
    elif "element" in act:
        shown = {int(m) for m in re.findall(r"^\s*\[(\d+)\]", r["prompt"], re.M)}
        if act["element"] not in shown: bad_eid += 1
    else:
        try: Action.from_dict(act)
        except Exception: bad_act += 1
if bad_eid: issue("动作样本的 element 编号不在提示词清单里", bad_eid)
if bad_act: issue("动作样本的动作解析不出来", bad_act)
if bad_target: issue("两段式样本的 target 是空的、或者还带着坐标", bad_target)

# 7 thought 里泄漏坐标
leak_xy = sum(1 for r in train + val if r["kind"] == "action"
              and re.search(r"\d\.\d{3,}", json.loads(r["response"]).get("thought", "")))
if leak_xy: issue("thought 里直接写了坐标（模型会学着抄）", leak_xy)

# 8 thought 为空
empty_thought = sum(1 for r in train + val if r["kind"] == "action"
                    and not json.loads(r["response"]).get("thought", "").strip())
if empty_thought: issue("动作样本的 thought 是空串", empty_thought)

# 9 定位样本：像素框要落在图里，归一化的点要落在 0~1
import PIL.Image
bad_box = 0
for r in (train + val):
    if r["kind"] != "grounding": continue
    body = json.loads(r["response"])
    if "point" in body:
        if not all(0.0 <= v <= 1.0 for v in body["point"]): bad_box += 1
        continue
    b = body["bbox_2d"]
    with PIL.Image.open(r["image"]) as im: w, h = im.size
    if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h): bad_box += 1
if bad_box: issue("定位样本的坐标超出范围", bad_box)

# 10 拆解样本：子任务非空且不重复
bad_plan = 0
for r in train + val:
    if r["kind"] != "plan": continue
    steps = json.loads(r["response"])
    if not steps or len(set(steps)) != len(steps): bad_plan += 1
if bad_plan: issue("拆解样本为空或有重复子任务", bad_plan)

print(f"训练 {len(train)} 条 {dict(Counter(r['kind'] for r in train))}")
print(f"验证 {len(val)} 条 {dict(Counter(r['kind'] for r in val))}")
print()
if problems:
    print("发现问题：")
    for s in problems: print("  -", s)
else:
    print("10 项检查全部通过")
