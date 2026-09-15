"""把 ScreenAgent test 353 步按应用类型拆开，看各配置在不同应用上的差异。

对应大纲第 7 周第 3 项「分析系统在不同应用下的表现差异」的离线部分。test 划分有 70 个
session，按任务描述里的关键词归成 7 类应用；每类里算动作类型准确率和「类型对且点准」
（坐标类动作要求预测点与真值点的归一化距离 ≤ 0.10）。

输出 logs/screenagent_by_app.json 和 docs/figures/screenagent_by_app.png。

用法：
    python scripts/analyze_by_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
TEST = ROOT / "data" / "screenagent" / "test.jsonl"

# 按顺序匹配，先命中的算数：数据表查询的描述里也有 table，要排在办公文档前面
RULES = [
    ("表格数据", ["_data table"]),
    ("图像编辑", ["gimp"]),
    ("终端与代码", ["command line", "terminal", "line of code"]),
    ("游戏", ["game"]),
    ("系统工具", ["calculator", "event viewer", "disk management", "device manager", "partition",
               "rename a target txt"]),
    ("办公文档", ["pdf", "slide", "presentation", "document", "paper size", "page number", "font",
               "text size", "table", "ruler", "underline", "bold", "text color", "insert a triangle"]),
]
DEFAULT_APP = "浏览器与网页"
APPS = [DEFAULT_APP] + [name for name, _ in RULES]

CONFIGS = [
    ("Qwen2.5 基座 一段式", ["screenagent_base_cases.json"]),
    ("Qwen2.5 微调 两段式", ["screenagent_lora2sp_2s_fix.json", "screenagent_lora2sp_2s.json"]),
    ("Qwen3.5 基座 一段式", ["screenagent_q35_base_cases.json"]),
    ("Qwen3.5 基座 两段式", ["screenagent_q35_base_2s.json"]),
    ("Qwen3.5 微调 两段式", ["screenagent_q35_2sp_2s.json"]),
]


def app_of(instruction: str) -> str:
    text = (instruction or "").lower()
    for name, words in RULES:
        if any(w in text for w in words):
            return name
    return DEFAULT_APP


def apps_of_test_steps() -> List[str]:
    """test.jsonl 第 i 行属于哪类应用；评测日志里的 cases[*]["i"] 就是这个行号。"""
    with TEST.open(encoding="utf-8") as f:
        return [app_of(json.loads(line).get("instruction", "")) for line in f]


def by_app(cases: list, apps: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, List[int]] = {}
    for c in cases:
        s = stats.setdefault(apps[c["i"]], [0, 0, 0])
        s[0] += 1
        ok = c["gt"] == c["pred"]
        s[1] += ok
        s[2] += ok and (c.get("dist") is None or c["dist"] <= 0.10)
    return {app: {"n": n, "type_accuracy": t / n, "joint_accuracy": j / n}
            for app, (n, t, j) in stats.items()}


def _load(names: List[str]):
    for name in names:
        try:
            d = json.loads((LOGS / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("cases"):
            return name, d
    return None, None


def main() -> None:
    apps = apps_of_test_steps()
    counts = {a: apps.count(a) for a in APPS}
    results = {}
    for label, names in CONFIGS:
        name, d = _load(names)
        if d:
            results[label] = {"log": name, "by_app": by_app(d["cases"], apps)}

    if not results:
        raise SystemExit("没有带逐条记录的 ScreenAgent 评测日志")

    labels = list(results)
    print("| 应用 | 步数 | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for app in APPS:
        cells = []
        for label in labels:
            s = results[label]["by_app"].get(app)
            cells.append("—" if not s else f"{s['type_accuracy']:.0%} / {s['joint_accuracy']:.0%}")
        print(f"| {app} | {counts[app]} | " + " | ".join(cells) + " |")
    print("\n格子里是「动作类型准确率 / 类型对且点准」")

    out = LOGS / "screenagent_by_app.json"
    out.write_text(json.dumps({"steps_per_app": counts, "configs": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"结果已存到 {out}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_charts as charts

    try:
        plt = charts._plt()
    except ImportError:
        return
    charts.OUT.mkdir(parents=True, exist_ok=True)
    series = [(label, [results[label]["by_app"].get(app, {}).get("type_accuracy") for app in APPS])
              for label in labels]
    charts._bars(plt, [f"{a}\n({counts[a]} 步)" for a in APPS], series,
                 "ScreenAgent test 353 步：按应用类型的动作类型准确率", "准确率",
                 charts.OUT / "screenagent_by_app.png")
    print(f"图已存到 {charts.OUT / 'screenagent_by_app.png'}")


if __name__ == "__main__":
    main()
