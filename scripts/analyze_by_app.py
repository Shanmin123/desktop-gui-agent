"""把 ScreenAgent test 353 步按应用类型拆开，看各配置在不同应用上的差异。

对应大纲第 7 周第 3 项「分析系统在不同应用下的表现差异」的离线部分。test 划分有 70 个
session，按任务描述里的关键词归成 7 类应用；每类里算 Op.F1 micro 和 Step SR ≤0.10
（坐标类动作要求预测点与真值点的归一化距离 ≤ 0.10）。

另给每个配置算整体指标，是第 7 周第 2 项的成功率、执行时间、错误率在离线评测上的口径：
  离线任务成功率  一个 session 里每一步都「类型对且点准」才算这条任务做对。评测时每一步喂的是
                真实截图、提示词里不带历史，所以是「每一步单独拿出来都做对」的严格口径，和 live
                里连着做完不是一回事（前一步错了，后面的屏幕就对不上）
  单步耗时        评测日志里每步的平均推理时间
  无法执行率      输出解析不了，或两段式里定位不到控件，这一步拿不出可执行的动作

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

# 只列现在这一版基座（Qwen3.5-4B）的配置：交付和汇报都只讲一套系统，上一代 Qwen2.5-VL 的
# 日志仍在 logs/ 里，要看就传文件名给 scripts/score_actions.py
# 三个配置的日志口径一致（键盘动作比输入内容），才能放在一张表里比
CONFIGS = [
    ("交付：加对齐层", ["screenagent_q25_proj_2s.json"]),
    ("对照：只训语言模型", ["screenagent_q25_lm_2s.json"]),
    ("基座", ["screenagent_q25_base_2s.json"]),
]

PARSE_FAILED = "解析失败"

# 指标一律用 eval_screenagent.py 的实现：键盘动作要求输入内容也对，和
# scripts/score_actions.py、《系统全面评估报告》完全同口径，避免同一份日志出两个数
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("eval_screenagent", ROOT / "scripts" / "eval_screenagent.py")
_E = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_E)  # eval_screenagent.py 在逐条记录里给拿不出动作的步写的预测类型


def app_of(instruction: str) -> str:
    text = (instruction or "").lower()
    for name, words in RULES:
        if any(w in text for w in words):
            return name
    return DEFAULT_APP


def _test_steps() -> List[dict]:
    with TEST.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def apps_of_test_steps() -> List[str]:
    """test.jsonl 第 i 行属于哪类应用；评测日志里的 cases[*]["i"] 就是这个行号。"""
    return [app_of(r.get("instruction", "")) for r in _test_steps()]


def sessions_of_test_steps() -> List[str]:
    """test.jsonl 第 i 行属于哪个 session（一条完整任务）。"""
    return [r["session_id"] for r in _test_steps()]


def _joint_ok(c: dict) -> bool:
    """这一步算不算做对：类型对、键盘动作内容也对、坐标类动作点在 0.10 以内。"""
    return _E.step_success([c], 0.10) == 1.0


def by_app(cases: list, apps: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, List[int]] = {}
    for c in cases:
        s = stats.setdefault(apps[c["i"]], [0, 0, 0])
        s[0] += 1
        s[1] += c["gt"] == c["pred"]
        s[2] += _joint_ok(c)
    return {app: {"n": n, "type_accuracy": t / n, "joint_accuracy": j / n}
            for app, (n, t, j) in stats.items()}


def by_app_strict(cases: list, apps: List[str]) -> Dict[str, Dict[str, float]]:
    """按应用分组后用同口径指标算：Op.F1 micro 与 Step SR ≤0.10。"""
    groups: Dict[str, list] = {}
    for c in cases:
        groups.setdefault(apps[c["i"]], []).append(c)
    return {app: {"n": len(g), "micro_f1": _E.op_f1(g)["micro_f1"],
                  "step_sr": _E.step_success(g, 0.10)}
            for app, g in groups.items()}


def overall(cases: list, sessions: List[str]) -> Dict[str, float]:
    """步级的类型准确率、类型对且点准、无法执行率，加上按 session 算的离线任务成功率。"""
    n = len(cases)
    done: Dict[str, bool] = {}
    for c in cases:
        sid = sessions[c["i"]]
        done[sid] = done.get(sid, True) and _joint_ok(c)
    return {
        "n": n,
        "type_accuracy": _E.op_f1(cases)["micro_f1"],
        "joint_accuracy": _E.step_success(cases, 0.10),
        "unexecutable_rate": sum(c["pred"] == PARSE_FAILED for c in cases) / n,
        "sessions": len(done),
        "sessions_done": sum(done.values()),
        "session_success": sum(done.values()) / len(done),
    }


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
    sessions = sessions_of_test_steps()
    counts = {a: apps.count(a) for a in APPS}
    results = {}
    for label, names in CONFIGS:
        name, d = _load(names)
        if d:
            results[label] = {"log": name,
                              "overall": {**overall(d["cases"], sessions), "sec_per_step": d.get("avg_latency_s")},
                              "by_app": by_app_strict(d["cases"], apps)}

    if not results:
        raise SystemExit("没有带逐条记录的 ScreenAgent 评测日志")

    labels = list(results)
    print("| 配置 | Op.F1 micro | Step SR ≤0.10 | 无法执行 | 离线任务成功（session 每步都对） | 单步耗时 |")
    print("|---|---|---|---|---|---|")
    for label in labels:
        o = results[label]["overall"]
        sec = f"{o['sec_per_step']:.2f} s" if o.get("sec_per_step") else "—"
        print(f"| {label} | {o['type_accuracy']:.1%} | {o['joint_accuracy']:.1%} | {o['unexecutable_rate']:.1%} "
              f"| {o['sessions_done']}/{o['sessions']} = {o['session_success']:.1%} | {sec} |")

    print("\n| 应用 | 步数 | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for app in APPS:
        cells = []
        for label in labels:
            s = results[label]["by_app"].get(app)
            cells.append("—" if not s else f"{s['micro_f1']:.0%} / {s['step_sr']:.0%}")
        print(f"| {app} | {counts[app]} | " + " | ".join(cells) + " |")
    print("\n格子里是「Op.F1 micro / Step SR ≤0.10」")

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
    # 图只画两段式（交付的执行路径），一段式的数在表里：六组柱子挤在一张图上读不清
    series = [(label, [results[label]["by_app"].get(app, {}).get("micro_f1") for app in APPS])
              for label in labels]
    charts._bars(plt, [f"{a}\n({counts[a]} 步)" for a in APPS], series,
                 "ScreenAgent test 353 步：按应用类型的 Op.F1 micro（两段式）", "Op.F1 micro",
                 charts.OUT / "screenagent_by_app.png")
    print(f"图已存到 {charts.OUT / 'screenagent_by_app.png'}")


if __name__ == "__main__":
    main()
