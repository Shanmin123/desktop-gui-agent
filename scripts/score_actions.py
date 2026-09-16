"""把动作评测日志按 GUI 智能体论文常用的口径汇总成一张表。

指标定义在 `scripts/eval_screenagent.py`：
  Op.F1     动作类型的 F1，macro 各类型同权、micro 等于类型准确率；键盘动作要内容也对
            （Mind2Web 对 TYPE 就是连输入内容一起比）
  Step SR   这一步整体做对的比例：类型对，坐标类动作的距离还要在阈值内。0.14 是 AITW 的口径，
            0.10 是更严的一档
  无法执行   模型没给出可执行动作的比例（解析失败、定位不到）

老日志里没有存输入内容，键盘动作那一项只比类型，会比重跑出来的略高，表里会标出来。

用法：
    python scripts/score_actions.py                      # logs/ 下所有 screenagent_*.json
    python scripts/score_actions.py logs/screenagent_q35_hist_2s.json
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
_spec = importlib.util.spec_from_file_location("eval_screenagent", ROOT / "scripts" / "eval_screenagent.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def score(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    cases = d.get("cases") or []
    if not cases:
        return {}
    f1 = E.op_f1(cases)
    has_text = any("pred_text" in c for c in cases)
    return {
        "name": path.stem.replace("screenagent_", ""),
        "model": "Qwen3.5-4B" if "q35" in path.stem or "qwen3" in str(d.get("model", "")).lower()
                 else str(d.get("model", "")).split("/")[-1],
        "adapter": (d.get("adapter") or "").replace("checkpoints/", "") or "基座",
        "two_stage": bool(d.get("locate_target")),
        "n": len(cases),
        "op_f1_macro": f1["macro_f1"],
        "op_f1_micro": f1["micro_f1"],
        "step_sr_010": E.step_success(cases, 0.10),
        "step_sr_014": E.step_success(cases, 0.14),
        "unexecutable": sum(1 for c in cases if c["pred"] == "解析失败") / len(cases),
        "latency": d.get("avg_latency_s"),
        "text_compared": has_text,
        "per_type": {t: v["f1"] for t, v in f1["per_type"].items() if v["gt"]},
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(LOGS.glob("screenagent_*.json"))
    rows = [r for r in (score(p) for p in paths) if r]
    if not rows:
        raise SystemExit("没有可汇总的动作评测日志")
    rows.sort(key=lambda r: -r["step_sr_010"])

    print("| 配置 | 路径 | 条数 | Op.F1 macro | Op.F1 micro | Step SR ≤0.10 | Step SR ≤0.14 | 无法执行 | 每步耗时 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        star = "" if r["text_compared"] else " *"
        lat = "—" if r["latency"] is None else f"{r['latency']:.2f} s"
        print(f"| {r['name']}{star} | {'两段式' if r['two_stage'] else '一段式'} | {r['n']} "
              f"| {r['op_f1_macro']:.1%} | {r['op_f1_micro']:.1%} | {r['step_sr_010']:.1%} "
              f"| {r['step_sr_014']:.1%} | {r['unexecutable']:.1%} | {lat} |")
    if any(not r["text_compared"] for r in rows):
        print("\n\* 这些日志没有存输入内容，键盘动作只比了类型，Op.F1 与 Step SR 略偏高")

    print("\n按动作类型的 F1（只列真值里出现过的）：")
    types = sorted({t for r in rows for t in r["per_type"]})
    print("| 配置 | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for r in rows:
        print(f"| {r['name']} | " + " | ".join(
            f"{r['per_type'][t]:.2f}" if t in r["per_type"] else "—" for t in types) + " |")


if __name__ == "__main__":
    main()
