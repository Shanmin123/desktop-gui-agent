"""汇总 run_tasks.py 的任务日志：成功率、平均执行时间、错误率，按难度档和按任务拆开。

对应大纲第 7 周第 2 项「从成功率、平均执行时间、错误率三方面定量评估」，以及第 3 项里
不同分辨率的 live 部分。一个日志是一组配置，模型、分辨率、开关都记在日志头里；标签取文件名
（虚拟机里走 API 调宿主机的模型服务，日志头里没有适配器，靠 tag 区分）。

指标
  成功率        验收通过的次数 / 运行次数，附 Wilson 95% 区间。25 个任务各跑 3 次也只有 75 次，
              两组配置差几个点时要看区间是否重叠
  平均执行时间  每次运行的墙钟时间；另给验收通过的那些单独算一个
  错误率        出错的步数 / 总步数。出错指这一步没拿到可执行的动作或执行失败（解析失败、定位
              不到、执行异常、注入的故障），后来重试成功的失败步也算
  异常终止      setup 失败或运行时抛异常、没有轨迹的运行；算进成功率的分母，不算进耗时

用法：
    python scripts/summarize_suite.py              # logs/ 下 live 跑的 tasks_*suite*.json
    python scripts/summarize_suite.py logs/tasks_vm_q35_2sp_suite.json logs/tasks_vm_q35_base_suite.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LEVELS = ("T1", "T2", "T3")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """二项比例的 Wilson 区间。样本少、比例靠近 0 或 1 时比正态近似可靠。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _steps(record: dict) -> list:
    return (record.get("trajectory") or {}).get("steps") or []


def summarize(records: List[dict]) -> Dict:
    ran = [r for r in records if r.get("trajectory")]  # 真正跑起来、留下轨迹的
    steps = [s for r in ran for s in _steps(r)]
    errors = sum(1 for s in steps if not s.get("ok", True))
    passed = sum(bool(r.get("passed")) for r in records)
    return {
        "runs": len(records),
        "passed": passed,
        "success_rate": passed / len(records) if records else None,
        "ci95": list(wilson(passed, len(records))),
        "avg_wall_time": _mean([r["wall_time"] for r in ran]),
        "avg_wall_time_passed": _mean([r["wall_time"] for r in ran if r.get("passed")]),
        "avg_steps": _mean([len(_steps(r)) for r in ran]),
        "steps": len(steps),
        "error_steps": errors,
        "error_rate": errors / len(steps) if steps else None,
        "retries": sum(r.get("retries", 0) for r in ran),
        "aborted": len(records) - len(ran),
    }


def by_level(records: List[dict]) -> Dict[str, Dict]:
    out = {}
    for level in LEVELS:
        group = [r for r in records if r.get("level") == level]
        if group:
            out[level] = summarize(group)
    return out


def by_task(records: List[dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in records:
        t = out.setdefault(r["task"], {"level": r.get("level", ""), "runs": 0, "passed": 0})
        t["runs"] += 1
        t["passed"] += bool(r.get("passed"))
    return out


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1%}"


def _sec(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1f} s"


def _rate(s: Dict) -> str:
    if not s["runs"]:
        return "—"
    lo, hi = s["ci95"]
    return f"{s['passed']}/{s['runs']} = {s['success_rate']:.0%}（{lo:.0%}~{hi:.0%}）"


def main(argv: Optional[List[str]] = None) -> Dict:
    args = sys.argv[1:] if argv is None else argv
    paths = [Path(p) for p in args] or sorted(LOGS.glob("tasks_*suite*.json"))
    configs = {}
    for path in paths:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"跳过 {path.name}：读不了（{e}）")
            continue
        if not d.get("live"):
            print(f"跳过 {path.name}：dry-run，动作没有真的执行")
            continue
        records = d.get("records") or []
        if not records:
            continue
        configs[path.stem.replace("tasks_", "", 1)] = {
            "log": path.name, "model": d.get("model"), "adapter": d.get("adapter"),
            "resolution": d.get("resolution"), "plan": d.get("plan"),
            "inject_failures": d.get("inject_failures"), "retry_limit": d.get("retry_limit"),
            "overall": summarize(records), "by_level": by_level(records), "by_task": by_task(records),
        }
    if not configs:
        print("没有 live 跑出来的任务日志")
        return {}

    names = list(configs)
    print("| 配置 | 运行 | 成功率（95% 区间） | 平均耗时 | 通过的平均耗时 | 平均步数 | 错误率 | 异常终止 |")
    print("|---|---|---|---|---|---|---|---|")
    for name in names:
        o = configs[name]["overall"]
        steps = "—" if o["avg_steps"] is None else f"{o['avg_steps']:.1f}"
        print(f"| {name} | {o['runs']} | {_rate(o)} | {_sec(o['avg_wall_time'])} "
              f"| {_sec(o['avg_wall_time_passed'])} | {steps} "
              f"| {_pct(o['error_rate'])}（{o['error_steps']}/{o['steps']}） | {o['aborted']} |")

    print("\n| 配置 | " + " | ".join(LEVELS) + " |")
    print("|---|" + "---|" * len(LEVELS))
    for name in names:
        cells = [_rate(s) if (s := configs[name]["by_level"].get(level)) else "—" for level in LEVELS]
        print(f"| {name} | " + " | ".join(cells) + " |")

    tasks = list(dict.fromkeys(t for name in names for t in configs[name]["by_task"]))
    print("\n| 任务 | 档 | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for t in tasks:
        level = next((configs[n]["by_task"][t]["level"] for n in names if t in configs[n]["by_task"]), "")
        cells = [f"{c['passed']}/{c['runs']}" if (c := configs[n]["by_task"].get(t)) else "—" for n in names]
        print(f"| `{t}` | {level} | " + " | ".join(cells) + " |")

    out = LOGS / "suite_summary.json"
    out.write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {out}")
    return configs


if __name__ == "__main__":
    main()
