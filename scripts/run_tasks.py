"""跑基础任务集，统计成功率。

对应大纲第 4 周第 4 项。每个任务跑完后用程序化验收条件判断成败，不靠人工看。

默认 dry-run，只打印每个任务模型想做什么，不动鼠标键盘。dry-run 下验收必然不通过，
因为动作没真的执行，这一轮是用来确认动作是否合理的。确认后加 --live。

用法：
    python scripts/run_tasks.py                    # dry-run
    python scripts/run_tasks.py --live             # 真实执行
    python scripts/run_tasks.py --live --only open_file
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.agent import Agent
from gui_agent.control import Controller, PyAutoGUIBackend
from gui_agent.models import DEFAULT_MODEL, LocalQwenVL
from gui_agent.perception import Perception
from gui_agent.tasks import basic_tasks

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="真的操作桌面，默认只打印")
    ap.add_argument("--only", default=None, help="只跑指定 id 的任务")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=1, help="每个任务重复跑几次")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="v1.0")
    args = ap.parse_args()

    tasks = [t for t in basic_tasks() if args.only in (None, t.id)]
    if not tasks:
        raise SystemExit(f"没有 id 为 {args.only} 的任务")

    if args.live:
        print("将真实操作桌面，5 秒后开始。鼠标甩到屏幕左上角可强制中断。\n")
        time.sleep(5)
    else:
        print("dry-run：只打印动作，验收必然不通过。确认动作合理后加 --live。\n")

    print(f"加载模型 {args.model} ……")
    vlm = LocalQwenVL(args.model)
    perception = Perception()
    controller = Controller(backend=PyAutoGUIBackend(), dry_run=not args.live)
    agent = Agent(perception, controller, vlm, max_steps=args.max_steps)

    records, n_ok = [], 0
    for task in tasks:
        for run in range(args.repeat):
            label = f"{task.id}" + (f" #{run + 1}" if args.repeat > 1 else "")
            print(f"\n── {label}：{task.instruction}")
            try:
                task.setup()
            except Exception as e:
                print(f"   setup 失败：{e}")
                continue

            traj = agent.run(task.instruction, task_id=task.id)
            for i, s in enumerate(traj.steps, 1):
                print(f"   {i:>2}. {s.action.type:<12} {'ok' if s.ok else s.error}")

            passed = False
            try:
                passed = bool(task.check())
            except Exception as e:
                print(f"   验收函数出错：{e}")
            try:
                task.teardown()
            except Exception as e:
                print(f"   teardown 失败：{e}")

            n_ok += passed
            print(f"   验收：{'通过' if passed else '不通过'}   {traj.n_steps} 步 "
                  f"{traj.wall_time:.1f}s")
            records.append({
                "task": task.id, "run": run + 1, "passed": passed,
                "steps": traj.n_steps, "wall_time": round(traj.wall_time, 2),
                "actions": [s.action.type for s in traj.steps],
            })

    perception.close()
    total = len(records)
    print(f"\n{'='*46}\n成功率 {n_ok}/{total} = {n_ok / total:.0%}" if total else "没有跑成任何任务")
    if total:
        print(f"平均步数 {sum(r['steps'] for r in records) / total:.1f}，"
              f"平均耗时 {sum(r['wall_time'] for r in records) / total:.1f}s")

    out = ROOT / "logs" / f"tasks_{args.tag}{'' if args.live else '_dryrun'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"live": args.live, "model": args.model,
         "success_rate": n_ok / total if total else 0, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已存到 {out}")


if __name__ == "__main__":
    main()
