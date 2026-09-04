"""命令行入口：给一句话，让智能体去做。

对应大纲第 4 周第 1、2、3 项：感知、控制、Agent 三方集成，打通
「用户指令 → 屏幕感知 → 任务规划 → 动作执行 → 结果反馈」的完整流程。

默认 dry-run，只打印每一步要做什么，不动鼠标键盘。确认动作合理后加 --live 真执行。

用法：
    python scripts/run_agent.py "打开计算器"
    python scripts/run_agent.py "打开计算器" --live
    python scripts/run_agent.py "在记事本里输入你好" --live --max-steps 8
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.agent import Agent
from gui_agent.control import Controller, PyAutoGUIBackend
from gui_agent.models import DEFAULT_MODEL, LocalQwenVL
from gui_agent.perception import Perception

ROOT = Path(__file__).resolve().parents[1]


def describe(action) -> str:
    """把动作写成一句人能读的话。"""
    t = action.type
    if t in ("click", "left_double", "right_single"):
        return f"{t} 于 ({action.point[0]:.3f}, {action.point[1]:.3f})"
    if t == "drag":
        return f"drag {action.point} → {action.point2}"
    if t == "scroll":
        return f"scroll {action.direction} 于 ({action.point[0]:.3f}, {action.point[1]:.3f})"
    if t in ("type", "hotkey"):
        return f"{t} {action.text!r}"
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", help="要完成的任务，用一句话描述")
    ap.add_argument("--live", action="store_true", help="真的操作桌面，默认只打印")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--delay", type=float, default=3.0, help="--live 时开始前的等待秒数")
    args = ap.parse_args()

    if args.live:
        print(f"将真实操作桌面，{args.delay} 秒后开始。鼠标甩到屏幕左上角可强制中断。")
        time.sleep(args.delay)
    else:
        print("dry-run：只打印动作，不操作桌面。确认无误后加 --live。")

    print(f"加载模型 {args.model} ……")
    t0 = time.perf_counter()
    vlm = LocalQwenVL(args.model, load_in_4bit=args.load_in_4bit)
    print(f"  耗时 {time.perf_counter() - t0:.1f}s\n")

    perception = Perception()
    controller = Controller(backend=PyAutoGUIBackend(), dry_run=not args.live)
    agent = Agent(perception, controller, vlm, max_steps=args.max_steps)

    print(f"任务：{args.instruction}\n")
    traj = agent.run(args.instruction)
    perception.close()

    for i, s in enumerate(traj.steps, 1):
        mark = "ok  " if s.ok else "失败"
        print(f"  {i:>2}. [{mark}] {describe(s.action):<40} {s.elapsed:.1f}s")
        if s.action.thought:
            print(f"       想法：{s.action.thought}")
        if not s.ok:
            print(f"       原因：{s.error}")

    print(f"\n共 {traj.n_steps} 步，耗时 {traj.wall_time:.1f}s，结果："
          f"{'完成' if traj.success else '未完成'}")

    out = ROOT / "logs" / f"traj_{int(traj.started_at)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(traj.to_json(), encoding="utf-8")
    print(f"执行记录已存到 {out}")


if __name__ == "__main__":
    main()
