"""跑基础任务集，统计成功率。

对应大纲第 4 周第 4 项。每个任务跑完后用程序化验收条件判断成败，不靠人工看。

默认 dry-run，模型给的动作只打印不执行，不动鼠标键盘。dry-run 下验收必然不通过，
因为动作没真的执行，这一轮是用来确认动作是否合理的。确认后加 --live。

注意 dry-run 只管住模型的动作，任务自身的准备和收尾照常执行：会在 logs/scratch/
下写文件，也会开关记事本（只开关脚本自己启动的那些）。

用法：
    python scripts/run_tasks.py                    # dry-run
    python scripts/run_tasks.py --live             # 真实执行
    python scripts/run_tasks.py --live --only open_file
    python scripts/run_tasks.py --live --set complex --plan   # 多步任务，先拆解
    python scripts/run_tasks.py --live --one-stage             # 一段式对照（提示词带 OCR 元素清单）

默认两段式：先问点什么、再问点哪，执行和拆解都不跑 OCR。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlib import nullcontext

from gui_agent.agent import Agent
from gui_agent.control import Controller, PyAutoGUIBackend
from gui_agent.monitor import Monitor
from gui_agent.display import resolution as use_resolution
from gui_agent.models import DEFAULT_MODEL, FlakyVLM, add_backend_args, load_vlm
from gui_agent.perception import Perception
from gui_agent.suite import suite_tasks
from gui_agent.tasks import basic_tasks, complex_tasks

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="真的操作桌面，默认只打印")
    ap.add_argument("--only", default=None, help="只跑指定 id 的任务")
    ap.add_argument("--set", default="basic", choices=["basic", "complex", "all", "suite"],
                    help="basic 是第 4 周的 5 个基础任务，complex 是第 6 周要拆解的多步任务，"
                         "suite 是第 7 周的 25 个任务评测集（分 T1/T2/T3 三档）")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=1, help="每个任务重复跑几次")
    add_backend_args(ap)
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--no-detect-change", action="store_true",
                    help="关掉执行后的屏幕变化检测")
    ap.add_argument("--inject-failures", type=float, default=0.0, metavar="RATE",
                    help="按比例把模型输出换成垃圾，测容错，如 0.3")
    ap.add_argument("--retry-limit", type=int, default=None,
                    help="单步连续失败几次还重试，0 表示一失败就放弃")
    ap.add_argument("--cache-ocr", action="store_true",
                    help="屏幕没变就复用上一次的 OCR 结果")
    ap.add_argument("--cv-elements", action="store_true",
                    help="额外用 OpenCV 找图标候选框，让没文字的控件也有编号")
    ap.add_argument("--locate-target", action="store_true", default=True,
                    help="两段式定位（默认，留着这个开关是为了旧命令还能跑）：先让模型说要操作哪个"
                         "控件，再用定位提示词解析坐标；执行和拆解都不跑 OCR")
    ap.add_argument("--one-stage", action="store_true",
                    help="一段式：提示词带 OCR 元素清单，模型回编号或坐标。只用来复现第 2、3 周的对照")
    ap.add_argument("--plan", action="store_true",
                    help="先把任务拆成子任务再逐个执行")
    ap.add_argument("--shots", action="store_true",
                    help="每步存一张截图，轨迹要当第 3 周的微调样本时打开")
    ap.add_argument("--resolution", default=None, metavar="WxH",
                    help="临时切到指定分辨率跑，结束后还原，如 1280x720")
    args = ap.parse_args()
    args.locate_target = not args.one_stage

    pool = {"basic": basic_tasks, "complex": complex_tasks,
            "all": lambda: basic_tasks() + complex_tasks(),
            "suite": suite_tasks}[args.set]()
    tasks = [t for t in pool if args.only in (None, t.id)]
    if not tasks:
        raise SystemExit(f"没有 id 为 {args.only} 的任务")

    if args.live:
        print("将真实操作桌面，5 秒后开始。鼠标甩到屏幕左上角可强制中断。\n")
        time.sleep(5)
    else:
        print("dry-run：只打印动作，验收必然不通过。确认动作合理后加 --live。\n")

    screen_ctx = nullcontext()
    if args.resolution:
        try:
            rw, rh = (int(v) for v in args.resolution.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--resolution 要写成 1280x720 的形式，收到 {args.resolution!r}")
        screen_ctx = use_resolution(rw, rh)

    print(f"加载模型 {args.model} ……")
    vlm = load_vlm(args)
    if args.inject_failures:
        vlm = FlakyVLM(vlm, rate=args.inject_failures)
        print(f"故障注入开启：{args.inject_failures:.0%} 的模型输出会被换成不可解析的文本")
    if (args.cv_elements or args.cache_ocr) and args.locate_target:
        # 两段式的动作提示词和拆解提示词都不带元素清单，整条任务不跑 OCR，这两个开关不起作用。
        # 不提示的话是静默失效的——第 3 周的实验就这么白开了一轮。
        print("注意：两段式不跑 OCR，--cv-elements / --cache-ocr 不起作用；要用就加 --one-stage。\n")
    perception = Perception(cache_ocr=args.cache_ocr,
                            cv_elements=args.cv_elements)
    controller = Controller(backend=PyAutoGUIBackend(), dry_run=not args.live)
    shot_dir = str(ROOT / "logs" / f"shots_{args.tag}") if args.shots else None
    agent = Agent(perception, controller, vlm, max_steps=args.max_steps,
                  shot_dir=shot_dir, plan=args.plan, locate_target=args.locate_target,
                  detect_change=not args.no_detect_change,
                  monitor=Monitor(str(ROOT / "logs" / f"run_{args.tag}.jsonl")),
                  **({} if args.retry_limit is None else {"retry_limit": args.retry_limit}))

    records, n_ok = [], 0
    with screen_ctx as actual:
        if args.resolution:
            print(f"分辨率切到 {args.resolution}，实际截图 {actual}，跑完还原\n")
        for task in tasks:
            for run in range(args.repeat):
                label = f"{task.id}" + (f" #{run + 1}" if args.repeat > 1 else "")
                print(f"\n── {label}：{task.instruction}")
                def record_failure(reason: str) -> None:
                    """跑不起来也要留一条记录，否则它从分母里消失，成功率会虚高。"""
                    print(f"   {reason}")
                    records.append({
                        "task": task.id, "level": task.level, "run": run + 1, "passed": False,
                        "steps": 0, "wall_time": 0.0, "actions": [], "error": reason,
                    })

                try:
                    baseline = task.setup() or {}
                except Exception as e:
                    record_failure(f"setup 失败：{type(e).__name__}: {e}")
                    continue

                try:
                    traj = agent.run(task.instruction, task_id=task.id)
                except Exception as e:  # 一个任务崩了不该带走整批的结果
                    record_failure(f"本次运行异常：{type(e).__name__}: {e}")
                    try:
                        task.teardown()
                    except Exception as te:
                        print(f"   teardown 失败：{te}")
                    continue
                for i, s in enumerate(traj.steps, 1):
                    print(f"   {i:>2}. {s.action.type:<12} {'ok' if s.ok else s.error}")

                passed = False
                try:
                    passed = bool(task.check(baseline))
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
                    "task": task.id, "level": task.level, "run": run + 1, "passed": passed,
                    "steps": traj.n_steps, "wall_time": round(traj.wall_time, 2),
                    "actions": [s.action.type for s in traj.steps],
                    "retries": traj.retries,
                    # 存完整轨迹，失败归因要看模型当时怎么想的
                    "trajectory": json.loads(traj.to_json()),
                })

    perception.close()
    total = len(records)
    print(f"\n{'='*46}\n成功率 {n_ok}/{total} = {n_ok / total:.0%}" if total else "没有跑成任何任务")
    if total:
        print(f"平均步数 {sum(r['steps'] for r in records) / total:.1f}，"
              f"平均耗时 {sum(r['wall_time'] for r in records) / total:.1f}s，"
              f"重试 {sum(r.get('retries', 0) for r in records)} 次")
        if args.inject_failures:
            print(f"实际注入 {vlm.injected}/{vlm.calls} 次")

    suffix = "" if args.set == "basic" else f"_{args.set}"
    out = ROOT / "logs" / f"tasks_{args.tag}{suffix}{'' if args.live else '_dryrun'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        # 把这一轮的配置一并记下来：之前几轮只记了 live/model，事后对不上是哪套开关
        # 跑出来的数，只能翻命令行历史。
        {"live": args.live, "model": args.model,
         "adapter": getattr(args, "adapter", None),
         "locate_target": args.locate_target, "plan": args.plan,
         "cache_ocr": args.cache_ocr, "cv_elements": args.cv_elements,
         "resolution": args.resolution, "max_steps": args.max_steps,
         "detect_change": not args.no_detect_change,
         "inject_failures": args.inject_failures,
         "retry_limit": args.retry_limit,
         "success_rate": n_ok / total if total else 0, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已存到 {out}")


if __name__ == "__main__":
    main()
