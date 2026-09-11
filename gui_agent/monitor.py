"""执行状态的实时监控与日志记录。

对应大纲第 6 周第 4 项。

原来轨迹只在任务跑完时落盘，中途崩掉就什么都看不到——第 2 周排查「规划没生效」
时就是因为轨迹里只有动作名、没有每步的细节，花了额外一轮才定位到原因。

这里做两件事：
1. 每一步立刻追加写进 JSONL，任务中途中断也留得下前面的记录
2. 把关键状态打到控制台，跑 live 时能看见 Agent 正在做什么

日志按行写，一行一个事件，跑到一半打开也能看。事件类型：
  plan       拆出的子任务清单
  step       一步执行完的结果
  retry      这一步失败了，准备重来
  reflect    子任务状态判定
  finish     任务结束
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .schema import Step, Trajectory


def describe_action(action) -> str:
    """把动作写成一句人能读的话。"""
    a = action
    if a.point is not None:
        pos = f"({a.point[0]:.3f}, {a.point[1]:.3f})"
    else:
        pos = ""
    if a.type in ("click", "left_double", "right_single"):
        return f"{a.type} {pos}"
    if a.type == "drag":
        return f"drag {pos} -> ({a.point2[0]:.3f}, {a.point2[1]:.3f})"
    if a.type == "scroll":
        return f"scroll {pos} {a.direction}"
    if a.type in ("type", "hotkey"):
        return f"{a.type} {a.text!r}"
    return a.type


class Monitor:
    """一条轨迹的实时记录。

    log_path 为空时只打印不落盘；quiet=True 时只落盘不打印。两个都关就什么
    都不做，Agent 里不需要为此加分支。
    """

    def __init__(self, log_path: Optional[str] = None, quiet: bool = False) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.quiet = quiet
        self.started = time.perf_counter()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # 覆盖式打开一次，之后都是追加，避免上一次运行的记录混进来
            self.log_path.write_text("", encoding="utf-8")

    # -- 内部 ---------------------------------------------------------------

    def _emit(self, event: str, line: str, **fields) -> None:
        if not self.quiet:
            print(f"  [{time.perf_counter() - self.started:6.1f}s] {line}", flush=True)
        if self.log_path:
            rec = {"t": round(time.perf_counter() - self.started, 3), "event": event, **fields}
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- 事件 ---------------------------------------------------------------

    def task_start(self, instruction: str, task_id: str = "") -> None:
        self._emit("start", f"任务：{instruction}", instruction=instruction, task_id=task_id)

    def plan(self, subtasks: list) -> None:
        if subtasks:
            self._emit("plan", f"拆成 {len(subtasks)} 个子任务：{' → '.join(subtasks)}",
                       subtasks=list(subtasks))
        else:
            self._emit("plan", "没拆出子任务，退回单步循环", subtasks=[])

    def step(self, n: int, step: Step, subtask: Optional[str] = None) -> None:
        if not step.ok:
            outcome = f"失败：{step.error}"
        elif step.changed is False:
            outcome = "界面无变化"
        elif step.changed is True:
            outcome = "界面已变化"
        else:
            outcome = "ok"
        tail = f"  [{subtask}]" if subtask else ""
        self._emit("step", f"{n:>2}. {describe_action(step.action):<34}{outcome}{tail}",
                   n=n, action=step.action.to_dict(), ok=step.ok,
                   error=step.error, changed=step.changed,
                   elapsed=round(step.elapsed, 3), subtask=subtask)

    def retry(self, n: int, reason: str) -> None:
        self._emit("retry", f"    第 {n} 次失败，重试：{reason}", n=n, reason=reason)

    def reflect(self, situation: str, subtask: Optional[str]) -> None:
        self._emit("reflect", f"    子任务判定：{situation}",
                   situation=situation, subtask=subtask)

    def finish(self, traj: Trajectory) -> None:
        mark = {True: "完成", False: "未完成", None: "未判定"}[traj.success]
        self._emit("finish", f"{mark}，{traj.n_steps} 步 {traj.wall_time:.1f}s",
                   success=traj.success, n_steps=traj.n_steps,
                   wall_time=round(traj.wall_time, 2),
                   changed_steps=sum(1 for s in traj.steps if s.changed),
                   unchanged_steps=sum(1 for s in traj.steps if s.changed is False),
                   retries=traj.retries)


def read_log(path: str) -> list:
    """读回一份运行日志。中途中断的文件也能读，坏行直接跳过。"""
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 进程被杀时最后一行可能只写了一半
    return out
