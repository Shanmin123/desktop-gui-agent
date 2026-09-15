"""第 6、7 周要在虚拟机里跑的几批 live 实验，一条命令按顺序跑完。

每批调用一次 host_jobs.py：起对应的模型服务 → 恢复快照开机 → 投任务 → 等回执 → 拷回日志，
批与批之间都恢复快照，起始状态一致。显卡队列的暂停文件从第一批挂到最后一批：中间要是放开，
消融队列会接着开一个几小时的训练，下一批就等不到显卡。

用法（宿主机）：
    python scripts/vm/run_batches.py --list                     # 只列出批次和命令
    python scripts/vm/run_batches.py                            # 全部按顺序跑
    python scripts/vm/run_batches.py --only q35_2sp_suite,q35_base_suite

跑完：python scripts/summarize_suite.py 出表，python scripts/make_charts.py 出图。
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
HOST_JOBS = Path(__file__).with_name("host_jobs.py")

_spec = importlib.util.spec_from_file_location("host_jobs", HOST_JOBS)
host_jobs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(host_jobs)

Q35, Q25 = "Qwen/Qwen3.5-4B", "Qwen/Qwen2.5-VL-3B-Instruct"
API = ["--api-base", "http://10.0.2.2:8000/v1", "--api-key", "local"]
GPU_WAIT = 12 * 3600  # 显卡上正在训练时等它这一步跑完；不想等就先停掉消融队列


def _batch(batch_id: str, what: str, serve_model: str, serve_adapter: Optional[str],
           coord_space: str, *args: str, tag: str) -> Dict:
    return {"id": batch_id, "what": what, "serve_model": serve_model, "serve_adapter": serve_adapter,
            "args": ["--live", *args, "--model", serve_model, *API, "--api-coord-space", coord_space,
                     "--tag", tag]}


# 日志名是 tasks_<tag><集合后缀>.json：suite 加 _suite，complex 加 _complex，basic 不加
BATCHES = [
    _batch("q35_2sp_suite", "第 7 周第 2 项：25 个任务评测集，Qwen3.5 微调", Q35, "checkpoints/q35_2sp",
           "rel1000", "--set", "suite", "--repeat", "3", "--resolution", "1280x720", tag="vm_q35_2sp"),
    _batch("q35_base_suite", "第 7 周第 2 项：25 个任务评测集，Qwen3.5 基座（微调前后对比）", Q35, None,
           "rel1000", "--set", "suite", "--repeat", "3", "--resolution", "1280x720", tag="vm_q35_base"),
    _batch("q35_robust_noretry", "第 6 周第 2 项：30% 故障注入，关重试", Q35, "checkpoints/q35_2sp",
           "rel1000", "--set", "basic", "--repeat", "3", "--resolution", "1280x720",
           "--inject-failures", "0.3", "--retry-limit", "0", tag="vm_q35_robust_noretry"),
    _batch("q35_robust_retry", "第 6 周第 2 项：30% 故障注入，开重试（额度 2）", Q35, "checkpoints/q35_2sp",
           "rel1000", "--set", "basic", "--repeat", "3", "--resolution", "1280x720",
           "--inject-failures", "0.3", "--retry-limit", "2", tag="vm_q35_robust_retry"),
    _batch("q35_2sp_complex_plan", "第 6 周第 1 项：4 个复杂任务，先拆解再执行", Q35, "checkpoints/q35_2sp",
           "rel1000", "--set", "complex", "--plan", "--repeat", "3", "--resolution", "1280x720",
           tag="vm_q35_2sp_plan"),
    _batch("q35_2sp_suite_1080p", "第 7 周第 3 项：评测集换到 1920×1080", Q35, "checkpoints/q35_2sp",
           "rel1000", "--set", "suite", "--resolution", "1920x1080", tag="vm_q35_2sp_1080p"),
    _batch("q25_2sp_suite", "对照：25 个任务评测集，Qwen2.5 微调", Q25, "checkpoints/lora_2sp",
           "pixel", "--set", "suite", "--repeat", "3", "--resolution", "1280x720", tag="vm_q25_2sp"),
]


def command(batch: Dict, vm: str, snapshot: str, python: str = sys.executable) -> List[str]:
    cmd = [python, str(HOST_JOBS), "--vm", vm, "--snapshot", snapshot, "--no-pause",
           "--serve-model", batch["serve_model"], "--id", batch["id"]]
    if batch["serve_adapter"]:
        cmd += ["--serve-adapter", batch["serve_adapter"]]
    return cmd + ["--"] + batch["args"]


def run_all(batches: List[Dict], vm: str, snapshot: str, run: Callable = subprocess.run,
            pause: Optional[Callable] = None) -> List[Dict]:
    pause = pause or (lambda: host_jobs.paused_gpu_queue(timeout=GPU_WAIT))
    results = []
    with pause():
        for b in batches:
            print(f"\n=== {b['id']}：{b['what']}", flush=True)
            rc = run(command(b, vm, snapshot), cwd=str(ROOT)).returncode
            results.append({"id": b["id"], "exit_code": rc})
            print(f"=== {b['id']} 结束 exit={rc}", flush=True)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", default="agent-win11")
    ap.add_argument("--snapshot", default="clean")
    ap.add_argument("--only", default=None, help="逗号分隔的批次 id，不给就全跑")
    ap.add_argument("--list", action="store_true", help="只列出批次和命令，不跑")
    args = ap.parse_args(argv)

    batches = BATCHES
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = sorted(set(wanted) - {b["id"] for b in BATCHES})
        if unknown:
            raise SystemExit(f"没有这些批次：{unknown}，可选 {[b['id'] for b in BATCHES]}")
        batches = [b for b in BATCHES if b["id"] in wanted]
    for b in batches:
        host_jobs.worker.validate_args(b["args"])

    if args.list:
        for b in batches:
            print(f"{b['id']}：{b['what']}\n    " + " ".join(command(b, args.vm, args.snapshot)))
        return 0

    results = run_all(batches, args.vm, args.snapshot)
    failed = [r["id"] for r in results if r["exit_code"] != 0]
    print("\n全部结束" + (f"，失败的批次：{failed}" if failed else "，每批都正常退出"))
    print("出表：python scripts/summarize_suite.py；出图：python scripts/make_charts.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
