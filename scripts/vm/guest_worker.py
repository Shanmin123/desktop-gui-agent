"""在虚拟机里常驻，从共享文件夹领任务、跑 run_tasks.py、把结果放回去。

宿主机不需要虚拟机的账号密码：宿主机往交换目录的 jobs/ 里写一个任务文件，这里领走、执行，
写 done/ 回执，并把这次新产生的 tasks_*.json、run_*.jsonl 拷到 results/<任务 id>/。

交换目录只放任务和结果：智能体在虚拟机里能看到这个文件夹，所以代码仓库要拷到虚拟机本地
盘上，不共享宿主机的项目目录。只接受 run_tasks.py 的已知参数，不执行任意命令。

目录约定（交换目录下）：
    jobs/<id>.json      {"id": ..., "args": ["--live", "--set", "suite", ...]}
    running/<id>.json   领走时写
    done/<id>.json      {"id", "exit_code", "seconds", "copied", "error"}
    out/<id>.txt        run_tasks.py 的输出
    results/<id>/       这次新产生的日志
    worker_alive.json   每轮刷新，宿主机据此判断 worker 在不在

用法（虚拟机里）：
    python scripts\\vm\\guest_worker.py --exchange Z:\\ --repo C:\\agent\\desktop-gui-agent
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

ALLOWED_FLAGS = {
    "--live", "--only", "--set", "--max-steps", "--repeat", "--model", "--api-base", "--api-key",
    "--api-qwen", "--api-coord-space", "--tag", "--no-detect-change", "--inject-failures",
    "--retry-limit", "--cache-ocr", "--cv-elements", "--locate-target", "--plan", "--shots",
    "--resolution",
}
VALUE = re.compile(r"^[\w.:/\-]+$")
SUBDIRS = ("jobs", "running", "done", "out", "results")


def validate_args(args) -> List[str]:
    """只放行 run_tasks.py 认识的参数，值里不许有空格、引号、分号这类字符。"""
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError("args 要是字符串列表")
    for a in args:
        if a.startswith("--"):
            if a not in ALLOWED_FLAGS:
                raise ValueError(f"不认识的参数：{a}")
        elif not VALUE.match(a):
            raise ValueError(f"参数值里有不允许的字符：{a!r}")
    return args


def ensure_dirs(exchange: Path) -> None:
    for name in SUBDIRS:
        (exchange / name).mkdir(parents=True, exist_ok=True)


def pending_jobs(exchange: Path) -> List[Path]:
    """还没领、也没做完的任务文件，按文件名排序（宿主机用时间戳开头命名）。"""
    out = []
    for job in sorted((exchange / "jobs").glob("*.json")):
        stem = job.stem
        if not (exchange / "running" / f"{stem}.json").exists() and \
                not (exchange / "done" / f"{stem}.json").exists():
            out.append(job)
    return out


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_one(exchange: Path, repo: Path, job_path: Path,
            runner: Callable = subprocess.run, python: Optional[str] = None) -> dict:
    """执行一个任务文件，返回回执。"""
    stem = job_path.stem
    _write_json(exchange / "running" / f"{stem}.json", {"id": stem, "started": time.time()})
    receipt = {"id": stem, "exit_code": None, "seconds": 0.0, "copied": [], "error": None}
    started = time.time()
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        args = validate_args(job.get("args"))
        cmd = [python or sys.executable, str(repo / "scripts" / "run_tasks.py"), *args]
        with (exchange / "out" / f"{stem}.txt").open("w", encoding="utf-8") as log:
            proc = runner(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT)
        receipt["exit_code"] = proc.returncode
        dest = exchange / "results" / stem
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted((repo / "logs").glob("*")):
            if f.is_file() and f.name.startswith(("tasks_", "run_")) and f.stat().st_mtime >= started - 1:
                shutil.copy2(f, dest / f.name)
                receipt["copied"].append(f.name)
    except Exception as e:  # 任务文件坏了、参数不合法：写进回执，worker 接着领下一个
        receipt["error"] = f"{type(e).__name__}: {e}"
    receipt["seconds"] = round(time.time() - started, 1)
    _write_json(exchange / "done" / f"{stem}.json", receipt)
    (exchange / "running" / f"{stem}.json").unlink(missing_ok=True)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True, help="共享的交换目录，如 Z:\\")
    ap.add_argument("--repo", required=True, help="虚拟机本地盘上的代码仓库")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="只处理当前已有的任务就退出")
    args = ap.parse_args()

    exchange, repo = Path(args.exchange), Path(args.repo)
    ensure_dirs(exchange)
    print(f"worker 启动：交换目录 {exchange}，仓库 {repo}")
    while True:
        _write_json(exchange / "worker_alive.json", {"time": time.time(), "repo": str(repo)})
        for job in pending_jobs(exchange):
            print(f"领到任务 {job.stem}")
            r = run_one(exchange, repo, job)
            print(f"  完成：exit={r['exit_code']} 用时 {r['seconds']}s 拷回 {len(r['copied'])} 个文件"
                  + (f" 出错 {r['error']}" if r["error"] else ""))
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
