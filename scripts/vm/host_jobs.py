"""宿主机这边：暂停显卡队列、起模型服务、恢复虚拟机快照、投任务、等回执、把结果拷回 logs/。

配合虚拟机里的 scripts/vm/guest_worker.py，整个过程不需要虚拟机的账号密码：
  恢复快照和开机    VBoxManage（VirtualBox 自带的命令行）
  在虚拟机里执行    worker 从共享的交换目录领任务
  模型推理          宿主机显卡上的 scripts/serve_vlm.py

用法（宿主机，-- 后面是交给虚拟机里 run_tasks.py 的参数）：
    python scripts/vm/host_jobs.py --vm agent-win11 --snapshot clean ^
        --serve-model Qwen/Qwen3.5-4B --serve-adapter checkpoints/q35_2sp --id q35_2sp_suite -- ^
        --live --set suite --repeat 3 --resolution 1280x720 ^
        --api-base http://10.0.2.2:8000/v1 --api-key local --api-coord-space rel1000 ^
        --tag vm_q35_2sp_suite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCHANGE = ROOT / "logs" / "vm_exchange"
PAUSE_FILE = ROOT / "logs" / "_q35_pause"
VBOXMANAGE = r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
GPU_SCRIPTS = ("train_lora.py", "eval_screenagent.py", "eval_grounding.py", "eval_plan.py",
               "tune_prompt.py", "serve_vlm.py")

_spec = importlib.util.spec_from_file_location("guest_worker", Path(__file__).with_name("guest_worker.py"))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def submit(exchange: Path, job_id: str, args: List[str], clock: Callable = time.time) -> Path:
    """写任务文件。文件名以时间戳开头，worker 按文件名顺序领。"""
    worker.validate_args(args)
    worker.ensure_dirs(exchange)
    stem = f"{int(clock())}_{job_id}"
    path = exchange / "jobs" / f"{stem}.json"
    path.write_text(json.dumps({"id": stem, "args": args}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def wait_for(check: Callable[[], bool], timeout: float, poll: float = 5.0,
             sleep: Optional[Callable] = None, clock: Optional[Callable] = None) -> bool:
    sleep = sleep or time.sleep
    clock = clock or time.time
    deadline = clock() + timeout
    while True:
        if check():
            return True
        if clock() >= deadline:
            return False
        sleep(poll)


def worker_alive(exchange: Path, max_age: float = 30.0, clock: Callable = time.time) -> bool:
    try:
        beat = json.loads((exchange / "worker_alive.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return clock() - float(beat.get("time", 0)) <= max_age


def collect(exchange: Path, stem: str, logs_dir: Path) -> List[str]:
    """把 results/<stem>/ 里的日志拷回宿主机的 logs/。"""
    src = exchange / "results" / stem
    copied = []
    for f in sorted(src.glob("*")) if src.is_dir() else []:
        if f.is_file():
            shutil.copy2(f, logs_dir / f.name)
            copied.append(f.name)
    return copied


def gpu_jobs_running(processes: Optional[Iterable] = None) -> List[str]:
    """宿主机上正在跑的训练 / 评测脚本（它们占着显卡）。"""
    if processes is None:
        import psutil

        processes = psutil.process_iter(["cmdline"])
    busy = []
    for p in processes:
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if "python" in cmd.lower() and any(s in cmd for s in GPU_SCRIPTS):
            busy.append(cmd)
    return busy


@contextmanager
def paused_gpu_queue(pause_file: Path = PAUSE_FILE, timeout: float = 4 * 3600,
                     busy: Callable[[], List[str]] = gpu_jobs_running):
    """建暂停文件，消融队列跑完当前这一步就停；等显卡空出来再往下走，结束后删掉暂停文件。"""
    pause_file.write_text("虚拟机 live 实验占用显卡", encoding="utf-8")
    try:
        if not wait_for(lambda: not busy(), timeout, poll=30):
            raise TimeoutError("等显卡空闲超时，还在跑：" + "; ".join(busy())[:300])
        yield
    finally:
        pause_file.unlink(missing_ok=True)


@contextmanager
def model_server(model: str, adapter: Optional[str], port: int = 8000, timeout: float = 300):
    cmd = [sys.executable, str(ROOT / "scripts" / "serve_vlm.py"), "--model", model, "--port", str(port)]
    if adapter:
        cmd += ["--adapter", adapter]
    log = (ROOT / "logs" / f"_serve_vlm_{port}.txt").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)

    def ready() -> bool:
        if proc.poll() is not None:
            raise RuntimeError(f"模型服务退出了，看 logs/_serve_vlm_{port}.txt")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                return r.status == 200
        except OSError:
            return False

    try:
        if not wait_for(ready, timeout, poll=5):
            raise TimeoutError("模型服务启动超时")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def restore_snapshot(vm: str, snapshot: str, vboxmanage: str = VBOXMANAGE,
                     run: Callable = subprocess.run) -> None:
    """关机 → 恢复快照 → 无界面开机。快照里要已经配好自动登录和 worker 开机自启。"""
    run([vboxmanage, "controlvm", vm, "poweroff"], capture_output=True)   # 本来就关着会报错，忽略
    time.sleep(3)
    run([vboxmanage, "snapshot", vm, "restore", snapshot], check=True)
    run([vboxmanage, "startvm", vm, "--type", "headless"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default=str(DEFAULT_EXCHANGE))
    ap.add_argument("--id", required=True, help="任务名，拼进任务文件名")
    ap.add_argument("--vm", default=None, help="VirtualBox 虚拟机名；给了 --snapshot 才会用")
    ap.add_argument("--snapshot", default=None, help="开跑前恢复到这个快照")
    ap.add_argument("--serve-model", default=None, help="给了就在宿主机起模型服务")
    ap.add_argument("--serve-adapter", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=6 * 3600, help="等回执的上限（秒）")
    ap.add_argument("--no-pause", action="store_true", help="不暂停显卡队列")
    ap.add_argument("run_tasks_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    run_args = [a for a in args.run_tasks_args if a != "--"]
    exchange = Path(args.exchange)
    worker.validate_args(run_args)

    pause = (lambda: contextmanager(lambda: (yield))()) if args.no_pause else paused_gpu_queue
    serve = (lambda: model_server(args.serve_model, args.serve_adapter, args.port)) if args.serve_model \
        else (lambda: contextmanager(lambda: (yield))())

    with pause():
        with serve():
            if args.snapshot:
                if not args.vm:
                    raise SystemExit("--snapshot 要配 --vm")
                print(f"恢复快照 {args.vm}:{args.snapshot} 并开机")
                restore_snapshot(args.vm, args.snapshot)
            print("等虚拟机里的 worker ……")
            if not wait_for(lambda: worker_alive(exchange), timeout=900, poll=10):
                raise SystemExit("15 分钟没等到 worker 心跳：检查虚拟机是否自动登录、worker 是否开机自启")
            job = submit(exchange, args.id, run_args)
            print(f"已投任务 {job.stem}")
            done = exchange / "done" / f"{job.stem}.json"
            if not wait_for(done.exists, timeout=args.timeout, poll=30):
                raise SystemExit(f"{args.timeout:.0f} 秒内没等到回执 {done}")
            receipt = json.loads(done.read_text(encoding="utf-8"))
            copied = collect(exchange, job.stem, ROOT / "logs")
            print(f"回执：exit={receipt['exit_code']} 用时 {receipt['seconds']}s 错误 {receipt['error']}")
            print(f"拷回 logs/：{copied}")


if __name__ == "__main__":
    main()
