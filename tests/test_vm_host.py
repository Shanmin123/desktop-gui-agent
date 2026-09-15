"""宿主机任务投递：任务文件、心跳判断、结果拷回、显卡队列暂停。不需要真的虚拟机和显卡。"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("host_jobs", ROOT / "scripts" / "vm" / "host_jobs.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def test_submit_writes_a_job_the_worker_will_pick_up(tmp_path):
    path = H.submit(tmp_path, "q35_suite", ["--live", "--set", "suite"], clock=lambda: 1700000000)
    assert path.name == "1700000000_q35_suite.json"
    job = json.loads(path.read_text(encoding="utf-8"))
    assert job == {"id": "1700000000_q35_suite", "args": ["--live", "--set", "suite"]}
    assert [p.stem for p in H.worker.pending_jobs(tmp_path)] == ["1700000000_q35_suite"]


def test_submit_rejects_arguments_the_worker_would_refuse(tmp_path):
    with pytest.raises(ValueError):
        H.submit(tmp_path, "bad", ["--live", "--exec", "calc"])


def test_worker_alive_checks_the_heartbeat_age(tmp_path):
    (tmp_path / "worker_alive.json").write_text(json.dumps({"time": 1000.0}), encoding="utf-8")
    assert H.worker_alive(tmp_path, max_age=30, clock=lambda: 1020.0)
    assert not H.worker_alive(tmp_path, max_age=30, clock=lambda: 1100.0)
    assert not H.worker_alive(tmp_path / "missing", clock=lambda: 0.0)


def test_wait_for_gives_up_at_the_deadline():
    now = [0.0]
    calls = []

    def sleep(seconds):
        now[0] += seconds

    assert H.wait_for(lambda: calls.append(1) or False, timeout=10, poll=5,
                      sleep=sleep, clock=lambda: now[0]) is False
    assert len(calls) == 3                                   # t=0, 5, 10


def test_collect_copies_results_into_logs(tmp_path):
    exchange, logs = tmp_path / "exchange", tmp_path / "logs"
    (exchange / "results" / "1_x").mkdir(parents=True)
    logs.mkdir()
    (exchange / "results" / "1_x" / "tasks_vm_suite.json").write_text("{}", encoding="utf-8")
    assert H.collect(exchange, "1_x", logs) == ["tasks_vm_suite.json"]
    assert (logs / "tasks_vm_suite.json").exists()
    assert H.collect(exchange, "missing", logs) == []


class _Proc:
    def __init__(self, cmdline):
        self.info = {"cmdline": cmdline}


def test_gpu_jobs_running_only_counts_training_and_eval_scripts():
    procs = [_Proc(["python", "scripts/train_lora.py", "--tag", "q35_2sp"]),
             _Proc(["python", "some_other_tool.py"]),
             _Proc(["notepad.exe"]),
             _Proc(None)]
    busy = H.gpu_jobs_running(procs)
    assert len(busy) == 1 and "train_lora.py" in busy[0]


def test_paused_gpu_queue_creates_and_removes_the_pause_file(tmp_path, monkeypatch):
    pause = tmp_path / "_q35_pause"
    states = [["python scripts/eval_screenagent.py"], []]      # 第一次还忙，第二次空闲
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    with H.paused_gpu_queue(pause, timeout=100, busy=lambda: states.pop(0) if states else []):
        assert pause.exists()
    assert not pause.exists()
