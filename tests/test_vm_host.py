"""宿主机任务投递：任务文件、心跳判断、结果拷回、显卡队列暂停。不需要真的虚拟机和显卡。"""

import importlib.util
import json
import types
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


# --- 恢复快照：等虚拟机真的关掉 -------------------------------------------------


class _Vbox:
    """假的 VBoxManage：showvminfo 按给定顺序报状态，其余命令只记下来。"""

    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[1] if len(cmd) > 1 else "")
        if "showvminfo" in cmd:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return types.SimpleNamespace(stdout=f'VMState="{state}"\nname="agent-win11"\n', returncode=0)
        return types.SimpleNamespace(stdout="", returncode=0)


def _fake_clock(step=10.0):
    now = [0.0]

    def clock():
        now[0] += step
        return now[0]

    return clock


def test_restore_snapshot_waits_until_the_vm_is_really_off():
    """poweroff 是异步的，没关干净就 restore 会被 VirtualBox 拒掉。"""
    vbox = _Vbox(["running", "running", "poweroff"])
    H.restore_snapshot("agent-win11", "clean", vboxmanage="vbox", run=vbox,
                       sleep=lambda _s: None, clock=_fake_clock())
    assert vbox.calls == ["controlvm", "showvminfo", "showvminfo", "showvminfo", "snapshot", "startvm"]


def test_restore_snapshot_gives_up_if_the_vm_never_powers_off():
    vbox = _Vbox(["running"])
    with pytest.raises(SystemExit):
        H.restore_snapshot("agent-win11", "clean", vboxmanage="vbox", run=vbox,
                           sleep=lambda _s: None, clock=_fake_clock(step=30.0))
    assert "snapshot" not in vbox.calls


def test_vm_state_reads_the_machinereadable_output():
    assert H.vm_state("agent-win11", "vbox", run=_Vbox(["saved"])) == "saved"
    assert H.vm_state("agent-win11", "vbox", run=lambda *a, **k: types.SimpleNamespace(stdout="")) == ""


# --- 回执 ---------------------------------------------------------------------


@pytest.mark.parametrize("receipt", [
    {"exit_code": 1, "error": None},
    {"exit_code": 0, "error": "ValueError: 参数不合法"},
    {"exit_code": None, "error": None},
])
def test_a_batch_that_failed_in_the_vm_exits_non_zero(receipt):
    """不抛的话 run_batches.py 会把失败的一批当成跑成了。"""
    with pytest.raises(SystemExit):
        H.check_receipt(receipt)


def test_a_clean_receipt_passes():
    H.check_receipt({"exit_code": 0, "error": None, "seconds": 12.3})
