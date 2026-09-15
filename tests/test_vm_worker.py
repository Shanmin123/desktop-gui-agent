"""虚拟机里的任务 worker：参数白名单、领任务、回执和结果拷回。不需要真的虚拟机。"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("guest_worker", ROOT / "scripts" / "vm" / "guest_worker.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def test_validate_args_accepts_a_normal_suite_run():
    args = ["--live", "--set", "suite", "--repeat", "3", "--locate-target",
            "--api-base", "http://10.0.2.2:8000/v1", "--api-key", "local",
            "--api-coord-space", "rel1000", "--model", "Qwen/Qwen3.5-4B", "--tag", "vm_q35_suite",
            "--resolution", "1280x720"]
    assert W.validate_args(args) == args


def test_validate_args_accepts_the_one_stage_baseline():
    args = ["--live", "--set", "basic", "--one-stage", "--tag", "vm_one_stage"]
    assert W.validate_args(args) == args


@pytest.mark.parametrize("args", [
    ["--live", "--exec", "calc"],               # 不认识的参数
    ["--tag", "a b"],                           # 值里有空格
    ["--tag", "x;del"],                         # 值里有分号
    ["--tag", "x&&y"],
    "--live --set suite",                       # 不是列表
])
def test_validate_args_rejects_anything_else(args):
    with pytest.raises(ValueError):
        W.validate_args(args)


def _setup(tmp_path):
    exchange, repo = tmp_path / "exchange", tmp_path / "repo"
    W.ensure_dirs(exchange)
    (repo / "logs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    return exchange, repo


def _job(exchange, stem, args):
    path = exchange / "jobs" / f"{stem}.json"
    path.write_text(json.dumps({"id": stem, "args": args}), encoding="utf-8")
    return path


def test_pending_jobs_skips_running_and_done(tmp_path):
    exchange, _ = _setup(tmp_path)
    for stem in ("001_a", "002_b", "003_c"):
        _job(exchange, stem, ["--live"])
    (exchange / "running" / "002_b.json").write_text("{}", encoding="utf-8")
    (exchange / "done" / "003_c.json").write_text("{}", encoding="utf-8")
    assert [p.stem for p in W.pending_jobs(exchange)] == ["001_a"]


def test_run_one_runs_run_tasks_and_copies_new_logs(tmp_path):
    exchange, repo = _setup(tmp_path)
    (repo / "logs" / "tasks_old.json").write_text("{}", encoding="utf-8")
    old = (repo / "logs" / "tasks_old.json")
    import os
    os.utime(old, (1, 1))                                   # 很早以前的日志，不该拷
    job = _job(exchange, "001_suite", ["--live", "--set", "suite", "--tag", "vm1"])
    seen = {}

    def fake_runner(cmd, cwd, stdout, stderr):
        seen["cmd"], seen["cwd"] = cmd, cwd
        stdout.write("成功率 1/1\n")
        (Path(cwd) / "logs" / "tasks_vm1_suite.json").write_text("{}", encoding="utf-8")
        (Path(cwd) / "logs" / "run_vm1.jsonl").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    receipt = W.run_one(exchange, repo, job, runner=fake_runner, python="python")
    assert seen["cmd"][:2] == ["python", str(repo / "scripts" / "run_tasks.py")]
    assert seen["cmd"][2:] == ["--live", "--set", "suite", "--tag", "vm1"]
    assert receipt["exit_code"] == 0 and receipt["error"] is None
    assert sorted(receipt["copied"]) == ["run_vm1.jsonl", "tasks_vm1_suite.json"]
    assert (exchange / "results" / "001_suite" / "tasks_vm1_suite.json").exists()
    assert json.loads((exchange / "done" / "001_suite.json").read_text(encoding="utf-8"))["exit_code"] == 0
    assert not (exchange / "running" / "001_suite.json").exists()
    assert "成功率" in (exchange / "out" / "001_suite.txt").read_text(encoding="utf-8")


def test_run_one_writes_an_error_receipt_for_bad_args(tmp_path):
    exchange, repo = _setup(tmp_path)
    job = _job(exchange, "002_bad", ["--rm", "-rf"])

    def runner(*a, **k):
        raise AssertionError("参数不合法时不该执行")

    receipt = W.run_one(exchange, repo, job, runner=runner, python="python")
    assert receipt["exit_code"] is None and "不认识的参数" in receipt["error"]
    assert (exchange / "done" / "002_bad.json").exists()
