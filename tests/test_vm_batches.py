"""虚拟机批次清单：参数都过得了 worker 白名单，暂停文件挂满整段，命令拼得对。不需要真的虚拟机。"""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_batches", ROOT / "scripts" / "vm" / "run_batches.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def _value(args, flag):
    return args[args.index(flag) + 1]


def test_every_batch_passes_the_worker_allowlist_with_unique_ids_and_tags():
    for b in B.BATCHES:
        assert B.host_jobs.worker.validate_args(b["args"]) == b["args"]
    assert len({b["id"] for b in B.BATCHES}) == len(B.BATCHES)
    assert len({_value(b["args"], "--tag") for b in B.BATCHES}) == len(B.BATCHES)


def test_coordinate_space_and_model_name_follow_the_served_model():
    for b in B.BATCHES:
        expected = "pixel" if "Qwen2.5" in b["serve_model"] else "rel1000"
        assert _value(b["args"], "--api-coord-space") == expected, b["id"]
        assert _value(b["args"], "--model") == b["serve_model"], b["id"]


def test_command_skips_host_jobs_own_pause_and_passes_an_adapter_only_when_set():
    base = next(b for b in B.BATCHES if not b["serve_adapter"])
    tuned = next(b for b in B.BATCHES if b["serve_adapter"])
    cmd = B.command(base, "vm1", "snap1", python="py")
    assert cmd[:2] == ["py", str(B.HOST_JOBS)]
    assert "--no-pause" in cmd and "--serve-adapter" not in cmd
    assert cmd[cmd.index("--") + 1:] == base["args"]
    assert _value(B.command(tuned, "vm1", "snap1"), "--serve-adapter") == tuned["serve_adapter"]


def test_run_all_holds_the_pause_across_every_batch():
    events = []

    @contextmanager
    def pause():
        events.append("pause")
        yield
        events.append("resume")

    def run(cmd, cwd=None):
        events.append(_value(cmd, "--id"))
        return SimpleNamespace(returncode=0)

    results = B.run_all(B.BATCHES[:2], "vm1", "snap1", run=run, pause=pause)
    assert events == ["pause", B.BATCHES[0]["id"], B.BATCHES[1]["id"], "resume"]
    assert [r["exit_code"] for r in results] == [0, 0]


def test_list_prints_commands_without_running_anything(capsys):
    assert B.main(["--list", "--only", B.BATCHES[0]["id"]]) == 0
    out = capsys.readouterr().out
    assert B.BATCHES[0]["id"] in out and "host_jobs.py" in out
