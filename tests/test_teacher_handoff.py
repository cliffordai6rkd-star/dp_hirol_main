"""Check external process waiting against real CPU teacher checkpoints."""

import os
from pathlib import Path
import subprocess
import sys
import time

import dill
from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.common import teacher_handoff


def _checkpoint(path, *, epoch=300, optimizer_step=100, num_epochs=300,
                num_optimizer_steps=None, early_enabled=False, should_stop=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cfg": OmegaConf.create({"training": {
            "num_epochs": num_epochs, "num_optimizer_steps": num_optimizer_steps,
            "early_stopping": {"enabled": early_enabled},
        }}),
        "state_dicts": {"model": {"weight": torch.ones(2)}},
        "pickles": {
            "epoch": dill.dumps(epoch),
            "optimizer_step": dill.dumps(optimizer_step),
            "early_stopping_state": dill.dumps({"should_stop": should_stop}),
        },
    }, path, pickle_module=dill)


def _exited_process():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=10)
    return process.pid


def test_process_identity_matches_proc_and_rejects_exited_process():
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    assert teacher_handoff.teacher_process_identity(os.getpid()) == stat[stat.rfind(")") + 2:].split()[19]
    with pytest.raises(ValueError, match="not running"):
        teacher_handoff.teacher_process_identity(_exited_process())
    with pytest.raises(ValueError, match="positive"):
        teacher_handoff.teacher_process_identity(0)


def test_process_identity_rejects_zombie():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        with pytest.raises(ValueError, match="exited"):
            teacher_handoff.teacher_process_identity(process.pid)
    finally:
        process.wait(timeout=10)


def test_wait_reads_latest_only_after_real_process_finishes(tmp_path, monkeypatch):
    latest = tmp_path / "checkpoints" / "latest.ckpt"
    complete = tmp_path / "checkpoints" / "complete.ckpt"
    _checkpoint(latest, epoch=50)
    _checkpoint(complete, epoch=300)
    process = subprocess.Popen([
        sys.executable, "-c",
        "import os,sys; sys.stdin.read(1); os.replace(sys.argv[1], sys.argv[2])",
        str(complete), str(latest),
    ], stdin=subprocess.PIPE)
    original_sleep = time.sleep
    waits = []

    def release_teacher(seconds):
        waits.append(seconds)
        if len(waits) == 1:
            process.stdin.write(b"x")
            process.stdin.flush()
        original_sleep(0.01)

    monkeypatch.setattr(teacher_handoff.time, "sleep", release_teacher)
    try:
        identity = teacher_handoff.teacher_process_identity(process.pid)
        result = teacher_handoff.wait_for_teacher(process.pid, identity, tmp_path)
        assert result == {
            "latest_checkpoint": str(latest), "epoch": 300,
            "optimizer_step": 100, "reason": "num_epochs",
        }
        assert waits and all(0 < delay <= 30 for delay in waits)
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdin.close()


@pytest.mark.parametrize("options,reason", [
    ({"num_epochs": None, "num_optimizer_steps": 100}, "num_optimizer_steps"),
    ({"epoch": 45, "early_enabled": True, "should_stop": True}, "early_stopping"),
])
def test_completed_budget_or_enabled_early_stop_is_accepted(tmp_path, options, reason):
    _checkpoint(tmp_path / "checkpoints" / "latest.ckpt", **options)
    result = teacher_handoff.wait_for_teacher(_exited_process(), "old-starttime", tmp_path)
    assert result["reason"] == reason


@pytest.mark.parametrize("options", [
    {"epoch": 50},
    {"epoch": 50, "should_stop": True, "early_enabled": False},
    {"epoch": 50, "early_enabled": True, "should_stop": False},
    {"num_epochs": None, "num_optimizer_steps": 101},
])
def test_failed_process_with_incomplete_checkpoint_does_not_handoff(tmp_path, options):
    _checkpoint(tmp_path / "checkpoints" / "latest.ckpt", **options)
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(2)"])
    process.wait(timeout=10)
    with pytest.raises(ValueError, match="before completing training"):
        teacher_handoff.wait_for_teacher(process.pid, "old-starttime", tmp_path)


def test_missing_latest_does_not_handoff(tmp_path):
    with pytest.raises(ValueError, match="without a latest checkpoint"):
        teacher_handoff.wait_for_teacher(_exited_process(), "old-starttime", tmp_path)


def test_reused_pid_does_not_wait_on_unrelated_process(tmp_path, monkeypatch):
    _checkpoint(tmp_path / "checkpoints" / "latest.ckpt")
    monkeypatch.setattr(teacher_handoff, "teacher_process_identity", lambda pid: "different-starttime")

    def unexpected_wait(seconds):
        pytest.fail("Must not wait for a reused PID")

    monkeypatch.setattr(teacher_handoff.time, "sleep", unexpected_wait)
    assert teacher_handoff.wait_for_teacher(os.getpid(), "old-starttime", tmp_path)["epoch"] == 300
