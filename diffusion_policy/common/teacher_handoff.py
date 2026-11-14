"""Wait for an existing Linux teacher process and check its saved progress."""

from pathlib import Path
import time

import dill
from omegaconf import OmegaConf
import torch


def teacher_process_identity(pid: int) -> str:
    """Return /proc starttime, rejecting absent or already-exited processes."""
    if int(pid) <= 0:
        raise ValueError("Teacher PID must be positive")
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except FileNotFoundError as exc:
        raise ValueError(f"Teacher process {pid} is not running") from exc
    # comm (field 2) may contain spaces or parentheses. The suffix begins at
    # field 3 (state), and starttime is field 22.
    fields = stat[stat.rfind(")") + 2:].split()
    if fields[0] in {"Z", "X"}:
        raise ValueError(f"Teacher process {pid} has exited")
    return fields[19]


def wait_for_teacher(pid: int, start_time: str, run_dir) -> dict:
    """Wait for this process to exit, then require a completed latest checkpoint.

    This is an external process, so its exit status is not available through
    waitpid. Saved epoch/step progress or an enabled early-stop decision is
    the completion evidence; process disappearance alone is insufficient.
    """
    while True:
        try:
            identity = teacher_process_identity(pid)
        except ValueError:
            break
        if identity != str(start_time):
            break
        time.sleep(5)

    checkpoint = Path(run_dir).expanduser().resolve() / "checkpoints" / "latest.ckpt"
    if not checkpoint.is_file():
        raise ValueError(f"Teacher exited without a latest checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    saved = payload["pickles"]
    epoch = int(dill.loads(saved["epoch"])) if "epoch" in saved else 0
    optimizer_step = int(dill.loads(saved["optimizer_step"])) if "optimizer_step" in saved else 0
    early_stop = dill.loads(saved["early_stopping_state"]) if "early_stopping_state" in saved else {}
    num_epochs = OmegaConf.select(cfg, "training.num_epochs")
    num_steps = OmegaConf.select(cfg, "training.num_optimizer_steps")
    if num_epochs is not None and epoch >= int(num_epochs):
        reason = "num_epochs"
    elif num_steps is not None and optimizer_step >= int(num_steps):
        reason = "num_optimizer_steps"
    elif (OmegaConf.select(cfg, "training.early_stopping.enabled", default=False)
          and early_stop.get("should_stop", False)):
        reason = "early_stopping"
    else:
        raise ValueError(
            f"Teacher exited before completing training: epoch={epoch}/{num_epochs}, "
            f"optimizer_step={optimizer_step}/{num_steps}; student will not start"
        )
    return {
        "latest_checkpoint": str(checkpoint),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "reason": reason,
    }
