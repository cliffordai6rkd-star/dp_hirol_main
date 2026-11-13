"""Checkpoint scores from fresh episode inference, shared by teacher and student."""

import json
import logging
import math
from pathlib import Path


ACTION_MSE_KEY = "checkpoint_action_mse"
SCORES_FILENAME = "action_mse.json"


def _available_scores(directory, records):
    return [
        record for record in records
        if record.get("checkpoint")
        and math.isfinite(float(record[ACTION_MSE_KEY]))
        and (directory / record["checkpoint"]).is_file()
    ]


def select_action_mse_teacher(run_dir):
    """Select the lowest scored checkpoint that is still retained on disk."""
    directory = Path(run_dir).expanduser().resolve() / "checkpoints"
    scores = directory / SCORES_FILENAME
    if not scores.is_file():
        raise FileNotFoundError(
            f"No checkpoint inference scores at {scores}; train the teacher with "
            "checkpoint.action_mse.enabled=true first"
        )
    records = json.loads(scores.read_text())["records"]
    candidates = _available_scores(directory, records)
    if not candidates:
        raise FileNotFoundError(f"No retained checkpoint with a finite action MSE in {scores}")
    best = min(candidates, key=lambda record: record[ACTION_MSE_KEY])
    return (directory / best["checkpoint"]).resolve(), best


def resolve_student_teacher(cfg):
    """Read optional teacher data YAML; prefer an explicit checkpoint to MSE selection."""
    from omegaconf import OmegaConf, open_dict
    from diffusion_policy.common.config_cli import configured_output_dir, load_config_reference

    if OmegaConf.is_missing(cfg, "teacher_checkpoint"):
        # New-teacher presets require an explicit checkpoint. Preserve null
        # as the historical opt-in to automatic action-MSE selection.
        raise ValueError("Set teacher_checkpoint=/path/to/new_teacher/checkpoints/best.ckpt")
    source = None
    if cfg.get("teacher_config"):
        source = load_config_reference(
            str(cfg.teacher_config), cfg.get("teacher_config_overrides", []), include_hydra=True,
        )
    teacher_output = score = None
    if cfg.get("teacher_checkpoint"):
        checkpoint = Path(str(cfg.teacher_checkpoint)).expanduser().resolve()
    elif source is not None:
        teacher_output = Path(cfg.get("teacher_output_dir") or configured_output_dir(source)).expanduser().resolve()
        checkpoint, score = select_action_mse_teacher(teacher_output)
    else:
        raise ValueError("Set teacher_config for MSE selection or specify teacher_checkpoint")
    if source is not None:
        with open_dict(source):
            del source["hydra"]
        OmegaConf.resolve(source)
    with open_dict(cfg):
        cfg.teacher_checkpoint = str(checkpoint)
        if teacher_output is not None:
            cfg.teacher_output_dir = str(teacher_output)
        cfg.teacher_selection = score
    return checkpoint, source


def record_checkpoint_action_mse(run_dir, metrics, checkpoint):
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    scores = directory / SCORES_FILENAME
    records = json.loads(scores.read_text())["records"] if scores.is_file() else []
    protocol = metrics.get("action_mse_protocol")
    if any(item.get("action_mse_protocol") != protocol for item in records):
        raise ValueError("Action MSE protocol changed; re-evaluate retained checkpoints before combining scores")
    previous = next((item for item in records if item["optimizer_step"] == metrics["optimizer_step"]), {})
    record = dict(metrics, checkpoint=Path(checkpoint).name if checkpoint else previous.get("checkpoint"))
    records = [item for item in records if item["optimizer_step"] != record["optimizer_step"]]
    records.append(record)
    candidates = _available_scores(directory, records)
    best = min(candidates, key=lambda item: item[ACTION_MSE_KEY]) if candidates else None
    payload = {"metric": ACTION_MSE_KEY, "mode": "min", "best": best, "records": records}
    if protocol is not None:
        payload["action_mse_protocol"] = protocol
    temporary = scores.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(scores)
    if protocol is not None and protocol.get("episode_selection") == "all":
        # Render in the training process so PNG refreshes survive future resumes.
        # Plotting is secondary to preserving the scores and training checkpoint.
        try:
            from diffusion_policy.scripts.plot_checkpoint_action_mse import plot_scores
            plot_scores(payload, Path(run_dir) / "action_mse.png")
        except Exception:
            logging.getLogger(__name__).exception("Could not update action_mse.png; checkpoint scores are saved")
