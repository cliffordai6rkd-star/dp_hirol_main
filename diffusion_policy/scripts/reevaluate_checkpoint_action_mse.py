"""Stage full-dataset rescoring of retained teacher checkpoints without training.

Original checkpoints and score files are read-only. After every evaluation
succeeds, a separate output directory contains replacement checkpoint payloads,
score history, aliases, and a resume config ready for explicit promotion.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dill
import hydra
from omegaconf import OmegaConf, open_dict
import torch

from diffusion_policy.common.checkpoint_action_mse import ACTION_MSE_KEY
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
    _NullLogger,
    _action_mse_protocol,
)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", pickle_module=dill,
                      weights_only=False, mmap=True)


def checkpoint_step(payload):
    return int(dill.loads(payload["pickles"]["optimizer_step"]))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reevaluate(args):
    run = args.run_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output == run or output == run / "checkpoints":
        raise ValueError("--output-dir must be a new staging directory, not the run/checkpoints directory")
    latest = run / "checkpoints/latest.ckpt"
    latest_payload = read_checkpoint(latest)
    cfg = OmegaConf.create(OmegaConf.to_container(latest_payload["cfg"], resolve=True))
    latest_step = checkpoint_step(latest_payload)
    del latest_payload
    with open_dict(cfg):
        cfg.output_dir = str(run)
        cfg.training.resume = True
        cfg.checkpoint.action_mse.enabled = True
        cfg.checkpoint.action_mse.episode_selection = "all"
        cfg.checkpoint.action_mse.seed = args.seed
        cfg.checkpoint.action_mse.batch_size = args.batch_size
        cfg.checkpoint.action_mse.every_optimizer_steps = args.every_optimizer_steps
        cfg.checkpoint.action_mse.pop("num_episodes", None)
    if cfg.checkpoint.topk.monitor_key != ACTION_MSE_KEY or cfg.checkpoint.topk.mode != "min":
        raise ValueError("Teacher Top-K must minimize checkpoint_action_mse")

    # Symlink aliases do not introduce duplicate evaluations. A standalone
    # latest checkpoint can be newer than every retained Top-K candidate.
    paths_by_step = {}
    for path in sorted((run / "checkpoints").glob("*.ckpt")):
        if path.is_symlink() or not path.is_file():
            continue
        payload = read_checkpoint(path)
        step = checkpoint_step(payload)
        del payload
        if step not in paths_by_step or path.name == "latest.ckpt":
            paths_by_step[step] = path
    if latest_step not in paths_by_step:
        paths_by_step[latest_step] = latest.resolve()
    if max(paths_by_step) != latest_step:
        raise ValueError("latest.ckpt is not the newest saved step; resolve this before migration")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "checkpoints"
    destination.mkdir()
    OmegaConf.save(cfg, output / "resume_config.yaml")
    protocol = _action_mse_protocol(cfg)
    print(f"STAGING {len(paths_by_step)} checkpoints, latest step {latest_step}, protocol {protocol}", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.training.amp.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg.training.amp.get("allow_tf32", True))
    workspace = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(run))
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    with open_dict(data_cfg):
        data_cfg.preload_images = True
        data_cfg.load_result_add = "ram"
        # Match the saved training decoder, including CUDA TorchCodec when
        # configured. CPU/GPU decoders can differ slightly in pixel values.
        data_cfg.image_randomer_config = None
    dataset = hydra.utils.instantiate(data_cfg)
    if hasattr(dataset, "set_contact_label_config"):
        dataset.set_contact_label_config(WrenchPreprocessor.from_policy(workspace.model).get_config())
    workspace._checkpoint_eval_dataset = dataset.get_all_episodes_dataset()
    print(f"DATASET_READY {len(workspace._checkpoint_eval_dataset)} windows; images decoded once", flush=True)
    records, timings = [], []
    for step, path in sorted(paths_by_step.items()):
        started = time.monotonic()
        payload = read_checkpoint(path)
        workspace.load_payload(
            payload, exclude_keys=("optimizer", "grad_scaler"),
            include_keys=("global_step", "optimizer_step", "epoch", "freeze_config"),
        )
        workspace.model.to(device)
        if workspace.ema_model is not None:
            workspace.ema_model.to(device)
        workspace.action_mse_metrics = {}
        del payload
        print(f"EVALUATING step={step} source={path.name}", flush=True)
        metrics = workspace._checkpoint_action_mse_and_log(cfg, device, _NullLogger(), _NullLogger())
        if metrics.get("action_mse_protocol") != protocol:
            raise RuntimeError("Evaluation did not use the requested fixed full-dataset protocol")
        filename = str(cfg.checkpoint.topk.format_str).format(**metrics)
        records.append(dict(metrics, checkpoint=filename))
        elapsed = time.monotonic() - started
        timings.append({"optimizer_step": step, "seconds": elapsed, "source_checkpoint": str(path)})
        write_json(output / "reevaluation_progress.json", {
            "status": "evaluating", "protocol": protocol, "records": records, "timings": timings,
        })
        print(f"SCORED step={step} mse={metrics[ACTION_MSE_KEY]:.8f} windows={metrics['num_windows']} elapsed={elapsed:.1f}s", flush=True)

    # Only now prepare the replacement training metadata. Full tensor state
    # comes from the original payload, not from the evaluation workspace.
    manager = TopKCheckpointManager(save_dir=str(run / "checkpoints"), **cfg.checkpoint.topk)
    best_records = sorted(records, key=lambda row: row[ACTION_MSE_KEY])[:int(cfg.checkpoint.topk.k)]
    manager.path_value_map = {str(run / "checkpoints" / row["checkpoint"]): row[ACTION_MSE_KEY]
                              for row in best_records}
    workspace._store_topk_state(manager, cfg)
    ranking = copy.deepcopy(workspace.topk_state)
    for record in records:
        step = record["optimizer_step"]
        payload = read_checkpoint(paths_by_step[step])
        payload["cfg"] = copy.deepcopy(cfg)
        payload["pickles"]["topk_state"] = dill.dumps(ranking)
        payload["pickles"]["action_mse_metrics"] = dill.dumps(
            {key: value for key, value in record.items() if key != "checkpoint"}
        )
        payload["pickles"]["_output_dir"] = dill.dumps(str(run))
        target = destination / record["checkpoint"]
        temporary = target.with_suffix(".tmp.ckpt")
        torch.save(payload, temporary, pickle_module=dill)
        temporary.replace(target)
        del payload
        print(f"STAGED_CHECKPOINT {target.name}", flush=True)
    best = min(records, key=lambda row: row[ACTION_MSE_KEY])
    newest = max(records, key=lambda row: row["optimizer_step"])
    write_json(destination / "action_mse.json", {
        "metric": ACTION_MSE_KEY, "mode": "min", "best": best, "records": records,
        "action_mse_protocol": protocol,
    })
    (destination / "best.ckpt").symlink_to(best["checkpoint"])
    (destination / "latest.ckpt").symlink_to(newest["checkpoint"])
    report = {"status": "staged", "run_dir": str(run), "output_dir": str(output),
              "protocol": protocol, "records": records, "timings": timings,
              "latest_step": newest["optimizer_step"], "best_step": best["optimizer_step"],
              "topk_state": ranking, "source_files_modified": False,
              "note": "Checkpoint tensor/optimizer states are preserved; only config and evaluation/ranking metadata change. Promote staging/checkpoints explicitly before resuming."}
    write_json(output / "reevaluation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--every-optimizer-steps", type=int, default=10000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.every_optimizer_steps < 1:
        parser.error("batch size and evaluation interval must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reevaluate(args)


if __name__ == "__main__":
    main()
