"""Re-evaluate saved students by contact phase without training or changing a run.

Replay the original full validation order and diffusion probes before splitting
per-sample errors by measured anchor phase. Only saved checkpoints are plotted.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def plot_result(result, history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = result["checkpoints"]
    epochs = [row["epoch"] for row in points]
    contact = [row["phases"]["contact"]["mse"] for row in points]
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    full_history = {0: result["logged_initial_all_phase_mse"]}
    full_history.update({row["epoch"]: row["val_mse"] for row in history if "val_mse" in row})
    ax.plot(list(full_history), list(full_history.values()), color="#9ca3af",
            linewidth=1.6, label="All phases: original validation log")
    ax.plot(epochs, contact, "o--", color="#1668b2", linewidth=2, markersize=8,
            label="Contact only: re-evaluated saved checkpoints")
    for row, value in zip(points, contact):
        offset = (-8, 20) if row["label"] == "latest" else (0, -36)
        ax.annotate(f'{row["label"]} (epoch {row["epoch"]})\n{value:.5f}',
                    (row["epoch"], value), xytext=offset, textcoords="offset points",
                    ha="right" if row["label"] == "latest" else "center", fontsize=10)
    count = points[0]["phases"]["contact"]["samples"]
    ax.set_title(f"Contact-stage validation loss\nMeasured contact anchors (phase=2), n={count}",
                 fontsize=14, pad=12)
    ax.set_xlabel("Completed training epoch")
    ax.set_ylabel("Frozen DiT output MSE")
    ax.set_xlim(-1.5, max(epochs) + 2)
    ax.margins(y=0.35)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("Same EMA weights, validation order, noise and timesteps as the original run",
                 fontsize=10, color="#4b5563")
    saved_epochs = ", ".join(str(epoch) for epoch in epochs)
    fig.text(0.5, -0.035,
             f"Only epochs {saved_epochs} are recoverable. Dashed segments do not represent intermediate measurements.",
             ha="center", fontsize=9, color="#4b5563")
    fig.savefig(destination / "contact_validation_loss.png", dpi=180, bbox_inches="tight")
    fig.savefig(destination / "contact_validation_loss.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = run.parent / "source"
    # Use the implementation saved with this experiment when available.
    sys.path.insert(0, str(source if source.is_dir() else ROOT))

    import dill
    import hydra
    import numpy as np
    from omegaconf import OmegaConf, open_dict
    import torch
    from torch.utils.data import DataLoader
    from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache
    from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace

    class PhaseEvaluationWorkspace(TrainDistillRenderNetWorkspace):
        def _denoiser_losses(self, action, student_tokens, teacher_tokens):
            predictions = []
            hook = self.teacher_policy.model.register_forward_hook(
                lambda module, inputs, value: predictions.append(value.detach()))
            try:
                losses = super()._denoiser_losses(action, student_tokens, teacher_tokens)
            finally:
                hook.remove()
            assert len(predictions) == 2, "Expected the matched teacher/student DiT pair"
            errors = (predictions[1].float() - predictions[0].float()).square().flatten(1).mean(1)
            self.sample_errors.extend(errors.cpu().tolist())
            return losses

    torch.set_num_threads(4)
    cfg = OmegaConf.load(run / "config.yaml")
    initial_path = Path(cfg.training.warm_start_from)
    summary = json.loads((run / "training_summary.json").read_text())
    history = [json.loads(line) for line in (run / "logs.json.txt").read_text().splitlines() if line.strip()]
    with open_dict(cfg):
        cfg.training.resume_from = None
        cfg.training.compile.enabled = False
        cfg.training.distributed.enabled = False
        cfg.training.max_val_batches = None
    workspace = PhaseEvaluationWorkspace(cfg, output_dir=str(output))
    assert workspace.objective == "denoiser_mse"
    workspace.training_stage = "rgb"
    workspace.model.eval().requires_grad_(False)
    workspace.ema_model.eval().requires_grad_(False)

    cache_path = Path(json.loads((run / "teacher_cache.json").read_text())["splits"]["validation"])
    meta = json.loads((cache_path / "metadata.json").read_text())
    dataset = hydra.utils.instantiate(OmegaConf.create(meta["identity"]["dataset"]))
    validation = dataset.get_validation_dataset()
    labels = validation.get_phase_labels()
    indices_hash = hashlib.sha256(np.ascontiguousarray(validation.indices).tobytes()).hexdigest()
    assert indices_hash == meta["identity"]["validation"]["indices_sha256"]
    assert len(validation) == meta["length"]
    cache = TeacherTargetCache(cache_path, meta["identity"], len(validation), token_dtype=meta["dtype"])
    assert cache.complete
    loader = DataLoader(CachedTeacherDataset(validation, cache), **OmegaConf.to_container(cfg.val_dataloader))
    checkpoints = [("initial", initial_path, 0, summary["initial_val_mse"]),
                   ("best", run / "checkpoints/best.ckpt", None, summary["best_val_mse"]),
                   ("latest", run / "checkpoints/latest.ckpt", None, summary["latest_val_mse"])]
    result = {"run_dir": str(run), "source_code": str(source), "objective": "denoiser_mse",
              "state_key": "ema_model", "scope": "measured observation anchor phase",
              "validation_samples": len(validation), "batch_size": loader.batch_size,
              "seed": int(cfg.distillation.validation_seed),
              "probe_protocol": "Original full validation batches; aggregate by phase after forward pass",
              "logged_initial_all_phase_mse": summary["initial_val_mse"], "checkpoints": [],
              "limitation": "Only initial, best and latest weights exist; no intermediate contact curve can be reconstructed."}
    arrays = {"phase": labels, "dataset_indices": validation.indices}
    try:
        for label, path, epoch, expected in checkpoints:
            payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
            workspace.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
            saved_epoch = dill.loads(payload["pickles"]["epoch"])
            epoch = saved_epoch if epoch is None else epoch
            step = 0 if label == "initial" else dill.loads(payload["pickles"]["global_step"])
            del payload
            workspace.epoch = epoch
            workspace.sample_errors = []
            all_phase_mse = workspace._validate(loader)
            errors = np.asarray(workspace.sample_errors, dtype=np.float64)
            assert len(errors) == len(labels) and np.isfinite(errors).all()
            np.testing.assert_allclose(errors.mean(), all_phase_mse, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(all_phase_mse, expected, rtol=0, atol=2e-5,
                                       err_msg=f"Original {label} validation must reproduce before phase analysis")
            phases = {name: {"samples": int((labels == phase).sum()),
                             "mse": float(errors[labels == phase].mean())}
                      for phase, name in enumerate(("free", "precontact", "contact"))}
            row = {"label": label, "checkpoint": str(path), "epoch": epoch, "step": step,
                   "all_phase_mse": all_phase_mse, "logged_all_phase_mse": expected,
                   "reproduction_abs_error": abs(all_phase_mse - expected), "phases": phases}
            result["checkpoints"].append(row)
            arrays[label] = errors
            print(json.dumps(row), flush=True)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
        cache.close()
        dataset.lerobot_dataset.close()

    baseline = result["checkpoints"][0]["phases"]["contact"]["mse"]
    for row in result["checkpoints"]:
        row["contact_reduction_from_initial_pct"] = 100 * (1 - row["phases"]["contact"]["mse"] / baseline)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "per_sample_mse.npz", **arrays)
    with (output / "phase_losses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint", "epoch", "step", "all_phase_mse", "free_mse", "precontact_mse", "contact_mse"])
        for row in result["checkpoints"]:
            writer.writerow([row["label"], row["epoch"], row["step"], row["all_phase_mse"],
                             *(row["phases"][p]["mse"] for p in ("free", "precontact", "contact"))])
    plot_result(result, history, output)
    print(json.dumps({"output_dir": str(output), "original_validation_reproduced": True}), flush=True)


if __name__ == "__main__":
    main()
