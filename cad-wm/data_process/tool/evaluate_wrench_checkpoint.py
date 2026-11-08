"""Replay trained CAD-WM forecasts and export aligned six-channel plots.

Run from cad-wm: python -m data_process.tool.evaluate_wrench_checkpoint
Only q/dq/tau histories enter inference. Recorded wrench is used for scoring.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_process.libero_wrench_dataset import LiberoWrenchDataset, STATE_KEYS
from model.wrench_predictor import WrenchPredictor


CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
PHASES = ("free", "precontact", "contact")
BLUE, YELLOW = "#1665D8", "#DDB500"


def channel_metrics(prediction, target):
    error = (prediction.astype(np.float64) - target.astype(np.float64)).reshape(-1, 6)
    truth = target.reshape(-1, 6).astype(np.float64)
    if not len(error):
        return {"forecast_points": 0}
    variance = truth.var(0)
    mse = np.square(error).mean(0)
    return {
        "forecast_points": len(error),
        "channels": {
            channel: {"unit": "N" if index < 3 else "N m",
                      "rmse": float(np.sqrt(mse[index])),
                      "mae": float(np.abs(error[:, index]).mean()),
                      "bias": float(error[:, index].mean()),
                      "p95_absolute_error": float(np.quantile(np.abs(error[:, index]), .95)),
                      "max_absolute_error": float(np.abs(error[:, index]).max()),
                      "truth_std": float(np.sqrt(variance[index])),
                      "rmse_over_truth_std": float(np.sqrt(mse[index] / max(variance[index], 1e-12)))}
            for index, channel in enumerate(CHANNELS)
        },
        "force_mse": float(mse[:3].mean()),
        "moment_mse": float(mse[3:].mean()),
    }


def draw_curves(output, label, horizon_index, selected, predictions, targets, target_times):
    """Save six individual plots and a six-panel overview at one fixed horizon."""
    output.mkdir(parents=True, exist_ok=True)
    times = target_times[selected, horizon_index]
    truth, predicted = targets[selected, horizon_index], predictions[selected, horizon_index]
    assert np.all(np.diff(times) > 0), "Do not connect points from different episodes"
    horizon = (horizon_index + 1) / 20
    overview, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    overview.suptitle(f"{label} | forecast +{horizon:.2f} s | training-data replay", fontsize=14)
    for index, channel in enumerate(CHANNELS):
        fig, axis = plt.subplots(figsize=(11.5, 3.8), constrained_layout=True)
        rmse = float(np.sqrt(np.square(predicted[:, index] - truth[:, index]).mean()))
        for ax in (axis, axes.flat[index]):
            ax.plot(times, truth[:, index], color=BLUE, linewidth=1.6, label="Ground truth (median-3)")
            ax.plot(times, predicted[:, index], color=YELLOW, linewidth=1.4, label="Prediction")
            ax.set_ylabel(f"{channel} ({'N' if index < 3 else 'N m'})")
            ax.set_xlabel("Target time within episode (s)")
            ax.grid(alpha=.2)
            ax.set_title(f"{channel} | RMSE {rmse:.4f}", fontsize=11)
            ax.legend(loc="best", fontsize=8)
        axis.set_title(f"{label} | {channel} | forecast +{horizon:.2f} s | RMSE {rmse:.4f}")
        fig.savefig(output / f"{channel}.png", dpi=190)
        fig.savefig(output / f"{channel}.svg")
        plt.close(fig)
    overview.savefig(output / "overview.png", dpi=160)
    overview.savefig(output / "overview.pdf")
    plt.close(overview)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/cad_wm_median3_200k/checkpoints/step_00200000.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[2]
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve() if args.output else root / "outputs" / datetime.now().strftime("%Y.%m.%d_%H.%M.%S_wrench_forecast_replay")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config, old_metrics = copy.deepcopy(saved["config"]), saved["metrics"]
    history = saved.get("trainer_state", {}).get("loss_history", [])
    validation_evidence = {
        "validation_enabled": bool(config["train"].get("validation_enabled", True)),
        "configured_val_ratio": config["train"].get("val_ratio"),
        "saved_val_loss": saved.get("val_loss"),
        "epochs_with_nonempty_val_loss": sum(row.get("val_loss") is not None for row in history),
        "saved_split": saved.get("split"),
        "configured_val_episode_indices": config["train"].get("val_episode_indices"),
        "interpretation": "All episodes were used for training; this is training-data replay, not held-out validation.",
    }
    if validation_evidence["validation_enabled"]:
        raise ValueError("This audit expects the current all-data checkpoint; explicitly implement its split before evaluating a held-out checkpoint")
    del saved, history
    predictor = WrenchPredictor(checkpoint, device=args.device)
    predictor.model.requires_grad_(False)
    dataset = LiberoWrenchDataset(config)  # Raw physical units; predictor owns checkpoint normalization.
    assert dataset.normalizer is None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    seed = int(config["train"].get("checkpoint_eval", {}).get("seed", 42))
    generator = torch.Generator(device="cpu")
    predictions, targets, anchors, phase_targets, phase_predictions = [], [], [], [], []
    for batch_index, batch in enumerate(loader):
        noise = []
        for anchor in batch["sample_idx"].tolist():
            generator.manual_seed(seed + anchor)
            noise.append(torch.randn(dataset.future_horizon, 6, generator=generator))
        result = predictor({key: batch[key] for key in STATE_KEYS},
                           source_noise=torch.stack(noise).to(predictor.device))
        predictions.append(result["future_wrench"].float().cpu().numpy())
        targets.append(batch["wrench_future_raw"].numpy())
        anchors.append(batch["sample_idx"].numpy())
        phase_targets.append(batch["contact_future"].numpy())
        phase_predictions.append(result["contact_state_pred"].cpu().numpy())
        if batch_index % 10 == 0 or batch_index + 1 == len(loader):
            print(f"Forecast batches {batch_index + 1}/{len(loader)}", flush=True)
    prediction, truth, anchor = np.concatenate(predictions), np.concatenate(targets), np.concatenate(anchors)
    phases, predicted_phases = np.concatenate(phase_targets), np.concatenate(phase_predictions)
    assert prediction.shape == truth.shape == (len(dataset), 8, 6)
    assert np.isfinite(prediction).all() and np.isfinite(truth).all()
    future_index = anchor[:, None] + np.arange(1, 9)[None]
    # Verify the exact native-row target alignment independently of the loader.
    np.testing.assert_array_equal(truth, dataset.high_tensors["wrench"].numpy()[future_index])
    episode_ids = np.full(len(anchor), -1, dtype=np.int64)
    local_anchor = np.empty_like(anchor)
    target_times = np.empty((len(anchor), 8), dtype=np.float64)
    episode_metrics = []
    dataset_path = Path(config["dataloader"]["root"])
    if not dataset_path.is_absolute():
        dataset_path = (root / dataset_path).resolve()
    with h5py.File(dataset_path, "r") as source:
        for episode in dataset.episodes:
            start, stop = episode["dataset_from_index"], episode["dataset_to_index"]
            chosen = (anchor >= start) & (anchor < stop)
            episode_ids[chosen] = episode["episode_index"]
            local_anchor[chosen] = anchor[chosen] - start
            assert (future_index[chosen] < stop).all(), "Forecast target crossed an episode boundary"
            timestamps = np.asarray(source[f"data/{episode['name']}/states"])[:, 0].astype(np.float64)
            np.testing.assert_allclose(np.diff(timestamps), 1 / dataset.sample_rate_hz, atol=1e-7, rtol=0)
            target_times[chosen] = timestamps[future_index[chosen] - start] - timestamps[0]
            scores = channel_metrics(prediction[chosen], truth[chosen])
            episode_metrics.append({"episode": episode["name"], "episode_index": episode["episode_index"],
                                    "windows": int(chosen.sum()), **scores})
    overall = channel_metrics(prediction, truth)
    by_horizon = {f"{(index + 1) / dataset.sample_rate_hz:.2f}s": channel_metrics(prediction[:, index], truth[:, index])
                  for index in range(8)}
    by_phase = {name: channel_metrics(prediction[phases == index], truth[phases == index]) for index, name in enumerate(PHASES)}
    raw_wrench = dataset.high_tensors["wrench"].numpy()
    force_changes = np.linalg.norm(raw_wrench[future_index, :3] - raw_wrench[future_index - 1, :3], axis=-1)
    change_threshold = float(np.quantile(force_changes, .9))
    sharp = force_changes >= change_threshold
    ranked = sorted(episode_metrics, key=lambda item: item["force_mse"], reverse=True)
    worst, typical = ranked[0], ranked[len(ranked) // 2]
    plot_specs = [
        (output, worst, 0, "Worst episode by force MSE"),
        (output / "worst_400ms", worst, 7, "Worst episode by force MSE"),
        (output / "typical_50ms", typical, 0, "Median episode by force MSE"),
    ]
    for folder, episode, horizon, description in plot_specs:
        selected = episode_ids == episode["episode_index"]
        draw_curves(folder, f"{episode['episode']} ({description})", horizon, selected,
                    prediction, truth, target_times)
    np.savez_compressed(output / "forecasts.npz", prediction=prediction, ground_truth=truth,
                        global_anchor=anchor, episode_id=episode_ids, local_anchor_frame=local_anchor,
                        local_target_frame=local_anchor[:, None] + np.arange(1, 9)[None],
                        target_time_s=target_times, future_offsets_s=np.arange(1, 9) / dataset.sample_rate_hz,
                        target_phase=phases, predicted_phase=predicted_phases,
                        channel_names=np.asarray(CHANNELS))
    combined_mse = (overall["force_mse"] + overall["moment_mse"]) / 2
    summary = {
        "checkpoint": str(checkpoint), "dataset": str(dataset_path), "output": str(output),
        "mode": "recorded-input forecast replay; no simulator steps", "validation_evidence": validation_evidence,
        "episodes": len(dataset.episodes), "windows": len(dataset), "seed": seed,
        "sampling": "every valid window once, without oversampling", "precision": "float32",
        "input_fields": list(STATE_KEYS), "ground_truth": "median-3 filtered wrench used to train this checkpoint",
        "time_alignment": "anchor t forecast for t+k*0.05 is compared with recorded wrench at t+k*0.05",
        "all_horizons": overall, "by_horizon": by_horizon, "by_target_phase": by_phase,
        "force_jump_diagnostic": {"top_decile_threshold_n_per_frame": change_threshold,
                                  "top_decile": channel_metrics(prediction[sharp], truth[sharp]),
                                  "other": channel_metrics(prediction[~sharp], truth[~sharp])},
        "future_phase_accuracy": float((phases == predicted_phases).mean()),
        "episodes_ranked_by_force_mse": ranked,
        "checkpoint_saved_metrics": old_metrics,
        "reproduced_combined_wrench_mse": combined_mse,
        "difference_from_checkpoint_saved_mse": combined_mse - old_metrics["full_dataset_wrench_mse"],
        "main_plots": {"episode": worst["episode"], "forecast_offset_s": .05,
                       "channels": [f"{channel}.png" for channel in CHANNELS], "overview": "overview.png"},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "channel_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["channel", *next(iter(overall["channels"].values())).keys()])
        writer.writeheader()
        writer.writerows({"channel": channel, **metrics} for channel, metrics in overall["channels"].items())
    (output / "README.md").write_text(
        "# CAD-WM wrench forecast replay\n\n"
        "This checkpoint trained on all 50 episodes; these figures are training-data replay, not held-out validation.\n\n"
        "Blue = recorded median-3 ground truth; yellow = model forecast. Each figure fixes a forecast horizon and aligns by target time. "
        "Input histories contain only q/dq/tau. Forecasts use full noise-to-wrench Heun integration and checkpoint normalization.\n\n"
        f"Root: six individual channels and overview for {worst['episode']} (highest force MSE across all eight horizons), +0.05 s.\n\n"
        f"worst_400ms/: the same episode at +0.40 s. typical_50ms/: {typical['episode']} (median force MSE), +0.05 s.\n\n"
        "summary.json contains errors by channel, horizon, phase and episode; forecasts.npz retains all aligned predictions and targets.\n"
    )
    print(json.dumps({"output": str(output), "windows": len(dataset), "episodes": len(dataset.episodes),
                      "force_mse": overall["force_mse"], "moment_mse": overall["moment_mse"],
                      "saved_mse_difference": summary["difference_from_checkpoint_saved_mse"],
                      "worst_episode": worst["episode"], "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
