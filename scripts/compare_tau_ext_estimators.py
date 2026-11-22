#!/usr/bin/env python3
"""Compare generalized momentum and causal Pinocchio RNEA torque estimates."""
from __future__ import annotations

import argparse
from collections import deque
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO))

import av
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq

from data_process.tau_ext_video import EpisodeVideoCursor, episode_video_spec
from diffusion_policy.common.momentum_observer import GeneralizedMomentumObserver
from diffusion_policy.common.xarm_tau_ext import CausalTauExt, TauExtConfig, gate_tau_ext


COLORS = {"rnea": "#1666aa", "momentum": "#d65a16"}
LABELS = {"rnea": "Pinocchio RNEA residual", "momentum": "Momentum observer (3 Hz)"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def plot_episode(output, task, episode, original_episode, times, estimates, threshold, gated=False):
    figure = Figure(figsize=(15, 10), dpi=150, facecolor="white")
    axes = figure.subplots(4, 2, sharex=True)
    stage = "gated" if gated else "unthresholded"
    for joint, axis in enumerate(axes.flat):
        for method, values in estimates.items():
            curve = values[f"tau_ext_{stage}"][:, joint] if joint < 7 else values[f"l1_{stage}"]
            axis.plot(times, curve, color=COLORS[method], label=LABELS[method], linewidth=1.2)
        axis.grid(alpha=.25)
        axis.set_xlim(times[0], times[-1])
        if joint < 7:
            axis.axhline(0, color="gray", linewidth=.6)
            axis.set_ylabel(f"J{joint + 1} torque (N m)")
        else:
            axis.axhline(threshold, color="#a33d3d", linestyle="--", linewidth=1,
                         label=f"L1 threshold = {threshold:g} N m")
            axis.set_ylabel("Seven-joint L1 (N m)")
        if joint >= 6:
            axis.set_xlabel("Episode time (s)")
    axes.flat[0].legend(fontsize=9)
    axes.flat[7].legend(fontsize=8)
    figure.suptitle(f"{task}: episode {episode} (original {original_episode})\n"
                   f"{'After L1 gate' if gated else 'Before L1 gate'} | causal inputs 3 Hz, trailing mean 5 frames | tool-on-environment", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    stem = f"episode_{episode:03d}_tau_ext_{'gated' if gated else 'comparison'}"
    figure.savefig(output / f"{stem}.png")
    figure.savefig(output / f"{stem}.pdf")


class LiveNormPlot:
    def __init__(self, times, estimates, threshold, width=960, height=300):
        self.height = height
        self.times, self.estimates = times, estimates
        self.font = ImageFont.truetype(findfont("DejaVu Sans"), 16)
        figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes([.075, .19, .9, .61])
        for method, values in estimates.items():
            axis.plot(times, values["l1_unthresholded"], color=COLORS[method],
                      label=LABELS[method], linewidth=1.5)
        axis.axhline(threshold, color="#a33d3d", linestyle="--", label=f"Threshold {threshold:g} N·m")
        axis.set_xlim(times[0], times[-1])
        axis.set_ylim(0, max(threshold, *(v["l1_unthresholded"].max() for v in estimates.values())) * 1.16)
        axis.set_xlabel("Episode time (s)")
        axis.set_ylabel(r"$\|\tau_{\mathrm{ext}}\|_1$ (N$\cdot$m)")
        axis.grid(alpha=.25)
        axis.legend(loc="upper right", fontsize=8)
        canvas.draw()
        self.background = Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        self.x = axis.transData.transform(np.column_stack((times, np.zeros(len(times)))))[:, 0]
        self.top, self.bottom = height - axis.bbox.ymax, height - axis.bbox.ymin

    def render(self, index):
        image = self.background.copy()
        draw = ImageDraw.Draw(image)
        draw.line((self.x[index], self.top, self.x[index], self.bottom), fill="black", width=2)
        draw.text((12, 8), f"t = {self.times[index]:.2f} s", fill="black", font=self.font)
        for name, x in [("rnea", 175), ("momentum", 530)]:
            label = "RNEA" if name == "rnea" else "Momentum"
            text = f"{label} ‖τ_ext‖₁ = {self.estimates[name]['l1_unthresholded'][index]:.2f} N·m"
            draw.text((x, 8), text, fill=COLORS[name], font=self.font)
        return image


def episode_video(root, info, metadata, times, estimates, output, threshold):
    keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    width = 320 * len(keys)
    plot = LiveNormPlot(times, estimates, threshold, width=width)
    font = plot.font
    cursors, target_times = [], []
    fps = int(info["fps"])
    try:
        for key in keys:
            relative, targets = episode_video_spec(info, metadata, key, times)
            cursor = EpisodeVideoCursor(root / relative, targets[0], fps)
            cursors.append(cursor)
            target_times.append(targets)
        with av.open(str(output), "w", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = width, 32 + 320 + 300
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 2
            stream.options = {"crf": "18", "preset": "veryfast"}
            for index in range(len(times)):
                panel = Image.new("RGB", (width, stream.height), "#171b21")
                draw = ImageDraw.Draw(panel)
                for camera, (key, cursor, targets) in enumerate(zip(keys, cursors, target_times)):
                    rgb = cursor.rgb_at(float(targets[index]))
                    tile = Image.fromarray(rgb).resize((320, 320), Image.Resampling.BILINEAR)
                    panel.paste(tile, (camera * 320, 32))
                    draw.text((camera * 320 + 10, 8), key.removeprefix("observation.images."), fill="white", font=font)
                panel.paste(plot.render(index), (0, 352))
                frame = av.VideoFrame.from_ndarray(np.asarray(panel), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    finally:
        for cursor in cursors:
            cursor.close()
    with av.open(str(output)) as container:
        assert container.streams.video[0].frames == len(times)
    return {"file": output.name, "frames": len(times), "fps": fps, "cameras": keys,
            "norm_label": "||tau_ext||_1", "norm_definition": "sum(abs(tau_ext[J1:J7]))", "units": "N m",
            "curves": "MA5 before threshold", "synchronization": "source frame PTS matched to state timestamp"}


def process_dataset(root, output, args):
    root = root.resolve()
    output.mkdir()
    info = json.loads((root / "meta/info.json").read_text())
    table = pa.concat_tables([pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))])
    metadata = pq.read_table(sorted((root / "meta/episodes").rglob("*.parquet"))).to_pylist()
    q = np.asarray(table["observation.joint"].to_pylist(), dtype=np.float64)
    dq = np.asarray(table["observation.velocity"].to_pylist(), dtype=np.float64)
    tau = np.asarray(table["observation.torque"].to_pylist(), dtype=np.float64)
    times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    ids = table["episode_index"].to_numpy()
    config = TauExtConfig(fps=info["fps"], cutoff_hz=3., contact_threshold_l1=args.threshold,
                          tau_ext_smoothing_window=5, convention="tool-on-environment")
    methods = {name: {"tau_ext_raw": np.empty_like(q), "tau_ext_unthresholded": np.empty_like(q),
                      "tau_ext_gated": np.empty_like(q), "l1_unthresholded": np.empty(len(q)),
                      "l1_gated": np.empty(len(q))} for name in ["rnea", "momentum"]}
    episode_metrics = []
    for row in metadata:
        ep = int(row["episode_index"])
        indices = np.flatnonzero(ids == ep)
        processor = CausalTauExt(args.urdf, config)
        observer = GeneralizedMomentumObserver(args.urdf, config, args.observer_bandwidth_hz)
        history = deque(maxlen=5)
        for index in indices:
            inverse = processor.update(q[index], dq[index], tau[index], times[index])
            momentum = observer.update(q[index], inverse["dq_filtered"], inverse["tau_filtered"], times[index])
            history.append(momentum["tau_ext_raw"].astype(np.float32))
            smoothed = np.mean(history, axis=0, dtype=np.float64).astype(np.float32)
            gated, l1, _ = gate_tau_ext(smoothed, args.threshold)
            methods["momentum"]["tau_ext_raw"][index] = momentum["tau_ext_raw"]
            methods["momentum"]["tau_ext_unthresholded"][index] = smoothed
            methods["momentum"]["tau_ext_gated"][index] = gated
            methods["momentum"]["l1_unthresholded"][index] = l1
            methods["momentum"]["l1_gated"][index] = np.abs(gated.astype(np.float64)).sum()
            methods["rnea"]["tau_ext_raw"][index] = (inverse["tau_filtered"]
                                                               - np.asarray(config.torque_bias) - inverse["tau_free"])
            methods["rnea"]["tau_ext_unthresholded"][index] = inverse["tau_ext_unthresholded"]
            methods["rnea"]["tau_ext_gated"][index] = inverse["tau_ext"]
            methods["rnea"]["l1_unthresholded"][index] = inverse["tau_ext_l1"]
            methods["rnea"]["l1_gated"][index] = np.abs(inverse["tau_ext"].astype(np.float64)).sum()
        metric = {"episode_index": ep, "frames": len(indices)}
        for name, values in methods.items():
            norms = values["l1_unthresholded"][indices]
            metric.update({f"{name}_mean_l1_Nm": float(norms.mean()),
                           f"{name}_max_l1_Nm": float(norms.max()),
                           f"{name}_fraction_above_threshold": float((norms >= args.threshold).mean())})
        episode_metrics.append(metric)
        print(f"{output.name}: episode {ep + 1}/{len(metadata)} estimated ({len(indices)} frames)", flush=True)
    for method, values in methods.items():
        for key, value in values.items():
            assert np.isfinite(value).all(), (method, key)
            arrow = pa.array(value.tolist(), type=pa.list_(pa.float32(), 7) if value.ndim == 2 else pa.float64())
            table = table.append_column(f"estimate.{method}.{key}", arrow)
    pq.write_table(table.replace_schema_metadata(None), output / "all_episodes_tau_ext.parquet", compression="snappy")
    with (output / "episode_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_metrics[0]))
        writer.writeheader()
        writer.writerows(episode_metrics)
    indices = np.flatnonzero(ids == args.episode)
    if not len(indices):
        raise ValueError(f"Episode {args.episode} absent")
    chosen = {name: {key: value[indices] for key, value in values.items()} for name, values in methods.items()}
    ep_times = times[indices] - times[indices[0]]
    manifest_path = root / "meta/selection_manifest.json"
    selection = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    original_episode = next((e["original_episode_index"] for e in selection.get("episodes", [])
                             if e["episode_index"] == args.episode), args.episode)
    LABELS["momentum"] = f"Momentum observer ({args.observer_bandwidth_hz:g} Hz)"
    for gated in [False, True]:
        plot_episode(output, output.name, args.episode, original_episode, ep_times, chosen, args.threshold, gated)
    csv_columns = {"timestamp_s": ep_times, "frame_index": table["frame_index"].to_numpy()[indices]}
    for method, values in chosen.items():
        for key, value in values.items():
            if value.ndim == 2:
                csv_columns.update({f"{method}_{key}_J{joint + 1}_Nm": value[:, joint] for joint in range(7)})
            else:
                csv_columns[f"{method}_{key}_Nm"] = value
    with (output / f"episode_{args.episode:03d}_tau_ext.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        writer.writerows(zip(*csv_columns.values()))
    chosen_metadata = next(row for row in metadata if row["episode_index"] == args.episode)
    video = episode_video(root, info, chosen_metadata, ep_times, chosen,
                          output / f"episode_{args.episode:03d}_three_cameras_tau_ext.mp4", args.threshold)
    record = {"source_dataset": str(root), "total_episodes": len(metadata), "total_frames": len(q),
              "visualized_episode": args.episode, "original_episode_index": original_episode,
              "config": config.to_dict(), "pinocchio_version": pin.__version__,
              "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
              "observer_bandwidth_hz": args.observer_bandwidth_hz,
              "observer_gain_per_s": 2 * np.pi * args.observer_bandwidth_hz,
              "observer_definition": "p=M(q)*dq; beta=tau_motor+C.T*dq-g; p_hat_dot=beta+r; r=K*(p-p_hat); tool torque=-r",
              "observer_discretization": "stable exponential update for linearly interpolated momentum and interval-average beta",
              "observer_initialization": "p_hat=p at each episode first frame; residual zero",
              "rnea_definition": "tau_motor_filtered-bias-RNEA(q,dq_filtered,backward_difference(dq_filtered))",
              "shared_preprocessing": "q unchanged; dq and motor torque causal 2nd-order Butterworth 3 Hz, steady-state initial conditions; signed output trailing mean 5 frames; seven-axis L1 gate 12.5 N m",
              "interpretation": "Model-based estimates, not force-sensor ground truth. Both include errors from uncalibrated friction, torque bias and payload. Observer adds its own response delay.",
              "results_are": "standalone comparison results, not modifications of the source training datasets",
              "video": video, "episode_metrics": episode_metrics[args.episode]}
    write_json(output / "processing.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--observer-bandwidth-hz", type=float, default=3.)
    parser.add_argument("--threshold", type=float, default=12.5)
    parser.add_argument("--urdf", type=Path, default=REPO / "scripts/assets/xarm7/xarm7_dynamics.urdf")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.urdf, args.output / "robot.urdf")
    results = []
    for root in args.dataset:
        task = root.name.removeprefix("xarm_").split("_25hz")[0]
        results.append(process_dataset(root, args.output / task, args))
    write_json(args.output / "summary.json", results)
    shutil.copy2(Path(__file__), args.output / "compare_tau_ext_estimators.py")
    shutil.copy2(REPO / "diffusion_policy/common/momentum_observer.py", args.output / "momentum_observer.py")
    print("Finished:", args.output, flush=True)


if __name__ == "__main__":
    main()
