"""Compare FiLM on/off for one RenderNet episode in a three-row video.

Rows 1 and 2 show each complete rendering path: input, predicted intensity,
applied gate, rendered RGB and residual. Row 3 plots the RGB MAE and RMSE
between both outputs, using floats before resizing or video encoding.
"""

from __future__ import annotations

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from omegaconf import OmegaConf, open_dict

from diffusion_policy.model.vision.wrench_preprocessing import compute_contact_alpha
from diffusion_policy.scripts import visualize_render_net_episode as visual


BRANCHES = ("film_on", "film_off")
METRICS = ("mae", "mse", "rmse", "max_abs_delta")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="RenderNet student checkpoint with FiLM enabled.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", choices=("wrist", "side", "both"), default="both")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--device", help="Default: checkpoint device, with CPU fallback.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, help="Default: dataset FPS divided by stride.")
    parser.add_argument("--video-codec", choices=("h264", "av1", "mpeg4"), default="h264")
    parser.add_argument("--panel-width", type=int, default=224, help="Display width per panel; metrics use original resolution.")
    parser.add_argument("--difference-gain", type=float, default=5.0, help="Display gain for residual panels only.")
    parser.add_argument("--output", type=Path, help="Output MP4; default: checkpoint run's attention_visualize directory.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.episode < 0 or args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--episode must be nonnegative; --stride and --max-frames must be positive")
    if args.panel_width < 128:
        parser.error("--panel-width must be at least 128")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    if not math.isfinite(args.difference_gain) or args.difference_gain <= 0:
        parser.error("--difference-gain must be positive and finite")
    if args.output is not None and args.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    return args


def output_paths(checkpoint, args):
    video = args.output.expanduser().resolve() if args.output else (
        checkpoint.parent.parent / "attention_visualize"
        / f"render_net_film_episode_{args.episode}_{args.camera}_comparison.mp4"
    )
    return {
        "comparison": video,
        "preview": video.with_name(video.stem + "_preview.png"),
        "metrics": video.with_name(video.stem + "_metrics.csv"),
        "metadata": video.with_name(video.stem + "_metadata.json"),
    }


@torch.inference_mode()
def compare_observations(model, preprocessor, obs, image_keys):
    """Run identical inputs twice, bypassing only decoder spatial modulation."""
    if not model.spatial_modulation_enabled or not len(model.spatial_modulations):
        raise ValueError("This checkpoint has no enabled FiLM path (render_net.spatial_modulation.enabled)")
    images = {key: obs[key][:, :preprocessor.n_obs_steps] for key in image_keys}
    predict_phase = bool(getattr(model, "phase_prediction_enabled", False))
    preprocessing_obs = ({key: value for key, value in obs.items()
                          if key not in {"contact_alpha", "contact_phase"}}
                         if predict_phase else obs)
    wrench, contact, future_contact = preprocessor(preprocessing_obs, return_future_contact=True)
    contact_gate = future_contact if future_contact is not None else contact
    if predict_phase:
        # Share one causal prediction so the comparison changes only FiLM.
        alpha = model.predict_contact_phase(images, wrench)["contact_alpha"]
    else:
        alpha = (obs["contact_alpha"][:, :preprocessor.n_obs_steps] if "contact_alpha" in obs else
                 compute_contact_alpha(contact_gate if contact_gate.ndim == 3 else contact_gate.unsqueeze(-1)))
    branches = {}
    original_setting = model.spatial_modulation_enabled
    try:
        for name, enabled in zip(BRANCHES, (True, False)):
            model.spatial_modulation_enabled = enabled
            rendered, masks = model(images, wrench, return_aux=True, future_contact=contact_gate,
                                    contact_alpha=alpha)
            gates, intensities = {}, {}
            for key, image in images.items():
                mask = masks.get(key, torch.ones_like(image[:, :, :1]))
                flat_gate = model._mask_gate(mask.flatten(0, 1), image) if model.high_attention_enabled else mask.flatten(0, 1)
                gate = flat_gate.reshape_as(mask)
                if contact_gate is not None:
                    gate = gate * alpha.to(gate.dtype)[..., None, None, None]
                intensities[key], gates[key] = mask, gate
            branches[name] = {"rendered": rendered, "intensity": intensities, "gate": gates}
    finally:
        model.spatial_modulation_enabled = original_setting
    return images, branches


def difference_metrics(branches, camera_keys):
    result = {}
    for key in camera_keys:
        delta = (branches["film_on"]["rendered"][key][0, -1].float()
                 - branches["film_off"]["rendered"][key][0, -1].float())
        mse = float(delta.square().mean())
        result[key] = {"mae": float(delta.abs().mean()), "mse": mse, "rmse": math.sqrt(mse),
                       "max_abs_delta": float(delta.abs().max()), "num_values": delta.numel()}
    count = sum(item["num_values"] for item in result.values())
    mse = sum(item["mse"] * item["num_values"] for item in result.values()) / count
    result["overall"] = {
        "mae": sum(item["mae"] * item["num_values"] for item in result.values()) / count,
        "mse": mse, "rmse": math.sqrt(mse),
        "max_abs_delta": max(item["max_abs_delta"] for item in result.values()), "num_values": count,
    }
    return result


def _field_rgb(field):
    gray = (field[0].detach().float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def path_row(images, branch, camera_keys, *, title, panel_width, difference_gain):
    panels = []
    for key in camera_keys:
        original = images[key][0, -1].float()
        rendered = branch["rendered"][key][0, -1].float()
        views = (
            ("Input RGB", visual._to_rgb(original)),
            ("Intensity M [0,1]", _field_rgb(branch["intensity"][key][0, -1])),
            ("Applied gate [0,1]", _field_rgb(branch["gate"][key][0, -1])),
            ("Rendered RGB", visual._to_rgb(rendered)),
            (f"RGB residual x{difference_gain:g}", visual._to_rgb(0.5 + difference_gain * (rendered - original))),
        )
        for label, rgb in views:
            height = max(1, round(rgb.shape[0] * panel_width / rgb.shape[1]))
            resized = cv2.resize(rgb, (panel_width, height), interpolation=cv2.INTER_AREA)
            panels.append(visual._label(resized, f"{key} | {label}"))
    return visual._label(visual._tile(panels, horizontal=True), title)


class DifferencePlot:
    """Progressive curves with a synchronized cursor and RGB-unit axes."""

    def __init__(self, camera_keys, width, end_time, height=260):
        self.camera_keys = camera_keys
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(1, 2)
        self.figure.subplots_adjust(left=0.045, right=0.985, bottom=0.20, top=0.76, wspace=0.17)
        self.times = []
        self.values = {key: {metric: [] for metric in ("mae", "rmse")} for key in camera_keys}
        self.lines, self.cursors = {}, []
        for ax, metric in zip(self.axes, ("mae", "rmse")):
            for key in camera_keys:
                self.lines[key, metric], = ax.plot([], [], label=key, linewidth=1.4, marker=".", markersize=2)
            ax.set(xlim=(0, max(end_time, 0.05)), ylim=(0, 1e-6),
                   xlabel="Episode time (s)", ylabel=f"RGB {metric.upper()} [0,1]")
            ax.grid(alpha=0.25)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            ax.legend(loc="upper left", fontsize=8)
            self.cursors.append(ax.axvline(0, color="#333333", linestyle="--", linewidth=1))
        self.status = self.figure.text(0.045, 0.91, "", fontsize=10)

    def frame(self, seconds, metrics):
        self.times.append(seconds)
        for ax, cursor, metric in zip(self.axes, self.cursors, ("mae", "rmse")):
            for key in self.camera_keys:
                self.values[key][metric].append(metrics[key][metric])
                self.lines[key, metric].set_data(self.times, self.values[key][metric])
            peak = max(max(self.values[key][metric]) for key in self.camera_keys)
            ax.set_ylim(0, max(peak * 1.15, 1e-6))
            cursor.set_xdata([seconds, seconds])
        overall = metrics["overall"]
        self.status.set_text(
            f"FiLM ON - OFF | float RGB before encoding | t={seconds:.2f}s | "
            f"MAE={overall['mae']:.6g}  RMSE={overall['rmse']:.6g}  max={overall['max_abs_delta']:.6g}"
        )
        self.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[..., :3])


def summarize(records, keys):
    result = {}
    for key in keys:
        count = sum(record[key]["num_values"] for record in records)
        mse = sum(record[key]["mse"] * record[key]["num_values"] for record in records) / count
        result[key] = {
            "mae": sum(record[key]["mae"] * record[key]["num_values"] for record in records) / count,
            "mse": mse, "rmse": math.sqrt(mse),
            "max_abs_delta": max(record[key]["max_abs_delta"] for record in records), "num_values": count,
        }
    return result


def run(args):
    checkpoint = visual.common.choose_checkpoint(args.checkpoint, None)
    model, preprocessor, cfg, teacher_cfg, device, info = visual.load_render_net(checkpoint, args.device)
    if not model.spatial_modulation_enabled or not len(model.spatial_modulations):
        raise ValueError("Choose a student checkpoint trained with render_net.spatial_modulation.enabled=true")
    image_keys = list(model.image_keys)
    camera_keys = [key for key in ("wrist", "side") if key in image_keys]
    camera_keys.extend(key for key in image_keys if key not in camera_keys)
    if args.camera != "both":
        if args.camera not in image_keys:
            raise KeyError(f"Camera {args.camera!r} unavailable: {image_keys}")
        camera_keys = [args.camera]
    source_path = checkpoint.parent.parent / "teacher_source_config.yaml"
    source = OmegaConf.load(source_path) if source_path.is_file() else teacher_cfg
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    with open_dict(dataset_cfg):
        dataset_cfg.task.dataset.update(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset = visual.common.instantiate_episode_dataset(dataset_cfg, args.dataset_path, args.video_backend)
    writer = None
    paths = output_paths(checkpoint, args)
    try:
        selected = visual.common.episode_sample_indices(dataset, args.episode)[::args.stride]
        if args.max_frames is not None:
            selected = selected[:args.max_frames]
        dataset_fps = visual.common.infer_dataset_fps(dataset)
        fps = args.fps if args.fps is not None else dataset_fps / args.stride
        times = [(anchor - selected[0][0]) / dataset_fps for anchor, _ in selected]
        visual.rendering.prepare_outputs(paths, overwrite=args.overwrite, save_frames=False)
        amp_enabled = device.type == "cuda" and bool(cfg.training.amp.enabled)
        amp_dtype = visual.resolve_amp_dtype(cfg.training.amp.dtype)
        records, plot = [], None
        metric_keys = [*camera_keys, "overall"]
        columns = ["frame", "anchor", "sample_index", "episode_time_sec"]
        columns += [f"{key}_{metric}" for key in metric_keys for metric in METRICS]
        print(f"RenderNet: {checkpoint}; weights={info['student_state_key']}; device={device}\n"
              f"Episode {args.episode}: {len(selected)} frames at {fps:g} FPS; cameras={camera_keys}", flush=True)
        with paths["metrics"].open("w", newline="") as stream:
            table = csv.DictWriter(stream, fieldnames=columns)
            table.writeheader()
            for index, ((anchor, sample_index), seconds) in enumerate(zip(selected, times)):
                sample = dataset[sample_index]
                obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()
                       if key in image_keys or key in preprocessor.required_keys}
                autocast = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with autocast:
                    images, branches = compare_observations(model, preprocessor, obs, image_keys)
                metrics = difference_metrics(branches, camera_keys)
                records.append(metrics)
                table.writerow({"frame": index, "anchor": anchor, "sample_index": sample_index,
                                "episode_time_sec": seconds,
                                **{f"{key}_{metric}": metrics[key][metric]
                                   for key in metric_keys for metric in METRICS}})
                rows = []
                for name in BRANCHES:
                    film = "ON" if name == "film_on" else "OFF (identity)"
                    rows.append(path_row(
                        images, branches[name], camera_keys, panel_width=args.panel_width,
                        difference_gain=args.difference_gain,
                        title=f"FiLM {film} | RGB+wrench -> CNN/LSTM -> self/cross-attention -> decoder -> gates -> RGB"
                              f" | ep {args.episode} | {index + 1}/{len(selected)} | {seconds:.2f}s",
                    ))
                if plot is None:
                    plot = DifferencePlot(camera_keys, rows[0].shape[1], times[-1])
                rows.append(plot.frame(seconds, metrics))
                frame = visual._tile(rows, horizontal=False)
                frame = np.ascontiguousarray(np.pad(frame, ((0, frame.shape[0] % 2),
                                                           (0, frame.shape[1] % 2), (0, 0))))
                if writer is None:
                    writer = visual.common.FFmpegVideoWriter(
                        paths["comparison"], frame.shape[1], frame.shape[0], fps, args.video_codec,
                    )
                    video_shape = list(frame.shape)
                    row_heights = [row.shape[0] for row in rows]
                writer.write(frame)
                if index == len(selected) // 2:
                    visual._save_rgb(paths["preview"], frame)
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(selected):
                    print(f"Compared {index + 1}/{len(selected)} frames; "
                          f"MAE={metrics['overall']['mae']:.6g}, RMSE={metrics['overall']['rmse']:.6g}", flush=True)
        finished_writer, writer = writer, None
        finished_writer.close()
        summary = summarize(records, metric_keys)
        metadata = {
            "checkpoint": str(checkpoint), **info, "episode": args.episode,
            "video_frames": len(selected), "fps": fps, "dataset_fps": dataset_fps,
            "stride": args.stride, "video_shape_hwc": video_shape, "row_heights": row_heights,
            "rows": ["FiLM on rendering path", "FiLM off rendering path", "RGB MAE/RMSE difference curves"],
            "panels_per_camera": ["input RGB", "predicted intensity", "applied contact/region gate", "rendered RGB", "RGB residual"],
            "input_camera_keys": image_keys, "display_camera_keys": camera_keys,
            "display_observation": "last", "observation_steps": preprocessor.n_obs_steps,
            "ablation": "Bypass decoder spatial_modulations; identical weights, inputs, wrench preprocessing, attention and gates",
            "spatial_modulation_layers": len(model.spatial_modulations),
            "region_threshold": model.region_threshold, "high_attention_render_enabled": model.high_attention_enabled,
            "difference": "film_on.rendered - film_off.rendered",
            "metrics_scope": "latest observation of displayed cameras, original resolution, float RGB in [0,1], before resizing/encoding",
            "overall_weighting": "all RGB elements, weighted by each camera's pixel count",
            "curve_axes": "fixed episode-time range; RGB error range expands with observed maximum",
            "difference_gain": args.difference_gain, "gain_scope": "residual display panels only; neutral gray means zero",
            "summary": summary,
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Video: {paths['comparison']}\nMetrics: {paths['metrics']}\n"
              f"Overall MAE={summary['overall']['mae']:.8g}, RMSE={summary['overall']['rmse']:.8g}", flush=True)
        return paths
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            dataset.lerobot_dataset.close()


if __name__ == "__main__":
    run(parse_args())
