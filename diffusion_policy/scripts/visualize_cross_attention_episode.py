"""Visualize force-to-image cross-attention for one LeRobot episode.

The script loads the EMA policy from a force-aware diffusion checkpoint,
replays the valid observation windows of one episode, and writes an MP4 with
the latest observation image overlaid by the attention paid to its DINO patch
tokens.  The attention is averaged over heads and the force-history query
tokens by default.  The output defaults to the dataset's AV1/yuv420p MP4
format and frame rate.  Heatmaps use one robust color range across the
selected episode by default, so colors remain comparable between frames;
use ``--attention-normalization frame`` to rescale each frame separately.

Example:
    python diffusion_policy/scripts/visualize_cross_attention_episode.py \
        --checkpoint-dir data/outputs/2026.10.17/21.11.38_train_dp_insert_usb_force_aware_insert_usb_force_aware/checkpoints \
        --episode 3 \
        --output data/outputs/attention_episode_3.mp4 \
        --camera both \
        --save-frames \
        --overwrite
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
import copy
import contextlib
import dill
import math
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render cross-attention heatmaps for one force-aware DP episode."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        help="Path to one .ckpt file. If omitted, choose the latest checkpoint in --checkpoint-dir.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        help="Checkpoint directory; the largest optimizer_step checkpoint is selected.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Zero-based episode number in the LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=None,
        help="Optional dataset override. By default use the checkpoint config.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "torchcodec"),
        default="torchcodec",
        help="LeRobot video decoder. Use pyav as a fallback if TorchCodec is unavailable.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output MP4 path. Defaults beside the checkpoint directory as attention_episode_<N>.mp4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output MP4 and its saved-frame directory.",
    )
    parser.add_argument(
        "--camera",
        choices=("wrist", "side", "both"),
        default="both",
        help="Camera(s) to render. 'both' concatenates the two overlays horizontally.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate. Defaults to the dataset frame rate (50 Hz for this dataset).",
    )
    parser.add_argument(
        "--video-codec",
        choices=("av1", "h264", "mpeg4"),
        default="av1",
        help="FFmpeg video codec. The dataset uses AV1 in an MP4 container, so AV1 is the default.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--attention-query",
        choices=("all", "last"),
        default="all",
        help="Average all force-history queries or use only the latest query token.",
    )
    parser.add_argument(
        "--attention-normalization",
        choices=("global", "frame"),
        default="global",
        help=(
            "Normalize heatmaps using one range across the selected episode "
            "(global) or independently per frame (frame)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0 or cpu. Defaults to the checkpoint training device when available.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save individual heatmap PNGs beside the MP4.",
    )
    return parser.parse_args()


def choose_checkpoint(
    checkpoint: Optional[pathlib.Path],
    checkpoint_dir: Optional[pathlib.Path],
) -> pathlib.Path:
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    assert checkpoint_dir is not None
    directory = checkpoint_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    candidates = list(directory.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {directory}")

    def checkpoint_step(path: pathlib.Path) -> int:
        match = re.search(r"optimizer_step=(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=lambda path: (checkpoint_step(path), path.stat().st_mtime))


def load_checkpoint_payload(checkpoint_path: pathlib.Path):
    payload = torch.load(
        checkpoint_path.open("rb"),
        map_location="cpu",
        pickle_module=dill,
    )
    cfg = payload.get("cfg")
    if cfg is None:
        config_path = checkpoint_path.parent.parent.joinpath(".hydra", "config.yaml")
        if not config_path.is_file():
            raise KeyError("Checkpoint has no cfg and its sibling .hydra/config.yaml is missing")
        cfg = OmegaConf.load(config_path)
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    return payload, cfg


def resolve_device(requested: Optional[str], cfg) -> torch.device:
    if requested:
        device = torch.device(requested)
    else:
        configured = str(OmegaConf.select(cfg, "training.device", default="cuda:0"))
        device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    return device


def instantiate_policy(payload, cfg, device: torch.device):
    policy: ForceAwareDiffusionTransformerPolicy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload.get("state_dicts", {})
    state_key = "ema_model" if "ema_model" in state_dicts else "model"
    if state_key not in state_dicts:
        raise KeyError(f"Checkpoint state_dicts has neither 'ema_model' nor 'model': {list(state_dicts)}")
    policy.load_state_dict(state_dicts[state_key], strict=True)

    pickles = payload.get("pickles", {})
    if "optimizer_step" in pickles:
        policy.set_optimizer_step(int(dill.loads(pickles["optimizer_step"])))
    policy.to(device)
    policy.eval()
    policy.obs_encoder.set_attention_capture(True)
    return policy


def instantiate_episode_dataset(
    cfg,
    dataset_override: Optional[pathlib.Path],
    video_backend: str,
):
    dataset_cfg = copy.deepcopy(cfg.task.dataset)
    with open_dict(dataset_cfg):
        # Include every episode so --episode is independent of the training
        # validation split.  Disable augmentation and whole-dataset preload;
        # only the requested episode is iterated below.
        dataset_cfg.val_ratio = 0.0
        dataset_cfg.max_train_episodes = None
        dataset_cfg.preload_images = False
        dataset_cfg.image_randomer_config = None
        dataset_cfg.video_backend = video_backend
        if dataset_override is not None:
            dataset_cfg.dataset_path = str(dataset_override.expanduser().resolve())
    return hydra.utils.instantiate(dataset_cfg)


def episode_sample_indices(dataset, episode: int) -> List[Tuple[int, int]]:
    episode_count = len(dataset.episode_ranges)
    if episode < 0 or episode >= episode_count:
        raise IndexError(f"episode={episode} is outside [0, {episode_count})")

    selected = []
    for sample_index, row in enumerate(dataset.indices):
        anchor = dataset._anchor_index_from_sampler_row(row)
        if int(dataset.episode_index[anchor]) == episode:
            selected.append((int(anchor), int(sample_index)))
    selected.sort(key=lambda item: item[0])
    if not selected:
        raise RuntimeError(
            f"Episode {episode} has no valid inference windows. "
            "The future wrench/tau_ext context may remove its final frames."
        )
    return selected


def patch_grid(
    patch_count: int,
    image_height: int,
    image_width: int,
    backbone,
) -> Tuple[int, int]:
    patch_size = getattr(getattr(backbone, "config", None), "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    elif patch_size is not None:
        patch_h = patch_w = int(patch_size)
    else:
        patch_h = patch_w = 16
    grid_h = image_height // patch_h
    grid_w = image_width // patch_w
    if grid_h * grid_w == patch_count:
        return grid_h, grid_w

    # Fall back to the factor pair whose aspect ratio is closest to the image.
    candidates = []
    for height in range(1, int(math.sqrt(patch_count)) + 1):
        if patch_count % height:
            continue
        width = patch_count // height
        candidates.extend(((height, width), (width, height)))
    if not candidates:
        raise RuntimeError(f"Cannot reshape {patch_count} DINO patches into a grid")
    return min(
        candidates,
        key=lambda pair: abs((pair[0] / pair[1]) - (image_height / image_width)),
    )


def normalize_heatmap(
    values: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if bounds is None:
        low = float(np.percentile(values, 1.0))
        high = float(np.percentile(values, 99.0))
    else:
        low, high = (float(bounds[0]), float(bounds[1]))
    if high <= low + 1e-8:
        low = float(values.min())
        high = float(values.max())
    if high <= low + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def overlay_attention(
    image: np.ndarray,
    attention,
    grid: Tuple[int, int],
    alpha: float,
    normalization_bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if torch.is_tensor(attention):
        attention = attention.detach().float().cpu().numpy()
    else:
        attention = np.asarray(attention, dtype=np.float32)
    attention = normalize_heatmap(
        attention.reshape(*grid),
        bounds=normalization_bounds,
    )
    attention = cv2.resize(
        attention,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    color = cv2.applyColorMap(np.uint8(attention * 255.0), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = np.clip(
        image.astype(np.float32) * (1.0 - alpha)
        + color.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return np.ascontiguousarray(blended)


def image_to_uint8(image_chw: torch.Tensor) -> np.ndarray:
    """Convert one normalized CHW image to a compact CPU RGB array."""
    image = image_chw.detach().float().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))


def attention_bounds(
    records: List[Dict[str, object]],
) -> Optional[Tuple[float, float]]:
    """Compute one robust color range over all selected camera/frame maps."""
    values = []
    for record in records:
        for camera in record["cameras"]:
            values.append(camera["attention"])
    if not values:
        return None
    flattened = np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    )
    flattened = np.nan_to_num(flattened, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(flattened, 1.0))
    high = float(np.percentile(flattened, 99.0))
    if high <= low + 1e-8:
        low = float(flattened.min())
        high = float(flattened.max())
    return low, high


def _reduce_attention_tensor(
    weights: torch.Tensor,
    observation_steps: int,
    obs_step: int,
    query_mode: str,
) -> torch.Tensor:
    if weights is None or weights.ndim != 4:
        raise RuntimeError(
            "Cross-attention capture returned no weights or an unexpected shape; "
            f"got {None if weights is None else tuple(weights.shape)}"
        )
    if weights.shape[0] % observation_steps != 0:
        raise RuntimeError(
            f"Attention batch dimension {weights.shape[0]} is not divisible by "
            f"n_obs_steps={observation_steps}"
        )
    batch_size = weights.shape[0] // observation_steps
    if batch_size != 1:
        raise RuntimeError("Episode visualization expects one sample per inference call")

    # The encoder uses wrench tokens as Q and image patches as K/V.  Therefore
    # each row is "which image patches this wrench token attends to":
    # [B*To, heads, wrench_history, image_patches].
    # Reshape it to [To, heads, wrench_history, image_patches].
    weights = weights.reshape(batch_size, observation_steps, *weights.shape[1:])[0]
    weights = weights[obs_step]
    if query_mode == "last":
        weights = weights[:, -1, :]
    else:
        weights = weights.mean(dim=1)
    return weights.mean(dim=0)


def attention_by_camera(
    weights,
    policy,
    obs_step: int,
    query_mode: str,
) -> Dict[str, torch.Tensor]:
    observation_steps = int(policy.n_obs_steps)
    camera_keys = tuple(policy.obs_encoder.image_keys)
    if isinstance(weights, dict):
        # New per-camera mode: each map was normalized over that camera's
        # patches by its own cross-attention call.
        missing = [key for key in camera_keys if key not in weights]
        if missing:
            raise RuntimeError(
                f"Cross-attention capture is missing camera weights for {missing}"
            )
        return {
            key: _reduce_attention_tensor(
                weights[key], observation_steps, obs_step, query_mode
            )
            for key in camera_keys
        }

    # Legacy/joint mode: one softmax covered all camera patches, so split the
    # final patch axis without renormalizing the camera slices.
    weights = _reduce_attention_tensor(
        weights, observation_steps, obs_step, query_mode
    )
    if not camera_keys or weights.shape[-1] % len(camera_keys) != 0:
        raise RuntimeError(
            f"Attention has {weights.shape[-1]} image tokens for cameras {camera_keys}"
        )
    patches_per_camera = weights.shape[-1] // len(camera_keys)
    return {
        key: weights[index * patches_per_camera : (index + 1) * patches_per_camera]
        for index, key in enumerate(camera_keys)
    }


def display_camera_keys(policy, selection: str) -> List[str]:
    available = set(policy.obs_encoder.image_keys)
    if selection != "both":
        if selection not in available:
            raise KeyError(f"Camera {selection!r} is not available; found {sorted(available)}")
        return [selection]
    preferred = [key for key in ("wrist", "side") if key in available]
    preferred.extend(key for key in policy.obs_encoder.image_keys if key not in preferred)
    return preferred


def infer_dataset_fps(dataset) -> float:
    """Read the dataset FPS, falling back to its timestamp spacing."""
    candidates = [
        getattr(dataset, "fps", None),
        getattr(getattr(dataset, "lerobot_dataset", None), "fps", None),
    ]
    for value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value

    timestamps = getattr(dataset, "timestamps", None)
    if timestamps is not None:
        differences = np.diff(np.asarray(timestamps, dtype=np.float64))
        differences = differences[np.isfinite(differences) & (differences > 0)]
        if differences.size:
            step = float(np.median(differences))
            if step > 0:
                return 1.0 / step
    return 25.0


class FFmpegVideoWriter:
    """Write RGB frames to an MP4 using the requested codec."""

    def __init__(
        self,
        output_path: pathlib.Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
    ):
        ffmpeg_candidates = [
            shutil.which("ffmpeg"),
            str(pathlib.Path(sys.executable).with_name("ffmpeg")),
        ]
        ffmpeg = next(
            (
                candidate
                for candidate in ffmpeg_candidates
                if candidate and pathlib.Path(candidate).is_file()
            ),
            None,
        )
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required to write the dataset-compatible MP4. "
                "Install it in the active environment, for example: "
                "conda install -c conda-forge ffmpeg"
            )

        codec_args = {
            # LeRobot's metadata declares video.codec=av1 and video.pix_fmt=yuv420p.
            "av1": ["-c:v", "libsvtav1", "-preset", "8", "-crf", "30"],
            "h264": ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
            "mpeg4": ["-c:v", "mpeg4", "-q:v", "3"],
        }[codec]
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{int(width)}x{int(height)}",
            "-framerate",
            str(float(fps)),
            "-i",
            "pipe:0",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(float(fps)),
            "-an",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is closed")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
        except BrokenPipeError as exc:
            message = self.process.stderr.read().decode("utf-8", errors="replace")
            self.process.wait()
            raise RuntimeError(f"FFmpeg stopped while writing the video: {message}") from exc

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        message = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed to encode the video (exit code {return_code}): {message}"
            )


def run(args: argparse.Namespace) -> pathlib.Path:
    checkpoint_path = choose_checkpoint(args.checkpoint, args.checkpoint_dir)
    payload, cfg = load_checkpoint_payload(checkpoint_path)
    device = resolve_device(args.device, cfg)
    policy = instantiate_policy(payload, cfg, device)
    dataset = instantiate_episode_dataset(
        cfg,
        args.dataset_path,
        video_backend=args.video_backend,
    )
    selected = episode_sample_indices(dataset, args.episode)
    selected = selected[:: max(1, int(args.stride))]
    if args.max_frames is not None:
        selected = selected[: max(0, int(args.max_frames))]
    if not selected:
        raise RuntimeError("No frames remain after --stride/--max-frames filtering")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent.parent.joinpath(
            f"attention_episode_{args.episode}.mp4"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.with_suffix("").with_name(output_path.stem + "_frames")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Pass --overwrite to replace it."
            )
        if output_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {output_path}")
        output_path.unlink()
    if frame_dir.exists():
        if not args.overwrite and args.save_frames:
            raise FileExistsError(
                f"Frame directory already exists: {frame_dir}. "
                "Pass --overwrite to replace it."
            )
        if args.overwrite:
            if frame_dir.is_file():
                frame_dir.unlink()
            else:
                shutil.rmtree(frame_dir)
    if args.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    camera_keys = display_camera_keys(policy, args.camera)
    output_fps = float(args.fps) if args.fps is not None else infer_dataset_fps(dataset)
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError(f"FPS must be positive and finite, got {output_fps}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    def autocast_context():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    writer = None
    frame_count = 0
    try:
        records = []
        for anchor, sample_index in selected:
            sample = dataset[sample_index]
            obs = {
                key: value.unsqueeze(0).to(device, non_blocking=True)
                for key, value in sample["obs"].items()
            }
            with torch.inference_mode(), autocast_context():
                policy.predict_action(obs, generator=generator)

            attention = attention_by_camera(
                policy.obs_encoder.last_cross_attention_weights_by_camera
                if policy.obs_encoder.last_cross_attention_weights_by_camera is not None
                else policy.obs_encoder.last_cross_attention_weights,
                policy,
                obs_step=-1,
                query_mode=args.attention_query,
            )
            camera_records = []
            tau = sample["obs"].get("tau_ext")
            tau_value = float(tau[-1].abs().sum()) if tau is not None else float("nan")
            for key in camera_keys:
                image = sample["obs"][key][-1]
                grid = patch_grid(
                    int(attention[key].numel()),
                    int(image.shape[-2]),
                    int(image.shape[-1]),
                    policy.obs_encoder.backbone,
                )
                camera_records.append(
                    {
                        "key": key,
                        "image": image_to_uint8(image),
                        "attention": attention[key].detach().float().cpu().numpy(),
                        "grid": grid,
                        "tau_value": tau_value,
                    }
                )
            records.append({"anchor": anchor, "cameras": camera_records})

        normalization_bounds = (
            attention_bounds(records)
            if args.attention_normalization == "global"
            else None
        )
        for frame_count, record in enumerate(records):
            rendered = []
            for camera in record["cameras"]:
                rendered_image = overlay_attention(
                    camera["image"],
                    camera["attention"],
                    grid=camera["grid"],
                    alpha=float(np.clip(args.alpha, 0.0, 1.0)),
                    normalization_bounds=normalization_bounds,
                )
                cv2.putText(
                    rendered_image,
                    f"{camera['key']} ep={args.episode} frame={record['anchor']} "
                    f"| tau_ext L1={camera['tau_value']:.3f}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                rendered.append(rendered_image)

            frame_rgb = (
                np.concatenate(rendered, axis=1)
                if len(rendered) > 1
                else rendered[0]
            )
            frame_rgb = np.ascontiguousarray(frame_rgb)
            if writer is None:
                writer = FFmpegVideoWriter(
                    output_path=output_path,
                    width=frame_rgb.shape[1],
                    height=frame_rgb.shape[0],
                    fps=output_fps,
                    codec=args.video_codec,
                )
            writer.write(frame_rgb)
            if args.save_frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_bgr = np.ascontiguousarray(frame_bgr)
                cv2.imwrite(
                    str(frame_dir.joinpath(f"frame_{frame_count:06d}.png")),
                    frame_bgr,
                )
    finally:
        if writer is not None:
            writer.close()
        dataset.lerobot_dataset.close()

    print(f"checkpoint: {checkpoint_path}")
    print(f"episode: {args.episode}, rendered frames: {frame_count + 1}")
    print(f"fps: {output_fps:g}, codec: {args.video_codec}")
    print(f"attention normalization: {args.attention_normalization}")
    if args.attention_normalization == "global":
        print(f"attention bounds: {normalization_bounds}")
    print(f"output: {output_path}")
    if args.save_frames:
        print(f"frames: {frame_dir}")
    return output_path


if __name__ == "__main__":
    run(parse_args())
