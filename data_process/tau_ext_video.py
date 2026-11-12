"""CPU-only, timestamp-aligned LeRobot video with external-torque norm traces."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import av
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq


def episode_video_spec(info, episode_metadata, camera_key, times):
    """Map episode-relative state timestamps into its concatenated video shard."""
    prefix = f"videos/{camera_key}"
    path = info["video_path"].format(
        video_key=camera_key,
        chunk_index=int(episode_metadata[f"{prefix}/chunk_index"]),
        file_index=int(episode_metadata[f"{prefix}/file_index"]),
    )
    targets = float(episode_metadata[f"{prefix}/from_timestamp"]) + np.asarray(times)
    return path, targets


class EpisodeVideoCursor:
    """Seek once, then select the nearest decoded PTS for increasing targets."""

    def __init__(self, path, first_time, fps):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.period = 1.0 / fps
        if first_time > 0:
            # AV1 shards may have long GOPs. A backward keyframe seek preserves
            # the preceding candidate instead of assuming episode == file.
            seek_time = max(0.0, float(first_time) - self.period)
            self.container.seek(int(seek_time / self.stream.time_base), stream=self.stream,
                                backward=True, any_frame=False)
        self.frames = iter(self.container.decode(self.stream))
        self.previous = None
        self.following = next(self.frames, None)

    def rgb_at(self, timestamp):
        while self.following is not None and float(self.following.time) < timestamp - 1e-7:
            self.previous = self.following
            self.following = next(self.frames, None)
        candidates = [f for f in (self.previous, self.following) if f is not None]
        if not candidates:
            raise ValueError(f"Video has no decoded frame for timestamp {timestamp:.6f}")
        frame = min(candidates, key=lambda f: abs(float(f.time) - timestamp))
        if abs(float(frame.time) - timestamp) > 0.55 * self.period + 1e-5:
            raise ValueError(f"Video/state timestamp mismatch: requested {timestamp:.6f}, "
                             f"nearest video PTS {float(frame.time):.6f}")
        return frame.to_ndarray(format="rgb24")

    def close(self):
        self.container.close()


class TauExtNormPlot:
    """Draw static norm traces once; overlay only the live cursor and readout."""

    def __init__(self, times, pre_gate_l1, final_l1, contact, threshold, width, height):
        self.times = np.asarray(times)
        self.pre_gate_l1 = np.asarray(pre_gate_l1)
        self.final_l1 = np.asarray(final_l1)
        self.contact = np.asarray(contact, dtype=bool)
        self.threshold = float(threshold)
        self.width, self.height = width, height
        self.font = ImageFont.truetype(findfont("DejaVu Sans"), 14 if width >= 600 else 10)
        figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_axes([0.085, 0.20, 0.89, 0.61])
        ax.plot(times, pre_gate_l1, color="#9aa2ab", linewidth=1.4,
                label="Smoothed, before gate")
        ax.plot(times, final_l1, color="#006fbb", linewidth=1.6,
                label="Final tau_ext")
        ax.axhline(threshold, color="#c64036", linestyle="--", linewidth=1.1,
                   label=f"Threshold {threshold:g} Nm")
        ax.set_xlabel("Episode time (s)", fontsize=9)
        ax.set_ylabel("L1 norm (Nm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, max(float(self.times[-1]), 1.0 / 25))
        ax.set_ylim(0, max(float(self.pre_gate_l1.max()), threshold, 1) * 1.14)
        ax.legend(loc="upper right", fontsize=8 if width >= 600 else 6, framealpha=0.9)
        canvas.draw()
        self.background = Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        self.cursor_xy = ax.transData.transform(np.column_stack((times, final_l1)))
        self.top = height - ax.bbox.ymax
        self.bottom = height - ax.bbox.ymin

    def render(self, index):
        frame = self.background.copy()
        draw = ImageDraw.Draw(frame)
        x, y = self.cursor_xy[index]
        y = self.height - y
        draw.line((x, self.top, x, self.bottom), fill="#202020", width=2)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#006fbb")
        state = "CONTACT" if self.contact[index] else "FREE"
        text = (f"t={self.times[index]:.2f}s  |  ||tau_ext||_1={self.final_l1[index]:.2f} Nm"
                f"  |  before gate={self.pre_gate_l1[index]:.2f} Nm  |  {state}")
        draw.text((12, 9), text, fill="#202020", font=self.font)
        return np.asarray(frame)


def _camera_panel(images, keys, width_per_camera, font):
    heights = [round(image.shape[0] * width_per_camera / image.shape[1]) for image in images]
    height = max(heights)
    panel = Image.new("RGB", (len(images) * width_per_camera, height + 28), "#151b22")
    draw = ImageDraw.Draw(panel)
    for index, (image, key, resized_height) in enumerate(zip(images, keys, heights)):
        x = index * width_per_camera
        tile = Image.fromarray(image).resize((width_per_camera, resized_height), Image.Resampling.BILINEAR)
        panel.paste(tile, (x, 28 + (height - resized_height) // 2))
        draw.text((x + 9, 5), key.removeprefix("observation.images."), fill="white", font=font)
    return np.asarray(panel)


def export_tau_ext_video(
    dataset_root: Path,
    episode: int,
    camera_keys: list[str],
    output: Path | None = None,
    width_per_camera=448,
    plot_height=280,
    codec="libx264",
) -> dict:
    """Export all episode frames with pre/post-gate L1 and a synchronized cursor.

    Cameras are read directly from their video shards. This function does not
    load a policy, use a GPU, smooth again, or backdate contact decisions.
    """
    dataset_root = Path(dataset_root)
    info = json.loads((dataset_root / "meta/info.json").read_text())
    processing = json.loads((dataset_root / "meta/tau_ext_processing.json").read_text())
    threshold = float(processing["config"]["contact_threshold_l1"])
    fps = float(info["fps"])
    if not camera_keys or width_per_camera < 64 or plot_height < 160:
        raise ValueError("Select at least one camera; require width >= 64 and plot height >= 160")
    for key in camera_keys:
        if info["features"].get(key, {}).get("dtype") != "video":
            raise ValueError(f"Not a video feature: {key}")
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    columns = ["frame_index", "timestamp", "observation.tau_ext",
               "observation.tau_ext_l1", "observation.contact"]
    table = pq.read_table(files, columns=columns, filters=[("episode_index", "=", episode)])
    table = table.sort_by([("frame_index", "ascending")])
    if len(table) == 0:
        raise ValueError(f"Episode {episode} not found")
    times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
    times -= times[0]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Episode timestamps must be finite and strictly increasing")
    tau = np.asarray(table["observation.tau_ext"].to_pylist(), dtype=np.float64)
    if tau.shape != (len(table), 7) or not np.isfinite(tau).all():
        raise ValueError("Expected finite seven-axis tau_ext")
    pre_gate_l1 = np.asarray(table["observation.tau_ext_l1"].to_pylist()).reshape(-1)
    final_l1 = np.abs(tau).sum(axis=1)
    contact = np.asarray(table["observation.contact"].to_pylist()).reshape(-1).astype(bool)
    metadata_files = sorted((dataset_root / "meta/episodes").rglob("*.parquet"))
    metadata = pq.read_table(metadata_files, filters=[("episode_index", "=", episode)]).to_pylist()
    if len(metadata) != 1:
        raise ValueError(f"Expected one metadata row for episode {episode}")
    output = (Path(output) if output is not None else
              dataset_root / "visualizations" / f"episode_{episode:03d}_tau_ext.mp4")
    if output.exists():
        raise FileExistsError(f"Video output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    width = len(camera_keys) * width_per_camera
    plot = TauExtNormPlot(times, pre_gate_l1, final_l1, contact, threshold, width, plot_height)
    cursors, targets, container = [], [], None
    succeeded = False
    try:
        for key in camera_keys:
            relative_path, camera_targets = episode_video_spec(info, metadata[0], key, times)
            cursors.append(EpisodeVideoCursor(dataset_root / relative_path, camera_targets[0], fps))
            targets.append(camera_targets)
        for index in range(len(times)):
            images = [cursor.rgb_at(float(camera_targets[index]))
                      for cursor, camera_targets in zip(cursors, targets)]
            cameras = _camera_panel(images, camera_keys, width_per_camera, plot.font)
            image = np.concatenate((cameras, plot.render(index)), axis=0)
            # yuv420p requires even spatial dimensions.
            if image.shape[0] % 2 or image.shape[1] % 2:
                image = np.pad(image, ((0, image.shape[0] % 2), (0, image.shape[1] % 2), (0, 0)))
            if container is None:
                container = av.open(str(output), mode="w", options={"movflags": "+faststart"})
                stream = container.add_stream(codec, rate=Fraction(str(fps)).limit_denominator(100000))
                stream.width, stream.height = image.shape[1], image.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.codec_context.thread_count = 2
                if codec == "libx264":
                    stream.options = {"crf": "18", "preset": "veryfast"}
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 1) / Fraction(str(fps))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        succeeded = True
    finally:
        for cursor in cursors:
            cursor.close()
        if container is not None:
            container.close()
        if not succeeded:
            output.unlink(missing_ok=True)
    return {"video": str(output), "episode": int(episode), "frames": len(times), "fps": fps,
            "duration_s": len(times) / fps, "threshold_l1_Nm": threshold,
            "norm": "L1 over seven joint torques", "camera_keys": list(camera_keys),
            "synchronization": "nearest video PTS to episode from_timestamp + relative source timestamp"}
