"""Export original camera frames with the actual FDP sampling-window labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import av
import hydra
import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw, ImageFont


PHASES = {0: "free", 1: "precontact", 2: "contact", -1: "无有效训练窗口"}
COLORS = {0: "#6f8195", 1: "#e3ae33", 2: "#e15c58", -1: "#b8bdc4"}


def decode_episode(path, start_s, length, fps):
    frames = []
    expected_times = start_s + np.arange(length) / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(start_s / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < start_s - 1e-5:
                continue
            if len(frames) == length:
                break
            if abs(timestamp - expected_times[len(frames)]) > 1e-4:
                raise ValueError(f"Unexpected video timestamp: {timestamp}")
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != length:
        raise ValueError(f"Decoded {len(frames)} frames, expected {length}")
    return frames


def write_video(path, frames, fps):
    height, width = frames[0].shape[:2]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"Video encoding failed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "data/outputs/civi/teacher/.hydra/config.yaml")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--phase-label-mode", choices=("window_contact", "current_wrench"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    dataset_cfg.preload_images = False
    dataset_cfg.video_decode_device = None
    dataset_cfg.image_randomer_config = None
    dataset_cfg.phase_resampling_enabled = True
    if args.phase_label_mode is not None:
        with open_dict(dataset_cfg):
            dataset_cfg.phase_label_mode = args.phase_label_mode
    dataset = hydra.utils.instantiate(dataset_cfg)
    root = Path(dataset.dataset_path)
    if not root.is_absolute():
        root = ROOT / root
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    metadata = [row for path in sorted((root / "meta/episodes").rglob("*.parquet"))
                for row in pq.read_table(path).to_pylist()]
    episode = next(row for row in metadata if row["episode_index"] == args.episode)
    start, stop = episode["dataset_from_index"], episode["dataset_to_index"]
    length = stop - start
    anchors = np.array([dataset._anchor_index_from_sampler_row(row) for row in dataset.indices])
    all_labels = dataset.get_phase_labels()
    labels = np.full(length, -1, dtype=np.int8)
    selected = (anchors >= start) & (anchors < stop)
    labels[anchors[selected] - start] = all_labels[selected]
    measured_force = dataset._phase_magnitude(dataset._phase_source_data)[start:stop]
    counts = np.bincount(all_labels, minlength=3)
    weights = np.array([cfg.phase_resampling.weights[name] for name in ("free", "precontact", "contact")])
    exposure = weights / weights.sum() * len(all_labels) / counts
    changes = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, length]
    spans = [{"phase_id": int(labels[a]), "phase": PHASES[int(labels[a])],
              "start_frame": int(a), "end_frame_exclusive": int(b),
              "start_s": float(a / fps), "end_s_exclusive": float(b / fps)}
             for a, b in zip(changes[:-1], changes[1:])]
    videos = {}
    for camera in ("side", "wrist"):
        key = dataset.image_feature_map[camera]
        prefix = f"videos/{key}/"
        path = root / info["video_path"].format(
            video_key=key, chunk_index=episode[prefix + "chunk_index"],
            file_index=episode[prefix + "file_index"])
        videos[camera] = decode_episode(path, episode[prefix + "from_timestamp"], length, fps)
    args.output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = {size: ImageFont.truetype(font_path, size) for size in (17, 19, 22, 27)}
    ratio = ":".join(f"{v:g}" for v in weights)

    def render(index, slow=False):
        phase = int(labels[index])
        canvas = Image.new("RGB", (1024, 640), "#101822")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 9), f"FDP 采样预览 · episode {args.episode} · 权重 {ratio}", font=fonts[27], fill="white")
        factor = f"单窗口相对均匀采样 {exposure[phase]:.2f} 倍" if phase >= 0 else "尾帧缺少完整未来窗口"
        draw.text((20, 51), f"帧 {index:03d}  |  原时间 {index / fps:.2f} s  |  {PHASES[phase]}  |  {factor}", font=fonts[22], fill=COLORS[phase])
        draw.text((20, 86), "侧视相机", font=fonts[19], fill="#c9d4df")
        draw.text((532, 86), f"腕部相机  |  当前实测力 {measured_force[index]:.1f} N / 阈值 {dataset.contact_threshold:g} N", font=fonts[19], fill="#c9d4df")
        for x, camera in ((0, "side"), (512, "wrist")):
            camera_image = Image.fromarray(videos[camera][index]).resize((512, 384), Image.Resampling.BICUBIC)
            canvas.paste(camera_image, (x, 116))
        draw.rectangle((0, 112, 1023, 115), fill=COLORS[phase])
        for span in spans:
            a = 20 + 984 * span["start_frame"] / length
            b = 20 + 984 * span["end_frame_exclusive"] / length
            draw.rectangle((a, 522, b, 543), fill=COLORS[span["phase_id"]])
        pointer = 20 + 984 * (index + 0.5) / length
        draw.polygon([(pointer, 517), (pointer - 6, 508), (pointer + 6, 508)], fill="white")
        draw.text((20, 552), "灰：free  黄：precontact  红：contact  浅灰：不参与采样", font=fonts[19], fill="white")
        rule = (f"新标签：当前实测力连续 {dataset.contact_enter_frames} 帧确认 contact；首次确认前 {dataset.precontact_seconds:g} 秒为 precontact。"
                if dataset.phase_label_mode == "current_wrench" else
                "旧标签：contact 可由历史或未来力触发，不等同当前已接触。")
        draw.text((20, 580), rule, font=fonts[17], fill="#becbd8")
        speed = "4 倍慢放；原时间不变。拼接片段之间时间会跳转。" if slow else "原速 20 Hz；原训练视频画面，未重现随机图像增强。"
        draw.text((20, 607), speed, font=fonts[17], fill="#becbd8")
        return np.asarray(canvas)

    frames = [render(i) for i in range(length)]
    full_path = args.output / f"episode_{args.episode}_full_annotated.mp4"
    write_video(full_path, frames, fps)
    outputs = {"full": full_path.name}
    for tag, mask in (("strengthened", labels > 0), ("precontact", labels == 1), ("contact", labels == 2)):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        path = args.output / f"episode_{args.episode}_{tag}_slow4x.mp4"
        write_video(path, [render(int(i), slow=True) for i in indices], fps / 4)
        outputs[tag] = path.name
    preview_index = int(np.flatnonzero(labels == 1)[len(np.flatnonzero(labels == 1)) // 2])
    Image.fromarray(render(preview_index, slow=True)).save(args.output / f"episode_{args.episode}_preview.png")
    manifest = json.loads((root / "meta/wrench_conversion.json").read_text())
    source = next(row for row in manifest["episodes"] if row["episode_index"] == args.episode)
    report = {"config": str(args.config.resolve()), "dataset": str(root.resolve()),
              "episode_index": args.episode, "source": source, "fps": fps,
              "phase_label_mode": dataset.phase_label_mode,
              "phase_label_mode_override": args.phase_label_mode,
              "measured_force_norm_N": measured_force.tolist(),
              "phase_weights": weights.tolist(), "all_dataset_phase_counts": counts.tolist(),
              "per_window_exposure_vs_uniform": exposure.tolist(), "spans": spans,
              "outputs": outputs, "note": "Actual sampler labels on window anchor frames; not WM phase labels or a trace of random minibatch draws. Camera frames decoded at exact episode timestamps from training videos; no random augmentation."}
    (args.output / "sampling_segments.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
