# python3 data_process/tool/humandata_images_to_mp4.py \
#   --input-root data/humandata \
#   --output-root outputs/humandata_mp4 \
#   --camera head_color \
#   --fps 30 \
#   --overwrite



from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_candidates() -> list[Path]:
    repo = repo_root()
    bases = [
        repo / "data",
        repo.parent / "data",
        Path("/workspace/data"),
        Path("/data"),
    ]
    names = ("humandata", "humanidodata")

    candidates: list[Path] = []
    env_root = os.environ.get("HUMANDATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    for base in bases:
        for name in names:
            candidate = base / name
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def default_input_root() -> Path:
    for candidate in default_input_candidates():
        if candidate.exists():
            return candidate
    return repo_root() / "data" / "humandata"


def default_output_root() -> Path:
    return repo_root() / "outputs" / "humandata_mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ordered image sequences under humandata episode folders into MP4 videos."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root(),
        help="Dataset root, episode folder, or a colors directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Where MP4 files will be written.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Only export these camera streams. Repeat the option to keep multiple streams.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 files.",
    )
    parser.add_argument(
        "--episode-subdirs",
        action="store_true",
        help="Write videos as category/episode/camera.mp4 instead of category/episode_camera.mp4.",
    )
    return parser.parse_args()


def contains_images(directory: Path) -> bool:
    return any(
        child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
        for child in directory.iterdir()
    )


def discover_colors_dirs(input_root: Path) -> list[Path]:
    input_root = input_root.expanduser().resolve()
    if not input_root.exists():
        candidates = "\n".join(f"  - {path}" for path in default_input_candidates())
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}\n"
            "Try passing the mounted data path explicitly, for example:\n"
            "  --input-root /workspace/data/humandata\n"
            f"Default candidates checked:\n{candidates}"
        )

    if input_root.is_dir() and input_root.name == "colors" and contains_images(input_root):
        return [input_root]

    colors_dirs: list[Path] = []
    for path in input_root.rglob("colors"):
        if path.is_dir() and contains_images(path):
            colors_dirs.append(path)

    if not colors_dirs:
        raise FileNotFoundError(f"No colors directories with images found under {input_root}")

    return sorted(colors_dirs, key=lambda path: path.as_posix())


def ensure_output_is_separate(input_root: Path, output_root: Path) -> None:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise ValueError(
            "Output root must be outside the source dataset. "
            f"Got output_root={output_root}, input_root={input_root}. "
            "Use something like --output-root outputs/humandata_mp4."
        )


def parse_frame_name(path: Path) -> tuple[int | None, str]:
    stem = path.stem
    if "_" not in stem:
        return None, "default"

    frame_text, camera = stem.split("_", 1)
    if frame_text.isdigit():
        return int(frame_text), camera
    return None, camera


def group_stream_frames(
    colors_dir: Path,
    camera_filter: set[str] | None,
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    grouped: dict[str, list[tuple[tuple[int, str], Path]]] = {}
    skipped_empty: dict[str, int] = {}

    for path in sorted(colors_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        frame_idx, camera = parse_frame_name(path)
        if camera_filter is not None and camera not in camera_filter:
            continue

        if path.stat().st_size == 0:
            skipped_empty[camera] = skipped_empty.get(camera, 0) + 1
            continue

        sort_key = (frame_idx if frame_idx is not None else 10**12, path.name)
        grouped.setdefault(camera, []).append((sort_key, path))

    ordered: dict[str, list[Path]] = {}
    for camera, items in grouped.items():
        items.sort(key=lambda item: item[0])
        ordered[camera] = [path for _, path in items]

    return dict(sorted(ordered.items(), key=lambda item: item[0])), skipped_empty


def category_for(colors_dir: Path, input_root: Path) -> str:
    input_root = input_root.expanduser().resolve()
    colors_dir = colors_dir.expanduser().resolve()
    episode_dir = colors_dir.parent

    try:
        rel = episode_dir.relative_to(input_root)
        for marker in ("humandata", "humanidodata"):
            if marker in rel.parts:
                marker_index = rel.parts.index(marker)
                if len(rel.parts) > marker_index + 1:
                    return rel.parts[marker_index + 1]
        if len(rel.parts) >= 2:
            return rel.parts[0]
        if len(rel.parts) == 1 and input_root.name not in {"data", "humandata", "humanidodata"}:
            return input_root.name
    except ValueError:
        pass

    parent_name = episode_dir.parent.name
    if parent_name and parent_name not in {"data", "humandata", "humanidodata"}:
        return parent_name
    return "uncategorized"


def output_path_for(
    colors_dir: Path,
    input_root: Path,
    output_root: Path,
    camera: str,
    episode_subdirs: bool,
) -> Path:
    category = category_for(colors_dir, input_root)
    episode = colors_dir.parent.name
    if episode_subdirs:
        return output_root / category / episode / f"{camera}.mp4"
    return output_root / category / f"{episode}_{camera}.mp4"


def link_sequence(frames: list[Path], seq_dir: Path) -> None:
    suffixes = {frame.suffix.lower() for frame in frames}
    if len(suffixes) != 1:
        raise ValueError(f"Mixed image suffixes are not supported: {sorted(suffixes)}")

    suffix = next(iter(suffixes))
    for index, src in enumerate(frames):
        dst = seq_dir / f"{index:06d}{suffix}"
        try:
            os.symlink(src, dst)
        except OSError:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def encode_mp4(frames: list[Path], output_path: Path, fps: float, overwrite: bool) -> None:
    if not frames:
        raise ValueError(f"No valid frames for {output_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "ffmpeg was not found in PATH. Install ffmpeg first, then rerun this script."
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="humandata_seq_") as tmp:
        seq_dir = Path(tmp)
        link_sequence(frames, seq_dir)

        cmd = [
            ffmpeg,
            "-y" if overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(seq_dir / ("%06d" + frames[0].suffix.lower())),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    camera_filter = set(args.camera) if args.camera else None

    ensure_output_is_separate(input_root, output_root)
    colors_dirs = discover_colors_dirs(input_root)
    print(f"found {len(colors_dirs)} colors folders under {input_root}")

    total_videos = 0
    for colors_dir in colors_dirs:
        streams, skipped_empty = group_stream_frames(colors_dir, camera_filter)
        for camera, count in sorted(skipped_empty.items()):
            print(f"skip {count} empty image(s): {colors_dir} [{camera}]")

        if not streams:
            print(f"skip {colors_dir}: no matching image streams")
            continue

        for camera, frames in streams.items():
            if not frames:
                continue

            output_path = output_path_for(
                colors_dir,
                input_root,
                output_root,
                camera,
                args.episode_subdirs,
            )
            encode_mp4(frames, output_path, args.fps, args.overwrite)
            total_videos += 1
            print(f"saved {output_path} ({len(frames)} frames)")

    print(f"done, wrote {total_videos} mp4 file(s)")


if __name__ == "__main__":
    main()
