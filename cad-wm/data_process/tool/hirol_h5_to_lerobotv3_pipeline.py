from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SHAPE_META = Path("config/shape_meta/shape_meta_hirol_pinocchio.yaml")


try:
    from data_process.tool.hirol_episode_to_factr_h5 import (  # type: ignore
        convert_episode,
        discover_episodes,
        inspect_h5,
    )
    from data_process.tool.h5_2_lerobotev3 import load_shape_meta, run_conversion  # type: ignore
except ModuleNotFoundError:
    from hirol_episode_to_factr_h5 import convert_episode, discover_episodes, inspect_h5
    from h5_2_lerobotev3 import load_shape_meta, run_conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline: HIROL EpisodeWriter data.json + async ATI JSON -> factr H5 -> LeRobot v3. "
            "The LeRobot schema is controlled by shape_meta."
        )
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="HIROL episode dir, data.json, or task dir.")
    parser.add_argument(
        "--shape-meta",
        type=Path,
        default=DEFAULT_SHAPE_META,
        help="Shape meta used for the H5 -> LeRobot v3 step.",
    )
    parser.add_argument("--h5-output", type=Path, default=None, help="Intermediate H5 directory.")
    parser.add_argument("--output-root", "-o", type=Path, default=None, help="Output LeRobot dataset root.")
    parser.add_argument("--repo-id", default=None, help="Output LeRobot repo id.")
    parser.add_argument("--task", default=None, help="Task label stored by LeRobot.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        help="Forwarded to HIROL -> H5 when --include-videos is used.",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Keep camera features from shape_meta and write camera frames into H5.",
    )
    parser.add_argument(
        "--use-ati-json",
        dest="prefer_async_ft",
        action="store_true",
        default=True,
        help="Use the explicit ATI JSON file beside data.json.",
    )
    parser.add_argument(
        "--no-use-ati-json",
        dest="prefer_async_ft",
        action="store_false",
        help="Use FT samples embedded in data.json instead of the explicit ATI JSON.",
    )
    parser.add_argument("--prefer-async-ft", dest="prefer_async_ft", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-prefer-async-ft", dest="prefer_async_ft", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ati-json-name",
        default="ee_ft_data.json",
        help="Exact ATI JSON filename beside data.json. Expected schema: data[*].ft_data + data[*].time_stamp.",
    )
    parser.add_argument(
        "--allow-missing-ft",
        action="store_true",
        help="Allow H5 episodes with no FT rows. Not recommended for pinocchio validation.",
    )
    parser.add_argument(
        "--epoch-timestamps",
        action="store_true",
        help="Forwarded to HIROL -> H5 for perf_counter-style timestamps.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing H5/LeRobot outputs first.")
    parser.add_argument("--inspect-h5", action="store_true", help="Print each intermediate H5 structure.")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def shape_meta_io(shape_meta: Mapping[str, Any]) -> dict[str, Any]:
    io_cfg = shape_meta.get("io") or {}
    if not isinstance(io_cfg, Mapping):
        return {}
    return dict(io_cfg)


def io_path(shape_meta: Mapping[str, Any], key: str) -> Path | None:
    value = shape_meta_io(shape_meta).get(key)
    return Path(value) if value is not None else None


def io_str(shape_meta: Mapping[str, Any], key: str) -> str | None:
    value = shape_meta_io(shape_meta).get(key)
    return str(value) if value is not None else None


def feature_dtype(spec: Mapping[str, Any]) -> str:
    return str(spec.get("dtype", spec.get("type", "float32"))).lower()


def is_video_feature(key: str, spec: Any) -> bool:
    if not isinstance(spec, Mapping):
        return False
    return key.startswith("observation.images.") or feature_dtype(spec) in {"image", "video"}


def strip_video_features(shape_meta: dict[str, Any]) -> None:
    features = shape_meta.get("features")
    if not isinstance(features, dict):
        return
    for key in list(features.keys()):
        if is_video_feature(str(key), features[key]):
            del features[key]


def prepare_empty_dir(path: Path, overwrite: bool, label: str) -> None:
    path = path.expanduser()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{label} already exists and is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_absent_path(path: Path, overwrite: bool, label: str) -> None:
    path = path.expanduser()
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{label} already exists: {path}. Use --overwrite.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_pipeline_shape_meta(
    *,
    source_shape_meta: dict[str, Any],
    h5_output: Path,
    output_root: Path,
    repo_id: str,
    task: str,
    max_episodes: int | None,
    include_videos: bool,
    push_to_hub: bool,
) -> Path:
    shape_meta = json.loads(json.dumps(source_shape_meta))
    if not include_videos:
        strip_video_features(shape_meta)

    io_cfg = dict(shape_meta.get("io") or {})
    io_cfg["input"] = str(h5_output)
    io_cfg["output"] = str(output_root)
    io_cfg["repo_id"] = repo_id
    io_cfg["no_videos"] = not include_videos
    io_cfg["push_to_hub"] = push_to_hub
    if max_episodes is not None:
        io_cfg["max_episodes"] = max_episodes
    shape_meta["io"] = io_cfg
    shape_meta["task"] = task

    config_path = h5_output / "_pipeline_shape_meta.json"
    config_path.write_text(json.dumps(shape_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def build_h5_args(args: argparse.Namespace, h5_output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=args.input,
        output=h5_output,
        output_name=None,
        max_episodes=args.max_episodes,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_map=args.camera_map,
        prefer_async_ft=args.prefer_async_ft,
        ati_json_name=args.ati_json_name,
        epoch_timestamps=args.epoch_timestamps,
        require_cameras=args.include_videos,
        require_ft=not args.allow_missing_ft,
        skip_cameras=not args.include_videos,
        overwrite=args.overwrite,
        inspect=False,
        num_episodes=0,
    )


def run_hirol_to_h5(args: argparse.Namespace, h5_output: Path) -> list[Path]:
    h5_args = build_h5_args(args, h5_output)
    episodes = discover_episodes(args.input, args.max_episodes)
    h5_args.num_episodes = len(episodes)

    written = []
    print(f"Found {len(episodes)} HIROL episode(s).")
    for episode_dir in episodes:
        h5_path = convert_episode(episode_dir, h5_output, h5_args)
        written.append(h5_path)
        print(f"Wrote H5: {h5_path}")
        if args.inspect_h5:
            inspect_h5(h5_path)
    return written


def run_h5_to_lerobot(shape_meta_path: Path) -> None:
    run_conversion(argparse.Namespace(config=shape_meta_path))


def main() -> None:
    args = parse_args()
    source_shape_meta = load_shape_meta(args.shape_meta)

    output_root = args.output_root or io_path(source_shape_meta, "output")
    if output_root is None:
        raise ValueError("--output-root is required when shape_meta io.output is missing.")
    output_root = output_root.expanduser()

    repo_id = args.repo_id or io_str(source_shape_meta, "repo_id")
    if repo_id is None:
        raise ValueError("--repo-id is required when shape_meta io.repo_id is missing.")

    task = args.task or str(source_shape_meta.get("task", "default_task"))
    h5_output = args.h5_output or output_root.with_name(f"{output_root.name}_h5")
    h5_output = h5_output.expanduser()

    prepare_empty_dir(h5_output, args.overwrite, "H5 output")
    prepare_absent_path(output_root, args.overwrite, "LeRobot output")

    run_hirol_to_h5(args, h5_output)
    pipeline_shape_meta = write_pipeline_shape_meta(
        source_shape_meta=source_shape_meta,
        h5_output=h5_output,
        output_root=output_root,
        repo_id=repo_id,
        task=task,
        max_episodes=args.max_episodes,
        include_videos=args.include_videos,
        push_to_hub=args.push_to_hub,
    )
    print(f"Pipeline shape_meta: {pipeline_shape_meta}")

    run_h5_to_lerobot(pipeline_shape_meta)
    print(f"Done. LeRobot root={output_root} repo_id={repo_id}")


if __name__ == "__main__":
    main()
