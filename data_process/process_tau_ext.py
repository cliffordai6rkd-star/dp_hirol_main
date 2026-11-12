#!/usr/bin/env python3
"""Rebuild causal tau_ext datasets and synchronized videos from one YAML config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("tau_ext_config.yaml")
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_repo_path(value, repo_root=REPO_ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(repo_root) / path).resolve()


def load_config(config_file=DEFAULT_CONFIG, *, repo_root=REPO_ROOT):
    """Dataset/model paths in the config are relative to the repository root."""
    config_file = Path(config_file).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, dict) or not config.get("datasets"):
        raise ValueError("Config must contain a nonempty datasets list")
    for dataset in config["datasets"]:
        for key in ("input", "output"):
            dataset[key] = str(resolve_repo_path(dataset[key], repo_root))
    processing = config.setdefault("processing", {})
    if "urdf" in processing:
        processing["urdf"] = str(resolve_repo_path(processing["urdf"], repo_root))
    return config


def build_export_args(dataset, processing):
    """Reuse the existing exporter's option validation; settings come from YAML."""
    from scripts.add_xarm_wrench_v3 import parse_args

    settings = {
        "bio_gripper_g2": True,
        "filter_mode": "causal",
        "convention": "tool-on-environment",
        **processing,
    }
    window = settings.pop("tau_ext_smoothing_window", 5)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("processing.tau_ext_smoothing_window must be a positive integer")
    if settings["filter_mode"] != "causal":
        raise ValueError("This pipeline requires processing.filter_mode: causal")
    argv = ["--dataset", str(dataset["input"]), "--output", str(dataset["output"])]
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, *map(str, value)])
        else:
            argv.extend([flag, str(value)])
    args = parse_args(argv)
    args.tau_ext_smoothing_window = window
    return args


def validate_outputs(datasets, *, overwrite=False):
    """Check all targets before starting either dataset."""
    outputs = set()
    sources = [Path(dataset["input"]) for dataset in datasets]
    for dataset in datasets:
        source, target = Path(dataset["input"]), Path(dataset["output"])
        if not (source / "meta/info.json").is_file():
            raise FileNotFoundError(f"Missing input dataset metadata: {source / 'meta/info.json'}")
        if any(target == root or root in target.parents or target in root.parents for root in sources):
            raise ValueError("Output must be outside all input datasets")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists; choose a new output path in the config or use --overwrite: {target}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"Output must be a dataset directory: {target}")
        if target in outputs:
            raise ValueError(f"Duplicate dataset output: {target}")
        outputs.add(target)


def run_pipeline(config_file=DEFAULT_CONFIG, *, overwrite=False):
    from scripts.add_xarm_wrench_v3 import export_dataset

    config_file = Path(config_file).expanduser().resolve()
    config = load_config(config_file)
    validate_outputs(config["datasets"], overwrite=overwrite)
    jobs = [build_export_args(dataset, config["processing"]) for dataset in config["datasets"]]
    outputs = []
    # Finish both data conversions before rendering any video.
    for args in jobs:
        print(f"\nProcessing: {args.dataset} -> {args.output}", flush=True)
        output = export_dataset(args, overwrite=overwrite)
        record = {
            "config_file": str(config_file),
            "configuration": config,
            "overwrite": overwrite,
            "resolved_export_parameters": vars(args),
        }
        (output / "meta/data_process_config.json").write_text(
            json.dumps(record, indent=2, default=str, allow_nan=False) + "\n"
        )
        outputs.append(output)

    video = config.get("visualization", {})
    if video.get("enabled", True):
        from data_process.tau_ext_video import export_tau_ext_video

        for output in outputs:
            episodes = video.get("episodes", [0])
            if episodes == "all":
                import pyarrow.parquet as pq
                episode_files = sorted((output / "meta/episodes").rglob("*.parquet"))
                episodes = sorted(pq.read_table(episode_files, columns=["episode_index"])
                                  ["episode_index"].to_pylist())
            if not isinstance(episodes, list) or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in episodes
            ):
                raise ValueError("visualization.episodes must be a list of episode indices or 'all'")
            for episode in episodes:
                result = export_tau_ext_video(
                    output,
                    episode,
                    video.get("cameras", ["observation.images.front", "observation.images.right_wrist"]),
                    width_per_camera=video.get("width_per_camera", 448),
                    plot_height=video.get("plot_height", 280),
                    codec=video.get("codec", "libx264"),
                )
                print(f"Video: {result}", flush=True)
    print("\nFinished. Dataset outputs:", flush=True)
    for output in outputs:
        print(output, flush=True)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output datasets")
    args = parser.parse_args(argv)
    run_pipeline(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
