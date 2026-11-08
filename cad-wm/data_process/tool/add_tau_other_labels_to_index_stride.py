"""Add 50 Hz causal RNEA residual ``tau_other`` labels to LeRobot datasets.

Each phase dataset is processed independently, using the same filtered dataset
view and target-generation function as ``TauOtherTrainer``.  No 100 Hz H5 signal is
used to estimate acceleration or inverse dynamics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.tau_other_target_generation import (  # noqa: E402
    build_causal_tau_other_target,
    resolve_tau_other_target_generation,
    timestamps_to_seconds,
)
from data_process.tool.lerobot_add_feature import (  # noqa: E402
    compute_stats,
    feature_arrow_type,
    replace_or_append_column,
    update_huggingface_schema_metadata,
    update_meta_files_in_place,
)
from physics.nero_dynamics import PinocchioDynamics  # noqa: E402


MANIFEST_NAME = "index_stride_manifest.json"
DEFAULT_PHASE_ROOTS = (
    REPO_ROOT / "data/train_episode/bg_data_filted_a_lbv3",
    REPO_ROOT / "data/train_episode/bg_data_filted_b_lbv3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute causal tau_other independently on each 50 Hz phase dataset "
            "with the same preprocessing contract as TauOtherTrainer."
        )
    )
    parser.add_argument(
        "--phase-roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_PHASE_ROOTS),
        help="Index-stride LeRobot roots; phase_index is read from each manifest.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/train_cfg/tau_other_sequence.yaml",
        help="Training config defining target_generation, filters, and Pinocchio.",
    )
    parser.add_argument("--tau-other-key", default="observation.tau_other")
    parser.add_argument("--tau-g-key", default="observation.tau_g")
    parser.add_argument("--ddq-key", default="observation.ddq")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files before replacing parquet/info/stats files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate both 50 Hz datasets and compute all labels without writing.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "meta" / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") != "index_stride":
        raise ValueError(f"Not an index-stride manifest: {path}")
    if int(data.get("stride", 0)) <= 0:
        raise ValueError(f"Invalid stride in {path}")
    return data


def validate_manifests(manifests: list[Mapping[str, Any]]) -> None:
    fps_values = {int(item.get("fps", 0)) for item in manifests}
    if fps_values != {50}:
        raise ValueError(f"This labeler requires 50 Hz phase datasets, got {fps_values}")
    phases = [int(item["phase_index"]) for item in manifests]
    if len(set(phases)) != len(phases):
        raise ValueError(f"Duplicate phase indices: {phases}")
    stride_values = {int(item["stride"]) for item in manifests}
    if len(stride_values) != 1:
        raise ValueError(f"Phase manifests disagree on stride: {stride_values}")
    episode_lists = [item.get("episodes") for item in manifests]
    if any(not isinstance(items, list) for items in episode_lists):
        raise ValueError("Every manifest must contain an episodes list.")
    reference = [item["source_h5"] for item in episode_lists[0]]
    for items in episode_lists[1:]:
        if [item["source_h5"] for item in items] != reference:
            raise ValueError("Phase manifests disagree on source episode ordering.")


def _column_to_tensor(dataset: Any, key: str) -> torch.Tensor:
    if key not in dataset.column_names:
        raise KeyError(f"Dataset is missing required column {key!r}")
    formatted = dataset.with_format(
        "torch", columns=[key], output_all_columns=False
    )
    value = formatted[:][key]
    if torch.is_tensor(value):
        return value
    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        return torch.stack(value)
    return torch.as_tensor(value)


def compute_phase_labels(
    *,
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[np.ndarray]], list[tuple[int, int]]]:
    from data_process.dataloader import PINNDataset

    phase_config = dict(config)
    data_config = dict(config.get("dataloader") or {})
    data_config.update(
        {
            "root": str(root),
            "repo_id": root.name,
            "load_images": False,
        }
    )
    phase_config["dataloader"] = data_config
    dataset = PINNDataset(phase_config, compute_normalizer=False)
    target_config = resolve_tau_other_target_generation(
        phase_config, dataset.filter_config
    )
    if not target_config.get("enabled"):
        raise ValueError("The selected config must enable target_generation.")
    lowdim_keys = data_config.get("lowdim_keys") or {}
    source_keys = target_config["source_keys"]
    columns = {
        name: _column_to_tensor(dataset.stats_dataset, lowdim_keys[alias])
        for name, alias in source_keys.items()
    }
    timestamp_values = _column_to_tensor(
        dataset.stats_dataset, target_config["timestamp_key"]
    )
    timestamps_s = timestamps_to_seconds(
        timestamp_values, target_config["timestamp_unit"]
    )
    result = build_causal_tau_other_target(
        timestamps_s=timestamps_s,
        q=columns["q"],
        dq=columns["dq"],
        tau_measured=columns["tau"],
        episodes=dataset.dataset.meta.episodes,
        target_config=target_config,
        dynamics=PinocchioDynamics(phase_config),
    )
    tau_other = result.tau_other.numpy()
    ddq = result.ddq.numpy()
    tau_measured = columns["tau"].detach().cpu().to(torch.float32).numpy()
    tau_g = result.tau_g.numpy()
    values_by_name = {
        "tau_other": [row for row in tau_other],
        "tau_g": [row for row in tau_g],
        "ddq": [row for row in ddq],
    }
    expected_rows: list[tuple[int, int]] = []
    for episode in dataset.dataset.meta.episodes:
        episode_index = int(episode["episode_index"])
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        expected_rows.extend(
            (episode_index, frame_index) for frame_index in range(stop - start)
        )
    if any(len(values) != len(expected_rows) for values in values_by_name.values()):
        raise ValueError("Generated labels do not cover all 50 Hz dataset rows.")
    if not np.allclose(
        tau_other, tau_measured - tau_g, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("Generated tau_other/tau_g residual identity check failed.")
    return values_by_name, expected_rows


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def write_phase(
    root: Path,
    values_by_key: Mapping[str, list[np.ndarray]],
    expected_rows: list[tuple[int, int]],
    *,
    backup: bool,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    features = {
        key: {"dtype": "float32", "shape": (7,)} for key in values_by_key
    }
    frame_count = len(expected_rows)
    if any(len(values) != frame_count for values in values_by_key.values()):
        raise ValueError("Every output feature must contain one value per dataset row.")
    by_index: dict[int, np.ndarray] = {}
    cursor = 0
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        required = {"index", "episode_index", "frame_index"}
        missing = required - set(table.column_names)
        if missing:
            raise KeyError(f"{parquet_path} is missing columns {sorted(missing)}")
        indices = [int(value) for value in table["index"].to_pylist()]
        episodes = [int(value) for value in table["episode_index"].to_pylist()]
        frames = [int(value) for value in table["frame_index"].to_pylist()]
        for dataset_index, episode_index, frame_index in zip(indices, episodes, frames):
            if dataset_index < 0 or dataset_index >= frame_count:
                raise ValueError(f"Dataset index out of range in {parquet_path}: {dataset_index}")
            if expected_rows[dataset_index] != (episode_index, frame_index):
                raise ValueError(
                    f"Row mapping mismatch at index {dataset_index}: parquet has "
                    f"({episode_index}, {frame_index}), expected {expected_rows[dataset_index]}"
                )
            by_index[dataset_index] = np.empty(0)
        updated = table
        for key, values in values_by_key.items():
            column_values = [
                np.asarray(values[dataset_index], dtype=np.float32).tolist()
                for dataset_index in indices
            ]
            array = pa.array(column_values, type=feature_arrow_type(features[key]))
            updated = replace_or_append_column(updated, key, array)
        updated = update_huggingface_schema_metadata(updated, features)
        if backup:
            backup_once(parquet_path)
        temporary = parquet_path.with_name(parquet_path.name + ".tmp")
        pq.write_table(updated, temporary)
        os.replace(temporary, parquet_path)
        cursor += len(table)

    if cursor != frame_count or len(by_index) != frame_count:
        raise ValueError(
            f"Parquet rows do not cover labels exactly: rows={cursor}, "
            f"unique_indices={len(by_index)}, labels={frame_count}"
        )
    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    if backup:
        backup_once(info_path)
        backup_once(stats_path)
    stats = {key: compute_stats(values) for key, values in values_by_key.items()}
    update_meta_files_in_place(root, features, stats)


def main() -> None:
    args = parse_args()
    output_keys = [args.tau_other_key, args.tau_g_key, args.ddq_key]
    if any(not key for key in output_keys) or len(set(output_keys)) != 3:
        raise ValueError("tau_other, tau_g, and ddq output keys must be non-empty and distinct.")
    roots = [path.resolve() for path in args.phase_roots]
    manifests = [load_manifest(root) for root in roots]
    validate_manifests(manifests)
    config = load_config(args.config.resolve())

    for root, manifest in zip(roots, manifests):
        values_by_name, expected_rows = compute_phase_labels(root=root, config=config)
        values_by_key = {
            args.tau_other_key: values_by_name["tau_other"],
            args.tau_g_key: values_by_name["tau_g"],
            args.ddq_key: values_by_name["ddq"],
        }
        phase = int(manifest["phase_index"])
        tau_other_array = np.asarray(values_by_name["tau_other"])
        ddq_array = np.asarray(values_by_name["ddq"])
        print(
            f"phase={phase} root={root} frames={len(expected_rows)} "
            f"tau_other_abs_p99={np.quantile(np.abs(tau_other_array), 0.99):.6f} "
            f"ddq_abs_p99={np.quantile(np.abs(ddq_array), 0.99):.6f}",
            flush=True,
        )
        if not args.dry_run:
            write_phase(
                root,
                values_by_key,
                expected_rows,
                backup=not args.no_backup,
            )
            print(
                f"updated {root}: keys={', '.join(values_by_key)}",
                flush=True,
            )
    if args.dry_run:
        print("dry-run complete; no files were changed", flush=True)


if __name__ == "__main__":
    main()
