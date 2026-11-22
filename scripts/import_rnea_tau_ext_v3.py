#!/usr/bin/env python3
"""Write previously computed RNEA torques into a LeRobot v3 dataset in place."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter

REPO = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO))

from scripts.add_xarm_wrench_v3 import add_columns, feature_stats, write_json


def import_estimates(dataset, estimates, backup, urdf):
    dataset, estimates, backup, urdf = [Path(p).resolve() for p in (dataset, estimates, backup, urdf)]
    info = json.loads((dataset / "meta/info.json").read_text())
    processing = json.loads((estimates.parent / "processing.json").read_text())
    if Path(processing["source_dataset"]).resolve() != dataset:
        raise ValueError("Estimates refer to a different dataset")
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != processing["urdf_sha256"]:
        raise ValueError("URDF does not match the estimation model")
    if backup.exists() or dataset == backup or dataset in backup.parents:
        raise ValueError("Choose a new backup directory outside the dataset")
    files = sorted((dataset / "data").rglob("*.parquet"))
    tables = [pq.read_table(path) for path in files]
    original = pa.concat_tables(tables)
    estimated = pq.read_table(estimates)
    if len(original) != info["total_frames"] or len(original) != len(estimated):
        raise ValueError("Frame counts differ")
    # Compare every original value, including timestamps and episode/frame IDs.
    for key in original.column_names:
        if not original[key].combine_chunks().equals(estimated[key].combine_chunks()):
            raise ValueError(f"Estimates do not align with dataset column {key}")
    config = processing["config"]
    if config["fps"] != info["fps"]:
        raise ValueError("Estimation FPS differs from the dataset")
    prefix = "estimate.rnea."
    before = np.asarray(estimated[prefix + "tau_ext_unthresholded"].to_pylist(), dtype=np.float32)
    after = np.asarray(estimated[prefix + "tau_ext_gated"].to_pylist(), dtype=np.float32)
    l1 = np.abs(before.astype(np.float64)).sum(axis=1)
    contact = l1 >= config["contact_threshold_l1"]
    np.testing.assert_array_equal(after, np.where(contact[:, None], before, np.zeros_like(before)))
    np.testing.assert_allclose(l1, estimated[prefix + "l1_unthresholded"].to_numpy(), rtol=0, atol=1e-10)
    columns = {
        "observation.tau_ext": after,
        "observation.tau_ext_unthresholded": before,
        "observation.tau_ext_l1": l1.astype(np.float32)[:, None],
        "observation.contact": contact.astype(np.float32)[:, None],
    }
    for key, values in columns.items():
        if key in info["features"] or key in original.column_names or not np.isfinite(values).all():
            raise ValueError(f"Field already exists or has invalid values: {key}")
    episode_ids = original["episode_index"].to_numpy()
    episodes = np.unique(episode_ids)
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode count differs from metadata")
    episode_stats = {
        int(ep): {key: feature_stats(values[episode_ids == ep]) for key, values in columns.items()}
        for ep in episodes
    }
    stats = json.loads((dataset / "meta/stats.json").read_text())
    for key, values in columns.items():
        stats[key] = feature_stats(values)
        info["features"][key] = {"dtype": "float32", "shape": [values.shape[1]], "names": None}
    with tempfile.TemporaryDirectory(prefix=".tau-ext-import-", dir=dataset.parent) as temporary:
        stage = Path(temporary)
        offset = 0
        for path, table in zip(files, tables):
            target = stage / path.relative_to(dataset)
            target.parent.mkdir(parents=True, exist_ok=True)
            end = offset + len(table)
            pq.write_table(add_columns(table, {key: value[offset:end] for key, value in columns.items()}),
                           target, compression="snappy")
            offset = end
        seen = []
        for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
            table = pq.read_table(path)
            ids = table["episode_index"].to_pylist()
            seen.extend(ids)
            for key in columns:
                for name in stats[key]:
                    table = table.append_column(f"stats/{key}/{name}", pa.array(
                        [episode_stats[ep][key][name] for ep in ids],
                        type=pa.list_(pa.int64() if name == "count" else pa.float64())))
            target = stage / path.relative_to(dataset)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target, compression="snappy")
        if sorted(seen) != episodes.tolist():
            raise ValueError("Episode metadata differs from data")
        write_json(stage / "meta/info.json", info)
        write_json(stage / "meta/stats.json", stats)
        model = stage / "meta/wrench_model"
        model.mkdir()
        shutil.copy2(urdf, model / "robot.urdf")
        license_file = REPO / "scripts/assets/xarm7/LICENSE.xarm_ros2"
        if license_file.is_file():
            shutil.copy2(license_file, model / license_file.name)
        record = {
            "config": config, "filter_mode": "causal", "estimator": "Pinocchio RNEA residual",
            "urdf_sha256": processing["urdf_sha256"],
            "sos_coefficients": butter(2, config["cutoff_hz"], fs=config["fps"], output="sos").tolist(),
            "pre_gate_field": "observation.tau_ext_unthresholded",
            "pre_gate_l1_field": "observation.tau_ext_l1",
            "tau_ext_field": "observation.tau_ext", "units": "N m",
            "postprocessing": processing["shared_preprocessing"],
            "threshold": {"norm": "L1 over J1..J7", "value_Nm": config["contact_threshold_l1"],
                          "rule": "L1 < threshold zeros the entire vector; equality retained"},
            "source_estimates": str(estimates),
            "source_estimates_sha256": hashlib.sha256(estimates.read_bytes()).hexdigest(),
            "source_processing": processing,
            "online_runtime": "diffusion_policy.common.xarm_tau_ext.CausalTauExt.from_bundle(dataset_root)",
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_files_backup": str(backup),
        }
        write_json(stage / "meta/tau_ext_processing.json", record)
        # Keep originals outside the dataset so recursive Parquet readers cannot load them.
        staged_files = sorted(path for path in stage.rglob("*") if path.is_file())
        backup.mkdir(parents=True)
        for path in staged_files:
            relative = path.relative_to(stage)
            current = dataset / relative
            if current.is_file():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, saved)
        shutil.copy2(Path(__file__), backup / "import_rnea_tau_ext_v3.py")
        write_json(backup / "import_manifest.json", {
            "dataset": str(dataset), "estimates": str(estimates),
            "written_files": [str(path.relative_to(stage)) for path in staged_files],
        })
        for path in staged_files:
            target = dataset / path.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    print(json.dumps({"dataset": str(dataset), "episodes": len(episodes), "frames": len(original),
                      "nonzero_tau_ext_frames": int(contact.sum()), "backup": str(backup)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=REPO / "scripts/assets/xarm7/xarm7_dynamics.urdf")
    args = parser.parse_args()
    import_estimates(args.dataset, args.estimates, args.backup, args.urdf)


if __name__ == "__main__":
    main()
