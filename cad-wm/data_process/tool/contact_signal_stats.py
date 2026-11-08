"""Print contact-signal statistics from raw NERO H5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from data_process.tool.contact_signal_rerun import _phase_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "--root", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("force_xyz_l2", "tau_ext_l1"),
        default=None,
        help="Only report one metric; by default both candidates are reported.",
    )
    parser.add_argument("--consecutive", type=int, default=5)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    return parser.parse_args()


def _quantiles(values):
    return {
        f"q{q:g}": float(np.quantile(values, q / 100.0))
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }


def main():
    args = parse_args()
    files = sorted(args.data_dir.glob("episode_*.h5"))
    files.extend(sorted(args.data_dir.glob("episode_*.hdf5")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no episode files found under {args.data_dir}")

    tau_parts = []
    force_parts = []
    for path in files:
        with h5py.File(path, "r") as h5:
            tau_parts.append(np.abs(np.asarray(h5["teleop/tau_ext_cal"])).sum(axis=1))
            wrench = np.asarray(h5["teleop/wrench_cal"])
            force_parts.append(np.linalg.norm(wrench[:, :3], axis=1))
    tau = np.concatenate(tau_parts)
    force = np.concatenate(force_parts)
    metrics = {
        "tau_ext_l1_Nm": (tau, 0.15, 0.75),
        "force_xyz_l2_N": (force, 0.3, 1.5),
    }
    if args.metric == "tau_ext_l1":
        metrics = {"tau_ext_l1_Nm": metrics["tau_ext_l1_Nm"]}
    elif args.metric == "force_xyz_l2":
        metrics = {"force_xyz_l2_N": metrics["force_xyz_l2_N"]}
    report = {"episodes": len(files), "rows": int(tau.size), "metrics": {}}
    for name, (values, off, on) in metrics.items():
        labels = _phase_labels(values, off, on, args.consecutive)
        counts = np.bincount(labels, minlength=3)
        report["metrics"][name] = {
            "threshold_off": off,
            "threshold_on": on,
            "quantiles": _quantiles(values),
            "phase_counts": counts.astype(int).tolist(),
            "phase_fractions": (counts / counts.sum()).tolist(),
        }
    report["log_correlation_tau_vs_force"] = float(
        np.corrcoef(np.log1p(tau), np.log1p(force))[0, 1]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
