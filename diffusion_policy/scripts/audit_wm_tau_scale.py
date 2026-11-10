"""Audit stored CAD-WM torque scales without changing data or model parameters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LIBERO = ROOT.parent / "LIBERO-FT"
sys.path.insert(0, str(LIBERO))
from inference.wrench_filter import lowpass_wrench


def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = checkpoint["config"]["dataloader"]["h5_fields"]["tau"]
    torque, force_raw, force_filtered, indices, episodes = [], [], [], [], []
    with h5py.File(args.dataset, "r") as source:
        for name in sorted(source["data"], key=lambda s: int(s.rsplit("_", 1)[-1])):
            group = source["data"][name]
            values = np.asarray(group[spec["path"]], dtype=np.float64)
            if spec.get("columns") is not None:
                values = values[:, spec["columns"]]
            wrench = np.asarray(group["obs/wrenches"], dtype=np.float64)
            filtered = lowpass_wrench(wrench, np.full(len(values), .05), 2.)
            torque.append(values)
            force_raw.append(wrench[:, :3])
            force_filtered.append(filtered[:, :3])
            indices.extend((name, frame) for frame in range(len(values)))
            episodes.append({"episode": name, "rows": len(values),
                "rows_any_abs_tau_gt_1000": int((np.abs(values) > 1000).any(1).sum()),
                "max_abs_tau": float(np.abs(values).max())})
        attrs = dict(source["data/demo_0"][spec["path"]].attrs)
    values = np.concatenate(torque)
    raw = np.concatenate(force_raw)
    filtered = np.concatenate(force_filtered)
    absolute = np.abs(values)
    centered_squares = (values-values.mean(0))**2
    sorted_squares = np.sort(centered_squares, axis=0)[::-1]
    quantiles = [50, 90, 95, 99, 99.9, 100]
    thresholds = {}
    for threshold in (100, 1000, 10000):
        mask = absolute > threshold
        thresholds[str(threshold)] = {"rows_any_joint": int(mask.any(1).sum()),
            "fraction_rows_any_joint": float(mask.any(1).mean()),
            "fraction_rows_per_joint": mask.mean(0).tolist()}
    variance = {}
    for fraction in (.001, .01, .05):
        count = max(1, int(np.ceil(len(values)*fraction)))
        variance[str(fraction)] = {"rows_per_joint": count,
            "fraction_of_total_centered_squared_deviation_per_joint":
                (sorted_squares[:count].sum(0)/sorted_squares.sum(0)).tolist()}
    extremes = []
    for index in np.argsort(absolute.max(1))[-12:][::-1]:
        name, frame = indices[index]
        extremes.append({"episode": name, "frame": frame, "tau": values[index].tolist(),
            "max_abs_tau": float(absolute[index].max()),
            "raw_force_norm_N": float(np.linalg.norm(raw[index])),
            "filtered_force_norm_N": float(np.linalg.norm(filtered[index]))})
    phases = {}
    for kind, force in (("raw", raw), ("causal_2hz_filtered", filtered)):
        contact = np.linalg.norm(force, axis=1) > 8
        for name, mask in (("below_or_equal_8N", ~contact), ("above_8N", contact)):
            phases[f"{kind}_{name}"] = {"rows": int(mask.sum()),
                "median_abs_tau_per_joint": np.median(absolute[mask], axis=0).tolist(),
                "p99_abs_tau_per_joint": np.percentile(absolute[mask], 99, axis=0).tolist(),
                "fraction_rows_any_abs_tau_gt_1000": float((absolute[mask] > 1000).any(1).mean())}
    saved = {k: v.tolist() for k, v in checkpoint["normalizer"]["stats"]["tau"].items()}
    report = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "rows": len(values), "episodes": len(episodes), "tau_field": spec,
        "tau_attributes": {k: v.item() if isinstance(v, np.generic) else v for k,v in attrs.items()},
        "checkpoint_tau_normalizer": saved,
        "computed_mean": values.mean(0).tolist(), "computed_population_std": values.std(0).tolist(),
        "checkpoint_std_matches_all_rows": bool(np.allclose(saved["std"], values.std(0), rtol=1e-6)),
        "absolute_quantiles_per_joint": {str(q): v.tolist() for q,v in
            zip(quantiles, np.percentile(absolute, quantiles, axis=0))},
        "thresholds": thresholds, "top_tail_variance_contribution": variance,
        "one_unit_tau_change_in_normalized_coordinates": (1/np.asarray(saved["std"])).tolist(),
        "extreme_rows": extremes, "phase_statistics": phases, "episode_statistics": episodes,
        "preprocessing": "H5 tau first seven columns pass unfiltered into Gaussian normalization. Only wrench is low-passed.",
        "limitations": [
            "Stored field is described as MuJoCo inverse joint torque; no explicit units attribute exists.",
            "Large numerical tails and normalization compression are verified; the source of the tails is not established.",
            "Large tau at low wrist force does not by itself prove a unit or joint-mapping bug.",
            "Phase groups use force magnitude only, not geometric contact or grasp success.",
            "Do not alter normalization of this checkpoint without retraining; that would change its input contract."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "rows": len(values), "std": saved["std"],
        "thresholds": thresholds, "top_tail_variance_contribution": variance}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"cad-wm/outputs/cad_wm_lowpass2hz/checkpoints/step_00200000.pt")
    parser.add_argument("--dataset", type=Path, default=LIBERO/"dataset/wm_data/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench_with_dq_tau.hdf5")
    parser.add_argument("--output", type=Path, default=LIBERO/"outputs/inference/offline_contact_diagnostics_20261110/tau_nominal/tau_scale_audit.json")
    run(parser.parse_args())
