#!/usr/bin/env python3
"""Measure the RNEA torque gap caused by causal versus RTS acceleration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from data_process.offline_tau_labels import causal_median_one_pole_filter
from data_process.tool.build_offline_tau_labels import (
    _batched_rnea,
    _build_reduced_model,
    _mapping,
    _read_vector,
    _torque_filter_config,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RNEA(q,dq,ddq_kf_causal) against the stored "
            "RNEA(q,dq,ddq_RTS) label baseline."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/data_process/offline_tau_labels.yaml"),
        help="Offline tau-label config used to resolve keys and the robot model.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="RTS-labelled HDF5 directory; defaults to config io.output_dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze only the first N sorted episodes.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=49,
        help=(
            "Exclude this many samples after every Kalman segment start. "
            "The default matches a 50-frame sequence-model warmup."
        ),
    )
    parser.add_argument(
        "--thresholds-nm",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2),
        help="Absolute torque-gap thresholds used for exceedance fractions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the full JSON report.",
    )
    return parser.parse_args()


def summarize_gap(
    gap_nm: np.ndarray,
    thresholds_nm: Sequence[float],
    joint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    gap_nm = np.asarray(gap_nm, dtype=np.float64)
    if gap_nm.ndim != 2 or gap_nm.shape[0] == 0:
        raise ValueError(f"gap_nm must have non-empty shape [time, joint], got {gap_nm.shape}.")
    if not np.isfinite(gap_nm).all():
        raise ValueError("gap_nm contains non-finite values.")

    thresholds = tuple(float(value) for value in thresholds_nm)
    if not thresholds or any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("thresholds_nm must contain positive finite values.")
    if joint_names is None:
        names = tuple(f"joint_{index + 1}" for index in range(gap_nm.shape[1]))
    else:
        names = tuple(str(name) for name in joint_names)
        if len(names) != gap_nm.shape[1]:
            raise ValueError(
                f"joint_names has {len(names)} entries, expected {gap_nm.shape[1]}."
            )

    def statistics(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        absolute = np.abs(values)
        return {
            "count": int(values.size),
            "bias_nm": float(np.mean(values)),
            "mae_nm": float(np.mean(absolute)),
            "rmse_nm": float(np.sqrt(np.mean(np.square(values)))),
            "p50_abs_nm": float(np.quantile(absolute, 0.50)),
            "p95_abs_nm": float(np.quantile(absolute, 0.95)),
            "p99_abs_nm": float(np.quantile(absolute, 0.99)),
            "max_abs_nm": float(np.max(absolute)),
            "fraction_above_threshold": {
                f"{threshold:g}": float(np.mean(absolute > threshold))
                for threshold in thresholds
            },
        }

    per_joint = []
    for joint_index, joint_name in enumerate(names):
        per_joint.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                **statistics(gap_nm[:, joint_index]),
            }
        )
    return {
        "overall": statistics(gap_nm),
        "per_joint": per_joint,
    }


def _joint_names(model) -> tuple[str, ...]:
    names = []
    for joint_id in range(1, model.njoints):
        joint = model.joints[joint_id]
        names.extend([str(model.names[joint_id])] * int(joint.nv))
    if len(names) != model.nv:
        return tuple(f"joint_{index + 1}" for index in range(model.nv))
    return tuple(names)


def _key_paths(config: Mapping[str, Any]) -> dict[str, str]:
    keys = _mapping(config, "keys")
    return {
        "q": str(keys.get("q", "teleop/q_follower")),
        "dq": str(keys.get("dq", "teleop/dq_follower")),
        "q_rts": str(keys.get("q_rts", "teleop/q_rts")),
        "dq_rts": str(keys.get("dq_rts", "teleop/dq_rts")),
        "ddq_causal": str(keys.get("ddq_causal", "teleop/ddq_kf_causal")),
        "tau_id_rts": str(keys.get("tau_id", "teleop/tau_id_rts")),
        "tau_id_rts_filtered": str(
            keys.get("tau_id_filtered", "teleop/tau_id_rts_filtered")
        ),
        "timestamp": str(keys.get("timestamp", "teleop/timestamp_us")),
    }


def _segment_starts(dataset: h5py.Dataset, frame_count: int) -> tuple[int, ...]:
    payload = dataset.attrs.get("segment_starts_json", "[0]")
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    starts = tuple(int(value) for value in json.loads(str(payload)))
    if not starts:
        starts = (0,)
    if starts[0] != 0 or any(
        start < 0 or start >= frame_count for start in starts
    ) or any(right <= left for left, right in zip(starts, starts[1:])):
        raise ValueError(f"Invalid Kalman segment starts: {starts}")
    return starts


def _evaluation_indices(
    frame_count: int,
    segment_starts: Sequence[int],
    warmup_frames: int,
) -> np.ndarray:
    keep = np.ones(frame_count, dtype=bool)
    for start in segment_starts:
        keep[start : min(start + warmup_frames, frame_count)] = False
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError(
            f"warmup_frames={warmup_frames} excludes all {frame_count} frames."
        )
    return indices


def analyze_episode(
    path: Path,
    *,
    pin,
    model,
    config: Mapping[str, Any],
    warmup_frames: int,
    thresholds_nm: Sequence[float],
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    keys = _key_paths(config)
    processing = _mapping(config, "processing")
    rnea_source = str(processing.get("rnea_state_source", "measured"))
    timestamp_scale = float(processing.get("timestamp_scale_to_s", 1.0e-6))
    cutoff_hz, median_window = _torque_filter_config(processing)
    if rnea_source not in {"measured", "smoothed"}:
        raise ValueError("processing.rnea_state_source must be measured or smoothed.")

    with h5py.File(path, "r") as h5_file:
        q_measured = _read_vector(h5_file, keys["q"])
        dq_measured = _read_vector(h5_file, keys["dq"])
        q_rts = _read_vector(h5_file, keys["q_rts"])
        dq_rts = _read_vector(h5_file, keys["dq_rts"])
        ddq_dataset = h5_file[keys["ddq_causal"]]
        if not isinstance(ddq_dataset, h5py.Dataset):
            raise TypeError(f"HDF5 path is not a dataset: {keys['ddq_causal']}")
        ddq_causal = np.asarray(ddq_dataset, dtype=np.float64)
        tau_id_rts = _read_vector(h5_file, keys["tau_id_rts"])
        tau_id_rts_filtered = _read_vector(
            h5_file,
            keys["tau_id_rts_filtered"],
        )
        timestamps_s = (
            np.asarray(h5_file[keys["timestamp"]], dtype=np.float64)
            * timestamp_scale
        )
        segment_starts = _segment_starts(ddq_dataset, len(q_measured))

    arrays = (
        q_measured,
        dq_measured,
        q_rts,
        dq_rts,
        ddq_causal,
        tau_id_rts,
        tau_id_rts_filtered,
    )
    if any(value.shape != q_measured.shape for value in arrays[1:]):
        shapes = [value.shape for value in arrays]
        raise ValueError(f"Episode arrays must share one shape, got {shapes} in {path}.")
    if rnea_source == "measured":
        q_rnea = np.where(np.isfinite(q_measured), q_measured, q_rts)
        dq_rnea = np.where(np.isfinite(dq_measured), dq_measured, dq_rts)
    else:
        q_rnea, dq_rnea = q_rts, dq_rts

    tau_id_causal = _batched_rnea(
        pin,
        model,
        q_rnea,
        dq_rnea,
        ddq_causal,
    )
    tau_id_causal_filtered = causal_median_one_pole_filter(
        timestamps_s,
        tau_id_causal,
        cutoff_hz=cutoff_hz,
        median_window=median_window,
    )
    indices = _evaluation_indices(len(q_rnea), segment_starts, warmup_frames)
    raw_gap_nm = tau_id_causal[indices] - tau_id_rts[indices]
    filtered_gap_nm = (
        tau_id_causal_filtered[indices] - tau_id_rts_filtered[indices]
    )
    raw_summary = summarize_gap(raw_gap_nm, thresholds_nm, joint_names)
    filtered_summary = summarize_gap(filtered_gap_nm, thresholds_nm, joint_names)
    return filtered_gap_nm, raw_gap_nm, {
        "file": path.name,
        "input_frames": int(len(q_rnea)),
        "evaluated_frames": int(len(indices)),
        "excluded_warmup_frames": int(len(q_rnea) - len(indices)),
        "segment_starts": list(segment_starts),
        "matched_filter": filtered_summary,
        "raw": raw_summary,
    }


def _print_summary(summary: Mapping[str, Any], thresholds_nm: Sequence[float]) -> None:
    thresholds = tuple(float(value) for value in thresholds_nm)
    threshold_headers = " ".join(f">{value:g}" for value in thresholds)
    print(
        "joint                 bias      MAE     RMSE      P95      P99      max "
        + threshold_headers
    )

    def print_row(label: str, values: Mapping[str, Any]) -> None:
        fractions = " ".join(
            f"{100.0 * values['fraction_above_threshold'][f'{value:g}']:6.2f}%"
            for value in thresholds
        )
        print(
            f"{label:<20} "
            f"{values['bias_nm']:8.4f} {values['mae_nm']:8.4f} "
            f"{values['rmse_nm']:8.4f} {values['p95_abs_nm']:8.4f} "
            f"{values['p99_abs_nm']:8.4f} {values['max_abs_nm']:8.4f} "
            f"{fractions}"
        )

    for joint in summary["per_joint"]:
        label = f"J{joint['joint_index'] + 1}:{joint['joint_name']}"
        print_row(label, joint)
    print_row("ALL", summary["overall"])


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")
    thresholds_nm = tuple(sorted(set(float(value) for value in args.thresholds_nm)))
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds_nm):
        raise ValueError("--thresholds-nm must contain positive finite values.")

    config = load_config(args.config)
    io_config = _mapping(config, "io")
    input_dir = args.input_dir or Path(io_config.get("output_dir", ""))
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    patterns = tuple(io_config.get("patterns", ("*.h5", "*.hdf5")))
    files = sorted({path for pattern in patterns for path in input_dir.glob(pattern)})
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No input episodes found under {input_dir}")

    pin, model, urdf_path = _build_reduced_model(_mapping(config, "robot"))
    joint_names = _joint_names(model)
    filtered_gaps = []
    raw_gaps = []
    episodes = []
    for path in files:
        filtered_gap_nm, raw_gap_nm, episode = analyze_episode(
            path,
            pin=pin,
            model=model,
            config=config,
            warmup_frames=args.warmup_frames,
            thresholds_nm=thresholds_nm,
            joint_names=joint_names,
        )
        filtered_gaps.append(filtered_gap_nm)
        raw_gaps.append(raw_gap_nm)
        episodes.append(episode)
        print(
            f"analyzed {path.name}: frames={episode['evaluated_frames']} "
            f"matched_mae={episode['matched_filter']['overall']['mae_nm']:.4f} Nm "
            f"matched_p99={episode['matched_filter']['overall']['p99_abs_nm']:.4f} Nm"
        )

    aggregate = summarize_gap(
        np.concatenate(filtered_gaps, axis=0),
        thresholds_nm,
        joint_names,
    )
    raw_aggregate = summarize_gap(
        np.concatenate(raw_gaps, axis=0),
        thresholds_nm,
        joint_names,
    )
    report = {
        "definition": (
            "causal_lowpass(RNEA(q,dq,ddq_kf_causal))-"
            "causal_lowpass(RNEA(q,dq,ddq_RTS))"
        ),
        "raw_definition": "RNEA(q,dq,ddq_kf_causal)-RNEA(q,dq,ddq_RTS)",
        "unit": "N*m",
        "config": str(args.config),
        "input_dir": str(input_dir),
        "urdf_path": str(urdf_path),
        "rnea_state_source": str(
            _mapping(config, "processing").get("rnea_state_source", "measured")
        ),
        "warmup_frames_per_segment": args.warmup_frames,
        "thresholds_nm": list(thresholds_nm),
        "episode_count": len(episodes),
        "input_frames": int(sum(item["input_frames"] for item in episodes)),
        "evaluated_frames": int(sum(item["evaluated_frames"] for item in episodes)),
        "aggregate": aggregate,
        "raw_aggregate": raw_aggregate,
        "episodes": episodes,
    }

    print("\nAggregate matched-filter causal-vs-RTS RNEA gap (Nm)")
    _print_summary(aggregate, thresholds_nm)
    print("\nAggregate raw causal-vs-RTS RNEA gap (Nm)")
    _print_summary(raw_aggregate, thresholds_nm)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {args.output}")


if __name__ == "__main__":
    main()
