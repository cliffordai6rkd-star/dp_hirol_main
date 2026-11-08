"""Create a LIBERO HDF5 copy with causal median-filtered wrench targets.

The CAD-WM LIBERO loader reads nested ``data/demo_i/obs/wrenches`` arrays
directly.  This utility keeps the source file untouched, copies it, and
replaces only those arrays with the existing episode-aware causal trailing
median implementation.  A copy of the unfiltered signal is retained as
``obs/wrenches_raw`` for auditability.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from data_process.causal_data_filter import filter_episode_values


FILTER_CONTRACT = "causal_trailing_median_v1"


def _episode_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group, key=lambda name: int(name.rsplit("_", 1)[-1]))


def filter_file(
    source: Path,
    output: Path,
    *,
    window: int = 3,
    sample_rate_hz: float = 20.0,
) -> dict[str, object]:
    if source.resolve() == output.resolve():
        raise ValueError("output must be different from input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if window < 1 or window % 2 == 0:
        raise ValueError("median window must be a positive odd integer")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".filtering")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")

    shutil.copy2(source, temporary)
    total_frames = 0
    changed_frames = 0
    raw_values: list[np.ndarray] = []
    filtered_values: list[np.ndarray] = []
    try:
        with h5py.File(temporary, "r+") as h5_file:
            data = h5_file["data"]
            names = _episode_names(data)
            if not names:
                raise ValueError("HDF5 data group contains no demos")
            for name in names:
                obs = data[name]["obs"]
                if "wrenches" not in obs:
                    raise KeyError(f"{name}/obs/wrenches is missing")
                if "wrenches_raw" in obs:
                    raise ValueError(f"{name}/obs/wrenches_raw already exists")

                source_dataset = obs["wrenches"]
                raw = np.asarray(source_dataset[...])
                if raw.ndim != 2 or raw.shape[1] != 6:
                    raise ValueError(
                        f"{name}/obs/wrenches must be [T, 6], got {raw.shape}"
                    )
                if raw.dtype.kind not in "fc" or not np.isfinite(raw).all():
                    raise ValueError(f"{name}/obs/wrenches must be finite floating point")

                timestamps = np.arange(len(raw), dtype=np.float64) / sample_rate_hz
                filtered = filter_episode_values(
                    timestamps,
                    raw,
                    [{"type": "median", "window": window}],
                )
                raw_backup = obs.create_dataset(
                    "wrenches_raw",
                    data=raw,
                    dtype=source_dataset.dtype,
                    compression=source_dataset.compression,
                    compression_opts=source_dataset.compression_opts,
                    shuffle=source_dataset.shuffle,
                    fletcher32=source_dataset.fletcher32,
                )
                for key, value in source_dataset.attrs.items():
                    raw_backup.attrs[key] = value
                raw_backup.attrs["preprocessing_filter"] = "none"

                source_dataset[...] = filtered
                source_dataset.attrs["preprocessing_filter"] = FILTER_CONTRACT
                source_dataset.attrs["filter_type"] = "median"
                source_dataset.attrs["filter_window"] = int(window)
                source_dataset.attrs["filter_causal"] = True
                source_dataset.attrs["filter_sample_rate_hz"] = float(sample_rate_hz)
                source_dataset.attrs["filter_timestamp_source"] = "native_row_index"

                total_frames += len(raw)
                changed_frames += int(np.any(np.abs(filtered - raw) > 1e-6, axis=1).sum())
                raw_values.append(raw.astype(np.float64, copy=False))
                filtered_values.append(filtered.astype(np.float64, copy=False))

            h5_file.attrs["wrench_preprocessed"] = True
            h5_file.attrs["wrench_filter_contract"] = FILTER_CONTRACT
            h5_file.attrs["wrench_filter_type"] = "median"
            h5_file.attrs["wrench_filter_window"] = int(window)
            h5_file.attrs["wrench_filter_causal"] = True
            h5_file.attrs["wrench_filter_sample_rate_hz"] = float(sample_rate_hz)
            h5_file.flush()

        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    raw_all = np.concatenate(raw_values, axis=0)
    filtered_all = np.concatenate(filtered_values, axis=0)
    raw_force = np.linalg.norm(raw_all[:, :3], axis=1)
    filtered_force = np.linalg.norm(filtered_all[:, :3], axis=1)
    return {
        "source": str(source),
        "output": str(output),
        "filter_contract": FILTER_CONTRACT,
        "window": int(window),
        "sample_rate_hz": float(sample_rate_hz),
        "frames": total_frames,
        "changed_frames": changed_frames,
        "raw_force_norm_percentiles": np.percentile(
            raw_force, [50, 90, 95, 99, 99.9, 100]
        ).tolist(),
        "filtered_force_norm_percentiles": np.percentile(
            filtered_force, [50, 90, 95, 99, 99.9, 100]
        ).tolist(),
        "raw_force_norm_counts_gt": {
            str(threshold): int((raw_force > threshold).sum())
            for threshold in (8, 100, 200, 300, 400)
        },
        "filtered_force_norm_counts_gt": {
            str(threshold): int((filtered_force > threshold).sum())
            for threshold in (8, 100, 200, 300, 400)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    args = parser.parse_args()
    result = filter_file(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        window=args.window,
        sample_rate_hz=args.sample_rate_hz,
    )
    manifest = args.output.expanduser().resolve().with_suffix(".median_filter.json")
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
