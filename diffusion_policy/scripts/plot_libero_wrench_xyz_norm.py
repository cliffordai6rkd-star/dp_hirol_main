"""Plot the XYZ force magnitude for the first N LeRobot v3 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def _read_episode_table(dataset_dir: Path, episode_count: int):
    episodes_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pq.read_table(
        episodes_path,
        columns=["episode_index", "dataset_from_index", "dataset_to_index"],
    ).to_pydict()
    selected = list(zip(
        episodes["episode_index"][:episode_count],
        episodes["dataset_from_index"][:episode_count],
        episodes["dataset_to_index"][:episode_count],
    ))
    if not selected:
        raise ValueError("The dataset has no episodes")

    data_root = dataset_dir / "data"
    data_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {data_root}")
    table = pq.read_table(data_files, columns=[
        "observation.wrench", "timestamp", "episode_index", "frame_index",
    ])
    wrench = np.stack(table["observation.wrench"].to_numpy(zero_copy_only=False))
    timestamp = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
    episode_index = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if wrench.ndim != 2 or wrench.shape[1] < 3:
        raise ValueError(f"Expected wrench shape [N, >=3], got {wrench.shape}")
    return selected, wrench[:, :3], timestamp, episode_index, frame_index


def plot(
    dataset_dir: Path,
    output: Path,
    csv_output: Path,
    episode_count: int = 10,
    contact_threshold: float | None = None,
) -> dict:
    selected, xyz, timestamp, episode_index, frame_index = _read_episode_table(
        dataset_dir, episode_count,
    )
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 15), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    csv_rows = []
    summary = []
    for plot_index, (episode, start, stop) in enumerate(selected):
        mask = episode_index == episode
        # The episode table is authoritative; the mask protects against a
        # malformed index range while preserving the displayed episode id.
        indices = np.arange(start, stop, dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < len(episode_index))]
        indices = indices[episode_index[indices] == episode]
        if len(indices) == 0:
            indices = np.flatnonzero(mask)
        t = timestamp[indices] - timestamp[indices[0]]
        norm = np.linalg.norm(xyz[indices], axis=1)
        ax = axes[plot_index]
        ax.plot(t, norm, linewidth=1.2, color="#1769aa")
        ax.fill_between(t, 0.0, norm, color="#1769aa", alpha=0.12)
        if contact_threshold is not None:
            contact = norm >= contact_threshold
            ax.axhline(
                contact_threshold, color="#c0392b", linestyle="--", linewidth=1.0,
                label=f"contact threshold = {contact_threshold:g}",
            )
            ax.fill_between(
                t, contact_threshold, norm, where=contact, interpolate=True,
                color="#c0392b", alpha=0.30,
            )
        ax.set_title(f"Episode {episode}  ({len(indices)} frames)")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$\|F_{xyz}\|_2")
        summary.append({
            "episode": int(episode),
            "frames": int(len(indices)),
            "duration_seconds": float(t[-1] if len(t) else 0.0),
            "min_xyz_norm": float(norm.min()),
            "max_xyz_norm": float(norm.max()),
            "mean_xyz_norm": float(norm.mean()),
        })
        for idx, time_value, force, value in zip(indices, t, xyz[indices], norm):
            csv_rows.append([int(episode), int(frame_index[idx]), float(time_value),
                             float(force[0]), float(force[1]), float(force[2]), float(value)])

    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(
        f"{dataset_dir.name}: first {len(selected)} episodes, XYZ force magnitude"
        + (f" (contact >= {contact_threshold:g})" if contact_threshold is not None else ""),
        fontsize=16,
    )
    if contact_threshold is not None:
        axes[0].legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_output,
        np.asarray(csv_rows, dtype=np.float64),
        delimiter=",",
        header="episode,frame_index,time_s,fx,fy,fz,xyz_norm",
        comments="",
    )
    return {"dataset": str(dataset_dir), "episodes": summary,
            "formula": "sqrt(Fx^2 + Fy^2 + Fz^2)",
            "contact_threshold": contact_threshold,
            "plot": str(output), "csv": str(csv_output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.dataset / "analysis" / "wrench_xyz_norm_first10_episodes.png"
    csv_output = output.with_suffix(".csv")
    result = plot(
        args.dataset, output, csv_output, args.episodes,
        contact_threshold=args.contact_threshold,
    )
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
