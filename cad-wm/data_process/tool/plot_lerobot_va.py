from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot velocity and acceleration for one LeRobot episode.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/train_episode/wrench_background/wrench_bg_lerobotv3_dv"),
        help="LeRobot dataset root.",
    )
    parser.add_argument("--repo-id", default="wrench_bg_lerobotv3_dv")
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--velocity-key", default="observation.velocity")
    parser.add_argument("--acceleration-key", default="observation.acceleration")
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_plots"))
    return parser.parse_args()


def to_numpy(value: Any):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def plot_channels(time, values, title: str, ylabel: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 5))
    for idx in range(values.shape[1]):
        axis.plot(time, values[:, idx], linewidth=0.9, label=f"{ylabel}{idx}")

    axis.set_title(title)
    axis.set_xlabel("timestamp (s)")
    axis.set_ylabel(ylabel)
    axis.grid(True)
    axis.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    episode = dataset.meta.episodes[args.episode_idx]
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])

    timestamps = []
    velocities = []
    accelerations = []

    for raw_idx in range(start, end):
        frame = dataset.hf_dataset[raw_idx]
        timestamps.append(float(frame[args.timestamp_key]))
        velocities.append(to_numpy(frame[args.velocity_key]))
        accelerations.append(to_numpy(frame[args.acceleration_key]))

    import numpy as np

    time = np.asarray(timestamps, dtype=np.float64)
    velocity = np.stack(velocities, axis=0)
    acceleration = np.stack(accelerations, axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    velocity_path = args.output_dir / f"episode_{args.episode_idx:03d}_velocity.png"
    acceleration_path = args.output_dir / f"episode_{args.episode_idx:03d}_acceleration.png"

    plot_channels(time, velocity, f"Episode {args.episode_idx} velocity", "v", velocity_path)
    plot_channels(time, acceleration, f"Episode {args.episode_idx} acceleration", "a", acceleration_path)

    print(f"saved velocity plot: {velocity_path}")
    print(f"saved acceleration plot: {acceleration_path}")


if __name__ == "__main__":
    main()
