r"""Plot raw and low-pass filtered six-axis LeRobot v3 wrench in separate panels.

Example:
    python -m diffusion_policy.scripts.visualize_lerobot_v3_wrench \
        --dataset data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3 \
        --episode 0 --cutoff-hz 2 --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pyarrow.parquet as pq

from diffusion_policy.common.wrench_filter import first_order_lowpass, relative_time_seconds


CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def load_episode(
    dataset: Path,
    episode: int,
    wrench_key: str = "observation.wrench",
    timestamp_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read one episode across v3 shards, ordered by its native frame index."""
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data Parquet files found under {dataset / 'data'}")
    names = pq.ParquetFile(files[0]).schema_arrow.names
    if timestamp_key is None:
        timestamp_key = (
            "timing.state_timestamp_ns" if "timing.state_timestamp_ns" in names else "timestamp"
        )
    columns = ["episode_index", "frame_index", wrench_key, timestamp_key]
    missing = set(columns) - set(names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    table = pq.read_table(files, columns=columns, filters=[("episode_index", "=", episode)])
    if not len(table):
        raise ValueError(f"Episode {episode} was not found in {dataset}")
    table = table.sort_by([("frame_index", "ascending")])
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    wrench = np.asarray(table[wrench_key].to_pylist(), dtype=np.float64)
    if wrench.shape != (len(table), 6):
        raise ValueError(f"{wrench_key} must have shape [N, 6], got {wrench.shape}")

    timestamps = np.asarray(table[timestamp_key].to_pylist()).reshape(-1)
    if timestamps.shape != (len(table),):
        raise ValueError(f"{timestamp_key} must contain one timestamp per frame")
    unit = next((suffix for suffix in ("ns", "us", "ms")
                 if timestamp_key.endswith("_" + suffix)), "s")
    times = relative_time_seconds(timestamps, unit)
    return times, wrench, frame_indices, timestamp_key


def visualize(
    dataset: Path,
    episode: int,
    cutoff_hz: float,
    *,
    output: Path | None = None,
    wrench_key: str = "observation.wrench",
    timestamp_key: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Save raw/filtered panels and a CSV without editing the dataset."""
    dataset = Path(dataset).expanduser().resolve()
    if episode < 0:
        raise ValueError("--episode must be a non-negative episode_index")
    times, raw, frame_indices, clock = load_episode(dataset, episode, wrench_key, timestamp_key)
    filtered = first_order_lowpass(times, raw, cutoff_hz)
    output = (
        Path(output).expanduser().resolve() if output is not None else
        dataset / "analysis" / f"wrench_episode_{episode}_lowpass_{cutoff_hz:g}hz.png"
    )
    if output.suffix.lower() != ".png":
        raise ValueError("--output must be a .png file")
    csv_output = output.with_suffix(".csv")
    for path in (output, csv_output):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace it")

    figure = Figure(figsize=(14, 9))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True, sharey=True)
    force_bound = 1.05 * (float(np.abs(raw[:, :3]).max()) or 1.0)
    moment_bound = 1.05 * (float(np.abs(raw[:, 3:]).max()) or 1.0)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    legend_lines = []
    for panel, (ax, values, title) in enumerate(zip(
        axes, (raw, filtered),
        ("Original wrench", f"Filtered wrench | 1st-order low-pass, cutoff {cutoff_hz:g} Hz"),
    )):
        moments = ax.twinx()
        for index, (name, color) in enumerate(zip(CHANNELS, colors)):
            channel_ax = ax if index < 3 else moments
            line, = channel_ax.plot(times, values[:, index], color=color, linewidth=1.25,
                                    linestyle="-" if index < 3 else "--", label=name)
            if panel == 0:
                legend_lines.append(line)
        ax.set_title(title)
        ax.set_ylabel("Force (N, solid lines)")
        moments.set_ylabel("Moment (N·m, dashed lines)")
        # Symmetric limits align both physical zeros; otherwise a zero moment
        # can appear at hundreds of newtons when read against the force axis.
        ax.set_ylim(-force_bound, force_bound)
        moments.set_ylim(-moment_bound, moment_bound)
        ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time from episode start (s)")
    figure.suptitle(
        f"{dataset.name} | episode {episode} | {len(times)} frames"
    )
    figure.legend(legend_lines, CHANNELS, loc="lower center", ncol=6)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    header = ["frame_index", "time_s"]
    header += [f"raw_{name.lower()}" for name in CHANNELS]
    header += [f"filtered_{name.lower()}" for name in CHANNELS]
    np.savetxt(csv_output, np.column_stack((frame_indices, times, raw, filtered)),
               delimiter=",", header=",".join(header), comments="", fmt=["%d"] + ["%.12g"] * 13)
    return {"plot": output, "csv": csv_output, "frames": len(times), "timestamp_key": clock}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3 dataset root (data/ and meta/)")
    parser.add_argument("--episode", type=int, required=True, help="Episode index, starting at 0")
    parser.add_argument("--cutoff-hz", type=float, required=True, help="First-order low-pass cutoff frequency in Hz")
    parser.add_argument("--wrench-key", default="observation.wrench", help="Six-axis column (default: observation.wrench)")
    parser.add_argument("--timestamp-key", default=None,
                        help="Clock column; auto: timing.state_timestamp_ns, then timestamp. _ns/_us/_ms suffixes specify units")
    parser.add_argument("--output", type=Path, default=None,
                        help="PNG path (default: DATASET/analysis/wrench_episode_ID_lowpass_FChz.png); also saves .csv")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing PNG/CSV outputs; source dataset is unchanged")
    args = parser.parse_args(argv)
    try:
        result = visualize(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Episode {args.episode}: {result['frames']} frames, clock={result['timestamp_key']}, cutoff={args.cutoff_hz:g} Hz")
    print(f"Plot: {result['plot']}")
    print(f"CSV:  {result['csv']}")


if __name__ == "__main__":
    main()
