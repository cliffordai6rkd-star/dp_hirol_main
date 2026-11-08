"""Interactive Rerun viewer for contact signals in raw NERO episodes.

The viewer is intentionally independent of the WM dataset/trainer. It reads
one raw H5 episode at a time, shows both cameras, and logs the two candidate
contact signals with the three-phase hysteresis label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review tau_ext/wrench contact signals and two cameras in Rerun."
    )
    parser.add_argument(
        "--data-dir",
        "--root",
        type=Path,
        required=True,
        help="Directory containing episode_*.h5 files.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Zero-based sorted file index; omit to select interactively.",
    )
    parser.add_argument("--off", type=float, default=None)
    parser.add_argument("--on", type=float, default=None)
    parser.add_argument(
        "--metric",
        choices=("force_xyz_l2", "tau_ext_l1"),
        default="force_xyz_l2",
        help="Signal used for the displayed three-phase hysteresis label.",
    )
    parser.add_argument("--consecutive", type=int, default=5)
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not launch a Rerun viewer process automatically.",
    )
    return parser.parse_args()


def _phase_labels(signal, off, on, consecutive):
    if off < 0.0 or on <= off or consecutive < 1:
        raise ValueError("require 0 <= off < on and consecutive >= 1")
    labels = np.zeros(signal.shape[0], dtype=np.int8)
    state = 0
    high_count = 0
    low_count = 0
    for i, value in enumerate(signal):
        if state == 2:
            labels[i] = 2
            low_count = low_count + 1 if value <= off else 0
            if low_count >= consecutive:
                state = 0
                labels[i - consecutive + 1 : i + 1] = 0
                low_count = 0
            continue
        if value <= off:
            state = 0
            high_count = 0
            labels[i] = 0
            continue
        labels[i] = 1
        high_count = high_count + 1 if value >= on else 0
        if high_count >= consecutive:
            state = 2
            labels[i - consecutive + 1 : i + 1] = 2
            high_count = 0
    return labels


def _camera_index(camera_timestamps, state_timestamp):
    index = int(np.searchsorted(camera_timestamps, state_timestamp, side="left"))
    return min(max(index, 0), len(camera_timestamps) - 1)


def _blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Wrist Camera", origin="cameras/wrist", contents="$origin/**"
                ),
                rrb.Spatial2DView(
                    name="Side Camera", origin="cameras/side", contents="$origin/**"
                ),
                row_shares=[1, 1],
            ),
            rrb.TimeSeriesView(
                name="Contact Signals", origin="signals", contents="$origin/**"
            ),
            column_shares=[1, 1],
        )
    )


def _log_episode(path: Path, metric: str, off: float, on: float, consecutive: int):
    import h5py
    import rerun as rr

    with h5py.File(path, "r") as h5:
        state_ts = np.asarray(h5["teleop/timestamp_us"], dtype=np.int64) * 1000
        tau_ext = np.asarray(h5["teleop/tau_ext_cal"], dtype=np.float64)
        wrench = np.asarray(h5["teleop/wrench_cal"], dtype=np.float64)
        wrist_ts = np.asarray(h5["cameras/wrist/timestamp_us"], dtype=np.int64) * 1000
        side_ts = np.asarray(h5["cameras/side/timestamp_us"], dtype=np.int64) * 1000
        wrist_frames = h5["cameras/wrist/frames"]
        side_frames = h5["cameras/side/frames"]

        tau_l1 = np.abs(tau_ext).sum(axis=1)
        force_xyz_l2 = np.linalg.norm(wrench[:, :3], axis=1)
        selected_signal = force_xyz_l2 if metric == "force_xyz_l2" else tau_l1
        phase = _phase_labels(selected_signal, off, on, consecutive)

        for idx, timestamp in enumerate(state_ts):
            rr.set_time("idx", sequence=int(idx))
            rr.log(
                "cameras/wrist",
                rr.Image(np.asarray(wrist_frames[_camera_index(wrist_ts, timestamp)])),
            )
            rr.log(
                "cameras/side",
                rr.Image(np.asarray(side_frames[_camera_index(side_ts, timestamp)])),
            )

            rr.log("signals/tau_ext_l1", rr.Scalars(float(tau_l1[idx])))
            rr.log("signals/force_xyz_l2", rr.Scalars(float(force_xyz_l2[idx])))
            rr.log("signals/selected_metric", rr.Scalars(float(selected_signal[idx])))
            rr.log("signals/phase", rr.Scalars(float(phase[idx])))
            rr.log("signals/threshold/off", rr.Scalars(float(off)))
            rr.log("signals/threshold/on", rr.Scalars(float(on)))
            for ch, name in enumerate(("fx", "fy", "fz")):
                rr.log(f"signals/wrench/{name}", rr.Scalars(float(wrench[idx, ch])))


def _run_one(path: Path, args):
    import rerun as rr

    rr.init(
        "contact_signal_review",
        recording_id=path.stem,
        spawn=not args.no_spawn,
        default_blueprint=_blueprint(),
    )
    _log_episode(path, args.metric, args.off, args.on, args.consecutive)
    if not args.no_spawn:
        input(f"{path.name} 已播放，回车返回 episode 选择，输入 q 退出: ")
    rr.disconnect()


def main():
    args = parse_args()
    defaults = {
        "force_xyz_l2": (0.3, 1.5),
        "tau_ext_l1": (0.15, 0.75),
    }
    default_off, default_on = defaults[args.metric]
    args.off = default_off if args.off is None else args.off
    args.on = default_on if args.on is None else args.on
    files = sorted(args.data_dir.glob("episode_*.h5"))
    files.extend(sorted(args.data_dir.glob("episode_*.hdf5")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no episode_*.h5 files found under {args.data_dir}")

    if args.episode is not None:
        if not 0 <= args.episode < len(files):
            raise IndexError(f"episode must be in [0, {len(files) - 1}]")
        _run_one(files[args.episode], args)
        return

    print("可用 episode（按文件名排序，编号从 0 开始）:")
    for index, path in enumerate(files):
        print(f"  {index:3d}: {path.name}")
    while True:
        answer = input("输入 episode 编号，或 q 退出: ").strip().lower()
        if answer in {"q", "quit", "exit"}:
            return
        try:
            index = int(answer)
            if not 0 <= index < len(files):
                raise ValueError
            _run_one(files[index], args)
        except ValueError:
            print(f"请输入 0 到 {len(files) - 1} 的整数，或 q。")


if __name__ == "__main__":
    main()
