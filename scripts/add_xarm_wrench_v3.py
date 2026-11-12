#!/usr/bin/env python3
"""Estimate flange wrench from xArm7 joint torque and export a new LeRobot v3 dataset.

Requires numpy, scipy, pin (Pinocchio), pyarrow and matplotlib. See
scripts/assets/xarm7/README.md for model provenance and physical assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, sosfiltfilt
from diffusion_policy.common.xarm_tau_ext import (
    CausalTauExt, TauExtConfig, build_arm_model, gate_tau_ext,
)


DEFAULT_URDF = Path(__file__).resolve().parent / "assets/xarm7/xarm7_dynamics.urdf"
COMPONENTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FIELD_DIMS = {
    "observation.tau_free": 7,
    "observation.tau_ext": 7,
    "observation.tau_ext_unthresholded": 7,
    "observation.tau_ext_l1": 1,
    "observation.contact": 1,
    "observation.wrench": 6,
    "observation.wrench_reconstruction_error": 1,
    "observation.wrench_sigma_min": 1,
}


def solve_wrench(jacobian, tau_ext, damping):
    """Solve min ||J.T w - tau_ext||² + damping² ||w||² by SVD.

    Pinocchio uses linear then angular rows: [Fx,Fy,Fz,Mx,My,Mz].
    The error is joint-torque reconstruction RMS in Nm, not force accuracy.
    """
    u, s, vt = np.linalg.svd(jacobian.T, full_matrices=False)
    wrench = vt.T @ ((s / (s * s + damping * damping)) * (u.T @ tau_ext))
    error = np.sqrt(np.mean((jacobian.T @ wrench - tau_ext) ** 2))
    return wrench, float(error), float(s[-1])


class WrenchEstimator:
    def __init__(self, args):
        self.args = args
        if args.filter_mode != "causal" and getattr(args, "tau_ext_smoothing_window", 1) != 1:
            raise ValueError("tau_ext smoothing requires filter_mode=causal")
        self.model = build_arm_model(args.urdf, TauExtConfig.from_args(args))
        frame = self.model.frames[self.model.getFrameId(args.ee_frame)]
        tcp = frame.placement * pin.SE3(np.eye(3), np.asarray(args.tcp_offset))
        self.frame_id = self.model.addFrame(pin.Frame(
            "wrench_estimation_tcp", frame.parentJoint,
            self.model.getFrameId(args.ee_frame), tcp, pin.FrameType.OP_FRAME,
        ))
        self.data = self.model.createData()
        self.reference = (pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                          if args.reference_frame == "base" else pin.ReferenceFrame.LOCAL)

    def estimate_episode(self, q, velocity, torque, timestamps):
        q, velocity, torque, timestamps = [np.asarray(x, dtype=np.float64)
                                          for x in (q, velocity, torque, timestamps)]
        n = len(timestamps)
        if n < 2 or any(x.shape != (n, 7) for x in (q, velocity, torque)):
            raise ValueError("Each episode needs >=2 samples of seven-dimensional state")
        if not all(np.isfinite(x).all() for x in (q, velocity, torque, timestamps)):
            raise ValueError("State, torque and timestamp must be finite")
        dt = np.diff(timestamps)
        if np.any(dt <= 0) or not np.allclose(dt, 1 / self.args.fps, rtol=0.01, atol=1e-5):
            raise ValueError("Expected monotonic, uniformly sampled episode timestamps at dataset FPS")
        causal = (CausalTauExt(self.args.urdf, TauExtConfig.from_args(self.args))
                  if self.args.filter_mode == "causal" else None)
        # Offline mode remains available for reproducing the earlier dataset.
        if causal is None and self.args.cutoff_hz:
            sos = butter(2, self.args.cutoff_hz, fs=self.args.fps, output="sos")
            padlen = min(n - 1, 9)
            velocity = sosfiltfilt(sos, velocity, axis=0, padlen=padlen)
            torque = sosfiltfilt(sos, torque, axis=0, padlen=padlen)
        if causal is None:
            acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=1)
            torque = torque - np.asarray(self.args.torque_bias)
        out = {key: np.empty((n, dim), np.float32) for key, dim in FIELD_DIMS.items()}
        for i in range(n):
            if causal is not None:
                sample = causal.update(q[i], velocity[i], torque[i], timestamps[i])
                free = sample["tau_free"]
                raw = sample["tau_ext_unthresholded"]
            elif self.args.compensation == "dynamics":
                free = pin.rnea(self.model, self.data, q[i], velocity[i], acceleration[i]).copy()
            elif self.args.compensation == "gravity":
                free = pin.computeGeneralizedGravity(self.model, self.data, q[i]).copy()
            else:
                free = np.zeros(7)
            # M*qdd + C + g = tau_actuator + J.T*w_environment_on_tool.
            if causal is None:
                raw = free - torque[i]
                if self.args.convention == "tool-on-environment":
                    raw = -raw
            ext, l1, contact = gate_tau_ext(raw, self.args.contact_threshold_l1)
            pin.computeJointJacobians(self.model, self.data, q[i])
            pin.updateFramePlacements(self.model, self.data)
            jacobian = pin.getFrameJacobian(self.model, self.data, self.frame_id, self.reference)
            wrench, error, sigma = solve_wrench(jacobian, ext, self.args.damping)
            out["observation.tau_free"][i] = free
            out["observation.tau_ext"][i] = ext
            out["observation.tau_ext_unthresholded"][i] = raw
            out["observation.tau_ext_l1"][i] = l1
            out["observation.contact"][i] = float(contact)
            out["observation.wrench"][i] = wrench
            out["observation.wrench_reconstruction_error"][i] = error
            out["observation.wrench_sigma_min"][i] = sigma
        if not all(np.isfinite(x).all() for x in out.values()):
            raise ValueError("Non-finite wrench estimate")
        return out


def array_column(table, key):
    return np.asarray(table[key].to_pylist(), dtype=np.float64)


def feature_stats(values):
    result = {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.astype(np.float64).std(axis=0).tolist(), "count": [len(values)],
    }
    for label, quantile in [("q01", .01), ("q10", .1), ("q50", .5), ("q90", .9), ("q99", .99)]:
        result[label] = np.quantile(values, quantile, axis=0).tolist()
    return result


def add_columns(table, columns):
    for key, values in columns.items():
        values = np.asarray(values, dtype=np.float32)
        table = table.append_column(key, pa.FixedSizeListArray.from_arrays(
            pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]))
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        features = hf.setdefault("info", {}).setdefault("features", {})
        # Match the installed datasets version's List / Sequence representation.
        vector_type = next((v.get("_type") for v in features.values()
                            if v.get("_type") in {"List", "Sequence"}), "List")
        for key, values in columns.items():
            features[key] = {"feature": {"dtype": "float32", "_type": "Value"},
                             "length": values.shape[1], "_type": vector_type}
        hf.pop("fingerprint", None)
        metadata[b"huggingface"] = json.dumps(hf).encode()
    return table.replace_schema_metadata(metadata)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_episode(path, episode_id, times, columns, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = times - times[0]
    tau = columns["observation.tau_ext"]
    l1 = np.abs(tau).sum(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for i in range(7):
        axes[0].plot(t, tau[:, i], label=f"J{i + 1}", lw=1)
    axes[0].set_ylabel("External joint torque [Nm]")
    axes[0].legend(ncol=7)
    axes[1].plot(t, l1, color="black", label="sum(abs(tau_ext[J1:J7]))")
    if args.contact_threshold_l1:
        axes[1].plot(t, columns["observation.tau_ext_l1"][:, 0],
                     color="tab:blue", alpha=.5, label="L1 before threshold")
        axes[1].axhline(args.contact_threshold_l1, color="tab:red", ls="--",
                        label=f"Threshold {args.contact_threshold_l1:g} Nm")
    axes[1].set_ylabel("Seven-axis L1 norm [Nm]")
    axes[1].set_xlabel("Episode time [s]")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.3)
    definition = ("RNEA - measured torque" if args.convention == "environment-on-tool"
                  else "measured torque - RNEA")
    fig.suptitle(f"Episode {episode_id}: tau_ext = {definition}\n"
                 f"{args.compensation}; payload {args.payload_mass:g} kg; {args.filter_mode} low-pass {args.cutoff_hz:g} Hz; URDF estimate")
    fig.tight_layout()
    fig.savefig(path.with_name(f"episode_{episode_id:03d}_tau_ext_l1.png"), dpi=160)
    plt.close(fig)


def export_dataset(args, *, overwrite=False):
    source, target = args.dataset.resolve(), args.output.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Output must be outside the input dataset")
    if target.exists() and not overwrite:
        raise ValueError("Output already exists; enable overwrite to replace it")
    if target.exists() and not target.is_dir():
        raise ValueError("Output must be a dataset directory")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected a LeRobot v3.0 dataset")
    args.fps = float(info["fps"])
    processing_config = TauExtConfig.from_args(args)
    if not 0 <= args.cutoff_hz < args.fps / 2:
        raise ValueError("Cutoff must be zero (disabled), or below Nyquist")
    files = sorted((source / "data").rglob("*.parquet"))
    tables = [pq.read_table(p) for p in files]
    table = pa.concat_tables(tables)
    collisions = set(FIELD_DIMS) & set(table.column_names)
    if collisions or set(FIELD_DIMS) & set(info["features"]):
        raise ValueError(f"Derived fields already exist; use the original dataset: {sorted(collisions)}")
    if len(table) != info["total_frames"] or not np.array_equal(
            array_column(table, "index"), np.arange(len(table))):
        raise ValueError("Data files must contain all rows in contiguous dataset index order")
    episode_ids = array_column(table, "episode_index").astype(np.int64)
    frame_ids = array_column(table, "frame_index").astype(np.int64)
    episodes = np.unique(episode_ids)
    if len(episodes) != info["total_episodes"] or args.plot_episode not in episodes:
        raise ValueError("Episode metadata mismatch or plot episode not present")
    q = array_column(table, args.joint_key)
    velocity = array_column(table, args.velocity_key)
    torque = array_column(table, args.torque_key)
    timestamps = array_column(table, "timestamp")
    estimator = WrenchEstimator(args)
    columns = {key: np.empty((len(table), dim), np.float32) for key, dim in FIELD_DIMS.items()}
    episode_stats = {}
    for episode in episodes:
        rows = np.flatnonzero(episode_ids == episode)
        if not np.array_equal(frame_ids[rows], np.arange(len(rows))):
            raise ValueError(f"Episode {episode}: invalid frame order")
        result = estimator.estimate_episode(q[rows], velocity[rows], torque[rows], timestamps[rows])
        for key, values in result.items():
            columns[key][rows] = values
        episode_stats[int(episode)] = {key: feature_stats(values) for key, values in result.items()}
        print(f"Episode {episode}: {len(rows)} frames, median force norm "
              f"{np.median(np.linalg.norm(result['observation.wrench'][:, :3], axis=1)):.2f} N", flush=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        # Copy videos and all other metadata; no symlinks/hardlinks to the source.
        shutil.copytree(source, staging, dirs_exist_ok=True)
        start = 0
        for path, original in zip(files, tables):
            end = start + len(original)
            pq.write_table(add_columns(original, {k: v[start:end] for k, v in columns.items()}),
                           staging / path.relative_to(source), compression="snappy")
            start = end
        stats = json.loads((staging / "meta/stats.json").read_text())
        for key, values in columns.items():
            stats[key] = feature_stats(values)
            info["features"][key] = {"dtype": "float32", "shape": [values.shape[1]],
                                     "names": COMPONENTS if key == "observation.wrench" else None}
        write_json(staging / "meta/stats.json", stats)
        metadata_episodes = []
        for path in sorted((staging / "meta/episodes").rglob("*.parquet")):
            et = pq.read_table(path)
            ids = et["episode_index"].to_pylist()
            metadata_episodes.extend(ids)
            for key in FIELD_DIMS:
                for name in stats[key]:
                    et = et.append_column(f"stats/{key}/{name}", pa.array(
                        [episode_stats[i][key][name] for i in ids],
                        type=pa.list_(pa.int64() if name == "count" else pa.float64())))
            pq.write_table(et, path, compression="snappy")
        if sorted(metadata_episodes) != episodes.tolist():
            raise ValueError("Episode parquet metadata does not match the data")
        write_json(staging / "meta/info.json", info)
        model_dir = staging / "meta/wrench_model"
        model_dir.mkdir()
        shutil.copy2(args.urdf, model_dir / "robot.urdf")
        for name in ["LICENSE.xarm_ros2"]:
            if (args.urdf.parent / name).is_file():
                shutil.copy2(args.urdf.parent / name, model_dir / name)
        manifest = {
            "method": "Pinocchio RNEA + damped SVD of J.T", "estimated": True,
            "equation": "M*qdd + C + g = tau_actuator + J.T*w_environment_on_tool",
            "tau_ext_definition": ("tau_free - (filtered_torque - bias)" if args.convention == "environment-on-tool"
                                   else "(filtered_torque - bias) - tau_free"),
            "wrench_convention": args.convention, "components": COMPONENTS,
            "units": ["N"] * 3 + ["Nm"] * 3,
            "reference_axes": args.reference_frame,
            "reference_point": "ee_frame origin + tcp_offset expressed in ee_frame",
            "joint_order": list(estimator.model.names)[1:], "joint_units": "rad, rad/s, Nm",
            "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
            "pinocchio_version": pin.__version__, "source_dataset": str(source),
            "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "filter": ("episode-local causal second-order Butterworth on velocity and torque; acceleration = backward difference(filtered velocity)"
                       if args.filter_mode == "causal" else
                       "episode-local zero-phase second-order Butterworth on velocity and torque; acceleration = gradient(filtered velocity)"),
            "filter_is_causal": args.filter_mode == "causal",
            "threshold": {"norm": "L1 over J1..J7", "value_Nm": args.contact_threshold_l1,
                          "rule": "L1 < threshold: all seven tau_ext and wrench components zero; otherwise retain"},
            "payload_provenance": {
                "bio_g2_mass_source": "https://www.ufactory.cc/upfiles/download/202610/WJY8KDIK8B1F.pdf (v2.8.0 section 6: 790g including adapter)",
                "bio_g2_default_com": "[0, 0, 0.05] m is an approximation, NOT manufacturer calibration",
                "bio_g2_default_inertia": "point mass approximation (zero COM rotational inertia)",
            } if args.bio_gripper_g2 else {"source": "user-supplied parameters"},
            "limitations": ["Assumes torque is joint actuator torque in Nm, not motor current or precompensated external torque",
                            "Single resultant wrench at the specified end effector; cannot separate other link contacts",
                            "Uncalibrated joint friction and torque bias contaminate the estimate; URDF damping/friction not used",
                            "Payload inertia is diagonal about its COM in ee_frame axes; zero means point-mass approximation",
                            "No additional camera, eraser or other tool load compensation",
                            "Torque reconstruction RMS is not a ground-truth wrench accuracy measurement"],
        }
        write_json(staging / "meta/tau_ext_processing.json", {
            "config": processing_config.to_dict(),
            "filter_mode": args.filter_mode,
            "urdf_sha256": manifest["urdf_sha256"],
            "sos_coefficients": (butter(2, args.cutoff_hz, fs=args.fps, output="sos").tolist()
                                 if args.cutoff_hz else None),
            "q_processing": "none; input radians in joint1..joint7 order",
            "dq_processing": "second-order Butterworth low-pass; no differencing q",
            "tau_processing": "same Butterworth low-pass, then subtract fixed torque_bias",
            "ddq_processing": ("backward difference of filtered dq using current minus previous timestamp; first sample zero"
                               if args.filter_mode == "causal" else "np.gradient(filtered dq, timestamps), central inside, one-sided at endpoints"),
            "causal_filter_initialization": "sosfilt_zi multiplied by first sample, independently for dq and tau; reset per episode/rollout",
            "tau_ext_smoothing": {
                "window_samples": processing_config.tau_ext_smoothing_window,
                "causal": True,
                "stage": "signed tau_ext after dynamics compensation, before L1 and threshold",
                "rule": "mean of current and previous window_samples-1 vectors; use available samples at episode start; reset per episode/rollout",
            },
            "postprocessing": (
                f"causal trailing mean of signed tau_ext over {processing_config.tau_ext_smoothing_window} samples; "
                "no clipping, normalization or abs per component; gate entire seven-vector when its L1 < threshold; equality retained"
            ),
            "pre_gate_field": "observation.tau_ext_unthresholded",
            "pre_gate_l1_field": "observation.tau_ext_l1",
            "online_runtime": "diffusion_policy.common.xarm_tau_ext.CausalTauExt.from_bundle(dataset_root)",
            "runtime_source": "diffusion_policy/common/xarm_tau_ext.py",
            "estimation": manifest,
            "sampling": "fixed saved FPS, dt tolerance 1% + 1e-5 s; resample before update if source rate differs; reset after gaps",
            "payload_com_is_approximate": bool(args.bio_gripper_g2),
        })
        plot_rows = np.flatnonzero(episode_ids == args.plot_episode)
        plot_dir = staging / "visualizations"
        plot_dir.mkdir(exist_ok=True)
        plot_episode(plot_dir / f"episode_{args.plot_episode:03d}_wrench.png", args.plot_episode,
                     timestamps[plot_rows], {k: v[plot_rows] for k, v in columns.items()}, args)
        if target.exists():
            # Keep the old dataset until the complete replacement is ready.
            with tempfile.TemporaryDirectory(prefix=f".{target.name}-old-", dir=target.parent) as backup_dir:
                backup = Path(backup_dir) / "dataset"
                target.rename(backup)
                try:
                    staging.rename(target)
                except BaseException:
                    backup.rename(target)
                    raise
        else:
            staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Saved {len(table)} frames / {len(episodes)} episodes to {target}")
    print(f"Plot: {target}/visualizations/episode_{args.plot_episode:03d}_tau_ext_l1.png")
    return target


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New v3 dataset directory (must not exist)")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--joint-key", default="observation.joint")
    p.add_argument("--velocity-key", default="observation.velocity")
    p.add_argument("--torque-key", default="observation.torque")
    p.add_argument("--ee-frame", default="link_eef")
    p.add_argument("--reference-frame", choices=["base", "local"], default="base",
                   help="Base-aligned axes at TCP (not a moment about base origin), or TCP-local axes")
    p.add_argument("--tcp-offset", type=float, nargs=3, default=[0., 0., 0.], metavar=("X", "Y", "Z"))
    p.add_argument("--gravity", type=float, nargs=3, default=[0., 0., -9.81])
    p.add_argument("--bio-gripper-g2", action="store_true",
                   help="Preset: official mass 0.79 kg; approximate COM [0,0,0.05] m; point-mass inertia. Explicit payload flags override")
    p.add_argument("--payload-mass", type=float, default=None, help="Additional mass in kg")
    p.add_argument("--payload-com", type=float, nargs=3, default=None, help="Payload COM in ee_frame, metres")
    p.add_argument("--payload-inertia", type=float, nargs=3, default=[0., 0., 0.], help="Diagonal COM inertia Ixx Iyy Izz in kg*m^2")
    p.add_argument("--compensation", choices=["dynamics", "gravity", "none"], default="dynamics")
    p.add_argument("--convention", choices=["environment-on-tool", "tool-on-environment"],
                   default="environment-on-tool", help="Sets sign of both tau_ext and wrench")
    p.add_argument("--torque-bias", type=float, nargs=7, default=[0.] * 7, help="Measured sensor bias J1..J7, Nm")
    p.add_argument("--cutoff-hz", type=float, default=3., help="Second-order Butterworth cutoff; 0 disables")
    p.add_argument("--filter-mode", choices=["offline", "causal"], default="offline",
                   help="causal shares the exact streaming processor used for online inference")
    p.add_argument("--contact-threshold-l1", type=float, default=0.,
                   help="Zero the entire seven-axis tau_ext vector when sum(abs(tau_ext)) < this Nm threshold")
    p.add_argument("--damping", type=float, default=.02, help="SVD gain s/(s^2+damping^2)")
    p.add_argument("--plot-episode", type=int, default=0)
    args = p.parse_args(argv)
    if args.payload_mass is None:
        args.payload_mass = .79 if args.bio_gripper_g2 else 0.
    if args.payload_com is None:
        args.payload_com = [0., 0., .05] if args.bio_gripper_g2 else [0., 0., 0.]
    numeric = [args.payload_mass, args.damping, args.cutoff_hz, args.contact_threshold_l1, *args.tcp_offset,
               *args.gravity, *args.payload_com, *args.payload_inertia, *args.torque_bias]
    if not np.isfinite(numeric).all() or args.payload_mass < 0 or args.damping <= 0:
        p.error("Parameters must be finite; payload mass >= 0 and damping > 0")
    if args.contact_threshold_l1 < 0:
        p.error("Contact threshold must be nonnegative")
    inertia = np.asarray(args.payload_inertia)
    if np.any(inertia < 0) or np.any(2 * inertia > inertia.sum() + 1e-12):
        p.error("Payload diagonal inertia must be nonnegative and satisfy the triangle inequalities")
    return args


if __name__ == "__main__":
    export_dataset(parse_args())
