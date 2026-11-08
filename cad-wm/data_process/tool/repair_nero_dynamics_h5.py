from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin


DEFAULT_DQ_SIGN = np.asarray([-1, -1, -1, -1, -1, 1, -1], dtype=np.float64)
LOCKED_JOINTS = ("gripper", "gripper_joint1", "gripper_joint2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Nero dq/ddq and dependent torque residuals in H5 episodes."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/train_episode/tau_refinement"),
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("data/train_episode/tau_refinement_before_state_repair_20260806"),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("sim_mesh/nero/nero_with_gripper.urdf"),
    )
    parser.add_argument("--ddq-lowpass-hz", type=float, default=3.0)
    parser.add_argument(
        "--reuse-existing-backup",
        action="store_true",
        help=(
            "Rebuild the repaired files from an existing backup directory. "
            "This makes filter-only corrections repeatable without applying "
            "the velocity sign correction twice."
        ),
    )
    return parser.parse_args()


def build_arm_model(urdf_path: Path):
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    locked_ids = []
    for name in LOCKED_JOINTS:
        joint_id = full_model.getJointId(name)
        if joint_id == full_model.njoints:
            raise ValueError(f"Joint {name!r} is missing from {urdf_path}.")
        locked_ids.append(joint_id)
    return pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))


def causal_derivative(values: np.ndarray, timestamps_s: np.ndarray) -> np.ndarray:
    dt = np.diff(timestamps_s)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("Episode timestamps must be finite and strictly increasing.")
    derivative = np.zeros_like(values, dtype=np.float64)
    derivative[1:] = np.diff(values, axis=0) / dt[:, None]
    return derivative


def causal_one_pole_lowpass(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    cutoff_hz: float,
) -> np.ndarray:
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("Low-pass cutoff must be positive and finite.")
    filtered = np.empty_like(values, dtype=np.float64)
    filtered[0] = values[0]
    for index in range(1, len(values)):
        dt = timestamps_s[index] - timestamps_s[index - 1]
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
        filtered[index] = filtered[index - 1] + alpha * (
            values[index] - filtered[index - 1]
        )
    return filtered


def batched_rnea(
    model,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
) -> np.ndarray:
    if q.shape[1] != model.nq or dq.shape[1] != model.nv:
        raise ValueError(
            f"H5 joint dimension {q.shape[1]} does not match model "
            f"nq={model.nq}, nv={model.nv}."
        )
    data = model.createData()
    tau_id = np.empty_like(dq, dtype=np.float64)
    for index in range(len(q)):
        tau_id[index] = pin.rnea(model, data, q[index], dq[index], ddq[index])
    return tau_id


def write_attrs(dataset, attrs: dict[str, object]) -> None:
    for key, value in attrs.items():
        dataset.attrs[key] = value


def repair_file(
    path: Path,
    backup_path: Path,
    model,
    ddq_lowpass_hz: float,
    *,
    reuse_existing_backup: bool,
) -> dict:
    if reuse_existing_backup:
        if not backup_path.is_file():
            raise FileNotFoundError(f"Existing backup is missing: {backup_path}")
        source_path = backup_path
    else:
        if backup_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup_path}")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        source_path = path

    temporary_path = path.with_suffix(path.suffix + ".repairing")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary repair file already exists: {temporary_path}")
    shutil.copy2(source_path, temporary_path)

    try:
        with h5py.File(temporary_path, "r+") as h5_file:
            teleop = h5_file["teleop"]
            timestamps_s = np.asarray(teleop["timestamp_us"], dtype=np.float64) * 1e-6
            q = np.asarray(teleop["q_follower"], dtype=np.float64)
            dq_source = np.asarray(teleop["dq_follower"], dtype=np.float64)
            tau_measured = np.asarray(teleop["tau_follower"], dtype=np.float64)
            tau_other_pred = np.asarray(teleop["tau_other_pred"], dtype=np.float64)

            dq = dq_source * DEFAULT_DQ_SIGN[None, :]
            ddq_unfiltered = causal_derivative(dq, timestamps_s)
            ddq = causal_one_pole_lowpass(
                ddq_unfiltered,
                timestamps_s,
                ddq_lowpass_hz,
            )
            tau_id = batched_rnea(model, q, dq, ddq)
            tau_g = batched_rnea(
                model,
                q,
                np.zeros_like(q),
                np.zeros_like(q),
            )
            tau_other_cal = tau_measured - tau_g
            tau_ext = tau_other_cal - tau_other_pred

            teleop["dq_follower"][:] = dq
            teleop["ddq_follower"][:] = ddq
            teleop["tau_other_cal"][:] = tau_other_cal
            teleop["tau_ext"][:] = tau_ext
            if "tau_g" in teleop:
                teleop["tau_g"][:] = tau_g
            else:
                teleop.create_dataset("tau_g", data=tau_g)
            if "tau_id" in teleop:
                teleop["tau_id"][:] = tau_id
            else:
                teleop.create_dataset("tau_id", data=tau_id)

            sign_json = json.dumps(DEFAULT_DQ_SIGN.astype(int).tolist())
            write_attrs(
                teleop["dq_follower"],
                {
                    "coordinate_sign_correction_json": sign_json,
                    "formula": "dq_repaired[k]=dq_firmware[k]*joint_sign",
                    "processing_method": "firmware_velocity_sign_corrected",
                    "repair_source": str(backup_path),
                },
            )
            write_attrs(
                teleop["ddq_follower"],
                {
                    "derived_from_json": json.dumps(["dq_follower"]),
                    "derivative_method": "causal_first_derivative_of_repaired_dq",
                    "formula": "ddq_raw[k]=(dq[k]-dq[k-1])/measured_dt",
                    "lowpass": True,
                    "lowpass_cutoff_hz": float(ddq_lowpass_hz),
                    "lowpass_discretization": "alpha=1-exp(-2*pi*cutoff_hz*dt)",
                    "processing_method": "causal_derivative_then_nero_one_pole_iir",
                    "repair_source": str(backup_path),
                },
            )
            residual_attrs = {
                "definition": "tau_follower-tau_g(q)",
                "formula": "tau_other=tau_measured-tau_g",
                "processing_method": "recomputed_measured_torque_minus_gravity_rnea",
                "state_derivative_method": "sign_corrected_velocity_and_causal_acceleration",
                "repair_source": str(backup_path),
            }
            write_attrs(
                teleop["tau_g"],
                {
                    "definition": "RNEA(q,0,0) = tau_g",
                    "processing_method": "pinocchio_gravity_rnea",
                    "dq_source": "zero",
                    "ddq_source": "zero",
                    "repair_source": str(backup_path),
                },
            )
            write_attrs(
                teleop["tau_id"],
                {
                    "definition": "RNEA(q,dq,ddq) = tau_id",
                    "processing_method": "pinocchio_full_rnea",
                    "dq_source": "dq_follower",
                    "ddq_source": "ddq_follower",
                    "repair_source": str(backup_path),
                },
            )
            write_attrs(teleop["tau_other_cal"], residual_attrs)
            write_attrs(
                teleop["tau_ext"],
                {
                    **residual_attrs,
                    "definition": "tau_other_cal_repaired-tau_other_pred",
                },
            )
            h5_file.attrs["nero_dynamics_state_repaired"] = True
            h5_file.attrs["nero_dynamics_dq_sign_json"] = sign_json
            h5_file.flush()

        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    qdot = causal_derivative(q, timestamps_s)
    dq_correlations = []
    for joint in range(q.shape[1]):
        moving = np.abs(qdot[:, joint]) > 1e-5
        dq_correlations.append(
            float(np.corrcoef(qdot[moving, joint], dq[moving, joint])[0, 1])
        )
    return {
        "file": path.name,
        "frames": int(len(q)),
        "dq_qdot_correlation": dq_correlations,
        "ddq_abs_max": float(np.max(np.abs(ddq))),
        "tau_other_cal_abs_max": float(np.max(np.abs(tau_other_cal))),
    }


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF does not exist: {args.urdf}")
    files = sorted((*args.input_dir.glob("*.h5"), *args.input_dir.glob("*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No H5 episodes found in {args.input_dir}")
    if args.backup_dir.exists() and not args.reuse_existing_backup:
        raise FileExistsError(
            f"Backup directory already exists; choose a new path: {args.backup_dir}"
        )
    if args.reuse_existing_backup and not args.backup_dir.is_dir():
        raise FileNotFoundError(
            f"Existing backup directory does not exist: {args.backup_dir}"
        )

    model = build_arm_model(args.urdf)
    results = []
    for path in files:
        results.append(
            repair_file(
                path,
                args.backup_dir / path.name,
                model,
                args.ddq_lowpass_hz,
                reuse_existing_backup=args.reuse_existing_backup,
            )
        )

    manifest = {
        "input_dir": str(args.input_dir),
        "backup_dir": str(args.backup_dir),
        "urdf": str(args.urdf),
        "dq_sign": DEFAULT_DQ_SIGN.astype(int).tolist(),
        "ddq_lowpass_hz": args.ddq_lowpass_hz,
        "ddq_lowpass_discretization": "alpha=1-exp(-2*pi*cutoff_hz*dt)",
        "residual_convention": "tau_other=tau_measured-tau_g",
        "results": results,
    }
    manifest_path = args.backup_dir / "repair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
