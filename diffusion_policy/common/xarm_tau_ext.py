"""Shared causal q/dq/tau -> seven-axis tau_ext processing for export and inference.

No dataset/plotting dependencies. Use CausalTauExt.from_bundle(dataset_root),
then update(q_rad, dq_rad_s, tau_Nm, timestamp_seconds) at the saved sample rate.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.signal import butter, sosfilt, sosfilt_zi


@dataclass
class TauExtConfig:
    version: int = 1
    fps: float = 25.
    cutoff_hz: float = 3.
    contact_threshold_l1: float = 14.
    convention: str = "tool-on-environment"
    compensation: str = "dynamics"
    ee_frame: str = "link_eef"
    gravity: tuple = (0., 0., -9.81)
    payload_mass: float = .79
    payload_com: tuple = (0., 0., .05)
    payload_inertia: tuple = (0., 0., 0.)
    torque_bias: tuple = (0.,) * 7

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__
                      if name != "version"})

    def to_dict(self):
        return asdict(self)


def build_arm_model(urdf, cfg):
    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 7 or model.nv != 7 or list(model.names)[1:] != [f"joint{i}" for i in range(1, 8)]:
        raise ValueError("Expected seven revolute joints in joint1..joint7 order")
    model.gravity.linear[:] = cfg.gravity
    if not model.existFrame(cfg.ee_frame):
        raise ValueError(f"Unknown end-effector frame: {cfg.ee_frame}")
    frame = model.frames[model.getFrameId(cfg.ee_frame)]
    if cfg.payload_mass:
        inertia = pin.Inertia(cfg.payload_mass, np.asarray(cfg.payload_com), np.diag(cfg.payload_inertia))
        model.appendBodyToJoint(frame.parentJoint, inertia, frame.placement)
    return model


def gate_tau_ext(raw, threshold):
    """Threshold total L1 once, never threshold each joint or subtract threshold.

    Evaluate on float32 values, matching the stored unthresholded channel.
    Equality is contact. First-order spikes are not debounced.
    """
    raw = np.asarray(raw, dtype=np.float32)
    l1 = float(np.abs(raw.astype(np.float64)).sum())
    contact = l1 >= threshold
    return (raw.copy() if contact else np.zeros(7, np.float32)), l1, contact


class CausalTauExt:
    def __init__(self, urdf, config):
        self.config = config
        if config.version != 1:
            raise ValueError("Unsupported tau_ext processing version")
        if (not np.isfinite([config.fps, config.cutoff_hz, config.contact_threshold_l1]).all()
                or config.fps <= 0 or not 0 <= config.cutoff_hz < config.fps / 2
                or config.contact_threshold_l1 < 0):
            raise ValueError("Invalid FPS, cutoff or threshold")
        if config.convention not in {"tool-on-environment", "environment-on-tool"}:
            raise ValueError("Invalid torque sign convention")
        if config.compensation not in {"dynamics", "gravity", "none"}:
            raise ValueError("Invalid compensation mode")
        self.model = build_arm_model(urdf, config)
        self.data = self.model.createData()
        self.sos = (butter(2, config.cutoff_hz, fs=config.fps, output="sos")
                    if config.cutoff_hz else None)
        self.reset()

    @classmethod
    def from_bundle(cls, dataset_root):
        root = Path(dataset_root) / "meta"
        record = json.loads((root / "tau_ext_processing.json").read_text())
        if record["filter_mode"] != "causal":
            raise ValueError("This dataset uses offline future-dependent processing; use the causal bundle")
        urdf = root / "wrench_model/robot.urdf"
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != record["urdf_sha256"]:
            raise ValueError("URDF differs from saved processing bundle")
        return cls(urdf, TauExtConfig(**record["config"]))

    def reset(self):
        """Reset on a new rollout/episode or after a sample gap before resuming."""
        self.last_time = None
        self.last_dq = None
        self.dq_state = self.tau_state = None

    def update(self, q, dq, tau, timestamp_s):
        q, dq, tau = [np.asarray(x, dtype=np.float64) for x in (q, dq, tau)]
        if any(x.shape != (7,) or not np.isfinite(x).all() for x in (q, dq, tau)):
            raise ValueError("q, dq, tau must be finite seven-vectors (rad, rad/s, Nm)")
        t = float(timestamp_s)
        if not np.isfinite(t):
            raise ValueError("timestamp must be finite seconds")
        first = self.last_time is None
        dt = None if first else t - self.last_time
        # Butterworth coefficients are designed for a fixed sampling frequency.
        if not first and not np.isclose(dt, 1 / self.config.fps, rtol=.01, atol=1e-5):
            raise ValueError("Expected saved fixed sample rate; reset after gaps and resample other rates")
        if self.sos is not None:
            if first:
                steady = sosfilt_zi(self.sos)[:, :, None]
                self.dq_state = steady * dq[None, None, :]
                self.tau_state = steady * tau[None, None, :]
            filtered_dq, dq_state = sosfilt(self.sos, dq[None, :], axis=0, zi=self.dq_state)
            filtered_tau, tau_state = sosfilt(self.sos, tau[None, :], axis=0, zi=self.tau_state)
            filtered_dq, filtered_tau = filtered_dq[0], filtered_tau[0]
        else:
            filtered_dq, filtered_tau = dq.copy(), tau.copy()
            dq_state = tau_state = None
        # Backward difference needs only the current and previous filtered velocity.
        ddq = np.zeros(7) if first else (filtered_dq - self.last_dq) / dt
        if self.config.compensation == "dynamics":
            free = pin.rnea(self.model, self.data, q, filtered_dq, ddq).copy()
        elif self.config.compensation == "gravity":
            free = pin.computeGeneralizedGravity(self.model, self.data, q).copy()
        else:
            free = np.zeros(7)
        raw = filtered_tau - np.asarray(self.config.torque_bias) - free
        if self.config.convention == "environment-on-tool":
            raw = -raw
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite external torque")
        gated, l1, contact = gate_tau_ext(raw, self.config.contact_threshold_l1)
        self.last_time, self.last_dq = t, filtered_dq.copy()
        self.dq_state, self.tau_state = dq_state, tau_state
        return {"tau_ext": gated, "tau_ext_unthresholded": raw.astype(np.float32),
                "tau_ext_l1": l1, "contact": contact, "tau_free": free,
                "dq_filtered": filtered_dq, "tau_filtered": filtered_tau,
                "ddq": ddq, "first_sample": first}
