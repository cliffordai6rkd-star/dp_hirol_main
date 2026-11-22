"""Generalized momentum observer using Pinocchio, without joint acceleration.

For M*qdd + C*qd + g = tau_motor + tau_environment, p = M*qd obeys
p_dot = tau_motor + C.T*qd - g + tau_environment.  The residual observer is
r = K*(p - p_hat), p_hat_dot = tau_motor + C.T*qd - g + r.

Inputs to update() must already share the causal filtering used by the RNEA
comparison.  This observer adds its own first-order residual response.
"""
from pathlib import Path

import numpy as np
import pinocchio as pin

from diffusion_policy.common.xarm_tau_ext import TauExtConfig, build_arm_model


class GeneralizedMomentumObserver:
    def __init__(self, urdf: Path, config: TauExtConfig, bandwidth_hz: float = 3.):
        if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
            raise ValueError("Observer bandwidth must be positive and finite")
        if config.convention not in {"tool-on-environment", "environment-on-tool"}:
            raise ValueError("Unknown external-torque sign convention")
        self.config = config
        self.gain = 2 * np.pi * float(bandwidth_hz)
        self.model = build_arm_model(urdf, config)
        self.data = self.model.createData()
        self.reset()

    def reset(self):
        self.last_time = None
        self.previous_momentum = self.previous_beta = None
        self.residual = np.zeros(7, dtype=np.float64)

    def update(self, q, dq_filtered, tau_filtered, timestamp_s):
        q, dq, tau = [np.asarray(x, dtype=np.float64) for x in (q, dq_filtered, tau_filtered)]
        if any(x.shape != (7,) or not np.isfinite(x).all() for x in (q, dq, tau)):
            raise ValueError("Expected finite seven-axis q, filtered dq and filtered motor torque")
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("Timestamp must be finite")
        dt = None if self.last_time is None else timestamp_s - self.last_time
        if dt is not None and not np.isclose(dt, 1 / self.config.fps, rtol=.01, atol=1e-5):
            raise ValueError("Reset observer at episode boundaries or sample gaps")
        mass = pin.crba(self.model, self.data, q).copy()
        coriolis = pin.computeCoriolisMatrix(self.model, self.data, q, dq).copy()
        gravity = pin.computeGeneralizedGravity(self.model, self.data, q).copy()
        momentum = mass @ dq
        beta = tau - np.asarray(self.config.torque_bias) + coriolis.T @ dq - gravity
        if dt is not None:
            # Exact stable first-order update for linearly interpolated p and
            # interval-average beta.  No acceleration estimate or RNEA call.
            one_minus_alpha = -np.expm1(-self.gain * dt)
            beta_average = .5 * (self.previous_beta + beta)
            innovation = (momentum - self.previous_momentum) - dt * beta_average
            self.residual = ((1 - one_minus_alpha) * self.residual
                             + (one_minus_alpha / dt) * innovation)
        self.last_time = timestamp_s
        self.previous_momentum = momentum
        self.previous_beta = beta
        sign = -1 if self.config.convention == "tool-on-environment" else 1
        return {"tau_ext_raw": (sign * self.residual).copy(), "momentum": momentum.copy(),
                "beta": beta.copy(), "first_sample": dt is None}
