"""Physical sign, convergence and free-motion checks for the torque observer."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from diffusion_policy.common.momentum_observer import GeneralizedMomentumObserver
from diffusion_policy.common.xarm_tau_ext import TauExtConfig


URDF = Path(__file__).resolve().parents[1] / "scripts/assets/xarm7/xarm7_dynamics.urdf"
Q = np.array([.1, -.05, .1, 1.5, 0., 1.5, .1])


@pytest.mark.parametrize("convention,sign", [("tool-on-environment", 1), ("environment-on-tool", -1)])
def test_known_static_torque_and_reset(convention, sign):
    observer = GeneralizedMomentumObserver(URDF, replace(TauExtConfig(), convention=convention))
    gravity = pin.computeGeneralizedGravity(observer.model, observer.data, Q).copy()
    extra = np.array([.5, -2., 1., 3., -.8, .4, -.3])
    assert np.array_equal(observer.update(Q, np.zeros(7), gravity + extra, 0.)["tau_ext_raw"], np.zeros(7))
    for index in range(1, 51):
        result = observer.update(Q, np.zeros(7), gravity + extra, index / 25)
    np.testing.assert_allclose(result["tau_ext_raw"], sign * extra, atol=1e-10)
    observer.reset()
    assert observer.update(Q, np.zeros(7), gravity + extra, 0.)["first_sample"]


def test_contact_step_has_correct_sign_and_observer_response():
    observer = GeneralizedMomentumObserver(URDF, TauExtConfig())
    gravity = pin.computeGeneralizedGravity(observer.model, observer.data, Q).copy()
    zero = np.zeros(7)
    observer.update(Q, zero, gravity, 0.)
    extra = np.array([0., 5., 0., -2., 0., 0., 0.])
    result = observer.update(Q, zero, gravity + extra, .04)
    np.testing.assert_allclose(result["tau_ext_raw"], .5 * (1 - np.exp(-observer.gain * .04)) * extra)
    for index in range(2, 26):
        result = observer.update(Q, zero, gravity + extra, index / 25)
    np.testing.assert_allclose(result["tau_ext_raw"], extra, atol=1e-6)


def test_free_motion_recovers_zero_without_acceleration_input():
    observer = GeneralizedMomentumObserver(URDF, TauExtConfig())
    data = observer.model.createData()
    errors = []
    amplitudes = np.array([.03, .05, .03, .05, .02, .03, .02])
    omega = 1.1
    for index in range(126):
        t = index / 25
        q = Q + amplitudes * np.sin(omega * t)
        dq = amplitudes * omega * np.cos(omega * t)
        ddq = -amplitudes * omega ** 2 * np.sin(omega * t)
        motor = pin.rnea(observer.model, data, q, dq, ddq).copy()
        result = observer.update(q, dq, motor, t)
        errors.append(result["tau_ext_raw"])
    assert np.max(np.abs(errors)) < .002
