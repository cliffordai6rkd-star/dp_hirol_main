from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from model.tau_other_sequence import TauOtherSequenceRegressor
from physics.nero_dynamics import (
    NeroDynamicsCache,
    PinocchioDynamics,
    RNEALinearization,
    load_tau_other_predictor,
    predict_nero_wrench,
)


def _linearization(reference, tau_id):
    joints = reference.shape[-1]
    identity = torch.eye(joints, dtype=reference.dtype).expand(
        *reference.shape[:-1], joints, joints
    )
    return RNEALinearization(
        q_reference=reference,
        dq_reference=reference.clone(),
        ddq_reference=reference.clone(),
        tau_id_reference=tau_id,
        d_tau_d_q=0.7 * identity,
        d_tau_d_dq=0.5 * identity,
        d_tau_d_ddq=1.2 * identity,
    )


def test_synthetic_cache_wrench_is_differentiable_to_every_state():
    dtype = torch.float64
    reference = torch.zeros(1, 2, 2, dtype=dtype)
    tau_id_reference = torch.tensor(
        [[[1.0, -0.5], [0.8, 0.6]]], dtype=dtype
    )
    jacobian = torch.tensor(
        [[[[1.0, 0.2], [0.1, 0.9]], [[0.8, -0.1], [0.2, 1.1]]]],
        dtype=dtype,
    )
    cache = NeroDynamicsCache(
        rnea=_linearization(reference, tau_id_reference),
        frame_jacobian=jacobian,
    )
    values = [
        torch.full_like(reference, offset, requires_grad=True)
        for offset in (0.04, -0.03, 0.02, 0.05, -0.01)
    ]
    q, dq, ddq, tau_measured, tau_other = values
    prediction = predict_nero_wrench(
        q=q,
        dq=dq,
        ddq=ddq,
        tau_measured=tau_measured,
        tau_other=tau_other,
        cache=cache,
        damping=0.02,
    )

    expected_tau_id = (
        tau_id_reference + 0.7 * q + 0.5 * dq + 1.2 * ddq
    )
    torch.testing.assert_close(prediction.tau_id, expected_tau_id)
    torch.testing.assert_close(
        prediction.tau_external,
        tau_measured - expected_tau_id - tau_other,
    )
    lhs = jacobian @ jacobian.transpose(-1, -2)
    lhs = lhs + 0.02**2 * torch.eye(2, dtype=dtype)
    expected_wrench = torch.linalg.solve(
        lhs,
        (jacobian @ prediction.tau_external.unsqueeze(-1)),
    ).squeeze(-1)
    torch.testing.assert_close(prediction.wrench, expected_wrench)

    prediction.wrench.square().mean().backward()
    for value in values:
        assert value.grad is not None
        assert value.grad.abs().sum() > 0


def test_pinocchio_is_not_imported_until_a_cache_is_requested(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()
    with patch("physics.nero_dynamics.importlib.import_module") as importer:
        dynamics = PinocchioDynamics(
            {"pinocchio": {"urdf_path": urdf_path, "locked_joint_names": []}}
        )
        assert dynamics.urdf_path == urdf_path
        importer.assert_not_called()


class _FakeModel:
    nq = 2
    nv = 2
    njoints = 4
    frames = [object(), object()]

    def getJointId(self, name):
        return {"gripper": 2}.get(name, self.njoints)

    def getFrameId(self, name):
        return 1 if name == "tool" else len(self.frames)

    def createData(self):
        return object()


def _fake_pinocchio():
    model = _FakeModel()
    return SimpleNamespace(
        ReferenceFrame=SimpleNamespace(
            LOCAL=0, WORLD=1, LOCAL_WORLD_ALIGNED=2
        ),
        buildModelFromUrdf=lambda path: model,
        buildReducedModel=lambda full, locked, neutral: model,
        neutral=lambda full: np.zeros(full.nq),
        rnea=lambda model, data, q, dq, ddq: q + 2.0 * dq + 3.0 * ddq,
        computeRNEADerivatives=lambda model, data, q, dq, ddq: (
            np.eye(2),
            2.0 * np.eye(2),
            3.0 * np.eye(2),
        ),
        computeJointJacobians=lambda model, data, q: None,
        framesForwardKinematics=lambda model, data, q: None,
        getFrameJacobian=lambda model, data, frame, reference: np.vstack(
            (np.eye(2), np.eye(2), np.eye(2))
        ),
    )


def test_batched_pinocchio_cache_uses_reduced_model_config(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()
    dynamics = PinocchioDynamics(
        {
            "physics": {
                "pinocchio": {
                    "urdf_path": urdf_path,
                    "frame_name": "tool",
                    "reference_frame": "LOCAL_WORLD_ALIGNED",
                    "locked_joint_names": ["gripper"],
                }
            }
        }
    )
    q = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    dq = 0.1 * q
    ddq = 0.01 * q
    with patch(
        "physics.nero_dynamics.importlib.import_module",
        return_value=_fake_pinocchio(),
    ):
        cache = dynamics.build_cache(q, dq, ddq)

    assert cache.rnea.tau_id_reference.shape == (1, 2, 2)
    assert cache.frame_jacobian.shape == (1, 2, 6, 2)
    torch.testing.assert_close(
        cache.rnea.tau_id_reference,
        q + 2.0 * dq + 3.0 * ddq,
    )
    torch.testing.assert_close(
        cache.rnea.d_tau_d_ddq,
        3.0 * torch.eye(2).expand(1, 2, 2, 2),
    )


def test_batched_inverse_dynamics_skips_derivative_computation(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()
    dynamics = PinocchioDynamics(
        {
            "pinocchio": {
                "urdf_path": urdf_path,
                "frame_name": "tool",
                "locked_joint_names": ["gripper"],
            }
        }
    )
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dq = 0.1 * q
    ddq = 0.01 * q
    pinocchio = _fake_pinocchio()
    with patch(
        "physics.nero_dynamics.importlib.import_module",
        return_value=pinocchio,
    ), patch.object(
        pinocchio,
        "computeRNEADerivatives",
    ) as derivatives:
        tau = dynamics.inverse_dynamics(q, dq, ddq)

    derivatives.assert_not_called()
    torch.testing.assert_close(tau, q + 2.0 * dq + 3.0 * ddq)


def test_batched_gravity_torque_uses_zero_velocity_and_acceleration(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.touch()
    dynamics = PinocchioDynamics(
        {
            "pinocchio": {
                "urdf_path": urdf_path,
                "frame_name": "tool",
                "locked_joint_names": ["gripper"],
            }
        }
    )
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pinocchio = _fake_pinocchio()
    with patch(
        "physics.nero_dynamics.importlib.import_module",
        return_value=pinocchio,
    ), patch.object(pinocchio, "rnea", wraps=pinocchio.rnea) as rnea:
        gravity = dynamics.gravity_torque(q)

    torch.testing.assert_close(gravity, q)
    assert rnea.call_count == len(q)
    for call in rnea.call_args_list:
        assert np.allclose(call.args[3], np.zeros(2))
        assert np.allclose(call.args[4], np.zeros(2))


def _tau_other_checkpoint(path: Path, architecture="gru"):
    config = {
        "dataloader": {
            "horizon": 3,
            "normalize_mode": "gaussian",
        },
        "model": {
            "architecture": architecture,
            "inputs": ["q", "dq", "ddq", "tau"],
            "input_dims": {key: 2 for key in ("q", "dq", "ddq", "tau")},
            "hidden_dim": 4,
            "num_layers": 1,
            "output_dim": 2,
            "head_num_layers": 1,
            "dropout": 0.0,
        },
    }
    torch.manual_seed(5)
    model = TauOtherSequenceRegressor(config)
    stats = {
        key: {
            "mean": torch.zeros(2),
            "std": torch.ones(2),
            "min": -torch.ones(2),
            "max": torch.ones(2),
            "q01": -torch.ones(2),
            "q99": torch.ones(2),
        }
        for key in ("q", "dq", "ddq", "tau", "tau_other")
    }
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "normalizer": {
                "stats": stats,
                "eps": 1e-6,
                "normalize_mode": "gaussian",
            },
        },
        path,
    )


@pytest.mark.parametrize("architecture", ["gru", "lstm", "tcn"])
def test_frozen_tau_other_checkpoint_uses_caller_histories_and_input_gradients(
    tmp_path,
    architecture,
):
    checkpoint_path = tmp_path / "tau_other.pt"
    _tau_other_checkpoint(checkpoint_path, architecture=architecture)
    predictor = load_tau_other_predictor(
        checkpoint_path,
        max_model_batch_size=2,
    )
    reference_predictor = load_tau_other_predictor(checkpoint_path)
    history = {
        key: torch.randn(2, 4, 2, requires_grad=True)
        for key in predictor.active_inputs
    }
    future = {
        key: torch.randn(2, 3, 2, requires_grad=True)
        for key in predictor.active_inputs
    }
    tau_other = predictor(history, future)
    with torch.no_grad():
        reference = reference_predictor(history, future)

    assert tau_other.shape == (2, 3, 2)
    torch.testing.assert_close(tau_other, reference)
    assert not any(parameter.requires_grad for parameter in predictor.parameters())
    tau_other.square().mean().backward()
    assert future["q"].grad is not None
    assert future["q"].grad.abs().sum() > 0

    incomplete_history = dict(history)
    incomplete_history.pop("ddq")
    with pytest.raises(KeyError, match="ddq"):
        predictor(incomplete_history, future)


@pytest.mark.parametrize("architecture", ["gru", "tcn"])
def test_frozen_predictor_matches_independent_sliding_windows(
    tmp_path,
    architecture,
):
    checkpoint_path = tmp_path / "tau_other.pt"
    _tau_other_checkpoint(checkpoint_path, architecture=architecture)
    predictor = load_tau_other_predictor(checkpoint_path)
    history = {
        key: torch.randn(2, 4, 2)
        for key in predictor.active_inputs
    }
    future = {
        key: torch.randn(2, 3, 2)
        for key in predictor.active_inputs
    }

    prediction = predictor(history, future)
    manual_batch = {}
    for key in predictor.active_inputs:
        complete = torch.cat((history[key][:, -3:], future[key]), dim=1)
        normalized = predictor.normalizer.normalize(key, complete)
        manual_batch[key] = torch.stack(
            [normalized[:, offset + 1 : offset + 4] for offset in range(3)],
            dim=1,
        ).reshape(6, 3, 2)
    expected = predictor.model(manual_batch)["tau_other_pred"].reshape(2, 3, 2)
    expected = predictor.normalizer.denormalize("tau_other", expected)

    torch.testing.assert_close(prediction, expected)
