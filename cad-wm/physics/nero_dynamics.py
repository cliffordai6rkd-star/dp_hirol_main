"""Nero inverse-dynamics utilities for torque world-model training.

Pinocchio is used only to cache inverse dynamics and frame Jacobians around
recorded states. The online loss path is implemented entirely in Torch so that
gradients flow from wrench supervision into predicted q, dq, ddq, and torque.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from model.tau_other_sequence import (
    TauOtherSequenceModelBase,
    build_tau_other_sequence_model,
)


def _same_state_shape(named_tensors: Mapping[str, Tensor]) -> tuple[int, ...]:
    shape = None
    for name, value in named_tensors.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.ndim < 2:
            raise ValueError(
                f"{name} must have shape [..., N], got {tuple(value.shape)}."
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {value.dtype}.")
        if shape is None:
            shape = tuple(value.shape)
        elif tuple(value.shape) != shape:
            raise ValueError(
                "Joint-state tensors must have the same shape; "
                f"{name} has {tuple(value.shape)}, expected {shape}."
            )
    if shape is None:
        raise ValueError("At least one joint-state tensor is required.")
    return shape


@dataclass(frozen=True)
class RNEALinearization:
    """RNEA values and derivatives evaluated at recorded reference states."""

    q_reference: Tensor
    dq_reference: Tensor
    ddq_reference: Tensor
    tau_id_reference: Tensor
    d_tau_d_q: Tensor
    d_tau_d_dq: Tensor
    d_tau_d_ddq: Tensor

    def to(self, *args, **kwargs) -> "RNEALinearization":
        return RNEALinearization(
            **{
                name: value.to(*args, **kwargs)
                for name, value in self.__dict__.items()
            }
        )


@dataclass(frozen=True)
class NeroDynamicsCache:
    """Recorded-state local dynamics cache used by the differentiable loss."""

    rnea: RNEALinearization
    frame_jacobian: Tensor
    # Optional for caches serialized before the explicit gravity branch was
    # added.  New caches always populate this field from RNEA(q, 0, 0).
    tau_g_reference: Tensor | None = None

    def to(self, *args, **kwargs) -> "NeroDynamicsCache":
        return NeroDynamicsCache(
            rnea=self.rnea.to(*args, **kwargs),
            frame_jacobian=self.frame_jacobian.to(*args, **kwargs),
            tau_g_reference=(
                None
                if self.tau_g_reference is None
                else self.tau_g_reference.to(*args, **kwargs)
            ),
        )


@dataclass(frozen=True)
class NeroWrenchPrediction:
    """Intermediate physical quantities exposed for losses and diagnostics."""

    tau_id: Tensor
    tau_g: Tensor
    tau_other: Tensor
    tau_external: Tensor
    wrench: Tensor


class PinocchioDynamics:
    """Lazy, config-driven Pinocchio cache builder.

    Constructing this class does not import Pinocchio. The dependency and URDF
    are required only when a cache method is called, which keeps data-only CPU
    training and model-only tests independent of the robotics stack.
    """

    _REFERENCE_FRAME_NAMES = {
        "LOCAL",
        "WORLD",
        "LOCAL_WORLD_ALIGNED",
    }

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = self._select_config(config or {})
        self.urdf_path = Path(
            self._value(
                config,
                "urdf_path",
                "pinocchio_urdf_path",
                default="sim_mesh/nero/nero_with_gripper.urdf",
            )
        )
        self.frame_name = str(
            self._value(
                config,
                "frame_name",
                "pinocchio_frame_name",
                default="gripper_tcp",
            )
        )
        self.locked_joint_names = tuple(
            self._value(
                config,
                "locked_joint_names",
                "pinocchio_locked_joint_names",
                default=("gripper", "gripper_joint1", "gripper_joint2"),
            )
        )
        self.reference_frame_name = str(
            self._value(
                config,
                "reference_frame",
                "pinocchio_reference_frame",
                default="LOCAL",
            )
        ).upper()
        if self.reference_frame_name not in self._REFERENCE_FRAME_NAMES:
            choices = ", ".join(sorted(self._REFERENCE_FRAME_NAMES))
            raise ValueError(
                f"Pinocchio reference_frame must be one of {choices}, got "
                f"{self.reference_frame_name!r}."
            )

        self._pin = None
        self._model = None
        self._data = None
        self._frame_id = None
        self._reference_frame = None

    @staticmethod
    def _select_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
        physics = config.get("physics")
        if isinstance(physics, Mapping):
            pinocchio = physics.get("pinocchio")
            return pinocchio if isinstance(pinocchio, Mapping) else physics
        pinocchio = config.get("pinocchio")
        if isinstance(pinocchio, Mapping):
            return pinocchio
        loss = config.get("loss")
        if isinstance(loss, Mapping):
            return loss
        return config

    @staticmethod
    def _value(config, key, legacy_key, default):
        if key in config:
            return config[key]
        return config.get(legacy_key, default)

    @property
    def model(self):
        self._ensure_initialized()
        return self._model

    def _ensure_initialized(self) -> None:
        if self._model is not None:
            return
        if not self.urdf_path.is_file():
            raise FileNotFoundError(
                f"Pinocchio URDF does not exist: {self.urdf_path}"
            )
        try:
            pin = importlib.import_module("pinocchio")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Pinocchio is required only to build Nero dynamics caches. "
                "Install pinocchio or disable/cache the physics loss first."
            ) from exc

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        locked_joint_ids = []
        for name in self.locked_joint_names:
            joint_id = full_model.getJointId(name)
            if joint_id == full_model.njoints:
                raise ValueError(f"Pinocchio joint not found: {name}")
            locked_joint_ids.append(joint_id)

        if locked_joint_ids:
            model = pin.buildReducedModel(
                full_model,
                locked_joint_ids,
                pin.neutral(full_model),
            )
        else:
            model = full_model
        frame_id = model.getFrameId(self.frame_name)
        if frame_id == len(model.frames):
            raise ValueError(f"Pinocchio frame not found: {self.frame_name}")

        self._pin = pin
        self._model = model
        self._data = model.createData()
        self._frame_id = frame_id
        self._reference_frame = getattr(
            pin.ReferenceFrame,
            self.reference_frame_name,
        )

    def _validated_numpy_states(
        self,
        q: Tensor,
        dq: Tensor,
        ddq: Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
        shape = _same_state_shape({"q": q, "dq": dq, "ddq": ddq})
        self._ensure_initialized()
        if shape[-1] != self._model.nq or shape[-1] != self._model.nv:
            raise ValueError(
                f"Joint dimension {shape[-1]} does not match reduced "
                f"Pinocchio nq={self._model.nq}, nv={self._model.nv}."
            )
        arrays = tuple(
            value.detach().cpu().to(torch.float64).numpy().reshape(-1, shape[-1])
            for value in (q, dq, ddq)
        )
        return (*arrays, shape)

    def rnea_linearization(
        self,
        q: Tensor,
        dq: Tensor,
        ddq: Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RNEALinearization:
        """Evaluate batched RNEA and q/dq/ddq derivatives on the CPU."""

        q_np, dq_np, ddq_np, shape = self._validated_numpy_states(q, dq, ddq)
        joint_dim = shape[-1]
        tau = np.empty_like(q_np)
        derivative_shape = (q_np.shape[0], joint_dim, joint_dim)
        derivatives = [
            np.empty(derivative_shape, dtype=np.float64)
            for _ in range(3)
        ]

        for index, (q_value, dq_value, ddq_value) in enumerate(
            zip(q_np, dq_np, ddq_np)
        ):
            tau[index] = np.asarray(
                self._pin.rnea(
                    self._model,
                    self._data,
                    q_value,
                    dq_value,
                    ddq_value,
                ),
                dtype=np.float64,
            )
            sources = self._pin.computeRNEADerivatives(
                self._model,
                self._data,
                q_value,
                dq_value,
                ddq_value,
            )
            for destination, source in zip(derivatives, sources):
                destination[index] = np.asarray(source, dtype=np.float64)

        output_device = q.device if device is None else torch.device(device)
        output_dtype = q.dtype if dtype is None else dtype

        def tensor(value, trailing_shape=()):
            return torch.as_tensor(
                value.reshape(*shape[:-1], *trailing_shape),
                device=output_device,
                dtype=output_dtype,
            )

        matrix_shape = (joint_dim, joint_dim)
        return RNEALinearization(
            q_reference=q.detach().to(device=output_device, dtype=output_dtype),
            dq_reference=dq.detach().to(device=output_device, dtype=output_dtype),
            ddq_reference=ddq.detach().to(device=output_device, dtype=output_dtype),
            tau_id_reference=tensor(tau, (joint_dim,)),
            d_tau_d_q=tensor(derivatives[0], matrix_shape),
            d_tau_d_dq=tensor(derivatives[1], matrix_shape),
            d_tau_d_ddq=tensor(derivatives[2], matrix_shape),
        )

    def inverse_dynamics(
        self,
        q: Tensor,
        dq: Tensor,
        ddq: Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Evaluate batched RNEA without derivative computation."""

        q_np, dq_np, ddq_np, shape = self._validated_numpy_states(q, dq, ddq)
        tau = np.empty_like(q_np)
        for index, (q_value, dq_value, ddq_value) in enumerate(
            zip(q_np, dq_np, ddq_np)
        ):
            tau[index] = np.asarray(
                self._pin.rnea(
                    self._model,
                    self._data,
                    q_value,
                    dq_value,
                    ddq_value,
                ),
                dtype=np.float64,
            )
        return torch.as_tensor(
            tau.reshape(shape),
            device=q.device if device is None else device,
            dtype=q.dtype if dtype is None else dtype,
        )

    def gravity_torque(
        self,
        q: Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Evaluate the RNEA gravity term ``g(q)`` for batched positions."""

        shape = _same_state_shape({"q": q})
        self._ensure_initialized()
        if shape[-1] != self._model.nq:
            raise ValueError(
                f"Joint dimension {shape[-1]} does not match reduced "
                f"Pinocchio nq={self._model.nq}."
            )

        q_np = q.detach().cpu().to(torch.float64).numpy().reshape(-1, shape[-1])
        zero_velocity = np.zeros(self._model.nv, dtype=np.float64)
        gravity = np.empty_like(q_np)
        for index, q_value in enumerate(q_np):
            # RNEA(q, 0, 0) is the generalized gravity torque g(q).
            gravity[index] = np.asarray(
                self._pin.rnea(
                    self._model,
                    self._data,
                    q_value,
                    zero_velocity,
                    zero_velocity,
                ),
                dtype=np.float64,
            )
        return torch.as_tensor(
            gravity.reshape(shape),
            device=q.device if device is None else device,
            dtype=q.dtype if dtype is None else dtype,
        )

    def frame_jacobians(
        self,
        q: Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Cache the configured 6D frame Jacobian for batched joint states."""

        shape = _same_state_shape({"q": q})
        self._ensure_initialized()
        if shape[-1] != self._model.nq:
            raise ValueError(
                f"Joint dimension {shape[-1]} does not match reduced "
                f"Pinocchio nq={self._model.nq}."
            )
        q_np = q.detach().cpu().to(torch.float64).numpy().reshape(-1, shape[-1])
        jacobians = np.empty((q_np.shape[0], 6, self._model.nv), dtype=np.float64)
        for index, q_value in enumerate(q_np):
            self._pin.computeJointJacobians(self._model, self._data, q_value)
            self._pin.framesForwardKinematics(self._model, self._data, q_value)
            jacobians[index] = np.asarray(
                self._pin.getFrameJacobian(
                    self._model,
                    self._data,
                    self._frame_id,
                    self._reference_frame,
                ),
                dtype=np.float64,
            )
        return torch.as_tensor(
            jacobians.reshape(*shape[:-1], 6, self._model.nv),
            device=q.device if device is None else device,
            dtype=q.dtype if dtype is None else dtype,
        )

    def frame_poses(
        self,
        q: Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return configured frame poses as ``[..., 7]`` ``xyz+xyzw``.

        This is forward kinematics only. It does not evaluate RNEA, torque
        dynamics, or any physics loss, so it remains usable for action
        condition construction when the RNEA rollout is disabled.
        """

        shape = _same_state_shape({"q": q})
        self._ensure_initialized()
        if shape[-1] != self._model.nq:
            raise ValueError(
                f"Joint dimension {shape[-1]} does not match reduced "
                f"Pinocchio nq={self._model.nq}."
            )
        q_np = q.detach().cpu().to(torch.float64).numpy().reshape(-1, shape[-1])
        poses = np.empty((q_np.shape[0], 7), dtype=np.float64)
        for index, q_value in enumerate(q_np):
            self._pin.framesForwardKinematics(self._model, self._data, q_value)
            placement = self._data.oMf[self._frame_id]
            quaternion = np.asarray(
                self._pin.Quaternion(placement.rotation).coeffs(),
                dtype=np.float64,
            )
            poses[index, :3] = np.asarray(placement.translation, dtype=np.float64)
            poses[index, 3:] = quaternion
        return torch.as_tensor(
            poses.reshape(*shape[:-1], 7),
            device=q.device if device is None else device,
            dtype=q.dtype if dtype is None else dtype,
        )

    def build_cache(
        self,
        q: Tensor,
        dq: Tensor,
        ddq: Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NeroDynamicsCache:
        """Build matching RNEA-linearization and frame-Jacobian caches."""

        return NeroDynamicsCache(
            rnea=self.rnea_linearization(
                q,
                dq,
                ddq,
                device=device,
                dtype=dtype,
            ),
            frame_jacobian=self.frame_jacobians(
                q,
                device=device,
                dtype=dtype,
            ),
            tau_g_reference=self.gravity_torque(q, device=device, dtype=dtype),
        )


class _CheckpointNormalizer(nn.Module):
    _MODES = {None, "gaussian", "limit", "quantile"}

    def __init__(self, payload: Mapping[str, Any] | None, fallback_mode=None):
        super().__init__()
        payload = payload or {}
        self.mode = payload.get("normalize_mode", fallback_mode)
        if self.mode not in self._MODES:
            raise ValueError(f"Unsupported tau_other normalize_mode: {self.mode!r}")
        self.eps = float(payload.get("eps", 1e-6))
        self._buffer_lookup: dict[tuple[str, str], str] = {}
        statistics = (payload.get("stats") or {}).items()
        for key_index, (key, values) in enumerate(statistics):
            for statistic, value in values.items():
                buffer_name = f"stat_{key_index}_{statistic}"
                self.register_buffer(
                    buffer_name,
                    torch.as_tensor(value).detach().clone(),
                )
                self._buffer_lookup[(str(key), str(statistic))] = buffer_name

    def _stat(self, key: str, statistic: str, value: Tensor) -> Tensor:
        try:
            buffer_name = self._buffer_lookup[(key, statistic)]
        except KeyError as exc:
            raise KeyError(
                f"tau_other checkpoint normalizer lacks {statistic!r} for {key!r}."
            ) from exc
        return getattr(self, buffer_name).to(device=value.device, dtype=value.dtype)

    def normalize(self, key: str, value: Tensor) -> Tensor:
        if self.mode is None:
            return value
        if self.mode == "gaussian":
            return (value - self._stat(key, "mean", value)) / (
                self._stat(key, "std", value) + self.eps
            )
        if self.mode == "limit":
            minimum = self._stat(key, "min", value)
            maximum = self._stat(key, "max", value)
            return 2.0 * (value - minimum) / (maximum - minimum + self.eps) - 1.0
        q01 = self._stat(key, "q01", value)
        q99 = self._stat(key, "q99", value)
        return torch.clamp(
            2.0 * (value - q01) / (q99 - q01 + self.eps) - 1.0,
            -1.0,
            1.0,
        )

    def denormalize(self, key: str, value: Tensor) -> Tensor:
        if self.mode is None:
            return value
        if self.mode == "gaussian":
            return value * (self._stat(key, "std", value) + self.eps) + self._stat(
                key, "mean", value
            )
        if self.mode == "limit":
            minimum = self._stat(key, "min", value)
            maximum = self._stat(key, "max", value)
            return (value + 1.0) * (maximum - minimum + self.eps) / 2.0 + minimum
        q01 = self._stat(key, "q01", value)
        q99 = self._stat(key, "q99", value)
        return (value + 1.0) * (q99 - q01 + self.eps) / 2.0 + q01


class FrozenTauOtherPredictor(nn.Module):
    """Frozen sequence branch that remains differentiable to its inputs."""

    def __init__(
        self,
        model: TauOtherSequenceModelBase,
        *,
        normalizer_payload: Mapping[str, Any] | None = None,
        normalize_mode: str | None = None,
        history_horizon: int = 50,
        max_model_batch_size: int = 1024,
        target_generation_method: str | None = None,
        target_key: str = "tau_other",
    ):
        super().__init__()
        if history_horizon <= 0:
            raise ValueError("tau_other history_horizon must be positive.")
        if max_model_batch_size <= 0:
            raise ValueError("tau_other max_model_batch_size must be positive.")
        self.model = model
        self.history_horizon = int(history_horizon)
        self.max_model_batch_size = int(max_model_batch_size)
        self.target_generation_method = target_generation_method
        self.target_key = str(target_key)
        if not self.target_key:
            raise ValueError("tau_other target_key must not be empty")
        self.normalizer = _CheckpointNormalizer(
            normalizer_payload,
            fallback_mode=normalize_mode,
        )
        self.register_buffer(
            "_window_offsets",
            torch.arange(self.history_horizon),
            persistent=False,
        )
        self.model.requires_grad_(False)
        if hasattr(self.model, "recurrent"):
            self.model.recurrent.dropout = 0.0
        self.train(False)

    @property
    def active_inputs(self) -> tuple[str, ...]:
        return tuple(self.model.active_inputs)

    def train(self, mode: bool = True):
        # cuDNN needs its RNN in training mode to retain the reserve space used
        # for gradients to inputs. Parameters stay frozen and dropout is zero.
        super().train(False)
        self.model.eval()
        if hasattr(self.model, "recurrent"):
            self.model.recurrent.train(True)
            self.model.recurrent.dropout = 0.0
        return self

    @staticmethod
    def _sequence_shape(name: str, value: Tensor) -> tuple[int, int]:
        if not torch.is_tensor(value) or value.ndim != 3:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value)
            raise ValueError(f"{name} must have shape [B, S, D], got {shape}.")
        return int(value.shape[0]), int(value.shape[1])

    def forward(
        self,
        history: Mapping[str, Tensor],
        future: Mapping[str, Tensor],
    ) -> Tensor:
        """Predict tau_other for each future step from caller-provided states.

        The caller must explicitly provide q/dq/ddq/tau histories whenever the
        checkpoint uses them. No finite differences or legacy world-model
        state are consulted inside this helper.
        """

        complete_inputs = {}
        batch_size = future_steps = None
        for key in self.active_inputs:
            if key not in history or key not in future:
                raise KeyError(
                    f"Frozen tau_other input {key!r} must be supplied in both "
                    "history and future mappings."
                )
            history_value = history[key]
            future_value = future[key]
            history_batch, history_steps = self._sequence_shape(
                f"history[{key!r}]", history_value
            )
            future_batch, key_future_steps = self._sequence_shape(
                f"future[{key!r}]", future_value
            )
            expected_dim = int(self.model.input_dims[key])
            if (
                history_value.shape[-1] != expected_dim
                or future_value.shape[-1] != expected_dim
            ):
                raise ValueError(
                    f"tau_other input {key!r} must have D={expected_dim}, got "
                    f"history D={history_value.shape[-1]} and future "
                    f"D={future_value.shape[-1]}."
                )
            if history_steps < self.history_horizon:
                raise ValueError(
                    f"tau_other history needs at least {self.history_horizon} steps, "
                    f"got {history_steps}."
                )
            if batch_size is None:
                batch_size = history_batch
                future_steps = key_future_steps
            if history_batch != batch_size or future_batch != batch_size:
                raise ValueError("All tau_other inputs must share the same batch size.")
            if key_future_steps != future_steps:
                raise ValueError("All future tau_other inputs must share the same horizon.")
            complete = torch.cat(
                (history_value[:, -self.history_horizon :], future_value),
                dim=1,
            )
            complete_inputs[key] = self.normalizer.normalize(key, complete)

        if self.model.architecture == "tcn":
            # An exact-H causal receptive field makes dense TCN execution
            # identical to independent sliding windows, without materializing
            # B * F copies of the history.
            dense_prediction = self.model.forward_sequence(complete_inputs)
            normalized_tau_other = dense_prediction[:, -future_steps:]
            return self.normalizer.denormalize(self.target_key, normalized_tau_other)

        # Recurrent training treats every H-step window as an independent sample
        # with a fresh state. Build those same windows for every future target
        # instead of carrying a hidden state beyond the trained horizon.
        total_windows = batch_size * future_steps
        flat_indices = torch.arange(
            total_windows,
            device=next(iter(complete_inputs.values())).device,
        )
        batch_indices = torch.div(
            flat_indices,
            future_steps,
            rounding_mode="floor",
        )
        future_indices = flat_indices - batch_indices * future_steps
        window_indices = (
            future_indices[:, None] + 1 + self._window_offsets[None, :]
        )
        chunk_outputs = []
        for start in range(0, total_windows, self.max_model_batch_size):
            end = min(start + self.max_model_batch_size, total_windows)
            model_batch = {
                key: value[
                    batch_indices[start:end, None],
                    window_indices[start:end],
                ]
                for key, value in complete_inputs.items()
            }
            chunk_outputs.append(self.model(model_batch)["tau_other_pred"])
        normalized_tau_other = torch.cat(chunk_outputs, dim=0).reshape(
            batch_size,
            future_steps,
            -1,
        )
        return self.normalizer.denormalize(self.target_key, normalized_tau_other)


def load_tau_other_predictor(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    max_model_batch_size: int = 1024,
) -> FrozenTauOtherPredictor:
    """Load a frozen sequence regressor and its training normalizer."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"tau_other checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("config", "model"):
        if key not in checkpoint:
            raise KeyError(f"tau_other checkpoint is missing required key {key!r}.")
    config = checkpoint["config"]
    model = build_tau_other_sequence_model(config)
    model.load_state_dict(checkpoint["model"])
    dataloader_config = config.get("dataloader") or {}
    normalizer_payload = checkpoint.get("normalizer")
    target_generation = checkpoint.get("derived_target_config")
    if not isinstance(target_generation, Mapping):
        target_generation = config.get("target_generation") or {}
    target_generation_method = None
    if bool(target_generation.get("enabled", False)):
            target_generation_method = str(
                target_generation.get("method", "causal_rnea_residual_v1")
            ).lower()
    predictor = FrozenTauOtherPredictor(
        model,
        normalizer_payload=normalizer_payload,
        normalize_mode=dataloader_config.get("normalize_mode"),
        history_horizon=int(dataloader_config.get("horizon", 50)),
        max_model_batch_size=max_model_batch_size,
        target_generation_method=target_generation_method,
        target_key=str((config.get("model") or {}).get("target_key", "tau_other")),
    )
    return predictor.to(device)


def linearized_rnea(
    q: Tensor,
    dq: Tensor,
    ddq: Tensor,
    linearization: RNEALinearization,
) -> Tensor:
    """Evaluate the differentiable first-order RNEA approximation."""

    shape = _same_state_shape({"q": q, "dq": dq, "ddq": ddq})
    references = {
        "q_reference": linearization.q_reference,
        "dq_reference": linearization.dq_reference,
        "ddq_reference": linearization.ddq_reference,
        "tau_id_reference": linearization.tau_id_reference,
    }
    for name, value in references.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} has {tuple(value.shape)}, expected {shape}.")
    matrix_shape = (*shape, shape[-1])
    for name in ("d_tau_d_q", "d_tau_d_dq", "d_tau_d_ddq"):
        value = getattr(linearization, name)
        if tuple(value.shape) != matrix_shape:
            raise ValueError(f"{name} has {tuple(value.shape)}, expected {matrix_shape}.")
    return (
        linearization.tau_id_reference
        + torch.einsum(
            "...ij,...j->...i",
            linearization.d_tau_d_q,
            q - linearization.q_reference,
        )
        + torch.einsum(
            "...ij,...j->...i",
            linearization.d_tau_d_dq,
            dq - linearization.dq_reference,
        )
        + torch.einsum(
            "...ij,...j->...i",
            linearization.d_tau_d_ddq,
            ddq - linearization.ddq_reference,
        )
    )


def damped_wrench_from_joint_torque(
    frame_jacobian: Tensor,
    tau_external: Tensor,
    *,
    damping: float = 0.02,
) -> Tensor:
    """Solve tau_external = J.T @ wrench using damped least squares."""

    if damping <= 0.0:
        raise ValueError("wrench damping must be positive.")
    if frame_jacobian.ndim < 3:
        raise ValueError(
            "frame_jacobian must have shape [..., W, N], got "
            f"{tuple(frame_jacobian.shape)}."
        )
    expected_tau_shape = (*frame_jacobian.shape[:-2], frame_jacobian.shape[-1])
    if tuple(tau_external.shape) != expected_tau_shape:
        raise ValueError(
            f"tau_external has {tuple(tau_external.shape)}, expected "
            f"{expected_tau_shape} for Jacobian {tuple(frame_jacobian.shape)}."
        )
    lhs = frame_jacobian @ frame_jacobian.transpose(-1, -2)
    identity = torch.eye(
        frame_jacobian.shape[-2],
        device=frame_jacobian.device,
        dtype=frame_jacobian.dtype,
    )
    lhs = lhs + float(damping) ** 2 * identity
    rhs = torch.einsum("...ij,...j->...i", frame_jacobian, tau_external)
    return torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)


def predict_nero_wrench(
    *,
    q: Tensor,
    dq: Tensor,
    ddq: Tensor,
    tau_measured: Tensor,
    tau_other: Tensor,
    cache: NeroDynamicsCache,
    damping: float = 0.02,
) -> NeroWrenchPrediction:
    """Map state and measured torque to wrench through local Nero dynamics.

    The residual model follows ``tau_other = tau_measured - tau_g`` on free-space
    data.  With external contact, the remaining joint torque is therefore
    ``tau_measured - tau_g - tau_other``.
    """

    shape = _same_state_shape(
        {
            "q": q,
            "dq": dq,
            "ddq": ddq,
            "tau_measured": tau_measured,
            "tau_other": tau_other,
        }
    )
    expected_jacobian_shape = (
        *shape[:-1],
        cache.frame_jacobian.shape[-2],
        shape[-1],
    )
    if tuple(cache.frame_jacobian.shape) != expected_jacobian_shape:
        raise ValueError(
            f"frame_jacobian has {tuple(cache.frame_jacobian.shape)}, expected "
            f"{expected_jacobian_shape}."
        )
    tau_id = linearized_rnea(q, dq, ddq, cache.rnea)
    # Legacy caches did not carry a separate gravity reference.  Their RNEA
    # reference represented the full free-space torque, so using tau_id keeps
    # those checkpoints numerically compatible while new caches use the
    # explicit gravity-only term.
    tau_g = cache.tau_g_reference
    if tau_g is None:
        tau_g = tau_id
    if tuple(tau_g.shape) != tuple(tau_id.shape):
        raise ValueError(
            f"tau_g_reference has {tuple(tau_g.shape)}, expected {tuple(tau_id.shape)}."
        )
    tau_external = tau_measured - tau_g - tau_other
    wrench = damped_wrench_from_joint_torque(
        cache.frame_jacobian,
        tau_external,
        damping=damping,
    )
    return NeroWrenchPrediction(
        tau_id=tau_id,
        tau_g=tau_g,
        tau_other=tau_other,
        tau_external=tau_external,
        wrench=wrench,
    )
