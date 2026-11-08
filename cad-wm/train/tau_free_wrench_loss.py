import math

import torch

from physics.nero_dynamics import damped_wrench_from_joint_torque


class TauFreeTorqueWrenchLoss:
    """Constrain free-space torque prediction and its implied tool wrench."""

    def __init__(self, config):
        loss_config = config.get("loss") or {}
        physics_config = config.get("physics") or {}
        self.torque_loss_space = str(
            loss_config.get("torque_loss_space", "normalized")
        ).lower()
        configured_joint_weights = loss_config.get("joint_weights")
        self.has_configured_joint_weights = configured_joint_weights is not None
        default_joint_weight_mode = (
            "manual" if configured_joint_weights is not None else "equal"
        )
        self.joint_weight_mode = str(
            loss_config.get("joint_weight_mode", default_joint_weight_mode)
        ).lower()
        self.tau_weight = float(loss_config.get("tau_weight", 1.0))
        self.wrench_weight = float(loss_config.get("wrench_weight", 0.1))
        self.force_scale_n = float(loss_config.get("force_scale_n", 1.0))
        self.moment_scale_nm = float(loss_config.get("moment_scale_nm", 0.1))
        self.wrench_damping = float(
            physics_config.get(
                "wrench_damping",
                loss_config.get("wrench_damping", 0.02),
            )
        )
        if self.joint_weight_mode == "equal":
            self.joint_weights = torch.ones(7, dtype=torch.float32)
        elif configured_joint_weights is None:
            self.joint_weights = None
        else:
            self.joint_weights = torch.as_tensor(
                configured_joint_weights,
                dtype=torch.float32,
            )
        self._validate()

    def _validate(self):
        if self.torque_loss_space not in {"normalized", "physical_nm"}:
            raise ValueError(
                "tau-free loss.torque_loss_space must be 'normalized' or "
                f"'physical_nm', got {self.torque_loss_space!r}"
            )
        allowed_weight_modes = {"equal", "manual", "mean_abs", "max_abs"}
        if self.joint_weight_mode not in allowed_weight_modes:
            raise ValueError(
                "tau-free loss.joint_weight_mode must be equal, manual, "
                f"mean_abs, or max_abs; got {self.joint_weight_mode!r}"
            )
        values = {
            "tau_weight": self.tau_weight,
            "wrench_weight": self.wrench_weight,
            "force_scale_n": self.force_scale_n,
            "moment_scale_nm": self.moment_scale_nm,
            "wrench_damping": self.wrench_damping,
        }
        invalid = [
            name
            for name, value in values.items()
            if not math.isfinite(value) or value < 0.0
        ]
        if invalid:
            raise ValueError(f"tau-free loss values must be finite and non-negative: {invalid}")
        if self.tau_weight + self.wrench_weight <= 0.0:
            raise ValueError("tau-free tau_weight or wrench_weight must be positive")
        if self.force_scale_n <= 0.0 or self.moment_scale_nm <= 0.0:
            raise ValueError("tau-free wrench scales must be positive")
        if self.wrench_damping <= 0.0:
            raise ValueError("tau-free wrench_damping must be positive")
        if self.joint_weight_mode == "manual" and self.joint_weights is None:
            raise ValueError(
                "tau-free loss.joint_weight_mode=manual requires joint_weights"
            )
        if (
            self.joint_weight_mode in {"equal", "mean_abs", "max_abs"}
            and self.has_configured_joint_weights
        ):
            raise ValueError(
                "tau-free automatic joint weighting requires joint_weights=null"
            )
        if self.joint_weights is not None:
            if self.joint_weights.ndim != 1 or self.joint_weights.numel() != 7:
                raise ValueError("tau-free loss.joint_weights must contain seven values")
            if (self.joint_weights < 0).any():
                raise ValueError("tau-free loss.joint_weights must be non-negative")

    @property
    def needs_joint_weight_statistics(self):
        return self.joint_weight_mode in {"mean_abs", "max_abs"}

    def resolve_joint_weights(self, target_tau_nm):
        """Resolve automatic axis weights from physical training targets only."""

        if not self.needs_joint_weight_statistics:
            return self.joint_weights
        if target_tau_nm.ndim < 2 or target_tau_nm.shape[-1] != 7:
            raise ValueError(
                "tau-free joint-weight statistics require a [..., 7] tau tensor"
            )
        flattened = target_tau_nm.detach().reshape(-1, 7).abs()
        if self.joint_weight_mode == "mean_abs":
            scale = flattened.mean(dim=0)
        else:
            scale = flattened.max(dim=0).values
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(
                "tau-free automatic joint-weight scales must be finite and positive"
            )
        self.joint_weights = (scale / scale.mean()).to(dtype=torch.float32).cpu()
        return self.joint_weights

    @staticmethod
    def _validate_torque_shapes(
        prediction,
        target,
        prediction_nm,
        target_nm,
    ):
        if prediction.shape != target.shape or prediction.shape[-1] != 7:
            raise ValueError(
                "tau-free prediction and target must have matching [..., 7] shapes"
            )
        if prediction_nm.shape != prediction.shape or target_nm.shape != target.shape:
            raise ValueError(
                "physical and normalized tau tensors must have matching shapes"
            )

    def torque_objective(
        self,
        prediction,
        target,
        prediction_nm,
        target_nm,
    ):
        """Compute seven per-joint MSE values and their weighted mean.

        The model can continue to predict normalized torque while the optimization
        objective is evaluated either in normalized coordinates or directly in
        physical Nm.  ``physical_nm`` therefore gives equal cost to the same
        absolute torque error on every joint.
        """

        self._validate_torque_shapes(
            prediction,
            target,
            prediction_nm,
            target_nm,
        )
        normalized_squared_error = (prediction - target).square()
        physical_squared_error_nm2 = (prediction_nm - target_nm).square()
        objective_squared_error = (
            physical_squared_error_nm2
            if self.torque_loss_space == "physical_nm"
            else normalized_squared_error
        )

        objective_joint_mse = objective_squared_error.reshape(-1, 7).mean(dim=0)
        if self.joint_weights is None:
            raise RuntimeError(
                "tau-free automatic joint weights have not been resolved from "
                "the training split"
            )
        joint_weights = self.joint_weights.to(
            device=objective_joint_mse.device,
            dtype=objective_joint_mse.dtype,
        )
        objective_joint_mse = objective_joint_mse * joint_weights
        tau_mse = objective_joint_mse.mean()

        physical_joint_mse_nm2 = physical_squared_error_nm2.reshape(-1, 7).mean(
            dim=0
        )
        normalized_joint_mse = normalized_squared_error.reshape(-1, 7).mean(dim=0)
        metrics = {
            "tau_mse": tau_mse.detach(),
            "tau_mse_normalized": normalized_joint_mse.mean().detach(),
            "tau_mse_nm2": physical_joint_mse_nm2.mean().detach(),
        }
        metrics.update(
            {
                f"tau_mse_nm2_j{joint_index}": value.detach()
                for joint_index, value in enumerate(
                    physical_joint_mse_nm2,
                    start=1,
                )
            }
        )
        metrics.update(
            {
                f"tau_joint_weight_j{joint_index}": value.detach()
                for joint_index, value in enumerate(joint_weights, start=1)
            }
        )
        return tau_mse, metrics

    def __call__(
        self,
        prediction,
        target,
        prediction_nm,
        target_nm,
        frame_jacobian,
    ):
        tau_mse, metrics = self.torque_objective(
            prediction,
            target,
            prediction_nm,
            target_nm,
        )

        # External torque follows the deployment convention: measured torque
        # minus the free-space torque predicted by the model.
        tau_ext_pred_nm = target_nm - prediction_nm
        wrench_pred = damped_wrench_from_joint_torque(
            frame_jacobian.to(tau_ext_pred_nm),
            tau_ext_pred_nm,
            damping=self.wrench_damping,
        )
        wrench_scale = wrench_pred.new_tensor(
            [self.force_scale_n] * 3 + [self.moment_scale_nm] * 3
        )
        wrench_mse_scaled = (wrench_pred / wrench_scale).square().mean()
        total = (
            self.tau_weight * tau_mse
            + self.wrench_weight * wrench_mse_scaled
        )

        metrics.update(
            {
                "wrench_mse_scaled": wrench_mse_scaled.detach(),
                "wrench_force_mse_n2": wrench_pred[..., :3]
                .square()
                .mean()
                .detach(),
                "wrench_moment_mse_nm2": wrench_pred[..., 3:]
                .square()
                .mean()
                .detach(),
            }
        )
        diagnostics = {
            "tau_ext_pred_nm": tau_ext_pred_nm,
            "wrench_pred": wrench_pred,
        }
        return total, metrics, diagnostics
