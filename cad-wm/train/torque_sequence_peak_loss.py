import math

import torch


class TorqueSequencePeakLoss:
    """Configurable MSE or CVaR objective for sequence-torque pipelines."""

    def __init__(self, config=None):
        config = config or {}
        self.loss_type = str(config.get("type", "peak_cvar")).lower()
        if self.loss_type not in {"mse", "peak_cvar"}:
            raise ValueError(
                "sequence torque loss.type must be 'mse' or 'peak_cvar', "
                f"got {self.loss_type!r}"
            )

        self.tail_fraction = float(config.get("tail_fraction", 0.02))
        self.peak_weight = float(config.get("peak_weight", 0.9))
        self.mean_weight = float(config.get("mean_weight", 0.1))
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("loss.tail_fraction must be in (0, 1]")
        if self.peak_weight < 0.0 or self.mean_weight < 0.0:
            raise ValueError("loss peak_weight and mean_weight must be non-negative")
        if self.peak_weight + self.mean_weight <= 0.0:
            raise ValueError("at least one loss weight must be positive")

    def _tail_mean(self, values):
        flat = values.reshape(-1)
        tail_count = max(1, math.ceil(flat.numel() * self.tail_fraction))
        return torch.topk(flat, k=tail_count, sorted=False).values.mean()

    def __call__(self, prediction, target, joint_weights=None):
        squared_error = (prediction - target).square()
        weighted_squared_error = squared_error
        if joint_weights is not None:
            weights = torch.as_tensor(
                joint_weights,
                device=squared_error.device,
                dtype=squared_error.dtype,
            )
            if weights.ndim != 1 or weights.numel() != squared_error.shape[-1]:
                raise ValueError(
                    f"loss.joint_weights has {weights.numel()} entries, "
                    f"expected {squared_error.shape[-1]}."
                )
            if (weights < 0).any():
                raise ValueError("loss.joint_weights must be non-negative")
            weighted_squared_error = squared_error * weights

        if self.loss_type == "mse":
            return weighted_squared_error.mean()

        peak_mse_nm2 = self._tail_mean(weighted_squared_error)
        mean_mse_nm2 = weighted_squared_error.mean()
        weight_sum = self.peak_weight + self.mean_weight
        objective = (
            self.peak_weight * peak_mse_nm2
            + self.mean_weight * mean_mse_nm2
        ) / weight_sum
        return objective

    @torch.no_grad()
    def metrics_from_absolute_error(self, absolute_error_nm):
        if absolute_error_nm.ndim < 2:
            raise ValueError(
                "absolute torque error must have a sample and joint dimension"
            )
        flattened_by_joint = absolute_error_nm.reshape(
            -1,
            absolute_error_nm.shape[-1],
        )
        squared_error_nm2 = flattened_by_joint.square()
        peak_cvar_rmse_nm = self._tail_mean(squared_error_nm2).sqrt()

        joint_tail_count = max(
            1,
            math.ceil(flattened_by_joint.shape[0] * self.tail_fraction),
        )
        joint_peak_cvar_rmse_nm = torch.topk(
            squared_error_nm2,
            k=joint_tail_count,
            dim=0,
            sorted=False,
        ).values.mean(dim=0).sqrt()

        metrics = {
            "peak_cvar_rmse_nm": peak_cvar_rmse_nm,
            "peak_p95_nm": torch.quantile(flattened_by_joint, 0.95),
            "peak_p99_nm": torch.quantile(flattened_by_joint, 0.99),
            "peak_max_nm": flattened_by_joint.max(),
        }
        metrics.update(
            {
                f"peak_cvar_rmse_nm_j{joint_index}": value
                for joint_index, value in enumerate(
                    joint_peak_cvar_rmse_nm,
                    start=1,
                )
            }
        )
        return metrics
