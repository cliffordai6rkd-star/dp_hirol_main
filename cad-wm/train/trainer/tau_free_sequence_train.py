import argparse
import logging
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from physics.nero_dynamics import PinocchioDynamics
from train.tau_free_wrench_loss import TauFreeTorqueWrenchLoss
from train.trainer.tau_other_sequence_train import (
    TauOtherTrainer,
    run_tau_sequence_training,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a NEXT-style free-space inverse-dynamics model from joint "
            "position history."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=REPO_ROOT / "config/train_cfg/tau_free_sequence/tau_free_sequence.yaml",
        help="Path to the NEXT training config.",
    )
    return parser.parse_args()


class TauFreeSequenceTrainer(TauOtherTrainer):
    """Train stateless proprioceptive windows against free-space torque."""

    def __init__(self, config):
        self._validate_contract(config)
        super().__init__(config)
        self.tau_wrench_loss = TauFreeTorqueWrenchLoss(config)
        self.torque_loss_space = self.tau_wrench_loss.torque_loss_space
        self.use_wrench_objective = self.tau_wrench_loss.wrench_weight > 0.0
        self.wrench_dynamics = (
            PinocchioDynamics(config) if self.use_wrench_objective else None
        )

    def _denormalize_key(self, key, value):
        dataset = getattr(self, "dataset", None)
        if dataset is None:
            return value
        if (
            getattr(dataset, "normalize_mode", None) is None
            or key not in getattr(dataset, "normalize_lowdim_keys", ())
        ):
            return value
        normalizer = getattr(dataset, "normalizer", None)
        if normalizer is None:
            raise RuntimeError(f"normalized tau-free key {key!r} has no normalizer")
        denormalize = getattr(
            normalizer,
            f"{dataset.normalize_mode}_denormalize",
            None,
        )
        if denormalize is None:
            raise ValueError(f"unknown normalize mode: {dataset.normalize_mode}")
        return denormalize(key, value)

    def _build_device_cache(self, training_frames):
        super()._build_device_cache(training_frames)
        if self.tau_wrench_loss.needs_joint_weight_statistics:
            training_indices = torch.as_tensor(
                training_frames,
                device=self.tensor_cache["tau"].device,
                dtype=torch.long,
            )
            tau_train_nm = self._denormalize_key(
                "tau",
                self.tensor_cache["tau"].index_select(0, training_indices),
            )
            joint_weights = self.tau_wrench_loss.resolve_joint_weights(tau_train_nm)
            self.config.setdefault("loss", {})[
                "resolved_joint_weights"
            ] = joint_weights.tolist()
            log.info(
                "tau-free joint MSE weights: mode=%s values=%s",
                self.tau_wrench_loss.joint_weight_mode,
                [round(float(value), 6) for value in joint_weights],
            )
        if not self.use_wrench_objective:
            log.info(
                "pure torque MSE enabled: skip Jacobian cache and wrench solve"
            )
            return
        q_physical = self._denormalize_key("q", self.tensor_cache["q"])
        frame_jacobian = self.wrench_dynamics.frame_jacobians(
            q_physical,
            device=q_physical.device,
            dtype=q_physical.dtype,
        ).contiguous()
        self.tensor_cache["frame_jacobian"] = frame_jacobian
        cache_mib = frame_jacobian.numel() * frame_jacobian.element_size() / (1024**2)
        log.info(
            "tau-free frame Jacobian cache ready: shape=%s memory=%.2f MiB",
            tuple(frame_jacobian.shape),
            cache_mib,
        )

    def _cached_batch(self, sample_indices):
        batch = super()._cached_batch(sample_indices)
        if not self.use_wrench_objective:
            return batch
        sample_indices = sample_indices.to(
            device=self.valid_raw_indices_device.device,
            dtype=torch.long,
        )
        raw_indices = self.valid_raw_indices_device.index_select(0, sample_indices)
        batch["frame_jacobian"] = self.tensor_cache["frame_jacobian"].index_select(
            0,
            raw_indices,
        )
        return batch

    def compute_loss(self, batch):
        if "sample_idx" in batch:
            batch = self._cached_batch(batch["sample_idx"])
        if (
            not self.use_wrench_objective
            and self.torque_loss_space == "normalized"
        ):
            return super().compute_loss(batch)

        out = self.model(batch)
        prediction = out["tau_other_pred"]
        target = out.get("tau_other_target")
        if target is None:
            raise KeyError("Batch is missing the configured tau-free torque target.")
        if self.use_wrench_objective and "frame_jacobian" not in batch:
            raise KeyError("tau-free wrench loss requires frame_jacobian")

        prediction_nm = self._denormalize_target(prediction)
        target_nm = self._denormalize_target(target)
        if self.use_wrench_objective:
            loss, metrics, diagnostics = self.tau_wrench_loss(
                prediction,
                target,
                prediction_nm,
                target_nm,
                batch["frame_jacobian"],
            )
        else:
            tau_mse, metrics = self.tau_wrench_loss.torque_objective(
                prediction,
                target,
                prediction_nm,
                target_nm,
            )
            loss = self.tau_wrench_loss.tau_weight * tau_mse
            diagnostics = {}
        absolute_error_nm = (prediction_nm - target_nm).abs()
        out.update(diagnostics)
        out["loss_dict"] = {
            (
                "tau_wrench_objective"
                if self.use_wrench_objective
                else "physical_mse_objective_nm2"
            ): loss.detach(),
            "mse": metrics["tau_mse_normalized"],
            "mae": (prediction - target).abs().mean().detach(),
            **self._physical_mae_metrics(prediction, target),
            **metrics,
        }
        out["_absolute_error_nm"] = absolute_error_nm.detach()
        out["_target_nm"] = target_nm.detach()
        if self.use_wrench_objective:
            out["_wrench_pred"] = diagnostics["wrench_pred"].detach()
        return loss, out

    @torch.no_grad()
    def evaluate_loader(self, loader, epoch, description):
        loss, metrics = super().evaluate_loader(loader, epoch, description)
        for mse_key, rmse_key in (
            ("tau_mse_nm2", "tau_rmse_nm"),
            ("wrench_mse_scaled", "wrench_rmse_scaled"),
            ("wrench_force_mse_n2", "wrench_force_rmse_n"),
            ("wrench_moment_mse_nm2", "wrench_moment_rmse_nm"),
        ):
            if mse_key in metrics:
                metrics[rmse_key] = math.sqrt(max(metrics[mse_key], 0.0))
        for joint_index in range(1, 8):
            mse_key = f"tau_mse_nm2_j{joint_index}"
            if mse_key in metrics:
                metrics[f"tau_rmse_nm_j{joint_index}"] = math.sqrt(
                    max(metrics[mse_key], 0.0)
                )
        return loss, metrics

    def build_dataset(self):
        dataset = super().build_dataset()
        available_columns = set(dataset.stats_dataset.column_names)
        data_config = self.config.get("dataloader") or {}
        model_config = self.config.get("model") or {}
        lowdim_keys = data_config.get("lowdim_keys") or {}
        required_keys = list(
            dict.fromkeys(
                list(model_config.get("inputs") or [])
                + [str(model_config.get("target_key", "tau"))]
                + list(data_config.get("normalize_lowdim_keys") or [])
            )
        )
        missing = {
            key: lowdim_keys.get(key)
            for key in required_keys
            if lowdim_keys.get(key) not in available_columns
        }
        if missing:
            details = ", ".join(
                f"{key} -> {column}" for key, column in missing.items()
            )
            hint = ""
            if "delta_q" in missing:
                hint = (
                    " Current data has no commanded joint position from which to "
                    "construct delta_q. Remove delta_q from model.inputs, or "
                    "reconvert data containing a meaningful q_cmd."
                )
            raise ValueError(
                f"tau-free dataset is missing required columns: {details}."
                f"{hint}"
            )
        return dataset

    @staticmethod
    def _validate_contract(config):
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        inputs = list(model_config.get("inputs") or [])
        target_key = str(model_config.get("target_key", ""))
        architecture = str(model_config.get("architecture", "")).lower()
        horizon = int(data_config.get("horizon", 0))
        pad_history = bool(data_config.get("pad_history", False))
        contact_free = data_config.get("contact_free")
        loss_config = config.get("loss") or {}
        torque_loss_space = str(
            loss_config.get("torque_loss_space", "normalized")
        ).lower()
        loss_type = str(loss_config.get("type", "mse")).lower()
        configured_joint_weights = loss_config.get("joint_weights")
        default_joint_weight_mode = (
            "manual" if configured_joint_weights is not None else "equal"
        )
        joint_weight_mode = str(
            loss_config.get("joint_weight_mode", default_joint_weight_mode)
        ).lower()

        allowed_inputs = {"q", "dq", "ddq", "tau_id", "tau_other", "delta_q"}
        unknown_inputs = sorted(set(inputs) - allowed_inputs)
        if not inputs or inputs[0] != "q" or unknown_inputs:
            raise ValueError(
                "tau-free model.inputs must start with q and contain only "
                "q, dq, and delta_q; measured torque and ddq must not be inputs."
            )
        if len(set(inputs)) != len(inputs):
            raise ValueError("tau-free model.inputs must not contain duplicates.")
        if target_key != "tau":
            raise ValueError(
                "tau-free requires model.target_key=tau so the label is the "
                "measured motor torque from contact-free motion."
            )
        if architecture not in {"lstm", "gru", "tcn"}:
            raise ValueError(
                "tau-free model.architecture must be lstm, gru, or tcn."
            )
        if horizon != 50:
            raise ValueError("tau-free requires dataloader.horizon=50.")
        if pad_history:
            raise ValueError(
                "tau-free requires dataloader.pad_history=false so every "
                "supervised target has 50 real history frames."
            )
        if contact_free is not True:
            raise ValueError(
                "tau-free requires dataloader.contact_free=true; measured "
                "torque is a valid tau_free label only on contact-free data."
            )
        if torque_loss_space not in {"normalized", "physical_nm"}:
            raise ValueError(
                "tau-free loss.torque_loss_space must be 'normalized' or "
                f"'physical_nm', got {torque_loss_space!r}"
            )
        if torque_loss_space == "physical_nm" and loss_type != "mse":
            raise ValueError(
                "tau-free physical_nm optimization currently requires "
                "loss.type=mse"
            )
        if joint_weight_mode not in {"equal", "manual", "mean_abs", "max_abs"}:
            raise ValueError(
                "tau-free loss.joint_weight_mode must be equal, manual, "
                f"mean_abs, or max_abs; got {joint_weight_mode!r}"
            )
        if (
            joint_weight_mode in {"mean_abs", "max_abs"}
            and torque_loss_space != "physical_nm"
        ):
            raise ValueError(
                "tau-free automatic joint weighting requires "
                "loss.torque_loss_space=physical_nm"
            )


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    log.info("tau-free sequence train config: %s", config)
    result = run_tau_sequence_training(
        config,
        trainer_class=TauFreeSequenceTrainer,
    )
    if result.get("workflow") == "purged_kfold":
        log.info(
            "purged K-fold workflow finished: report=%s production_epochs=%d",
            result["report_path"],
            result["production"]["num_epochs"],
        )


if __name__ == "__main__":
    main()
