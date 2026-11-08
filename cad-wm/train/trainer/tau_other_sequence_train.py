import argparse
import copy
import json
import logging
import math
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

from data_process.tau_other_target_generation import (
    build_causal_tau_other_target,
    normalize_tau_other_target_generation,
    resolve_tau_other_target_generation,
    timestamps_to_seconds,
)
from model.tau_other_sequence import build_tau_other_sequence_model
from physics.nero_dynamics import PinocchioDynamics
from train.base_trainer import BaseTrainer
from train.nomalizer import Normalizer
from train.torque_sequence_peak_loss import TorqueSequencePeakLoss


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SampleIndexDataset(torch.utils.data.Dataset):
    """Return lightweight sample indices instead of materializing CPU windows."""

    def __init__(self, sample_indices):
        self.sample_indices = [int(index) for index in sample_indices]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        return {"sample_idx": self.sample_indices[index]}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a NEXT-style stateless-window LSTM, GRU, or TCN for tau_other."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/tau_other_sequence.yaml"),
        help="Path to the training config.",
    )
    return parser.parse_args()


class TauOtherTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        loss_config = config.get("loss") or {}
        joint_weights = loss_config.get("joint_weights")
        self.joint_weights = (
            torch.as_tensor(joint_weights, dtype=torch.float32)
            if joint_weights is not None
            else None
        )
        self.peak_loss = TorqueSequencePeakLoss(loss_config)
        self.tensor_cache = None
        self.valid_raw_indices_device = None
        self.episode_starts_device = None
        self.derived_target_config = normalize_tau_other_target_generation(config)
        self.tau_other_dynamics = None
        if self.derived_target_config["enabled"]:
            model_target = str((config.get("model") or {}).get("target_key", "tau_other"))
            derived_target = self.derived_target_config["target_key"]
            if model_target != derived_target:
                raise ValueError(
                    "model.target_key must match target_generation.target_key; "
                    f"got {model_target!r} and {derived_target!r}"
                )
            active_inputs = list((config.get("model") or {}).get("inputs") or [])
            tau_source = self.derived_target_config["source_keys"]["tau"]
            # if tau_source in active_inputs:
            #     raise ValueError(
            #         "Measured tau is a target-generation source and must not be a "
            #         "model input; doing so leaks the tau_other supervision target."
            #     )

    def build_dataset(self):
        # Keep the optional LeRobot dependency out of model-only imports and tests.
        from data_process.dataloader import PINNDataset

        return PINNDataset(self.config, compute_normalizer=False)

    def fit_dataset_normalizer(self, train_dataset):
        if isinstance(train_dataset, torch.utils.data.Subset):
            sample_indices = train_dataset.indices
        else:
            sample_indices = range(len(self.dataset))
        training_frames = self.dataset.covered_raw_indices(sample_indices)
        log.info(
            "fit normalizer from training windows only: raw_frames=%d",
            len(training_frames),
        )
        self._build_device_cache(training_frames)

    def setup(self):
        super().setup()
        self.loader = self._index_loader(self.loader.dataset, shuffle=True)
        if self.val_loader is not None:
            self.val_loader = self._index_loader(
                self.val_loader.dataset,
                shuffle=False,
            )

    def _index_loader(self, source_dataset, shuffle):
        if isinstance(source_dataset, torch.utils.data.Subset):
            sample_indices = source_dataset.indices
        else:
            sample_indices = range(len(self.dataset))

        return torch.utils.data.DataLoader(
            SampleIndexDataset(sample_indices),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _build_device_cache(self, training_frames):
        model_config = self.config.get("model") or {}
        active_inputs = list(model_config.get("inputs") or ["q", "dq", "delta_q"])
        target_key = str(model_config.get("target_key", "tau_other"))
        requested_keys = active_inputs + list(self.dataset.normalize_lowdim_keys)
        if self.derived_target_config["enabled"]:
            requested_keys = [key for key in requested_keys if key != target_key]
            requested_keys.extend(
                self.derived_target_config["source_keys"].values()
            )
        else:
            requested_keys.append(target_key)
        cache_keys = list(dict.fromkeys(requested_keys))
        missing_keys = [
            key for key in cache_keys if key not in self.dataset.lowdim_keys
        ]
        if missing_keys:
            raise KeyError(
                f"Missing dataloader.lowdim_keys for cached inputs: {missing_keys}"
            )

        cpu_cache = self._load_lowdim_columns(cache_keys)
        if self.derived_target_config["enabled"]:
            self._build_derived_target(cpu_cache, target_key)
            retained_keys = list(
                dict.fromkeys(
                    active_inputs
                    + [target_key]
                    + list(self.dataset.normalize_lowdim_keys)
                )
            )
            cpu_cache = {key: cpu_cache[key] for key in retained_keys}
        stats = self._normalizer_stats(cpu_cache, training_frames)
        if stats:
            self.dataset.set_normalizer(Normalizer(stats))

        cache_device = torch.device(self.device)
        self.tensor_cache = {}
        for key, value in cpu_cache.items():
            value = value.to(device=cache_device, dtype=torch.float32)
            if key in stats:
                value = self._normalize_cached_tensor(value, stats[key])
            self.tensor_cache[key] = value.contiguous()

        valid_raw_indices = self.dataset.valid_indices
        episode_starts = [
            self.dataset.raw_idx_to_episode_start[raw_idx]
            for raw_idx in valid_raw_indices
        ]
        self.valid_raw_indices_device = torch.as_tensor(
            valid_raw_indices,
            device=cache_device,
            dtype=torch.long,
        )
        self.episode_starts_device = torch.as_tensor(
            episode_starts,
            device=cache_device,
            dtype=torch.long,
        )

        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.tensor_cache.values()
        )
        log.info(
            "low-dimensional sequence cache ready: device=%s keys=%s memory=%.2f MiB",
            cache_device,
            list(self.tensor_cache),
            cache_bytes / (1024**2),
        )

    def _build_derived_target(self, cpu_cache, target_key):
        self.derived_target_config = resolve_tau_other_target_generation(
            self.config,
            self.dataset.filter_config,
        )
        timestamp_key = self.derived_target_config["timestamp_key"]
        timestamp_values = self._load_dataset_column(timestamp_key)
        timestamps_s = timestamps_to_seconds(
            timestamp_values,
            self.derived_target_config["timestamp_unit"],
        )
        source_keys = self.derived_target_config["source_keys"]
        if self.tau_other_dynamics is None:
            self.tau_other_dynamics = PinocchioDynamics(self.config)
        result = build_causal_tau_other_target(
            timestamps_s=timestamps_s,
            q=cpu_cache[source_keys["q"]],
            dq=cpu_cache[source_keys["dq"]],
            tau_measured=cpu_cache[source_keys["tau"]],
            episodes=self.dataset.dataset.meta.episodes,
            target_config=self.derived_target_config,
            dynamics=self.tau_other_dynamics,
        )
        cpu_cache[source_keys["dq"]] = result.dq
        cpu_cache[target_key] = result.tau_other
        ddq_abs_p99 = torch.quantile(result.ddq.abs().reshape(-1), 0.99).item()
        target_abs_p99 = torch.quantile(result.tau_other.abs().reshape(-1), 0.99).item()
        log.info(
            "derived causal tau_other target ready: frames=%d ddq_abs_p99=%.6f "
            "tau_other_abs_p99=%.6f formula=%s operations=%s",
            len(result.tau_other),
            ddq_abs_p99,
            target_abs_p99,
            self.derived_target_config["residual_formula"],
            self.derived_target_config["torque_filter_operations"],
        )

    def _load_lowdim_columns(self, cache_keys):
        dataset_keys = [self.dataset.lowdim_keys[key] for key in cache_keys]
        hf_dataset = self.dataset.stats_dataset

        try:
            formatted_dataset = hf_dataset.with_format(
                "torch",
                columns=dataset_keys,
                output_all_columns=False,
            )
            columns = formatted_dataset[:]
            cache = {
                key: self._column_to_tensor(columns[dataset_key])
                for key, dataset_key in zip(cache_keys, dataset_keys)
            }
        except (AttributeError, TypeError, ValueError) as exc:
            log.warning(
                "bulk tensor loading unavailable, falling back to one-pass rows: %s",
                exc,
            )
            buffers = {key: [] for key in cache_keys}
            for raw_idx in range(len(hf_dataset)):
                frame = hf_dataset[raw_idx]
                for key, dataset_key in zip(cache_keys, dataset_keys):
                    buffers[key].append(torch.as_tensor(frame[dataset_key]))
            cache = {key: torch.stack(values, dim=0) for key, values in buffers.items()}

        expected_length = len(hf_dataset)
        for key, value in cache.items():
            if value.shape[0] != expected_length:
                raise ValueError(
                    f"Cached field {key!r} has {value.shape[0]} frames, "
                    f"expected {expected_length}."
                )
        return cache

    def _load_dataset_column(self, dataset_key):
        hf_dataset = self.dataset.stats_dataset
        if dataset_key not in hf_dataset.column_names:
            raise KeyError(
                f"target_generation timestamp column is missing: {dataset_key!r}"
            )
        formatted_dataset = hf_dataset.with_format(
            "torch",
            columns=[dataset_key],
            output_all_columns=False,
        )
        columns = formatted_dataset[:]
        value = self._column_to_tensor(columns[dataset_key])
        if value.shape[0] != len(hf_dataset):
            raise ValueError(
                f"Target-generation column {dataset_key!r} has {value.shape[0]} "
                f"frames, expected {len(hf_dataset)}."
            )
        return value

    @staticmethod
    def _column_to_tensor(column):
        if torch.is_tensor(column):
            return column
        if isinstance(column, list) and column and torch.is_tensor(column[0]):
            return torch.stack(column, dim=0)
        return torch.as_tensor(column)

    def _normalizer_stats(self, cpu_cache, training_frames):
        if self.dataset.normalize_mode is None:
            return {}

        training_indices = torch.as_tensor(training_frames, dtype=torch.long)
        stats = {}
        for key in self.dataset.normalize_lowdim_keys:
            if key not in cpu_cache:
                continue
            values = (
                cpu_cache[key]
                .to(dtype=torch.float32)
                .index_select(
                    0,
                    training_indices,
                )
            )
            stats[key] = {
                "mean": values.mean(dim=0),
                "std": values.std(dim=0),
                "min": values.min(dim=0).values,
                "max": values.max(dim=0).values,
                "q01": torch.quantile(values, 0.01, dim=0),
                "q99": torch.quantile(values, 0.99, dim=0),
            }
        return stats

    def _normalize_cached_tensor(self, value, stats):
        eps = float(self.dataset.normalizer.eps)
        mode = self.dataset.normalize_mode
        stats = {
            name: statistic.to(device=value.device, dtype=value.dtype)
            for name, statistic in stats.items()
        }
        if mode == "gaussian":
            return (value - stats["mean"]) / (stats["std"] + eps)
        if mode == "limit":
            return 2 * (value - stats["min"]) / (stats["max"] - stats["min"] + eps) - 1
        if mode == "quantile":
            normalized = (
                2 * (value - stats["q01"]) / (stats["q99"] - stats["q01"] + eps) - 1
            )
            return torch.clamp(normalized, -1.0, 1.0)
        raise ValueError(f"unknown normalize mode: {mode}")

    def _cached_batch(self, sample_indices):
        if self.tensor_cache is None:
            raise RuntimeError("device tensor cache has not been initialized")

        sample_indices = sample_indices.to(
            device=self.valid_raw_indices_device.device,
            dtype=torch.long,
        )
        raw_indices = self.valid_raw_indices_device.index_select(0, sample_indices)
        episode_starts = self.episode_starts_device.index_select(0, sample_indices)
        offsets = torch.arange(
            -self.dataset.horizon + 1,
            1,
            device=raw_indices.device,
        )
        window_indices = raw_indices[:, None] + offsets[None, :]
        padding_mask = window_indices < episode_starts[:, None]
        window_indices = torch.maximum(window_indices, episode_starts[:, None])

        model = self.config.get("model") or {}
        active_inputs = list(model.get("inputs") or ["q", "dq", "delta_q"])
        target_key = str(model.get("target_key", "tau_other"))
        batch = {
            key: self.tensor_cache[key][window_indices].masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )
            for key in active_inputs
        }
        batch[target_key] = self.tensor_cache[target_key].index_select(0, raw_indices)
        return batch

    def build_model(self):
        return build_tau_other_sequence_model(self.config)

    def _denormalize_target(self, value):
        """Convert the configured torque target back to its physical unit."""
        dataset = getattr(self, "dataset", None)
        if dataset is None:
            return value

        target_key = str(
            (self.config.get("model") or {}).get("target_key", "tau_other")
        )
        normalize_mode = getattr(dataset, "normalize_mode", None)
        normalized_keys = getattr(dataset, "normalize_lowdim_keys", ())
        if normalize_mode is None or target_key not in normalized_keys:
            return value

        normalizer = getattr(dataset, "normalizer", None)
        if normalizer is None:
            raise RuntimeError(
                f"Target {target_key!r} is configured as normalized, but the "
                "dataset normalizer has not been fitted."
            )

        denormalize = getattr(
            normalizer,
            f"{normalize_mode}_denormalize",
            None,
        )
        if denormalize is None:
            raise ValueError(f"unknown normalize mode: {normalize_mode}")
        return denormalize(target_key, value)

    def _physical_mae_metrics(self, prediction, target):
        with torch.no_grad():
            prediction_nm = self._denormalize_target(prediction.detach())
            target_nm = self._denormalize_target(target.detach())
            absolute_error_nm = (prediction_nm - target_nm).abs()
            joint_mae_nm = absolute_error_nm.reshape(
                -1,
                absolute_error_nm.shape[-1],
            ).mean(dim=0)

        metrics = {"mae_nm": joint_mae_nm.mean()}
        metrics.update(
            {
                f"mae_nm_j{joint_index}": joint_mae
                for joint_index, joint_mae in enumerate(joint_mae_nm, start=1)
            }
        )
        return metrics

    def compute_loss(self, batch):
        if "sample_idx" in batch:
            batch = self._cached_batch(batch["sample_idx"])
        out = self.model(batch)
        prediction = out["tau_other_pred"]
        target = out.get("tau_other_target")
        if target is None:
            raise KeyError("Batch is missing the configured tau_other supervision target.")

        squared_error = F.mse_loss(prediction, target, reduction="none")
        prediction_nm = self._denormalize_target(prediction)
        target_nm = self._denormalize_target(target)
        absolute_error_nm = (prediction_nm - target_nm).abs()
        objective_prediction, objective_target = (
            (prediction, target)
            if self.peak_loss.loss_type == "mse"
            else (prediction_nm, target_nm)
        )
        loss = self.peak_loss(
            objective_prediction,
            objective_target,
            joint_weights=self.joint_weights,
        )
        out["loss_dict"] = {
            (
                "mse_objective"
                if self.peak_loss.loss_type == "mse"
                else "peak_objective_nm2"
            ): loss.detach(),
            "mse": squared_error.mean().detach(),
            "mae": F.l1_loss(prediction, target).detach(),
            **self._physical_mae_metrics(prediction, target),
        }
        out["_absolute_error_nm"] = absolute_error_nm.detach()
        out["_target_nm"] = target_nm.detach()
        return loss, out

    @staticmethod
    def _normalized_torque_metrics(absolute_error_nm, target_nm):
        absolute_error_nm = absolute_error_nm.reshape(
            -1, absolute_error_nm.shape[-1]
        )
        target_nm = target_nm.reshape(-1, target_nm.shape[-1])
        if absolute_error_nm.shape != target_nm.shape:
            raise ValueError(
                "absolute torque errors and targets must have matching shapes"
            )
        if len(target_nm) == 0:
            return {}

        joint_mae_nm = absolute_error_nm.mean(dim=0)
        joint_tau_std_nm = (
            target_nm.std(dim=0)
            if len(target_nm) > 1
            else torch.zeros_like(joint_mae_nm)
        )
        p05 = torch.quantile(target_nm, 0.05, dim=0)
        p95 = torch.quantile(target_nm, 0.95, dim=0)
        joint_tau_p90_range_nm = p95 - p05
        eps = target_nm.new_tensor(1.0e-8)
        joint_nmae_std = joint_mae_nm / (joint_tau_std_nm + eps)
        joint_nmae_p90 = joint_mae_nm / (joint_tau_p90_range_nm + eps)

        metrics = {
            "nmae_std": float(joint_nmae_std.mean().item()),
            "nmae_p90": float(joint_nmae_p90.mean().item()),
        }
        for joint_index in range(target_nm.shape[-1]):
            suffix = f"j{joint_index + 1}"
            metrics[f"mae_nm_{suffix}"] = float(joint_mae_nm[joint_index].item())
            metrics[f"joint_mae_nm_{suffix}"] = float(
                joint_mae_nm[joint_index].item()
            )
            metrics[f"joint_tau_std_nm_{suffix}"] = float(
                joint_tau_std_nm[joint_index].item()
            )
            metrics[f"joint_nmae_std_{suffix}"] = float(
                joint_nmae_std[joint_index].item()
            )
            metrics[f"joint_tau_p90_range_nm_{suffix}"] = float(
                joint_tau_p90_range_nm[joint_index].item()
            )
            metrics[f"joint_nmae_p90_{suffix}"] = float(
                joint_nmae_p90[joint_index].item()
            )
        return metrics

    @torch.no_grad()
    def evaluate_loader(self, loader, epoch, description):
        """Aggregate peak metrics over the full split, not per mini-batch."""
        training_model = self.model
        if self.ema is not None and self.ema_use_for_validation:
            self.model = self.ema.model
        self.model.eval()

        total_loss = 0.0
        num_samples = 0
        metric_sums = {}
        metric_counts = {}
        absolute_errors_nm = []
        targets_nm = []
        wrench_predictions = []
        pbar = tqdm(
            loader,
            desc=f"{description} epoch {epoch}",
            unit="batch",
            leave=False,
        )

        try:
            for batch in pbar:
                batch = self.batch_to_device(batch)
                loss, out = self.compute_loss(batch)
                batch_size = self._batch_size(batch)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                self._accumulate_scalar_metrics(
                    metric_sums,
                    metric_counts,
                    out.get("loss_dict") or {},
                    batch_size,
                )
                absolute_errors_nm.append(out["_absolute_error_nm"].cpu())
                targets_nm.append(out["_target_nm"].cpu())
                if "_wrench_pred" in out:
                    wrench_predictions.append(out["_wrench_pred"].cpu())
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        finally:
            self.model = training_model

        metrics = self._average_scalar_metrics(metric_sums, metric_counts)
        if absolute_errors_nm:
            global_absolute_error_nm = torch.cat(absolute_errors_nm, dim=0)
            global_peak_metrics = self.peak_loss.metrics_from_absolute_error(
                global_absolute_error_nm
            )
            metrics.update(
                {
                    key: float(value.item())
                    for key, value in global_peak_metrics.items()
                }
            )
            metrics.update(
                self._normalized_torque_metrics(
                    global_absolute_error_nm,
                    torch.cat(targets_nm, dim=0),
                )
            )
        if wrench_predictions:
            wrench = torch.cat(wrench_predictions, dim=0)
            force_norm = torch.linalg.vector_norm(wrench[:, :3], dim=-1)
            moment_norm = torch.linalg.vector_norm(wrench[:, 3:], dim=-1)
            metrics.update(
                {
                    "wrench_force_norm_mean_n": float(force_norm.mean().item()),
                    "wrench_force_norm_p95_n": float(
                        torch.quantile(force_norm, 0.95).item()
                    ),
                    "wrench_force_norm_max_n": float(force_norm.max().item()),
                    "wrench_moment_norm_mean_nm": float(moment_norm.mean().item()),
                    "wrench_moment_norm_p95_nm": float(
                        torch.quantile(moment_norm, 0.95).item()
                    ),
                    "wrench_moment_norm_max_nm": float(moment_norm.max().item()),
                }
            )
        loss = total_loss / max(num_samples, 1)
        return loss, metrics

    def validate_one_epoch(self, epoch):
        val_loss = super().validate_one_epoch(epoch)
        if val_loss is not None:
            metrics = self.last_val_epoch_metrics
            log.info(
                "validation peak torque: cvar_rmse=%.6f Nm p95=%.6f Nm "
                "p99=%.6f Nm max=%.6f Nm",
                metrics["peak_cvar_rmse_nm"],
                metrics["peak_p95_nm"],
                metrics["peak_p99_nm"],
                metrics["peak_max_nm"],
            )
            metric_groups = (
                ("joint_mae_nm", "joint_mae_nm_j"),
                ("joint_tau_std_nm", "joint_tau_std_nm_j"),
                ("joint_nmae_std", "joint_nmae_std_j"),
                ("joint_tau_p90_range_nm", "joint_tau_p90_range_nm_j"),
                ("joint_nmae_p90", "joint_nmae_p90_j"),
            )
            normalized_metric_text = " ".join(
                f"{label}=["
                + ",".join(
                    f"{metrics[f'{prefix}{joint_index}']:.6f}"
                    for joint_index in range(1, 8)
                )
                + "]"
                for label, prefix in metric_groups
            )
            log.info("validation normalized torque: %s", normalized_metric_text)
        return val_loss


def _json_compatible(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _configure_stage_wandb(config, stage_name, group_name):
    wandb_config = config["train"].get("wandb") or {}
    if not wandb_config:
        return
    wandb_config = dict(wandb_config)
    base_name = wandb_config.get("name") or "tau-sequence"
    wandb_config["name"] = f"{base_name}-{stage_name}"
    wandb_config.setdefault("group", group_name)
    config["train"]["wandb"] = wandb_config


def _aggregate_fold_metrics(fold_results):
    metric_names = sorted(
        {
            key
            for fold in fold_results
            for key, value in fold["best_metrics"].items()
            if key.startswith("val_") and isinstance(value, (int, float))
        }
    )
    aggregate = {}
    for metric_name in metric_names:
        values = [
            float(fold["best_metrics"][metric_name])
            for fold in fold_results
            if metric_name in fold["best_metrics"]
        ]
        if len(values) != len(fold_results):
            continue
        aggregate[metric_name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "worst": max(values),
            "values": values,
        }
    return aggregate


def _write_cross_validation_report(output_dir, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "purged_kfold_summary.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(
            _json_compatible(report),
            report_file,
            indent=2,
            sort_keys=True,
        )
        report_file.write("\n")
    return report_path


def run_purged_kfold_workflow(config, trainer_class=TauOtherTrainer):
    """Run all purged folds, aggregate them, then fit one full-data model."""
    train_config = config.get("train") or {}
    fold_config = train_config.get("purged_kfold") or {}
    num_folds = int(fold_config.get("num_folds", 3))
    if num_folds < 2:
        raise ValueError("train.purged_kfold.num_folds must be at least 2")
    output_dir = Path(
        train_config.get("output_dir", "outputs/tau_other_sequence")
    )
    group_name = str(
        (train_config.get("wandb") or {}).get("group")
        or (train_config.get("wandb") or {}).get("name")
        or output_dir.name
    )

    fold_results = []
    for fold_index in range(num_folds):
        current_config = copy.deepcopy(config)
        current_train = current_config["train"]
        current_fold = dict(current_train.get("purged_kfold") or {})
        current_fold["fold_index"] = fold_index
        current_fold["workflow_stage"] = "cross_validation"
        current_train["purged_kfold"] = current_fold
        current_train["output_dir"] = str(
            output_dir
            / "cross_validation"
            / f"fold_{fold_index + 1:02d}_of_{num_folds:02d}"
        )
        _configure_stage_wandb(
            current_config,
            f"cv-fold-{fold_index + 1}-of-{num_folds}",
            group_name,
        )

        log.info("starting purged K-fold %d/%d", fold_index + 1, num_folds)
        trainer = trainer_class(current_config)
        summary = trainer.train()
        if not summary["best_checkpoints"]:
            raise RuntimeError(
                f"fold {fold_index + 1} produced no checkpoint for "
                f"monitor_key={summary['monitor_key']!r}"
            )
        best_checkpoint = summary["best_checkpoints"][0]
        fold_results.append(
            {
                "fold_index": fold_index,
                "best_epoch": int(best_checkpoint["epoch"]),
                "best_epoch_count": int(best_checkpoint["epoch"]) + 1,
                "best_monitor_score": float(best_checkpoint["score"]),
                "best_checkpoint": str(best_checkpoint["path"]),
                "best_metrics": dict(best_checkpoint.get("metrics") or {}),
                "split": summary.get("split"),
            }
        )
        del trainer

    aggregate = _aggregate_fold_metrics(fold_results)
    best_epoch_counts = [fold["best_epoch_count"] for fold in fold_results]
    configured_production_epochs = fold_config.get("production_epochs")
    if (
        configured_production_epochs is not None
        and int(configured_production_epochs) < 1
    ):
        raise ValueError(
            "train.purged_kfold.production_epochs must be positive or null"
        )
    production_epochs = (
        int(configured_production_epochs)
        if configured_production_epochs is not None
        else max(1, math.ceil(statistics.median(best_epoch_counts)))
    )

    report = {
        "workflow": "purged_kfold",
        "num_folds": num_folds,
        "monitor_key": train_config.get("monitor_key", "val_loss"),
        "folds": fold_results,
        "aggregate": aggregate,
        "production": {
            "enabled": bool(fold_config.get("production_retrain", True)),
            "epoch_selection": (
                "configured"
                if configured_production_epochs is not None
                else "median_fold_best_epoch_count"
            ),
            "num_epochs": production_epochs,
            "output_dir": str(output_dir),
        },
    }
    report_path = _write_cross_validation_report(output_dir, report)
    log.info("purged K-fold report written before production: %s", report_path)

    if report["production"]["enabled"]:
        production_config = copy.deepcopy(config)
        production_train = production_config["train"]
        production_train["split_mode"] = "all"
        production_train["val_ratio"] = 0.0
        production_train["num_epochs"] = production_epochs
        production_train["output_dir"] = str(output_dir)
        production_train["early_stopping"] = {
            **(production_train.get("early_stopping") or {}),
            "enabled": False,
        }
        train_eval_enabled = bool(
            (production_train.get("train_eval") or {}).get("enabled", False)
        )
        production_train["monitor_key"] = (
            "train_eval_loss" if train_eval_enabled else "avg_loss"
        )
        production_fold_config = dict(
            production_train.get("purged_kfold") or {}
        )
        production_fold_config["workflow_stage"] = "production_full_data"
        production_fold_config["selected_num_epochs"] = production_epochs
        production_train["purged_kfold"] = production_fold_config
        _configure_stage_wandb(
            production_config,
            "production-full-data",
            group_name,
        )

        log.info(
            "starting full-data production retraining for %d epochs",
            production_epochs,
        )
        production_trainer = trainer_class(production_config)
        production_summary = production_trainer.train()
        report["production"]["summary"] = production_summary
        report_path = _write_cross_validation_report(output_dir, report)

    report["report_path"] = str(report_path)
    return report


def run_tau_sequence_training(config, trainer_class=TauOtherTrainer):
    train_config = config.get("train") or {}
    fold_config = train_config.get("purged_kfold") or {}
    if (
        train_config.get("split_mode") == "purged_kfold"
        and bool(fold_config.get("run_all_folds", True))
    ):
        return run_purged_kfold_workflow(config, trainer_class=trainer_class)

    trainer = trainer_class(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))
    return summary

def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    log.info("tau_other sequence train config: %s", config)
    result = run_tau_sequence_training(config)
    if result.get("workflow") == "purged_kfold":
        log.info(
            "purged K-fold workflow finished: report=%s production_epochs=%d",
            result["report_path"],
            result["production"]["num_epochs"],
        )


if __name__ == "__main__":
    main()
