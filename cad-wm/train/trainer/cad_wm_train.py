"""CAD-WM training: conditional flow matching, NEXT auxiliary loss, phase sampling."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

from data_process.libero_wrench_dataset import LiberoWrenchDataset
from model.cad_wm import CADWorldModel
from train.base_trainer import BaseTrainer


log = logging.getLogger(__name__)
CHECKPOINT_MSE_KEY = "full_dataset_wrench_mse"


class CADWMTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        loss = config.get("loss") or {}
        self.contact_weight = float(loss.get("contact_weight", 0.1))
        self.free_dynamics_weight = float(loss.get("free_dynamics_weight", 0.1))
        self.class_weights = tuple(float(v) for v in config["contact_gate"].get("class_weights", [1, 1, 1]))
        checkpoint_eval = self.train_config.get("checkpoint_eval") or {}
        self.checkpoint_eval_enabled = bool(checkpoint_eval.get("enabled", False))
        self.checkpoint_eval_batch_size = int(checkpoint_eval.get("batch_size", self.batch_size))
        self.checkpoint_eval_seed = int(checkpoint_eval.get("seed", self.seed))
        if self.checkpoint_eval_batch_size < 1:
            raise ValueError("train.checkpoint_eval.batch_size must be positive")
        if self.checkpoint_eval_enabled:
            self.checkpoint_monitor_key = CHECKPOINT_MSE_KEY
            if self.val_every <= 0:
                raise ValueError("train.val_every must be positive for full-dataset evaluation")
        self._checkpoint_eval_cache = None
        if (any(not math.isfinite(v) or v < 0 for v in (self.contact_weight, self.free_dynamics_weight))
                or len(self.class_weights) != 3
                or any(not math.isfinite(v) or v <= 0 for v in self.class_weights)):
            raise ValueError("invalid contact/free-dynamics loss weights")
        if self.device_batch_keys is not None:
            self.device_batch_keys.update({"q", "dq", "tau", "wrench_future", "contact_future",
                                           "free_dynamics_mask", "importance_weight", "sample_idx",
                                           "wrench_future_raw"})

    def build_dataset(self):
        return LiberoWrenchDataset(self.config)

    def build_model(self):
        return CADWorldModel(self.config)

    @staticmethod
    def _sample_indices(dataset):
        return list(dataset.indices) if isinstance(dataset, torch.utils.data.Subset) else list(range(len(dataset)))

    def fit_dataset_normalizer(self, train_dataset):
        self.dataset.fit_normalizer(self._sample_indices(train_dataset))

    def build_train_sampler(self, train_dataset):
        sampling = self.train_config.get("contact_sampling") or {}
        self.dataset.importance_weight_by_sample_index = {}
        if not sampling.get("enabled", False):
            return None
        indices = self._sample_indices(train_dataset)
        phase_weights = torch.as_tensor(sampling.get("phase_weights", [1.0, 5.0, 5.0]), dtype=torch.double)
        if phase_weights.shape != (3,) or not torch.isfinite(phase_weights).all() or (phase_weights <= 0).any():
            raise ValueError("contact_sampling.phase_weights must contain three finite positive weights")
        if sampling.get("future_phase_reduction", "max") != "max":
            raise ValueError("contact sampling uses max over the existing future phase labels")
        phases = torch.tensor([
            int(self.dataset.contact[anchor + 1:anchor + self.dataset.future_horizon + 1].max())
            for anchor in (self.dataset.valid_indices[index] for index in indices)
        ], dtype=torch.long)
        weights = phase_weights[phases]
        if sampling.get("importance_correction", False):
            corrections = weights.sum() / (len(weights) * weights)
            self.dataset.importance_weight_by_sample_index = {
                self.dataset.valid_indices[index]: float(value)
                for index, value in zip(indices, corrections)
            }
        probabilities = weights / weights.sum()
        counts = torch.bincount(phases, minlength=3)
        expected = [float(probabilities[phases == phase].sum()) for phase in range(3)]
        self.checkpoint_metadata = {"contact_sampling": {
            "source_counts": counts.tolist(), "expected_phase_ratio": expected,
            "phase_weights": phase_weights.tolist(),
            "importance_correction": bool(sampling.get("importance_correction", False)),
        }}
        log.info("CAD-WM phase sampler: free/precontact/contact counts=%s expected_ratio=%s",
                 counts.tolist(), [round(value, 4) for value in expected])
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)

    @staticmethod
    def free_dynamics_loss(out, batch):
        prediction = out["free_tau_pred"].float()
        target = batch["tau"][:, -1].detach().float()
        mask = batch["free_dynamics_mask"].bool() & torch.isfinite(target).all(-1)
        if prediction.shape != target.shape or mask.shape != prediction.shape[:1]:
            raise ValueError("free tau target and mask must align with current torque")
        weights = batch.get("importance_weight", prediction.new_ones(len(prediction))).float()
        selected_weights = weights[mask]
        # Index first, so no non-free target can contaminate the zero-free case.
        mse = (prediction[mask] - target[mask]).square().mean(-1)
        loss = (mse * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)
        return loss, mask.sum()

    def _validation_source(self, batch, *, seed=None):
        # Fixed per-anchor noise keeps evaluation stable across batch layouts.
        if seed is None:
            seed = int(self.train_config.get("validation_seed", self.seed))
        anchors = batch["sample_idx"].detach().cpu().tolist()
        generator = torch.Generator(device="cpu")
        noise = []
        for anchor in anchors:
            generator.manual_seed(seed + int(anchor))
            noise.append(torch.randn(self.model.future_horizon, 6, generator=generator))
        return torch.stack(noise).to(batch["q"])

    def evaluate_checkpoint(self):
        if not self.checkpoint_eval_enabled:
            return {}
        if self._checkpoint_eval_cache is not None:
            step, metrics = self._checkpoint_eval_cache
            if step == self.global_step:
                return dict(metrics)
        metrics = self._evaluate_full_dataset()
        self._checkpoint_eval_cache = (self.global_step, dict(metrics))
        self.inference_history.append({"epoch": self.current_epoch,
                                       "global_step": self.global_step, **metrics})
        self.save_loss_plot()
        return metrics

    def evaluate_epoch_metrics(self, epoch):
        if self.checkpoint_eval_enabled and (epoch + 1) % self.val_every == 0:
            return self.evaluate_checkpoint()
        return {}

    @torch.inference_mode()
    def _evaluate_full_dataset(self):
        # Iterate the original dataset exactly once, bypassing train subsets
        # and the replacement sampler. Keep loader/source RNG independent.
        kwargs = self._dataloader_kwargs(
            shuffle=False, num_workers=self.val_num_workers,
            prefetch_factor=self.val_prefetch_factor, pin_memory=self.val_pin_memory,
            persistent_workers=False,
        )
        kwargs.update(batch_size=self.checkpoint_eval_batch_size,
                      generator=torch.Generator().manual_seed(self.checkpoint_eval_seed))
        if self.dataset_on_device and hasattr(self.dataset, "batch_collate"):
            kwargs["collate_fn"] = self.dataset.batch_collate
        loader = torch.utils.data.DataLoader(self.dataset, **kwargs)
        # Evaluate exactly the weights stored under `model` for DP inference.
        model = self.ema.model if self.ema is not None else self.model
        was_training = model.training
        model.eval()
        force_squared_error, moment_squared_error = 0.0, 0.0
        elements, samples = 0, 0
        try:
            for batch in tqdm(loader, desc=f"full-dataset inference step {self.global_step}",
                              unit="batch", leave=False):
                batch = self.batch_to_device(batch)
                conditions = {key: batch[key] for key in ("q", "dq", "tau")}
                source = self._validation_source(batch, seed=self.checkpoint_eval_seed)
                with self.autocast_context():
                    prediction = model.predict(conditions, source_noise=source)["wrench_pred"].float()
                if self.dataset.normalizer is not None:
                    prediction = self.dataset.normalizer.gaussian_denormalize("wrench", prediction)
                error = prediction - batch["wrench_future_raw"].to(prediction)
                # Accumulate in double precision where supported so batch size
                # and a short final batch do not change the ranking metric.
                if error.device.type in {"cpu", "cuda"}:
                    error = error.double()
                force_squared_error += error[..., :3].square().sum().item()
                moment_squared_error += error[..., 3:].square().sum().item()
                elements += error.numel()
                samples += len(prediction)
        finally:
            model.train(was_training)
        if samples != len(self.dataset) or elements == 0:
            raise RuntimeError("checkpoint inference must cover every dataset window exactly once")
        mse = (force_squared_error + moment_squared_error) / elements
        if not math.isfinite(mse):
            raise RuntimeError("checkpoint full-dataset wrench MSE is not finite")
        metrics = {CHECKPOINT_MSE_KEY: mse, "checkpoint_eval_samples": samples,
                   "checkpoint_eval_elements": elements,
                   "full_dataset_force_mse": force_squared_error / (elements // 2),
                   "full_dataset_moment_mse": moment_squared_error / (elements // 2)}
        log.info("CAD-WM checkpoint step=%d: full-dataset physical wrench MSE=%.8g (%d windows)",
                 self.global_step, mse, samples)
        self.log_wandb({f"inference/{key}": value for key, value in metrics.items()},
                       step=self.global_step)
        return metrics

    def save_topk_checkpoint(self, epoch, metrics, checkpoint_label=None):
        if self.checkpoint_eval_enabled:
            return self.save_scheduled_epoch_checkpoint(epoch + 1, metrics)
        return super().save_topk_checkpoint(epoch, metrics, checkpoint_label)

    def compute_loss(self, batch):
        source = None if self.model.training else self._validation_source(batch)
        out = self.model(batch, source_noise=source, flow_time=None if self.model.training else 0.5)
        weights = batch.get("importance_weight", out["flow_velocity_pred"].new_ones(len(batch["q"]))).float()
        flow_ps = (out["flow_velocity_pred"].float() - out["flow_velocity_target"].float()).square().mean((1, 2))
        flow_loss = (flow_ps * weights).mean()
        logits = out["contact_logits"].float()
        ce = F.cross_entropy(logits.transpose(1, 2), batch["contact_future"].long(),
                             weight=logits.new_tensor(self.class_weights), reduction="none").mean(1)
        contact_loss = (ce * weights).mean()
        free_loss, free_count = self.free_dynamics_loss(out, batch)
        loss = flow_loss + self.contact_weight * contact_loss + self.free_dynamics_weight * free_loss
        metrics = {
            "total_loss": loss.detach(), "flow_matching_loss": flow_loss.detach(),
            "contact_ce": contact_loss.detach(), "free_dynamics_loss": free_loss.detach(),
            "free_dynamics_count": free_count.detach(),
            "contact_accuracy": (logits.argmax(-1) == batch["contact_future"]).float().mean().detach(),
        }
        if not self.model.training:
            # Score genuine noise-to-wrench generation. CFM's one-step training
            # endpoint uses the target and is never reported as inference error.
            generated = self.model.predict(batch, source_noise=source)["wrench_pred"].float()
            error = generated - batch["wrench_future"].float()
            metrics["wrench_mse"] = error.square().mean().detach()
            if self.dataset.normalizer is not None:
                normalizer = self.dataset.normalizer
                error = error * (normalizer.stats["wrench"]["std"].to(error) + normalizer.eps)
            metrics.update(force_mae_n=error[..., :3].abs().mean().detach(),
                           moment_mae_nm=error[..., 3:].abs().mean().detach())
        out["loss_dict"] = metrics
        return loss, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path(__file__).resolve().parents[2]
                        / "config/bench_train_config/libero_drawer_wrench_h5.yaml")
    parser.add_argument("--device", help="cpu, cuda:0, or npu:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-val", action="store_true", help="Train on all demos without validation")
    parser.add_argument("--smoke-test", action="store_true", help="One real-H5 CPU update, optional validation, and checkpoint")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train = config.setdefault("train", {})
    if args.smoke_test:
        if args.resume:
            parser.error("--smoke-test cannot be combined with --resume")
        torch.set_num_threads(1)
        train.update(device="cpu", num_workers=0, val_num_workers=0, batch_size=2,
                     max_optimizer_steps=1, checkpoint_every_steps=1, output_dir="outputs/cad_wm_smoke",
                     amp={"enabled": False}, wandb={"enabled": False}, scheduler={"name": "none"})
    if args.device:
        train["device"] = args.device
    if args.output_dir:
        train["output_dir"] = str(args.output_dir)
    if args.max_optimizer_steps is not None:
        train["max_optimizer_steps"] = args.max_optimizer_steps
    if args.resume:
        train["resume_from"] = str(args.resume)
    if args.no_val:
        train["validation_enabled"] = False
    trainer = CADWMTrainer(config)
    if args.smoke_test:
        trainer.setup()
        trainer.checkpoint_every_steps = 0
        trainer.loader = [next(iter(trainer.loader))]
        if trainer.val_loader is not None:
            trainer.val_loader = [next(iter(trainer.val_loader))]
        train_loss = trainer.train_one_epoch(0)
        val_loss = trainer.validate_one_epoch(0)
        losses = [train_loss] + ([] if val_loss is None else [val_loss])
        if trainer.global_step != 1 or not all(math.isfinite(v) for v in losses):
            raise RuntimeError("CAD-WM real-H5 smoke test failed")
        trainer.save_step_checkpoint(0, {"avg_loss": train_loss, "val_loss": val_loss})
        val_status = f"val_loss={val_loss:.6f}" if val_loss is not None else "validation=disabled"
        print(f"CAD-WM real-H5 smoke passed: train_loss={train_loss:.6f}, {val_status}")
        print(f"Checkpoint: {trainer.ckpt_dir / 'step_00000001.pt'}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
