"""Measure real cached-target RenderNet optimizer steps on one CUDA device.

Example (run eager and compiled measurements in separate processes):
    python -m diffusion_policy.scripts.benchmark_render_net_training \
        --checkpoint path/to/teacher.ckpt --output-dir /tmp/render_bench_eager \
        --batch-size 32 --warmup 5 --steps 20 --no-compile

The benchmark decodes distinct real dataset samples and computes teacher
context/attention targets once. Timings exclude decoding, target preparation,
host-to-device transfers and plotting, and include the actual training loss,
backward pass, gradient clipping, GradScaler and AdamW update. Each timed step
uses fresh diffusion noise/timesteps with the same resident observation batch.
"""

from __future__ import annotations

if __name__ == "__main__":
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import copy
import gc
import json
import logging
import pathlib
import statistics
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import default_collate

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.render_net_attention import teacher_attention_targets
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


log = logging.getLogger(__name__)
DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config/train_distill_render_net.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Trained FDP teacher checkpoint.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="New directory for report/config/logs.")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-path", type=pathlib.Path)
    parser.add_argument("--video-backend", help="Override video decoding backend; excluded from step timings.")
    parser.add_argument("--decode-device", default="cpu",
                        help="Dataset decoder device (default: cpu); decoding is outside timed steps.")
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true")
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=None)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.warmup, args.steps) < 1:
        parser.error("--batch-size, --warmup and --steps must be positive")
    if torch.device(args.device).type != "cuda":
        parser.error("This benchmark requires a CUDA device")
    return args


def _close_dataset(dataset):
    """Release decoder references before measuring model memory/throughput."""
    if dataset is None:
        return
    decoder = getattr(dataset, "_video_decoder", None)
    if decoder is not None:
        close = getattr(decoder, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.exception("Could not close optional CUDA video decoder")
        decoders = getattr(decoder, "_decoders", None)
        if hasattr(decoders, "clear"):
            decoders.clear()
        dataset._video_decoder = None
    close = getattr(getattr(dataset, "lerobot_dataset", None), "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            log.exception("Could not close dataset reader")


def _write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "report.json")


@torch.no_grad()
def _prepare_targets(workspace, batch):
    """Use the same context precision and teacher rollout as the training cache."""
    chunk_size = int(workspace.cfg.teacher_cache.batch_size)
    token_dtype = getattr(torch, str(workspace.cfg.teacher_cache.get("dtype", "float16")))
    tokens, attention_chunks = [], {}
    count = batch["action"].shape[0]
    for start in range(0, count, chunk_size):
        obs = {key: value[start:start + chunk_size] for key, value in batch["obs"].items()}
        with workspace._autocast():
            if workspace.attention_enabled:
                target, attention = teacher_attention_targets(
                    workspace.teacher_policy, obs,
                    percentiles=workspace.attention_config["percentiles"],
                    seed=int(workspace.attention_config.get("teacher_seed", 42)),
                    progress_callback=workspace._attention_progress,
                    return_patch_attention=True,
                )
            else:
                target = workspace.teacher_policy._encode_context(obs, apply_curriculum_mask=False)
                attention = {}
        tokens.append(target.to(device="cpu", dtype=token_dtype))
        for key, value in attention.items():
            attention_chunks.setdefault(key, []).append(value.float().cpu())
        log.info("Teacher targets prepared: %d/%d samples", min(start + chunk_size, count), count)
    return {"tokens": torch.cat(tokens),
            "patch_attention": {key: torch.cat(values) for key, values in attention_chunks.items()}}


def _optimizer_step(workspace, batch):
    workspace.optimizer.zero_grad(set_to_none=True)
    loss = workspace._batch_loss(batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite benchmark training loss")
    workspace.grad_scaler.scale(loss).backward()
    workspace.grad_scaler.unscale_(workspace.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        workspace.model.parameters(), float(workspace.cfg.training.max_grad_norm), error_if_nonfinite=True,
    )
    workspace.grad_scaler.step(workspace.optimizer)
    workspace.grad_scaler.update()
    workspace.global_step += 1
    return loss, grad_norm


def run(args):
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "benchmark.log")], force=True)
    report = {"status": "preparing", "batch_size": args.batch_size, "warmup_steps": args.warmup,
              "measured_steps": args.steps, "teacher_checkpoint": str(args.checkpoint.expanduser().resolve()),
              "device": args.device, "timing_scope": "GPU-resident cached batch: loss + backward + clip + AdamW",
              "excluded_from_step_time": ["data decode", "teacher context/attention cache preparation",
                                          "host-to-device transfer", "plotting", "checkpoint writes"]}
    _write_report(output, report)
    dataset = None
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; an actual GPU measurement is required")
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        cfg = OmegaConf.load(args.config.expanduser().resolve())
        if "hydra" in cfg:
            del cfg["hydra"]
        with open_dict(cfg):
            cfg.teacher_checkpoint = str(checkpoint)
            cfg.training.device = args.device
            cfg.training.resume_from = None
            cfg.training.distributed.enabled = False
            cfg.teacher_cache.enabled = False  # Prepare only this benchmark batch below.
            cfg.dataset.preload_images = False
            cfg.dataset.image_randomer_config = None
            cfg.dataset.load_result_add = "ram"
            if args.compile is not None:
                cfg.training.compile.enabled = args.compile
            if args.video_backend:
                cfg.dataset.video_backend = args.video_backend
            if args.decode_device:
                cfg.dataset.video_decode_device = args.decode_device
            cfg.logging.progress_bar = False
            cfg.dataloader.batch_size = args.batch_size
        OmegaConf.resolve(cfg)
        workspace = TrainDistillRenderNetWorkspace(cfg, output_dir=str(output))
        if workspace.objective != "denoiser_mse" or workspace.model.rendering_mode != "gaussian_glow":
            raise ValueError("Benchmark config must use denoiser_mse with gaussian_glow rendering")
        if not workspace.attention_enabled:
            raise ValueError("Benchmark requires student region prediction and teacher attention supervision")
        torch.cuda.synchronize(workspace.device)
        report["model_load_seconds"] = time.perf_counter() - started
        report["teacher_state_key"] = workspace.teacher_state_key
        report["compile_enabled"] = bool(cfg.training.compile.enabled)
        report["objective"] = workspace.objective
        report["rendering_mode"] = workspace.model.rendering_mode
        report["mask_supervision"] = True
        report["teacher_num_inference_steps_for_attention"] = workspace.teacher_policy.num_inference_steps
        report["teacher_trainable_parameters"] = sum(p.numel() for p in workspace.teacher_policy.parameters()
                                                       if p.requires_grad)
        report["renderer_trainable_parameters"] = sum(p.numel() for p in workspace.model.parameters()
                                                        if p.requires_grad)
        properties = torch.cuda.get_device_properties(workspace.device)
        report["hardware"] = {"gpu": properties.name, "total_vram_mib": properties.total_memory / 2**20,
                              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                              "cudnn_version": torch.backends.cudnn.version(),
                              "torch_num_threads": torch.get_num_threads(),
                              "torch_num_interop_threads": torch.get_num_interop_threads()}
        dataset_cfg = OmegaConf.create(OmegaConf.to_container(copy.deepcopy(workspace.teacher_cfg.task.dataset),
                                                            resolve=True))
        with open_dict(dataset_cfg):
            dataset_cfg.update(OmegaConf.to_container(cfg.dataset, resolve=True))
            if args.dataset_path:
                dataset_cfg.dataset_path = str(args.dataset_path.expanduser().resolve())
        report["dataset_config"] = OmegaConf.to_container(dataset_cfg, resolve=True)
        setup_started = time.perf_counter()
        dataset = hydra.utils.instantiate(dataset_cfg)
        report["dataset_init_seconds"] = time.perf_counter() - setup_started
        if len(dataset) < args.batch_size:
            raise ValueError(f"Dataset has {len(dataset)} samples; need {args.batch_size} distinct samples")
        indices = np.linspace(0, len(dataset) - 1, args.batch_size, dtype=np.int64)
        if len(np.unique(indices)) != args.batch_size:
            raise RuntimeError("Benchmark sample selection unexpectedly contains duplicates")
        report["dataset_train_samples"] = len(dataset)
        report["sample_indices"] = indices.tolist()
        if hasattr(dataset, "_anchor_index_from_sampler_row"):
            anchors = [int(dataset._anchor_index_from_sampler_row(dataset.indices[index])) for index in indices]
            report["sample_anchor_indices"] = anchors
            if hasattr(dataset, "episode_index"):
                report["sample_episode_indices"] = [int(dataset.episode_index[anchor]) for anchor in anchors]
        decode_started = time.perf_counter()
        examples = []
        for index in indices:
            examples.append(dataset[int(index)])
            log.info("Decoded real sample %d/%d (dataset index %d)", len(examples), args.batch_size, index)
        cpu_batch = default_collate(examples)
        del examples
        torch.cuda.synchronize(workspace.device)
        report["input_decode_seconds"] = time.perf_counter() - decode_started
        _close_dataset(dataset)
        dataset = None
        gc.collect()
        torch.cuda.empty_cache()
        transfer_started = time.perf_counter()
        batch = dict_apply(cpu_batch, lambda value: value.to(workspace.device, non_blocking=False))
        del cpu_batch
        torch.cuda.synchronize(workspace.device)
        report["input_transfer_seconds"] = time.perf_counter() - transfer_started
        report["input_shapes"] = {key: list(value.shape) for key, value in batch["obs"].items()}
        report["action_shape"] = list(batch["action"].shape)
        with torch.no_grad():
            _, normalized_wrench, contact, future_contact = workspace.teacher_policy.prepare_context_inputs(batch["obs"])
            report["normalized_wrench_shape"] = list(normalized_wrench.shape)
            report["normalized_wrench_rms"] = float(normalized_wrench.float().square().mean().sqrt())
            report["normalized_wrench_nonzero_fraction"] = float((normalized_wrench != 0).float().mean())
            report["current_contact_fraction"] = float(contact.float().mean()) if contact is not None else None
            report["future_contact_fraction"] = float(future_contact.float().mean()) if future_contact is not None else None
            del normalized_wrench
        prep_started = time.perf_counter()
        cpu_targets = _prepare_targets(workspace, batch)
        torch.cuda.synchronize(workspace.device)
        report["teacher_target_prep_seconds"] = time.perf_counter() - prep_started
        transfer_started = time.perf_counter()
        batch["teacher_targets"] = dict_apply(cpu_targets,
                                              lambda value: value.to(workspace.device, non_blocking=False))
        del cpu_targets
        torch.cuda.synchronize(workspace.device)
        report["target_transfer_seconds"] = time.perf_counter() - transfer_started
        report["cached_context_shape"] = list(batch["teacher_targets"]["tokens"].shape)
        report["cached_context_dtype"] = str(batch["teacher_targets"]["tokens"].dtype)
        report["cached_attention_shapes"] = {key: list(value.shape)
                                             for key, value in batch["teacher_targets"]["patch_attention"].items()}
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        OmegaConf.save(workspace.teacher_cfg, output / "teacher_config.yaml", resolve=True)
        report["resolved_config"] = OmegaConf.to_container(cfg, resolve=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(int(cfg.training.seed))
        workspace._setup_acceleration()
        workspace.model.train()
        report["status"] = "warming_up"
        _write_report(output, report)
        warmup_ms, measured_ms = [], []
        for index in range(args.warmup + args.steps):
            if index == args.warmup:
                torch.cuda.reset_peak_memory_stats(workspace.device)
                report["measurement_baseline_allocated_mib"] = torch.cuda.memory_allocated(workspace.device) / 2**20
                report["measurement_baseline_reserved_mib"] = torch.cuda.memory_reserved(workspace.device) / 2**20
                report["status"] = "measuring"
            torch.cuda.synchronize(workspace.device)
            step_started = time.perf_counter()
            loss, grad_norm = _optimizer_step(workspace, batch)
            torch.cuda.synchronize(workspace.device)
            elapsed_ms = (time.perf_counter() - step_started) * 1000
            if index < args.warmup:
                warmup_ms.append(elapsed_ms)
            else:
                measured_ms.append(elapsed_ms)
            log.info("%s step %d: %.3f ms, loss=%.8f, grad_norm=%.5f",
                     "Warmup" if index < args.warmup else "Measured", index + 1,
                     elapsed_ms, float(loss.detach()), float(grad_norm))
            if any(parameter.requires_grad or parameter.grad is not None
                   for parameter in workspace.teacher_policy.parameters()):
                raise RuntimeError("Frozen teacher unexpectedly acquired gradients/trainable parameters")
            if not workspace.loss_metrics.get("mask_supervised", False):
                raise RuntimeError("Benchmark failed to include region supervision")
            if not any(parameter.grad is not None for parameter in workspace.model.parameters()):
                raise RuntimeError("Renderer did not receive gradients")
            report["warmup_step_ms"] = list(warmup_ms)
            report["measured_step_ms"] = list(measured_ms)
            _write_report(output, report)
            del loss, grad_norm
        mean_ms = statistics.mean(measured_ms)
        report.update(status="completed", first_step_ms=warmup_ms[0],
                      mean_step_ms=mean_ms, median_step_ms=statistics.median(measured_ms),
                      p90_step_ms=float(np.percentile(measured_ms, 90)),
                      min_step_ms=min(measured_ms), max_step_ms=max(measured_ms),
                      samples_per_second=args.batch_size * 1000 / mean_ms,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(workspace.device) / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(workspace.device) / 2**20,
                      final_allocated_mib=torch.cuda.memory_allocated(workspace.device) / 2**20,
                      final_loss_metrics=dict(workspace.loss_metrics),
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(output, report)
        log.info("Completed: mean %.3f ms/step, median %.3f, p90 %.3f, %.2f samples/s; peak %.1f MiB",
                 report["mean_step_ms"], report["median_step_ms"], report["p90_step_ms"],
                 report["samples_per_second"], report["peak_allocated_mib"])
        return report
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(output, report)
        raise
    finally:
        _close_dataset(dataset)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
