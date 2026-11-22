"""Full-split paired A/B distillation of a frozen-gate CIVI RenderNet.

All precontact/contact image windows are used. A optimizes absolute context
token MSE under real, zero and cross-episode contact torque histories. B adds
the paired teacher/student response-difference MSE. Phase and spatial gates,
teacher, DINO and the teacher-FFN readout are fixed for both arms.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.response_distillation import masked_token_mse, phase_token_mask
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net
from tools.evaluate_torque_response import aggregate_response, contact_donors, response_values
from tools.fit_paired_torque_response import paired_losses, triplet_context, targets
from tools.quick_validate_student_ffn import ExperimentArm, PhaseGatedFFN
from tools.quick_validate_target_renderer import choose_samples, emit, select_batch, write_json


def triplet_context(arm, encoder, batch, *, input_grad=False):
    device = next(arm.parameters()).device
    keep = ("wrench", "other_wrench", "phase_gate")
    batch = {key: value.to(device, dtype=(torch.float32 if key.startswith("image_") else value.dtype),
                           non_blocking=True)
             for key, value in batch.items()
             if key in keep or key.startswith(("image_", "gate_"))}
    size = batch["phase_gate"].shape[0]
    tripled = {key: torch.cat([value] * 3) for key, value in batch.items()}
    tripled["wrench"] = torch.cat((batch["wrench"], torch.zeros_like(batch["wrench"]),
                                   batch["other_wrench"]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rendered = arm(tripled)
    if not input_grad:
        for key in arm.image_keys:
            support = tripled["gate_" + key] * tripled["phase_gate"][..., None, None, None]
            delta = rendered[key] - tripled["image_" + key]
            if torch.where(support == 0, delta, 0).abs().max() != 0:
                raise AssertionError("RGB changed outside fixed phase/spatial support")
    tokens = encoder.encode_image_context(
        rendered, input_grad=input_grad,
        context_adapter=lambda context: arm.readout(context, tripled["phase_gate"]),
    )
    return tokens.reshape(3, size, *tokens.shape[1:]).float()


def targets(batch, device):
    return torch.stack([batch[key].to(device, non_blocking=True)
                        for key in ("target", "zero_target", "other_target")])


def cache_split(teacher, gate_model, dataset, records, device, *, batch_size, workers, split):
    """Cache just precontact/contact windows in host memory at FP16 precision."""
    chunks: dict[str, list[torch.Tensor]] = {}
    loader = DataLoader(
        Subset(dataset, [r["dataset_index"] for r in records]), batch_size=batch_size,
        num_workers=workers, shuffle=False, pin_memory=device.type == "cuda",
    )
    for index, raw in enumerate(loader):
        obs = dict_apply(raw["obs"], lambda value: value.to(device, non_blocking=True))
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
        _, masks, condition = gate_model(
            images, wrench, phase_inputs=obs["phase_inputs"], intensity_only=True,
            return_condition=True,
        )
        phases = obs["contact_phase"][:, :teacher.n_obs_steps]
        fusion_gate = (phases == 1) | (phases == 2)
        target = teacher._encode_context(obs, apply_curriculum_mask=False)
        if index == 0:
            replay = teacher.obs_encoder(images=images, wrench_history=wrench,
                                         fusion_gate=fusion_gate)
            torch.testing.assert_close(target, replay, rtol=1e-5, atol=1e-6)
        batch = {
            "phase": phases,
            "phase_gate": condition["contact_alpha"].float(),
            "predicted_phase": condition["phase_prediction"],
            "wrench": wrench.float(),
            "target": target.float(),
        }
        for key in teacher.obs_encoder.image_keys:
            batch["image_" + key] = images[key].float()
            batch["gate_" + key] = gate_model._mask_gate(
                masks[key].flatten(0, 1), images[key],
            ).reshape_as(masks[key]).float()
        for key, value in batch.items():
            value = value.detach().cpu()
            value = value.to(torch.float16) if key != "phase" and key != "predicted_phase" else value
            chunks.setdefault(key, []).append(value)
        emit("cache", split=split, samples=min((index + 1) * batch_size, len(records)),
             total=len(records))
    cache = {key: torch.cat(parts) for key, parts in chunks.items()}
    donor_indices = contact_donors(
        cache["phase"], torch.tensor([record["episode"] for record in records]),
    )
    cache["other_wrench"] = cache["wrench"].flatten(0, 1)[donor_indices.flatten()].reshape_as(
        cache["wrench"]
    )
    # Freeze all counterfactual teacher targets with phase/gating held to the
    # original sample's true phase, matching the bounded paired experiment.
    for mode in ("zero", "other"):
        outputs = []
        for start in range(0, len(records), batch_size):
            batch = select_batch(cache, slice(start, start + batch_size))
            images = {key: batch["image_" + key].to(device, dtype=torch.float32,
                                                      non_blocking=True)
                      for key in teacher.obs_encoder.image_keys}
            wrench = (torch.zeros_like(batch["wrench"]) if mode == "zero"
                      else batch["other_wrench"])
            output = teacher.obs_encoder(
                images=images, wrench_history=wrench.to(device, dtype=torch.float32,
                                                        non_blocking=True),
                fusion_gate=batch["phase"].to(device, non_blocking=True) != 0,
            )
            outputs.append(output.float().cpu().to(torch.float16))
        cache[mode + "_target"] = torch.cat(outputs)
    return cache


@torch.no_grad()
def evaluate(arm, encoder, cache, *, batch_size):
    arm.eval()
    device = next(arm.parameters()).device
    token_count = 0
    absolute_sum = response_sum = 0.0
    absolute_parts = {"real": 0.0, "zero": 0.0, "other": 0.0}
    response_acc = {
        name: {"teacher_energy": 0.0, "student_energy": 0.0,
              "error_energy": 0.0, "cosine_sum": 0.0,
              "responsive": 0, "observations": 0}
        for name in ("real_vs_zero", "real_vs_other")
    }
    for start in range(0, len(cache["phase"]), batch_size):
        batch = select_batch(cache, slice(start, start + batch_size))
        prediction = triplet_context(arm, encoder, batch)
        target = targets(batch, device)
        phases = batch["phase"].to(device, non_blocking=True)
        mask = phase_token_mask(target[0], phases)
        terms = paired_losses(prediction, target, mask)
        count = int(mask.sum())
        absolute_sum += float(terms["absolute"]) * count
        response_sum += float(terms["response"]) * count
        token_count += count
        for cond, name in enumerate(("real", "zero", "other")):
            absolute_parts[name] += float(masked_token_mse(prediction[cond], target[cond], mask)[0]) * count
        selected = phases == 2
        n_obs = phases.shape[1]
        for pair, baseline in (("real_vs_zero", 1), ("real_vs_other", 2)):
            teacher_delta = (target[0] - target[baseline]).reshape(len(phases), n_obs, -1)
            student_delta = (prediction[0] - prediction[baseline]).reshape_as(teacher_delta)
            values = response_values(teacher_delta, student_delta)
            acc = response_acc[pair]
            te = values["teacher_energy"][selected].double()
            se = values["student_energy"][selected].double()
            ee = values["error_energy"][selected].double()
            cos = values["cosine"][selected].double()
            responsive = te > 1e-14
            acc["teacher_energy"] += float(te.sum())
            acc["student_energy"] += float(se.sum())
            acc["error_energy"] += float(ee.sum())
            acc["cosine_sum"] += float(cos[responsive].sum())
            acc["responsive"] += int(responsive.sum())
            acc["observations"] += int(selected.sum())
    responses = {}
    for name, acc in response_acc.items():
        energy = acc["teacher_energy"]
        count = max(acc["observations"], 1)
        responses[name] = {
            "observations": acc["observations"],
            "responsive_observations": acc["responsive"],
            "teacher_rms": (energy / count) ** 0.5,
            "student_rms": (acc["student_energy"] / count) ** 0.5,
            "difference_mse": acc["error_energy"] / count,
            "relative_difference_mse": acc["error_energy"] / energy if energy > 1e-14 else None,
            "amplitude_rms_ratio": (acc["student_energy"] / energy) ** 0.5 if energy > 1e-14 else None,
            "cosine_mean": acc["cosine_sum"] / acc["responsive"] if acc["responsive"] else None,
        }
    return {
        "absolute_mse": absolute_sum / max(token_count, 1),
        "response_mse": response_sum / max(token_count, 1),
        "real_context_mse": absolute_parts["real"] / max(token_count, 1),
        "zero_context_mse": absolute_parts["zero"] / max(token_count, 1),
        "other_context_mse": absolute_parts["other"] / max(token_count, 1),
        "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=39)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--cache-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--smoke-samples", type=int, default=None,
                        help="Evenly spaced rows per split for a short end-to-end smoke run")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.microbatch_size, args.cache_batch_size,
           args.workers, args.eval_batch_size) < 1:
        parser.error("Epoch, batch, worker, and evaluation sizes must be positive")
    if args.batch_size % args.microbatch_size:
        parser.error("batch-size must be divisible by microbatch-size")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Full data A/B distillation requires CUDA")

    gate_model, _, config, teacher_cfg, device, _ = load_render_net(args.student_checkpoint, "cuda:0")
    payload, teacher_cfg = common.load_checkpoint_payload(Path(config.teacher_checkpoint))
    teacher = common.instantiate_policy(payload, teacher_cfg, device, capture_attention=False)
    del payload
    teacher.eval().requires_grad_(False)
    encoder = teacher.obs_encoder
    ds_cfg = copy.deepcopy(teacher_cfg.task.dataset)
    with open_dict(ds_cfg):
        ds_cfg.preload_images = False
        ds_cfg.image_randomer_config = None
        ds_cfg.video_backend = "pyav"
        ds_cfg.phase_predictor = config.render_net.phase_predictor
    train_dataset = hydra.utils.instantiate(ds_cfg)
    validation_dataset = train_dataset.get_validation_dataset()
    _, train_records = choose_samples(train_dataset, 100000, max(1, len(train_dataset)))
    _, validation_records = choose_samples(validation_dataset, 100000,
                                           max(1, len(validation_dataset)))
    train_records = [record for record in train_records if record["phase"] in (1, 2)]
    validation_records = [record for record in validation_records if record["phase"] in (1, 2)]
    if args.smoke_samples is not None:
        if args.smoke_samples < 2:
            parser.error("smoke-samples must be at least 2 to include distinct episodes")
        train_ids = np.linspace(0, len(train_records) - 1,
                                min(args.smoke_samples, len(train_records))).round().astype(int)
        validation_ids = np.linspace(0, len(validation_records) - 1,
                                     min(args.smoke_samples, len(validation_records))).round().astype(int)
        train_records = [train_records[index] for index in train_ids]
        validation_records = [validation_records[index] for index in validation_ids]
    if not train_records or not validation_records:
        raise RuntimeError("Both train and episode-held-out validation splits need precontact/contact samples")
    if {r["episode"] for r in train_records} & {r["episode"] for r in validation_records}:
        raise RuntimeError("Train and validation episodes overlap")
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "teacher_checkpoint": str(config.teacher_checkpoint),
        "dataset": str(ds_cfg.dataset_path),
        "history_steps": int(encoder.wrench_history_steps),
        "train_samples": len(train_records), "validation_samples": len(validation_records),
        "smoke_samples_per_split": args.smoke_samples,
        "train_episodes": sorted({r["episode"] for r in train_records}),
        "validation_episodes": sorted({r["episode"] for r in validation_records}),
        "condition_triplets": ["actual_tau_ext_history", "zero_tau_ext_history",
                               "other_episode_contact_tau_ext_history"],
        "loss_A": "mean absolute final-context token MSE over real/zero/other conditions; phase 1/2 tokens only",
        "loss_B": "loss_A + mean paired response MSE(real-zero, real-other_contact)",
        "fixed": "Teacher, DINO, teacher-FFN image readout, phase head, spatial gate and torque donors",
        "initialization": "same resolved student checkpoint for both groups; RGB stage trains, phase and intensity stages frozen",
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 1e-4,
                      "betas": [0.9, 0.999], "global_batch": args.batch_size,
                      "microbatch": args.microbatch_size, "gradient_clip": 1.0},
        "schedule": {"epochs_per_group": args.epochs, "early_stopping": False,
                     "validation_selection": "none; fixed epoch endpoint"},
    }
    write_json(args.output / "manifest.json", manifest)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("fit_paired_torque_response.py"),
                 Path(__file__).with_name("evaluate_torque_response.py"),
                 Path(__file__).with_name("quick_validate_student_ffn.py")):
        shutil.copy2(path, snapshot / path.name)

    train_cache = cache_split(teacher, gate_model, train_dataset, train_records, device,
                              batch_size=args.cache_batch_size, workers=args.workers, split="train")
    validation_cache = cache_split(teacher, gate_model, validation_dataset, validation_records, device,
                                   batch_size=args.cache_batch_size, workers=args.workers,
                                   split="validation")
    del train_dataset, validation_dataset
    # Initialize both arms byte-identically from the same full-data student.
    reference = ExperimentArm(copy.deepcopy(gate_model), PhaseGatedFFN(encoder),
                              encoder.image_keys).to(device)
    reference.renderer.set_training_stage("rgb")
    del gate_model
    history, results = [], []
    global_batch = args.batch_size
    accumulation = global_batch // args.microbatch_size
    steps_per_epoch = (len(train_cache["phase"]) + global_batch - 1) // global_batch

    for arm_index, arm_name in enumerate(("absolute_only", "absolute_plus_response")):
        torch.manual_seed(args.seed)
        arm = copy.deepcopy(reference)
        params = [parameter for parameter in arm.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        arm_history = []
        arm.eval()
        initial = {
            "train": evaluate(arm, encoder, train_cache, batch_size=args.eval_batch_size),
            "validation": evaluate(arm, encoder, validation_cache, batch_size=args.eval_batch_size),
        }
        arm_history.append({"epoch": 0, **initial, "elapsed_seconds": time.monotonic() - started})
        emit("initial", arm=arm_name, train=initial["train"]["absolute_mse"],
             validation=initial["validation"]["absolute_mse"])

        for epoch in range(1, args.epochs + 1):
            arm.renderer.set_stage_mode(True)
            arm.train()
            order = torch.randperm(len(train_cache["phase"]), generator=torch.Generator().manual_seed(
                args.seed + epoch * 1009,
            ))
            epoch_abs = epoch_resp = epoch_weight = 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch_number, start in enumerate(range(0, len(order), global_batch), 1):
                ids = order[start:start + global_batch]
                ids_parts = ids.split(args.microbatch_size)
                counts = []
                for part in ids_parts:
                    phases = train_cache["phase"][part]
                    # Mask count is the number of selected observation-patch tokens.
                    patches = train_cache["target"].shape[1] // phases.shape[1]
                    counts.append(int((((phases == 1) | (phases == 2)).sum()) * patches))
                total_count = max(sum(counts), 1)
                for micro_index, (part, count) in enumerate(zip(ids_parts, counts)):
                    batch = select_batch(train_cache, part)
                    prediction = triplet_context(arm, encoder, batch, input_grad=True)
                    target = targets(batch, device=prediction.device)
                    phases = batch["phase"].to(device, non_blocking=True)
                    terms = paired_losses(prediction, target,
                                          phase_token_mask(prediction[0], phases))
                    loss = terms["absolute"]
                    if arm_name == "absolute_plus_response":
                        loss = loss + terms["response"]
                    (loss * (count / total_count)).backward()
                    epoch_abs += float(terms["absolute"].detach()) * count
                    epoch_resp += float(terms["response"].detach()) * count
                    epoch_weight += count
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if batch_number % 100 == 0 or batch_number == steps_per_epoch:
                    emit("train", arm=arm_name, epoch=epoch, epochs=args.epochs,
                         step=batch_number, steps=steps_per_epoch,
                         absolute=epoch_abs / max(epoch_weight, 1),
                         response=epoch_resp / max(epoch_weight, 1),
                         elapsed_seconds=time.monotonic() - started)
            arm.eval()
            validation_metrics = evaluate(arm, encoder, validation_cache,
                                          batch_size=args.eval_batch_size)
            row = {
                "epoch": epoch,
                "train_absolute_mse": epoch_abs / max(epoch_weight, 1),
                "train_response_mse": epoch_resp / max(epoch_weight, 1),
                "validation": validation_metrics,
                "elapsed_seconds": time.monotonic() - started,
            }
            arm_history.append(row)
            history.append({"arm": arm_name, **row})
            write_json(args.output / "metrics.json", history)
            checkpoint_path = args.output / f"{arm_name}_latest.pt"
            torch.save({"model": arm.state_dict(), "epoch": epoch,
                        "arm": arm_name, "manifest": manifest}, checkpoint_path)
            emit("epoch_complete", arm=arm_name, epoch=epoch,
                 train_absolute=row["train_absolute_mse"],
                 validation_absolute=validation_metrics["absolute_mse"],
                 validation_real_other=validation_metrics["responses"]["real_vs_other"],
                 elapsed_seconds=time.monotonic() - started)
        torch.save({"model": arm.state_dict(), "epoch": args.epochs,
                    "arm": arm_name, "manifest": manifest},
                   args.output / f"{arm_name}_endpoint.pt")
        results.append({"arm": arm_name, "initial": initial, "epochs": arm_history,
                        "endpoint": arm_history[-1]})
        write_json(args.output / "results.json", {"manifest": manifest,
                                                   "results": results,
                                                   "elapsed_seconds": time.monotonic() - started})
        del arm, optimizer
    emit("complete", output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
