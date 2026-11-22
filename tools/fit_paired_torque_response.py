"""Fit 32 contact windows with identical absolute-only / response-loss arms.

Both arms see the same real/zero/cross-episode-contact torque triplets, start
from the same frozen-FFN experiment checkpoint, and take the same updates.
Only the additional paired response objective differs. Teacher, DINO, FFNs,
phase predictions, and spatial support remain frozen.
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
from omegaconf import OmegaConf, open_dict
import torch
from torch import nn

from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.response_distillation import masked_token_mse, phase_token_mask
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net
from tools.evaluate_torque_response import aggregate_response, contact_donors, make_cache, response_values
from tools.quick_validate_student_ffn import ExperimentArm, PhaseGatedFFN, encode_rendered
from tools.quick_validate_target_renderer import choose_samples, emit, select_batch, write_json


def paired_losses(prediction, target, token_mask):
    """Condition order is real, zero, other_contact; all terms share the mask."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[0] != 3:
        raise ValueError("Expected matching [3 conditions, B, tokens, features] contexts")
    absolute = masked_token_mse(prediction.flatten(0, 1), target.flatten(0, 1), token_mask.repeat(3, 1))[0]
    zero = masked_token_mse(prediction[0] - prediction[1], target[0] - target[1], token_mask)[0]
    contact = masked_token_mse(prediction[0] - prediction[2], target[0] - target[2], token_mask)[0]
    return dict(absolute=absolute, zero_response=zero, contact_response=contact,
                response=(zero + contact) / 2)


def triplet_context(arm, encoder, batch, *, input_grad=False):
    size = len(batch["phase"])
    tripled = {key: torch.cat([value] * 3) for key, value in batch.items()}
    tripled["wrench"] = torch.cat((batch["wrench"], torch.zeros_like(batch["wrench"]), batch["other_wrench"]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rendered = arm(tripled)
    if not input_grad:
        for key in arm.image_keys:
            support = tripled["gate_" + key] * tripled["phase_gate"][..., None, None, None]
            delta = rendered[key] - tripled["image_" + key]
            if torch.where(support == 0, delta, 0).abs().max() != 0:
                raise AssertionError("RGB changed outside fixed support")
    z = encode_rendered(encoder, arm, rendered, tripled["phase_gate"], input_grad=input_grad)
    return z.reshape(3, size, *z.shape[1:]).float()


def targets(batch):
    return torch.stack([batch["target"], batch["zero_target"], batch["other_target"]])


@torch.no_grad()
def evaluate(arm, encoder, cache, batch_size):
    arm.eval()
    predictions = []
    for start in range(0, len(cache["phase"]), batch_size):
        predictions.append(triplet_context(arm, encoder, select_batch(cache, slice(start, start + batch_size))).cpu())
    prediction = torch.cat(predictions, dim=1)
    target = targets(cache).cpu()
    mask = phase_token_mask(target[0], cache["phase"].cpu())
    losses = paired_losses(prediction, target, mask)
    steps = cache["phase"].shape[1]
    selected = cache["phase"].cpu() == 2
    responses = {}
    for pair, baseline in (("real_vs_zero", 1), ("real_vs_other", 2)):
        t = (target[0] - target[baseline]).reshape(len(selected), steps, -1)
        s = (prediction[0] - prediction[baseline]).reshape_as(t)
        responses[pair] = aggregate_response(response_values(t, s), selected)
    return dict(**{key + "_mse": float(value) for key, value in losses.items()},
                real_context_mse=float(masked_token_mse(prediction[0], target[0], mask)[0]),
                zero_context_mse=float(masked_token_mse(prediction[1], target[1], mask)[0]),
                other_context_mse=float(masked_token_mse(prediction[2], target[2], mask)[0]),
                responses=responses)


@torch.no_grad()
def prepare_split(teacher, gate_model, dataset, records, device, args):
    # Only anchors known to be contact are candidates; require both observation
    # slots to be true contact AND predicted open, eliminating a fixed gate floor.
    records = [r for r in records if r["phase"] == 2]
    cache = make_cache(teacher, gate_model, dataset, records, device, args.batch_size, args.workers)
    eligible = torch.where((cache["phase"] == 2).all(1) & (cache["phase_gate"] > 0).all(1))[0]
    if len(eligible) < args.windows:
        raise ValueError(f"Need {args.windows} fully contact/open windows; found {len(eligible)}")
    ids = eligible[torch.linspace(0, len(eligible) - 1, args.windows, device=device).round().long()]
    chosen = [records[i] for i in ids.tolist()]
    cache = select_batch(cache, ids)
    donors = contact_donors(cache["phase"], torch.tensor([r["episode"] for r in chosen]))
    cache["other_wrench"] = cache["wrench"].flatten(0, 1)[donors.flatten().to(device)].reshape_as(cache["wrench"])
    for mode in ("zero", "other"):
        parts = []
        for start in range(0, len(chosen), args.batch_size):
            b = select_batch(cache, slice(start, start + args.batch_size))
            images = {key: b["image_" + key] for key in teacher.obs_encoder.image_keys}
            wrench = torch.zeros_like(b["wrench"]) if mode == "zero" else b["other_wrench"]
            parts.append(teacher.obs_encoder(images=images, wrench_history=wrench,
                                             fusion_gate=b["phase"] != 0).float())
        cache[mode + "_target"] = torch.cat(parts)
    return cache, dict(records=chosen, eligible_windows=len(eligible),
                       donor_flat_observation_indices=donors.tolist())


def plot_history(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for row, split in enumerate(("train", "validation")):
        for column, metric in enumerate(("absolute", "nmse", "cosine")):
            ax = axes[row, column]
            for arm, label in (("absolute_only", "A: absolute MSE"), ("absolute_plus_response", "B: + paired response")):
                steps = sorted({r["step"] for r in history if r["arm"] == arm})
                if not steps:
                    continue
                values = []
                for step in steps:
                    vals = [r[split]["absolute_mse"] if metric == "absolute" else
                            r[split]["responses"]["real_vs_other"]["relative_difference_mse" if metric == "nmse" else "cosine_mean"]
                            for r in history if r["arm"] == arm and r["step"] == step]
                    values.append((np.mean(vals), np.std(vals)))
                mean, std = np.asarray(values).T
                ax.plot(steps, mean, label=label)
                ax.fill_between(steps, mean - std, mean + std, alpha=.15)
            ax.set_title(split + ": " + {"absolute": "three-condition token MSE", "nmse": "contact-history response NMSE", "cosine": "contact-history response cosine"}[metric], fontsize=9)
            ax.set_xlabel("Additional updates")
            if metric == "absolute":
                ax.set_yscale("log")
            if metric == "nmse":
                ax.axhline(1, color="gray", ls="--", lw=.7)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, default=Path("data/debug/20261119_student_ffn_quick_01"))
    p.add_argument("--output", type=Path, default=Path("data/debug") / time.strftime("%Y%m%d_%H%M%S_paired_contact_fit"))
    p.add_argument("--windows", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--evaluate-every", type=int, default=250)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    previous = json.loads((args.experiment_dir / "results.json").read_text())
    old_manifest = previous["manifest"]
    gate_model, _, config, _, device, _ = load_render_net(Path(old_manifest["arguments"]["student_checkpoint"]), "cuda:0")
    payload, teacher_cfg = common.load_checkpoint_payload(Path(old_manifest["teacher_checkpoint"]))
    teacher = common.instantiate_policy(payload, teacher_cfg, device, capture_attention=False).eval().requires_grad_(False)
    del payload
    encoder = teacher.obs_encoder
    ds_cfg = copy.deepcopy(teacher_cfg.task.dataset)
    with open_dict(ds_cfg):
        ds_cfg.preload_images = False
        ds_cfg.image_randomer_config = None
        ds_cfg.video_backend = "pyav"
        ds_cfg.phase_predictor = config.render_net.phase_predictor
    dataset = hydra.utils.instantiate(ds_cfg)
    validation = dataset.get_validation_dataset()
    _, val_records = choose_samples(validation, 100000, 20)
    train_cache, train_info = prepare_split(teacher, gate_model, dataset, old_manifest["train"], device, args)
    validation_cache, validation_info = prepare_split(teacher, gate_model, validation, val_records, device, args)
    assert not ({r["episode"] for r in train_info["records"]} & {r["episode"] for r in validation_info["records"]})
    del gate_model, dataset, validation
    manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    teacher_checkpoint=old_manifest["teacher_checkpoint"], train=train_info, validation=validation_info,
                    history_steps=encoder.wrench_history_steps,
                    initialization="Identical previous frozen_teacher_ffn best checkpoint for A/B within each seed",
                    fixed="Teacher/DINO/FFN, original causal phase and spatial gates; same three torque conditions and update plan",
                    loss_A="Mean absolute context MSE over real, zero, other_contact, offline precontact/contact tokens only",
                    loss_B="A + 0.5 * (MSE of real-zero response + MSE of real-other_contact response)",
                    selection="Fixed-budget endpoint, no early stopping or checkpoint selection on validation",
                    optimizer="AdamW lr=2e-4, weight_decay=1e-4, max_grad_norm=1",
                    scope="32 training contact windows, 32 episode-held-out contact windows; both observation slots contact and predicted open")
    write_json(args.output / "manifest.json", manifest)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("fit_paired_torque_response.py", "evaluate_torque_response.py", "quick_validate_student_ffn.py", "quick_validate_target_renderer.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    history, results = [], []
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        checkpoint = args.experiment_dir / f"frozen_teacher_ffn_seed{seed}_best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        reference = ExperimentArm(build_render_net(encoder.wrench_projection.in_features,
                                  OmegaConf.to_container(config.render_net, resolve=True)).to(device),
                                  PhaseGatedFFN(encoder), encoder.image_keys).to(device)
        reference.load_state_dict(state["model"], strict=True)
        reference.renderer.set_training_stage("rgb")
        generator = torch.Generator(device=device).manual_seed(seed + 100)
        plan = [torch.randint(args.windows, (args.batch_size,), generator=generator, device=device) for _ in range(args.steps)]
        for arm_name in ("absolute_only", "absolute_plus_response"):
            arm = copy.deepcopy(reference)
            parameters = [p for p in arm.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)

            def record(step):
                result = dict(seed=seed, arm=arm_name, step=step, elapsed_seconds=time.monotonic() - started,
                              train=evaluate(arm, encoder, train_cache, args.batch_size),
                              validation=evaluate(arm, encoder, validation_cache, args.batch_size))
                history.append(result)
                write_json(args.output / "metrics.json", history)
                plot_history(args.output, history)
                emit("evaluation", seed=seed, arm=arm_name, step=step,
                     train_absolute=result["train"]["absolute_mse"],
                     train_response=result["train"]["responses"]["real_vs_other"],
                     validation_response=result["validation"]["responses"]["real_vs_other"],
                     elapsed_seconds=time.monotonic() - started)
                return result

            initial = record(0)
            for step, ids in enumerate(plan, 1):
                arm.train(); optimizer.zero_grad(set_to_none=True)
                batch = select_batch(train_cache, ids)
                prediction = triplet_context(arm, encoder, batch, input_grad=True)
                terms = paired_losses(prediction, targets(batch), phase_token_mask(prediction[0], batch["phase"]))
                loss = terms["absolute"] + (terms["response"] if arm_name == "absolute_plus_response" else 0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite paired contact loss")
                loss.backward()
                if step == 1:
                    if arm.renderer.output.weight.grad.abs().sum() <= 0:
                        raise AssertionError("No renderer gradient")
                    if any(p.grad is not None for p in arm.readout.parameters()):
                        raise AssertionError("Frozen FFN received parameter gradients")
                nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    emit("train", seed=seed, arm=arm_name, step=step, loss=float(loss.detach()),
                         absolute=float(terms["absolute"].detach()), response=float(terms["response"].detach()),
                         elapsed_seconds=time.monotonic() - started)
                if step % args.evaluate_every == 0 or step == args.steps:
                    endpoint = record(step)
            torch.save(dict(model=arm.state_dict(), seed=seed, arm=arm_name, step=args.steps, manifest=manifest),
                       args.output / f"{arm_name}_seed{seed}_endpoint.pt")
            results.append(dict(seed=seed, arm=arm_name, initial=initial, endpoint=endpoint))
            write_json(args.output / "results.json", dict(manifest=manifest, results=results,
                       elapsed_seconds=time.monotonic() - started))
            del arm, optimizer
        del reference, state
    assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    emit("complete", output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
