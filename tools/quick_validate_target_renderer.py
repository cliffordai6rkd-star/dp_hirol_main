"""Compare raw-torque and explicit-target RGB decoders with one frozen FDP.

This is a bounded experiment, independent of the production training entry
points. Training and validation episodes stay disjoint. Teacher conditioning
uses the existing causal phase predictor; offline phases are only loss labels
and evaluation strata. The pretrained candidate gets extra, reported updates.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.response_distillation import masked_token_mse, phase_token_mask
from diffusion_policy.model.vision.target_conditioned_renderer import TargetConditionedRenderer
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs), ensure_ascii=False), flush=True)


def choose_samples(dataset, max_episodes, per_phase):
    anchors = np.asarray([dataset._anchor_index_from_sampler_row(row) for row in dataset.indices])
    episodes = dataset.episode_index[anchors]
    available = np.unique(episodes)
    if max_episodes < len(available):
        available = available[np.linspace(0, len(available) - 1, max_episodes).round().astype(int)]
    chosen = []
    records = []
    for episode in available:
        for phase in range(3):
            ids = np.flatnonzero((episodes == episode) & (dataset.phase_labels == phase))
            if not len(ids):
                continue
            ids = ids[np.linspace(0, len(ids) - 1, min(per_phase, len(ids))).round().astype(int)]
            for index in ids:
                chosen.append(int(index))
                records.append(dict(dataset_index=int(index), episode=int(episode), phase=phase,
                                    anchor=int(anchors[index])))
    if not chosen:
        raise ValueError("No selected dataset windows")
    return chosen, records


@torch.no_grad()
def make_cache(teacher, gate_model, dataset, indices, device, args, split):
    chunks = {}
    keys = teacher.obs_encoder.image_keys
    for raw in DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                          num_workers=args.workers, shuffle=False):
        obs = dict_apply(raw["obs"], lambda x: x.to(device))
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
        _, masks, condition = gate_model(images, wrench, phase_inputs=obs["phase_inputs"],
                                        intensity_only=True, return_condition=True)
        # No offline phase/alpha is passed to the renderer or target conditioner.
        phase_prediction = condition["phase_prediction"]
        causal = dict(obs)
        causal.pop("contact_alpha", None)
        causal["contact_phase"] = phase_prediction
        visual = teacher.obs_encoder.encode_image_context(images).float()
        target = teacher._encode_context(obs, apply_curriculum_mask=False).float()
        causal_target = teacher._encode_context(causal, apply_curriculum_mask=False).float()
        values = {"visual": visual, "target": target, "condition": causal_target,
                  "wrench": wrench.float(), "phase": obs["contact_phase"],
                  "predicted_phase": phase_prediction,
                  "phase_gate": condition["contact_alpha"].float()}
        for key in keys:
            values["image_" + key] = images[key].float()
            values["gate_" + key] = gate_model._mask_gate(masks[key].flatten(0, 1), images[key]).reshape_as(masks[key]).float()
        for key, value in values.items():
            chunks.setdefault(key, []).append(value.detach())
        emit("cache", split=split, completed=sum(len(x) for x in chunks["target"]), total=len(indices))
    return {key: torch.cat(values) for key, values in chunks.items()}


def select_batch(cache, indices):
    return {key: value[indices] for key, value in cache.items()}


def images_and_gates(batch, keys):
    return ({key: batch["image_" + key] for key in keys},
            {key: batch["gate_" + key] for key in keys})


def render_raw_torque(model, batch):
    images, gates = images_and_gates(batch, model.image_keys)
    wrench_tokens = model.encode_wrench_history(batch["wrench"])
    outputs = {}
    for camera, key in enumerate(model.image_keys):
        image = images[key]
        options = {}
        if model.spatial_modulation_enabled:
            options = dict(spatial_modulations=model.spatial_modulations,
                           modulation_gate=gates[key].flatten(0, 1))
        raw, _ = model.backbone(image.flatten(0, 1), wrench_tokens,
                                model.camera_embedding.weight[camera].reshape(1, 1, -1), **options)
        support = gates[key] * batch["phase_gate"][..., None, None, None]
        candidate = (image + model.residual_scale * support * raw.tanh().reshape_as(image).float()).clamp(0, 1)
        outputs[key] = torch.where(support > 0, candidate, image)
    return outputs


def render_candidate(model, batch, condition=None):
    if not isinstance(model, TargetConditionedRenderer):
        return render_raw_torque(model, batch)
    images, gates = images_and_gates(batch, model.image_keys)
    return model(images, batch["visual"], batch["condition"] if condition is None else condition,
                 gates, batch["phase_gate"])


def context_error(z, target, phase):
    batch, steps = phase.shape
    return (z.float() - target.float()).square().reshape(batch, steps, -1).mean(-1)


@torch.no_grad()
def evaluate(model, cache, encoder, batch_size, condition_mode=None):
    model.eval()
    collected = {key: [] for key in ("mse", "identity_mse", "cosine", "relative_response_mse",
                                    "phase", "prediction", "gate", "rgb_rms", "rgb_max", "outside_max")}
    for start in range(0, len(cache["target"]), batch_size):
        b = select_batch(cache, slice(start, start + batch_size))
        condition = None
        if condition_mode == "zero_response":
            condition = b["visual"]
        elif condition_mode == "shuffled_response":
            # Fixed across chunks; images and visual baselines are unchanged.
            ids = (torch.arange(start, start + len(b["target"]), device=b["visual"].device)
                   + max(1, len(cache["target"]) // 2)) % len(cache["target"])
            condition = b["visual"] + cache["condition"][ids] - cache["visual"][ids]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rendered = render_candidate(model, b, condition)
        z = encoder.encode_image_context(rendered).float()
        batch, steps = b["phase"].shape
        delta = (z - b["visual"]).reshape(batch, steps, -1)
        target_delta = (b["target"] - b["visual"]).reshape_as(delta)
        mse = context_error(z, b["target"], b["phase"])
        identity = context_error(b["visual"], b["target"], b["phase"])
        rgb = torch.cat([(rendered[key] - b["image_" + key]).flatten(2) for key in model.image_keys], dim=-1)
        outside = []
        for key in model.image_keys:
            support = b["gate_" + key] * b["phase_gate"][..., None, None, None]
            difference = (rendered[key] - b["image_" + key]).abs()
            outside.append(torch.where(support == 0, difference, 0).flatten(2).amax(-1))
        values = dict(mse=mse, identity_mse=identity,
                      cosine=F.cosine_similarity(delta, target_delta, dim=-1, eps=1e-8),
                      relative_response_mse=mse / identity.clamp_min(1e-12),
                      phase=b["phase"], prediction=b["predicted_phase"], gate=b["phase_gate"],
                      rgb_rms=rgb.square().mean(-1).sqrt(), rgb_max=rgb.abs().amax(-1),
                      outside_max=torch.stack(outside).amax(0))
        for key, value in values.items():
            collected[key].append(value.cpu())
    values = {key: torch.cat(value) for key, value in collected.items()}
    def aggregate(mask):
        if not mask.any():
            return dict(observations=0)
        mse, base = values["mse"][mask].mean().item(), values["identity_mse"][mask].mean().item()
        result = dict(observations=int(mask.sum()), token_mse=mse, identity_mse=base,
                      remaining_gap=mse / base if base > 1e-10 else None,
                      gap_reduction=1 - mse / base if base > 1e-10 else None,
                      response_cosine=values["cosine"][mask].mean().item() if base > 1e-10 else None,
                      rgb_rms=values["rgb_rms"][mask].mean().item(),
                      rgb_max=values["rgb_max"][mask].max().item(),
                      outside_support_max=values["outside_max"][mask].max().item())
        if result["outside_support_max"] != 0:
            raise AssertionError("Renderer modified pixels outside its frozen support")
        return result
    phase = values["phase"]
    return {"all": aggregate(torch.ones_like(phase, dtype=torch.bool)),
            **{name: aggregate(phase == index) for index, name in enumerate(("free", "precontact", "contact"))},
            "true_open": aggregate(phase != 0),
            "true_and_predicted_open": aggregate((phase != 0) & (values["gate"] > 0)),
            "predicted_closed": aggregate(values["gate"] == 0),
            "phase_accuracy": (phase == values["prediction"]).float().mean().item(),
            "precontact_open_recall": (values["gate"][phase == 1] > 0).float().mean().item()
            if (phase == 1).any() else None}


def synthetic_pair(batch, encoder, keys, generator):
    images, gates = images_and_gates(batch, keys)
    altered = {}
    for key, image in images.items():
        count = image.shape[0] * image.shape[1]
        smooth = torch.randn(count, 3, 7, 7, device=image.device, generator=generator)
        smooth = F.interpolate(smooth, size=image.shape[-2:], mode="bilinear", align_corners=False).reshape_as(image)
        texture = torch.randn(image.shape, device=image.device, generator=generator)
        strength = torch.rand(count, 1, 1, 1, device=image.device, generator=generator) * 0.25 + 0.05
        perturbation = (smooth * 0.8 + texture * 0.2).tanh() * strength.reshape(image.shape[0], image.shape[1], 1, 1, 1)
        support = gates[key] * batch["phase_gate"][..., None, None, None]
        altered[key] = (image + support * perturbation).clamp(0, 1)
    with torch.no_grad():
        target = encoder.encode_image_context(altered).float()
    return altered, target


def plot_history(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for arm in dict.fromkeys(row["arm"] for row in history):
        rows = [row for row in history if row["arm"] == arm and row["stage"] == "distill"]
        axes[0].plot([r["step"] for r in rows], [r["train"]["true_open"]["remaining_gap"] for r in rows], label=arm)
        axes[1].plot([r["step"] for r in rows], [r["validation"]["true_open"]["remaining_gap"] for r in rows], label=arm)
    for ax, title in zip(axes, ("Training episodes", "Held-out episodes")):
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.set(title=title, xlabel="Distillation updates", ylabel="Remaining open-phase token gap")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("data/debug") / time.strftime("%Y%m%d_%H%M%S_target_renderer"),
                        help="Result directory (default: data/debug/<timestamp>_target_renderer)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--validation-episodes", type=int, default=6)
    parser.add_argument("--per-phase", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.train_episodes, args.validation_episodes, args.per_phase, args.batch_size) < 1:
        parser.error("Steps and sample counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gate_model, _, config, teacher_config, device, _ = load_render_net(args.student_checkpoint, "cuda:0")
    payload, teacher_config = common.load_checkpoint_payload(Path(config.teacher_checkpoint))
    teacher = common.instantiate_policy(payload, teacher_config, device, capture_attention=False)
    del payload
    teacher.eval().requires_grad_(False)
    encoder = teacher.obs_encoder
    dataset_config = copy.deepcopy(teacher_config.task.dataset)
    with open_dict(dataset_config):
        dataset_config.preload_images = False
        dataset_config.image_randomer_config = None
        dataset_config.video_backend = "pyav"
        dataset_config.phase_predictor = config.render_net.phase_predictor
    dataset = hydra.utils.instantiate(dataset_config)
    validation_dataset = dataset.get_validation_dataset()
    train_ids, train_records = choose_samples(dataset, args.train_episodes, args.per_phase)
    val_ids, val_records = choose_samples(validation_dataset, args.validation_episodes, args.per_phase)
    assert not ({r["episode"] for r in train_records} & {r["episode"] for r in val_records})
    manifest = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    teacher_checkpoint=str(config.teacher_checkpoint),
                    history_steps=int(encoder.wrench_history_steps), context_dim=int(encoder.n_emb),
                    patch_counts=dict(encoder.patch_counts),
                    train=train_records, validation=val_records,
                    conditioning="Frozen teacher evaluated with causal predicted phases; never offline phase labels",
                    gate="Identical frozen checkpoint phase/spatial gates for all arms",
                    loss_scope="Offline precontact/contact observations only, normalized by selected token count",
                    initialization="Identical fresh RGB weights; teacher-initialized torque encoder for baseline",
                    evaluation="Sampled held-out episodes; not a full dataset or robot success evaluation",
                    pretraining="Synthetic RGB perturbations from TRAIN images only; extra updates reported separately")
    write_json(args.output / "manifest.json", manifest)
    emit("selection", train=len(train_ids), validation=len(val_ids), history_steps=encoder.wrench_history_steps)
    cache = {"train": make_cache(teacher, gate_model, dataset, train_ids, device, args, "train"),
             "validation": make_cache(teacher, gate_model, validation_dataset, val_ids, device, args, "validation")}
    del gate_model, dataset, validation_dataset
    # Only phase/spatial weights are reused from the supplied student;
    # all three RGB arms start from the same fresh backbone initialization.
    torch.manual_seed(args.seed)
    reference = build_render_net(encoder.wrench_projection.in_features,
                                OmegaConf.to_container(config.render_net, resolve=True)).to(device)
    reference.initialize_wrench_encoder(encoder)
    reference.set_training_stage("rgb")
    arms = {"raw_torque": reference,
            "explicit_target": TargetConditionedRenderer(reference, encoder.n_emb, encoder.patch_counts).to(device)}
    arms["explicit_target_pretrained"] = copy.deepcopy(arms["explicit_target"])
    manifest["trainable_parameters"] = {key: sum(p.numel() for p in model.parameters() if p.requires_grad)
                                         for key, model in arms.items()}
    write_json(args.output / "manifest.json", manifest)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    active = torch.nonzero(cache["train"]["phase_gate"].any(dim=1), as_tuple=True)[0]
    if not len(active):
        raise RuntimeError("No training windows with an open causal gate")
    # Same supervised batch sequence for every experimental arm.
    plan = [active[torch.randint(len(active), (args.batch_size,), generator=generator, device=device)]
            for _ in range(args.steps)]
    train_probe = cache["train"]
    history, results = [], {}
    def record(arm, model, step, stage):
        row = dict(arm=arm, step=step, stage=stage, elapsed_seconds=time.monotonic() - started,
                   train=evaluate(model, train_probe, encoder, args.batch_size),
                   validation=evaluate(model, cache["validation"], encoder, args.batch_size))
        history.append(row)
        write_json(args.output / "metrics.json", history)
        emit("evaluation", arm=arm, stage=stage, step=step,
             train_open=row["train"]["true_open"], validation_open=row["validation"]["true_open"])
        return row
    for arm, model in arms.items():
        torch.manual_seed(args.seed + 2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=2e-4, weight_decay=1e-4)
        record(arm, model, 0, "distill")
        if arm == "explicit_target_pretrained":
            pre_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
            for step in range(1, args.pretrain_steps + 1):
                ids = active[torch.randint(len(active), (args.batch_size,), generator=pre_generator, device=device)]
                b = select_batch(cache["train"], ids)
                altered, synthetic_target = synthetic_pair(b, encoder, model.image_keys, pre_generator)
                model.train(); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rendered = render_candidate(model, b, synthetic_target)
                z = encoder.encode_image_context(rendered, input_grad=True)
                rgb_loss = torch.stack([F.l1_loss(rendered[key], altered[key]) for key in model.image_keys]).mean()
                loss = masked_token_mse(z, synthetic_target, phase_token_mask(z, b["phase"]))[0] + rgb_loss
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                if step == 1 or step % 25 == 0:
                    emit("pretrain", step=step, loss=float(loss.detach()), elapsed_seconds=time.monotonic() - started)
            record(arm, model, 0, "after_pretrain")
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        best = float("inf")
        for step, ids in enumerate(plan, 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            b = select_batch(cache["train"], ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rendered = render_candidate(model, b)
            z = encoder.encode_image_context(rendered, input_grad=True)
            loss, _, _ = masked_token_mse(z, b["target"], phase_token_mask(z, b["phase"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite token loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step == 1 and ((b["phase"] != 0) & (b["phase_gate"] > 0)).any()
                    and (not torch.isfinite(norm) or norm <= 0)):
                raise AssertionError("No finite RGB training gradient")
            optimizer.step()
            if step == 1 or step % 25 == 0:
                emit("train", arm=arm, step=step, loss=float(loss.detach()), elapsed_seconds=time.monotonic() - started)
            if step % args.evaluate_every == 0 or step == args.steps:
                row = record(arm, model, step, "distill")
                score = row["validation"]["true_open"]["token_mse"]
                if score < best:
                    best = score
                    results[arm] = dict(best_step=step, train=row["train"], validation=row["validation"])
                    torch.save(dict(model=model.state_dict(), arm=arm, step=step, manifest=manifest),
                               args.output / (arm + "_best.pt"))
        if isinstance(model, TargetConditionedRenderer):
            # Endpoint controls use endpoint weights; do not mix with best-step metrics.
            results[arm]["endpoint_controls"] = {
                mode: evaluate(model, cache["validation"], encoder, args.batch_size, mode)
                for mode in ("zero_response", "shuffled_response")}
            results[arm]["endpoint"] = history[-1]["validation"]
        write_json(args.output / "results.json", dict(manifest=manifest, results=results,
                   elapsed_seconds=time.monotonic() - started))
    assert not any(p.grad is not None or p.requires_grad for p in teacher.parameters())
    plot_history(args.output, history)
    emit("complete", output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
