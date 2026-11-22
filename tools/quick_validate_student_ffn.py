"""Small, episode-held-out test of student FFN readouts with one frozen FDP.

RGB architecture, targets, support, minibatch plan, and update count are shared.
The extra FFN-only arm and same-readout original-image controls distinguish
feature calibration from force information carried by rendered RGB.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.response_distillation import masked_token_mse, phase_token_mask
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net
from tools.quick_validate_target_renderer import (
    choose_samples, emit, images_and_gates, render_raw_torque, select_batch, write_json,
)


class PhaseGatedFFN(nn.Module):
    """Independent copies of teacher FFNs, with causal free-phase bypass.

    Input/output are before the teacher's frozen final context norm. This head
    reads only image tokens and the causal phase gate, never torque tokens.
    """

    def __init__(self, encoder, *, trainable=False):
        super().__init__()
        if (encoder.cross_attn_query != "image" or not encoder.cross_attn_query_residual
                or encoder.pre_cross_self_attn_layers or not encoder.cross_attn_layers):
            raise ValueError("FFN replay needs residual image-query fusion without joint self-attention")
        self.learnable = bool(trainable)
        self.layers = nn.ModuleList([
            nn.Sequential(copy.deepcopy(block.cross_output_norm), copy.deepcopy(block.cross_ffn))
            for block in (encoder, *encoder.extra_cross_attention_layers)
        ])
        # Teacher targets use eval-mode dropout; keep all readout arms deterministic.
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.requires_grad_(self.learnable)

    def train(self, mode=True):
        return super().train(bool(mode) and self.learnable)

    def forward(self, tokens, phase_gate):
        batch, steps = phase_gate.shape
        if tokens.ndim != 3 or tokens.shape[0] != batch or tokens.shape[1] % steps:
            raise ValueError("Expected [B, observation*patches, D] image features")
        transformed = tokens
        for layer in self.layers:
            transformed = transformed + layer(transformed)
        opened = (phase_gate > 0)[..., None, None].expand(
            batch, steps, tokens.shape[1] // steps, 1,
        ).reshape(batch, tokens.shape[1], 1)
        return torch.where(opened, transformed, tokens)


class ExperimentArm(nn.Module):
    def __init__(self, renderer, readout, image_keys):
        super().__init__()
        self.renderer = renderer
        self.readout = readout
        self.image_keys = tuple(image_keys)

    def forward(self, batch):
        if self.renderer is None:
            return images_and_gates(batch, self.image_keys)[0]
        return render_raw_torque(self.renderer, batch)


def apply_readout(encoder, arm, tokens, phase_gate):
    if arm.readout is not None:
        tokens = arm.readout(tokens, phase_gate)
    return encoder.context_norm(tokens)


def encode_rendered(encoder, arm, images, phase_gate, *, input_grad=False):
    adapter = (None if arm.readout is None else lambda z: arm.readout(z, phase_gate))
    return encoder.encode_image_context(images, input_grad=input_grad, context_adapter=adapter)


@torch.no_grad()
def make_cache(teacher, gate_model, dataset, indices, device, args, split):
    chunks = {}
    for raw in DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                          num_workers=args.workers, shuffle=False):
        obs = dict_apply(raw["obs"], lambda x: x.to(device))
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
        _, masks, condition = gate_model(images, wrench, phase_inputs=obs["phase_inputs"],
                                        intensity_only=True, return_condition=True)
        captured = []
        def capture(tokens):
            captured.append(tokens.detach())
            return tokens
        visual = teacher.obs_encoder.encode_image_context(images, context_adapter=capture).float()
        target = teacher._encode_context(obs, apply_curriculum_mask=False).float()
        values = dict(pre_norm=captured[0].float(), visual=visual, target=target,
                      wrench=wrench.float(), phase=obs["contact_phase"],
                      predicted_phase=condition["phase_prediction"],
                      phase_gate=condition["contact_alpha"].float())
        for key in teacher.obs_encoder.image_keys:
            values["image_" + key] = images[key].float()
            values["gate_" + key] = gate_model._mask_gate(
                masks[key].flatten(0, 1), images[key],
            ).reshape_as(masks[key]).float()
        for key, value in values.items():
            chunks.setdefault(key, []).append(value.detach())
        emit("cache", split=split, completed=sum(len(x) for x in chunks["target"]), total=len(indices))
    return {key: torch.cat(value) for key, value in chunks.items()}


@torch.no_grad()
def evaluate(arm, cache, encoder, batch_size, *, intervention=None):
    arm.eval()
    collected = {}
    for start in range(0, len(cache["target"]), batch_size):
        b = select_batch(cache, slice(start, start + batch_size))
        if intervention == "zero_torque":
            b["wrench"] = torch.zeros_like(b["wrench"])
        elif intervention == "shuffled_torque":
            # Keep images, phase predictions and spatial gates fixed. The saved
            # preprocessing preserves raw physical zeros as normalized zeros.
            ids = (torch.arange(start, start + len(b["target"]), device=b["target"].device)
                   + max(1, len(cache["target"]) // 2)) % len(cache["target"])
            b["wrench"] = cache["wrench"][ids]
        original_z = apply_readout(encoder, arm, b["pre_norm"], b["phase_gate"]).float()
        if arm.renderer is None:
            images = images_and_gates(b, arm.image_keys)[0]
            z = original_z
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                images = arm(b)
            z = encode_rendered(encoder, arm, images, b["phase_gate"]).float()
        batch, steps = b["phase"].shape
        error = (z - b["target"]).square().reshape(batch, steps, -1).mean(-1)
        identity = (original_z - b["target"]).square().reshape(batch, steps, -1).mean(-1)
        delta = (z - original_z).reshape(batch, steps, -1)
        desired_delta = (b["target"] - original_z).reshape_as(delta)
        rgb = torch.cat([(images[key] - b["image_" + key]).flatten(2) for key in arm.image_keys], dim=-1)
        outside = []
        for key in arm.image_keys:
            support = b["gate_" + key] * b["phase_gate"][..., None, None, None]
            difference = (images[key] - b["image_" + key]).abs()
            outside.append(torch.where(support == 0, difference, 0).flatten(2).amax(-1))
        values = dict(mse=error, original_mse=identity,
                      response_cosine=F.cosine_similarity(delta, desired_delta, dim=-1, eps=1e-8),
                      phase=b["phase"], predicted_phase=b["predicted_phase"], gate=b["phase_gate"],
                      rgb_rms=rgb.square().mean(-1).sqrt(), rgb_max=rgb.abs().amax(-1),
                      outside_max=torch.stack(outside).amax(0))
        for key, value in values.items():
            collected.setdefault(key, []).append(value.cpu())
    values = {key: torch.cat(value) for key, value in collected.items()}
    def aggregate(mask):
        if not mask.any():
            return dict(observations=0)
        mse = values["mse"][mask].mean().item()
        original = values["original_mse"][mask].mean().item()
        outside = values["outside_max"][mask].max().item()
        if outside != 0:
            raise AssertionError("RGB changed outside frozen phase/spatial support")
        return dict(observations=int(mask.sum()), token_mse=mse,
                    original_same_readout_mse=original,
                    rendering_gain=1 - mse / original if original > 1e-10 else None,
                    rendering_response_cosine=values["response_cosine"][mask].mean().item()
                    if original > 1e-10 else None,
                    rgb_rms=values["rgb_rms"][mask].mean().item(),
                    rgb_max=values["rgb_max"][mask].max().item(), outside_support_max=outside)
    phase = values["phase"]
    return {"all": aggregate(torch.ones_like(phase, dtype=torch.bool)),
            **{name: aggregate(phase == k) for k, name in enumerate(("free", "precontact", "contact"))},
            "true_open": aggregate(phase != 0),
            "true_and_predicted_open": aggregate((phase != 0) & (values["gate"] > 0)),
            "predicted_closed": aggregate(values["gate"] == 0)}


def write_plot(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for name in dict.fromkeys(row["arm"] for row in history):
        steps = sorted({row["step"] for row in history if row["arm"] == name})
        for axis, split in zip(axes, ("train", "validation")):
            vals = [[row[split]["true_open"]["token_mse"] for row in history
                     if row["arm"] == name and row["step"] == step] for step in steps]
            mean = np.asarray([np.mean(v) for v in vals])
            std = np.asarray([np.std(v) for v in vals])
            axis.plot(steps, mean, label=name)
            axis.fill_between(steps, mean - std, mean + std, alpha=0.13)
            axis.set_yscale("log")
    for axis, title in zip(axes, ("Training episodes", "Held-out episodes")):
        axis.set(title=title, xlabel="Updates", ylabel="Open-phase final context MSE (log)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("data/debug") / time.strftime("%Y%m%d_%H%M%S_student_ffn"),
                   help="Result directory (default: data/debug/<timestamp>_student_ffn)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--train-episodes", type=int, default=24)
    p.add_argument("--validation-episodes", type=int, default=6)
    p.add_argument("--per-phase", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--evaluate-every", type=int, default=250)
    args = p.parse_args()
    if min(args.steps, args.train_episodes, args.validation_episodes, args.per_phase,
           args.batch_size, args.evaluate_every) < 1:
        p.error("Steps and sample counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    dataset = hydra.utils.instantiate(ds_cfg)
    validation = dataset.get_validation_dataset()
    train_ids, train_records = choose_samples(dataset, args.train_episodes, args.per_phase)
    val_ids, val_records = choose_samples(validation, args.validation_episodes, args.per_phase)
    assert not ({r["episode"] for r in train_records} & {r["episode"] for r in val_records})
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    teacher_checkpoint=str(config.teacher_checkpoint), history_steps=int(encoder.wrench_history_steps),
                    teacher_context_shape=[teacher.n_obs_steps * sum(encoder.patch_counts.values()), encoder.n_emb],
                    train=train_records, validation=val_records,
                    target="Unmasked frozen teacher final context_norm; same target in every arm",
                    initialization="Fresh identical RenderNet per seed; FFN norms/weights cloned from teacher",
                    ffn="Two residual LayerNorm-MLP blocks (teacher width), before frozen context_norm; dropout disabled",
                    phase="Frozen causal phase prediction gates both RGB and FFNs; free bypasses both",
                    controls="Original RGB through the SAME readout; FFN-only arm; torque interventions fix phase/spatial gates",
                    loss_scope="Offline precontact/contact observations only, normalized by selected token count",
                    evaluation="Sampled, episode-disjoint validation; one selected checkpoint per seed/arm, no robot evaluation")
    write_json(args.output / "manifest.json", manifest)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for file in [Path(__file__), Path(__file__).with_name("quick_validate_target_renderer.py")]:
        shutil.copy2(file, snapshot / file.name)
    cache = dict(train=make_cache(teacher, gate_model, dataset, train_ids, device, args, "train"),
                 validation=make_cache(teacher, gate_model, validation, val_ids, device, args, "validation"))
    del gate_model, dataset, validation
    active = torch.nonzero(cache["train"]["phase_gate"].any(dim=1), as_tuple=True)[0]
    if not len(active):
        raise ValueError("No open causal training windows")
    history, results = [], []
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        reference = build_render_net(encoder.wrench_projection.in_features,
                                    OmegaConf.to_container(config.render_net, resolve=True)).to(device)
        reference.initialize_wrench_encoder(encoder)
        reference.set_training_stage("rgb")
        variants = {
            "no_ffn": ExperimentArm(copy.deepcopy(reference), None, encoder.image_keys),
            "frozen_teacher_ffn": ExperimentArm(copy.deepcopy(reference), PhaseGatedFFN(encoder), encoder.image_keys),
            "learnable_student_ffn": ExperimentArm(copy.deepcopy(reference), PhaseGatedFFN(encoder, trainable=True), encoder.image_keys),
            "ffn_only_no_render": ExperimentArm(None, PhaseGatedFFN(encoder, trainable=True), encoder.image_keys),
        }
        del reference
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        plan = [active[torch.randint(len(active), (args.batch_size,), generator=generator, device=device)]
                for _ in range(args.steps)]
        for name, arm in variants.items():
            arm.to(device)
            torch.manual_seed(seed + 2)
            parameters = [v for v in arm.parameters() if v.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
            checkpoint = args.output / f"{name}_seed{seed}_best.pt"
            def record(step):
                row = dict(seed=seed, arm=name, step=step, elapsed_seconds=time.monotonic() - started,
                           train=evaluate(arm, cache["train"], encoder, args.batch_size),
                           validation=evaluate(arm, cache["validation"], encoder, args.batch_size))
                history.append(row)
                write_json(args.output / "metrics.json", history)
                emit("evaluation", seed=seed, arm=name, step=step,
                     train_open=row["train"]["true_open"], validation_open=row["validation"]["true_open"])
                return row
            best_row = record(0)
            best = best_row["validation"]["true_open"]["token_mse"]
            torch.save(dict(model=arm.state_dict(), arm=name, seed=seed, step=0, manifest=manifest), checkpoint)
            for step, ids in enumerate(plan, 1):
                arm.train(); optimizer.zero_grad(set_to_none=True)
                b = select_batch(cache["train"], ids)
                if arm.renderer is None:
                    z = apply_readout(encoder, arm, b["pre_norm"], b["phase_gate"])
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        rendered = arm(b)
                    z = encode_rendered(encoder, arm, rendered, b["phase_gate"], input_grad=True)
                loss, _, _ = masked_token_mse(z, b["target"], phase_token_mask(z, b["phase"]))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite context loss")
                loss.backward()
                if step == 1 and ((b["phase"] != 0) & (b["phase_gate"] > 0)).any():
                    if arm.renderer is not None and arm.renderer.output.weight.grad.abs().sum() <= 0:
                        raise AssertionError("No RGB output gradient")
                    if arm.readout is not None and arm.readout.learnable:
                        if not any(v.grad is not None and v.grad.abs().sum() > 0 for v in arm.readout.parameters()):
                            raise AssertionError("No student FFN gradient")
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    emit("train", seed=seed, arm=name, step=step, loss=float(loss.detach()),
                         elapsed_seconds=time.monotonic() - started)
                if step % args.evaluate_every == 0 or step == args.steps:
                    row = record(step)
                    score = row["validation"]["true_open"]["token_mse"]
                    if score < best:
                        best, best_row = score, row
                        torch.save(dict(model=arm.state_dict(), arm=name, seed=seed, step=step, manifest=manifest), checkpoint)
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            arm.load_state_dict(state["model"])
            result = dict(seed=seed, arm=name, best_step=state["step"],
                          trainable_parameters=sum(v.numel() for v in parameters),
                          train=best_row["train"], validation=best_row["validation"])
            if arm.renderer is not None:
                result["torque_interventions"] = {
                    mode: evaluate(arm, cache["validation"], encoder, args.batch_size, intervention=mode)
                    for mode in ("zero_torque", "shuffled_torque")}
            results.append(result)
            write_json(args.output / "results.json", dict(manifest=manifest, results=results,
                       elapsed_seconds=time.monotonic() - started))
        del variants
    assert not any(v.requires_grad or v.grad is not None for v in teacher.parameters())
    write_plot(args.output, history)
    emit("complete", output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
