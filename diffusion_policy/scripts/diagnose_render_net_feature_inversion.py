"""Run small-set RenderNet and direct-pixel feature-inversion diagnostics.

The diagnostic fixes a teacher context target ``Z(W)`` and asks four questions:

* A: can the current RenderNet fit a small set when only those samples train?
* B: can direct RGB optimization fit the target through the student's fixed mask?
* C: can direct RGB optimization fit it through the teacher's > threshold region?
* D: can direct RGB optimization fit it when the whole image is available?

All direct pixel experiments use the same bounded residual budget and the same
frozen teacher image encoder.  The selected samples are deterministic and
cover pre-contact and persistent-contact windows.
"""

from __future__ import annotations

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import copy
import json
import math
import pathlib
import random
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net_attention import (
    context_effect_patch_weights,
    context_effect_regions,
    teacher_context_effect_targets,
)
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--student-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=12,
                        help="Total samples: half pre-contact and half persistent contact.")
    parser.add_argument("--steps", type=int, default=300,
                        help="Optimization steps for each selected experiment.")
    parser.add_argument("--pixel-lr", type=float, default=0.05)
    parser.add_argument("--student-lr", type=float, default=3e-4)
    parser.add_argument("--chunk-size", type=int, default=2,
                        help="Samples per frozen-DINO forward during gradient accumulation.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--a-loss", choices=("training", "token"), default="training",
                        help="A loss: existing token+region objective, or token MSE only.")
    parser.add_argument(
        "--experiments", nargs="+",
        choices=("A", "A_dynamic", "A_fixed", "B", "B_fixed", "C", "D"),
                        default=("A", "B", "C", "D"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 4 or args.samples % 2:
        parser.error("--samples must be an even number >= 4")
    if args.steps < 1 or args.chunk_size < 1 or args.log_every < 1:
        parser.error("--steps, --chunk-size and --log-every must be positive")
    if not math.isfinite(args.pixel_lr) or args.pixel_lr <= 0:
        parser.error("--pixel-lr must be positive and finite")
    if not math.isfinite(args.student_lr) or args.student_lr <= 0:
        parser.error("--student-lr must be positive and finite")
    return args


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_teacher(checkpoint, device):
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    teacher = common.instantiate_policy(payload, cfg, device, capture_attention=False)
    del payload
    teacher.eval().requires_grad_(False)
    return teacher, cfg


def _episode_and_anchor(dataset, sample_index):
    row = dataset.indices[sample_index]
    anchor = int(dataset._anchor_index_from_sampler_row(row))
    return int(dataset.episode_index[anchor]), anchor


def _window_lowdim(dataset, sample_index, key):
    sequence = dataset._sample_indices_to_sequence(sample_index)
    if dataset.window_sampling_strategy == "timestamp":
        sequence = dataset._retime_sequence_indices(sequence)
    observations = sequence[: int(dataset.n_obs_steps)]
    return dataset.windowed_lowdim_data[key][observations]


def _sample_phase(dataset, sample_index, *, force_dims, threshold):
    current = _window_lowdim(dataset, sample_index, "wrench")
    future = _window_lowdim(dataset, sample_index, "future_wrench")
    current_contact = bool(np.linalg.norm(current[..., list(force_dims)], axis=-1).max() > threshold)
    future_contact = bool(np.linalg.norm(future[..., list(force_dims)], axis=-1).max() > threshold)
    if future_contact and not current_contact:
        return "precontact"
    if future_contact and current_contact:
        return "persistent_contact"
    return "other"


def select_samples(dataset, teacher, count):
    detector = teacher.contact_detector
    force_dims = tuple(int(x) for x in detector.force_dims)
    threshold = float(detector.contact_threshold)
    groups = {"precontact": [], "persistent_contact": []}
    for sample_index in range(len(dataset)):
        phase = _sample_phase(dataset, sample_index, force_dims=force_dims, threshold=threshold)
        if phase in groups:
            episode, anchor = _episode_and_anchor(dataset, sample_index)
            groups[phase].append((episode, anchor, sample_index))
    per_group = count // 2
    episodes = sorted(set(item[0] for values in groups.values() for item in values))
    if len(episodes) < per_group:
        raise RuntimeError(
            f"Need {per_group} episodes with contact examples, found {len(episodes)}"
        )
    episode_positions = np.linspace(0, len(episodes) - 1, per_group).round().astype(int)
    selected = []
    for phase in ("precontact", "persistent_contact"):
        by_episode = {}
        for item in groups[phase]:
            by_episode.setdefault(item[0], []).append(item)
        phase_episodes = [episode for episode in episodes if episode in by_episode]
        if len(phase_episodes) < per_group:
            raise RuntimeError(f"Only {len(phase_episodes)} episodes have {phase} samples")
        positions = np.linspace(0, len(phase_episodes) - 1, per_group).round().astype(int)
        chosen_episodes = [phase_episodes[int(pos)] for pos in positions]
        for episode in chosen_episodes:
            values = sorted(by_episode[episode])
            if phase == "precontact":
                item = values[-1]  # closest valid pre-contact anchor
            else:
                item = values[len(values) // 2]  # clearly inside contact, not transition noise
            selected.append({"phase": phase, "episode": item[0], "anchor": item[1],
                             "sample_index": item[2]})
    selected.sort(key=lambda item: (item["episode"], item["phase"], item["anchor"]))
    return selected


def load_batch(dataset, selected, device):
    samples = [dataset[item["sample_index"]] for item in selected]
    keys = tuple(samples[0]["obs"])
    obs = {key: torch.stack([sample["obs"][key] for sample in samples]).to(device)
           for key in keys}
    return obs


def _chunks(size, chunk_size):
    for start in range(0, size, chunk_size):
        yield slice(start, min(size, start + chunk_size))


def _images_and_wrench(preprocessor, obs):
    images = {key: obs[key][:, :preprocessor.n_obs_steps]
              for key in ("side", "wrist") if key in obs}
    wrench, _, _ = preprocessor(obs)
    return images, wrench


def _encode_image_context(teacher, images):
    return teacher.obs_encoder.encode_image_context(images, input_grad=True)


def _encode_no_grad(teacher, images, chunk_size):
    values = []
    with torch.no_grad():
        for chunk in _chunks(next(iter(images.values())).shape[0], chunk_size):
            values.append(teacher.obs_encoder.encode_image_context(
                {key: value[chunk] for key, value in images.items()}, input_grad=False,
            ).float())
    return torch.cat(values, dim=0)


def _per_sample_mse(prediction, target):
    return (prediction.float() - target.float()).square().mean(dim=(1, 2))


def _mask_gate(mask, threshold):
    mask = mask.detach().float()
    return torch.where(mask > float(threshold), mask, torch.zeros_like(mask))


def _render_with_gate(original, parameter, gate, budget):
    delta = float(budget) * torch.tanh(parameter) * gate
    return (original + delta).clamp(0.0, 1.0), delta


def _render_with_fixed_gate(model, images, wrench_history, fixed_gate):
    """Render with a detached, precomputed continuous gate.

    This deliberately replaces the gate inside RenderNet's residual formula.
    Applying a mask after rendering would test a different function because it
    would leave the network's own gate in the RGB path.
    """
    if model.rendering_mode != "residual":
        raise ValueError("Fixed-gate diagnostics currently require rendering_mode=residual")
    wrench_tokens = model.encode_wrench_history(wrench_history)
    rendered = {}
    for key, image in images.items():
        if key not in model.image_keys:
            raise KeyError(f"Camera {key!r} is absent from configured image_keys {model.image_keys}")
        camera = model.camera_embedding.weight[model.image_keys.index(key)].reshape(1, 1, -1)
        backbone_forward = (
            model._compiled_backbone_forward
            if model._compiled_backbone_forward is not None
            and model.training and torch.is_grad_enabled()
            else model.backbone
        )
        raw_residual, _ = backbone_forward(
            image.flatten(0, 1), wrench_tokens, camera,
        )
        residual = raw_residual.tanh().reshape_as(image)
        gate = fixed_gate[key].to(device=image.device, dtype=residual.dtype)
        expected_shape = tuple(image.shape[:2]) + (1,) + tuple(image.shape[-2:])
        if tuple(gate.shape) != expected_shape:
            raise ValueError(
                f"Fixed gate for {key!r} has shape {tuple(gate.shape)}, "
                f"expected {expected_shape}"
            )
        rendered[key] = (
            image + model.residual_scale * residual * gate
        ).clamp(0.0, 1.0)
    return rendered


def _direct_pixel_inversion(
    name, teacher, original, target, gate, budget, args, output,
    *, initial_raw_mse=None, initial_student_mse=None, gate_mode=None,
):
    parameter = torch.nn.ParameterDict({
        key: torch.nn.Parameter(torch.zeros_like(value)) for key, value in original.items()
    })
    optimizer = torch.optim.Adam(parameter.parameters(), lr=args.pixel_lr)
    n_samples = next(iter(original.values())).shape[0]
    curve = []
    start_time = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = target.new_zeros(())
        for chunk in _chunks(n_samples, args.chunk_size):
            rendered = {}
            for key in original:
                rendered[key], _ = _render_with_gate(
                    original[key][chunk], parameter[key][chunk], gate[key][chunk], budget,
                )
            prediction = _encode_image_context(
                teacher, rendered,
            )
            loss = F.mse_loss(prediction.float(), target[chunk].float())
            weight = (chunk.stop - chunk.start) / n_samples
            (loss * weight).backward()
            total = total + loss.detach() * weight
        optimizer.step()
        curve.append(float(total))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"[{name}] step {step}/{args.steps} token_mse={float(total):.8f}", flush=True)

    with torch.no_grad():
        rendered, deltas = {}, {}
        for key in original:
            rendered[key], deltas[key] = _render_with_gate(
                original[key], parameter[key], gate[key], budget,
            )
        prediction = _encode_no_grad(teacher, rendered, args.chunk_size)
        per_sample = _per_sample_mse(prediction, target)
    abs_values = torch.cat([value.abs().reshape(n_samples, -1) for value in deltas.values()], dim=1)
    gate_values = torch.cat([value.expand(-1, -1, 3, -1, -1).reshape(n_samples, -1)
                             for value in gate.values()], dim=1)
    result = {
        "name": name,
        "initial_mse": (float(initial_raw_mse) if initial_raw_mse is not None else
                         float(_per_sample_mse(_encode_no_grad(teacher, original, args.chunk_size), target).mean())),
        "initial_student_mse": (None if initial_student_mse is None else float(initial_student_mse)),
        "final_mse": float(per_sample.mean()),
        "per_sample_final_mse": per_sample.cpu().tolist(),
        "curve": curve,
        "steps": args.steps,
        "lr": args.pixel_lr,
        "residual_budget": float(budget),
        "mean_abs_rgb_delta": float(abs_values.mean()),
        "p95_abs_rgb_delta": float(torch.quantile(abs_values.flatten(), 0.95)),
        "max_abs_rgb_delta": float(abs_values.max()),
        "gate_support_fraction": float((gate_values > 0).float().mean()),
        "gate_mode": gate_mode,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    torch.save({"rendered": {key: value.cpu() for key, value in rendered.items()},
                "delta": {key: value.cpu() for key, value in deltas.items()},
                "gate": {key: value.cpu() for key, value in gate.items()},
                "per_sample_mse": per_sample.cpu()}, output / f"{name}_tensors.pt")
    del parameter, optimizer, prediction, rendered, deltas
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result


def _student_overfit(
    teacher, student, preprocessor, batch_obs, original, target,
    target_patch_weights, args, output, *, fixed_gate=None, result_name=None,
    artifact_name=None, initial_raw_mse=None,
):
    if fixed_gate is not None and args.a_loss != "token":
        raise ValueError("A-fixed is defined only for the token-only objective")
    model = copy.deepcopy(student).to(next(iter(original.values())).device)
    # cuDNN RNN backward requires training mode.  Keep that mode for the
    # LSTM, but remove stochasticity from every dropout path for a deterministic
    # small-set capacity test.
    model.train().requires_grad_(True)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    if isinstance(getattr(model, "wrench_encoder", None), torch.nn.LSTM):
        model.wrench_encoder.dropout = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.student_lr)
    n_samples = next(iter(original.values())).shape[0]
    curve = []
    start_time = time.monotonic()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = target.new_zeros(())
        token_total = target.new_zeros(())
        patch_total = target.new_zeros(())
        smooth_total = target.new_zeros(())
        for chunk in _chunks(n_samples, args.chunk_size):
            images = {key: value[chunk] for key, value in original.items()}
            wrench, _, _ = preprocessor({key: value[chunk] for key, value in batch_obs.items()})
            if fixed_gate is None:
                rendered, masks = model(images, wrench, return_aux=True, use_attention_mask=True)
            else:
                masks = {}
                rendered = _render_with_fixed_gate(
                    model, images, wrench,
                    {key: value[chunk] for key, value in fixed_gate.items()},
                )
            prediction = _encode_image_context(teacher, rendered)
            token = F.mse_loss(prediction.float(), target[chunk].float())
            patch_losses = []
            smooth_losses = []
            if args.a_loss == "training" and fixed_gate is None:
                for key, mask in masks.items():
                    target_patch = target_patch_weights[key][chunk].to(mask)
                    batch_size, steps, _, height, width = mask.shape
                    grid = common.patch_grid(
                        target_patch.shape[-1], height, width, teacher.obs_encoder.backbone,
                    )
                    pooled = F.adaptive_avg_pool2d(
                        mask.reshape(batch_size * steps, 1, height, width), grid,
                    ).reshape(batch_size, steps, -1)
                    patch_losses.append(F.smooth_l1_loss(pooled, target_patch))
                    smooth_losses.extend((mask[..., 1:, :].sub(mask[..., :-1, :]).abs().mean(),
                                          mask[..., :, 1:].sub(mask[..., :, :-1]).abs().mean()))
            patch = torch.stack(patch_losses).mean() if patch_losses else token.new_zeros(())
            smooth = torch.stack(smooth_losses).mean() if smooth_losses else token.new_zeros(())
            loss = token + patch + 0.02 * smooth
            weight = (chunk.stop - chunk.start) / n_samples
            (loss * weight).backward()
            total = total + loss.detach() * weight
            token_total = token_total + token.detach() * weight
            patch_total = patch_total + patch.detach() * weight
            smooth_total = smooth_total + smooth.detach() * weight
        optimizer.step()
        curve.append({"loss": float(total), "token_mse": float(token_total),
                      "context_patch_mse": float(patch_total), "smoothness": float(smooth_total)})
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"[A] step {step}/{args.steps} loss={float(total):.8f} token_mse={float(token_total):.8f}", flush=True)

    with torch.no_grad():
        predictions = []
        predicted_masks = {key: [] for key in original}
        for chunk in _chunks(n_samples, args.chunk_size):
            images = {key: value[chunk] for key, value in original.items()}
            wrench, _, _ = preprocessor({key: value[chunk] for key, value in batch_obs.items()})
            if fixed_gate is None:
                rendered, masks = model(images, wrench, return_aux=True, use_attention_mask=True)
            else:
                masks = {}
                rendered = _render_with_fixed_gate(
                    model, images, wrench,
                    {key: value[chunk] for key, value in fixed_gate.items()},
                )
            predictions.append(_encode_no_grad(teacher, rendered, args.chunk_size))
            for key in masks:
                predicted_masks[key].append(masks[key].cpu())
        prediction = torch.cat(predictions, dim=0)
        per_sample = _per_sample_mse(prediction, target)
    if fixed_gate is None:
        mask_values = torch.cat([torch.cat(values, dim=0).reshape(n_samples, -1)
                                 for values in predicted_masks.values()], dim=1)
        threshold = float(model.region_threshold)
    else:
        mask_values = torch.cat([
            value.detach().float().reshape(n_samples, -1)
            for value in fixed_gate.values()
        ], dim=1)
        threshold = 0.0
    result_name = result_name or f"A_student_overfit_{args.a_loss}"
    result = {
        "name": result_name,
        "initial_mse": (float(initial_raw_mse) if initial_raw_mse is not None else
                         float(_per_sample_mse(_encode_no_grad(teacher, original, args.chunk_size), target).mean())),
        # curve[0] is measured before the first optimizer.step().  Keep this
        # separate from initial_mse, which is the raw-image baseline.
        "initial_student_mse": float(curve[0]["token_mse"]),
        "final_mse": float(per_sample.mean()),
        "per_sample_final_mse": per_sample.cpu().tolist(),
        "curve": curve,
        "steps": args.steps,
        "lr": args.student_lr,
        "residual_budget": float(model.residual_scale),
        "predicted_mask_mean": float(mask_values.mean()),
        "predicted_mask_gt_threshold_fraction": float((mask_values > threshold).float().mean()),
        "gate_mode": "dynamic_model_gate" if fixed_gate is None else "fixed_initial_student_gate",
        "fixed_gate_support_fraction": (
            None if fixed_gate is None else float((mask_values > 0).float().mean())
        ),
        "elapsed_seconds": time.monotonic() - start_time,
    }
    artifact_name = artifact_name or "A_student_overfit_model"
    torch.save({"model": model.state_dict(), "per_sample_mse": per_sample.cpu()},
               output / f"{artifact_name}.pt")
    del model, optimizer, prediction
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result


def _save_plot(results, path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not create plot: {exc}", flush=True)
        return
    plt.figure(figsize=(10, 6))
    for result in results:
        curve = result["curve"]
        values = ([item["token_mse"] for item in curve]
                  if curve and isinstance(curve[0], dict) else curve)
        plt.plot(np.arange(1, len(values) + 1), values, label=result["name"])
    plt.yscale("log")
    plt.xlabel("optimization step")
    plt.ylabel("context token MSE")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(args):
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    student_checkpoint = args.student_checkpoint.expanduser().resolve()
    if not teacher_checkpoint.is_file() or not student_checkpoint.is_file():
        raise FileNotFoundError("teacher or student checkpoint does not exist")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is nonempty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    teacher, teacher_cfg = _load_teacher(teacher_checkpoint, device)
    student, preprocessor, student_cfg, _, _, student_info = load_render_net(student_checkpoint, str(device))
    student.eval().requires_grad_(False)
    if not student.high_attention_enabled:
        raise ValueError("B requires a student checkpoint with high_attention_render.enabled=true")
    dataset = common.instantiate_episode_dataset(teacher_cfg, None, "pyav")
    try:
        selected = select_samples(dataset, teacher, args.samples)
        print("Selected samples:", json.dumps(selected, indent=2), flush=True)
        obs = load_batch(dataset, selected, device)
    finally:
        dataset.lerobot_dataset.close()

    images, wrench = _images_and_wrench(preprocessor, obs)
    # Raw context effects provide both the fixed target and the teacher spatial gate.
    target, raw_effect = teacher_context_effect_targets(teacher, obs, baseline="raw_zero")
    target = target.detach().float()
    teacher_masks = context_effect_regions(
        raw_effect, images, teacher.obs_encoder.backbone,
        selection="soft", normalization="image_minmax",
    )
    teacher_gates = {key: _mask_gate(value, 0.5) for key, value in teacher_masks.items()}
    with torch.no_grad():
        _, student_masks = student(images, wrench, return_aux=True, use_attention_mask=True)
    student_threshold = float(student.region_threshold)
    student_gates = {key: _mask_gate(value, student_threshold) for key, value in student_masks.items()}
    full_gates = {key: torch.ones_like(value) for key, value in images.items()}
    target_patch_weights = context_effect_patch_weights(
        raw_effect, selection="soft", normalization="image_minmax",
    )
    budget = float(student.residual_scale)
    original = {key: value.detach().clone() for key, value in images.items()}
    # Keep both baselines explicit.  The raw image is the common reference for
    # all experiments; the trained EMA student's output is the initialization
    # reference for A and a useful scale for the fixed-gate comparison.
    raw_prediction = _encode_no_grad(teacher, original, args.chunk_size)
    raw_mse_per_sample = _per_sample_mse(raw_prediction, target)
    with torch.no_grad():
        student_rendered, _ = student(
            original, wrench, return_aux=True, use_attention_mask=True,
        )
    student_prediction = _encode_no_grad(teacher, student_rendered, args.chunk_size)
    student_mse_per_sample = _per_sample_mse(student_prediction, target)
    raw_initial_mse = float(raw_mse_per_sample.mean())
    student_initial_mse = float(student_mse_per_sample.mean())
    fixed_gate = {
        key: value.detach().float().clone() for key, value in student_gates.items()
    }
    if any(label in args.experiments for label in ("A_fixed", "B_fixed")):
        torch.save(
            {key: value.cpu() for key, value in fixed_gate.items()},
            output / "fixed_initial_student_gate.pt",
        )
    # A uses the same current student parameterization; its loss can be
    # restricted to token MSE for a like-for-like comparison with B-D.
    results = []
    if "A" in args.experiments:
        results.append(_student_overfit(
            teacher, student, preprocessor, obs, original, target, target_patch_weights,
            args, output, initial_raw_mse=raw_initial_mse,
        ))
    if "A_dynamic" in args.experiments:
        results.append(_student_overfit(
            teacher, student, preprocessor, obs, original, target, target_patch_weights,
            args, output, result_name="A_dynamic_student_overfit_token",
            artifact_name="A_dynamic_student_overfit_model",
            initial_raw_mse=raw_initial_mse,
        ))
    if "A_fixed" in args.experiments:
        results.append(_student_overfit(
            teacher, student, preprocessor, obs, original, target, target_patch_weights,
            args, output, fixed_gate=fixed_gate,
            result_name="A_fixed_student_overfit_token",
            artifact_name="A_fixed_student_overfit_model",
            initial_raw_mse=raw_initial_mse,
        ))
    if "B_fixed" in args.experiments:
        results.append(_direct_pixel_inversion(
            "B_fixed_gate_pixels", teacher, original, target, fixed_gate, budget, args, output,
            initial_raw_mse=raw_initial_mse, initial_student_mse=student_initial_mse,
            gate_mode="fixed_initial_student_gate",
        ))
    for label, name, gate in (("B", "B_student_mask_pixels", student_gates),
                              ("C", "C_teacher_gt05_pixels", teacher_gates),
                              ("D", "D_full_image_pixels", full_gates)):
        if label in args.experiments:
            results.append(_direct_pixel_inversion(
                name, teacher, original, target, gate, budget, args, output,
                initial_raw_mse=raw_initial_mse, initial_student_mse=student_initial_mse,
                gate_mode={
                    "B": "fixed_initial_student_gate",
                    "C": "fixed_teacher_gt05_gate",
                    "D": "full_image",
                }[label],
            ))
    report = {
        "teacher_checkpoint": str(teacher_checkpoint),
        "student_checkpoint": str(student_checkpoint),
        "student_state_key": student_info.get("student_state_key"),
        "device": str(device),
        "samples": selected,
        "sample_count": len(selected),
        "steps": args.steps,
        "chunk_size": args.chunk_size,
        "pixel_lr": args.pixel_lr,
        "student_lr": args.student_lr,
        "a_loss": args.a_loss,
        "experiments": args.experiments,
        "direct_pixel_residual_budget": budget,
        "student_mask_threshold": student_threshold,
        "teacher_mask_threshold": 0.5,
        "initial_baselines": {
            "raw_image_mse": raw_initial_mse,
            "raw_image_per_sample_mse": raw_mse_per_sample.cpu().tolist(),
            "student_checkpoint_mse": student_initial_mse,
            "student_checkpoint_per_sample_mse": student_mse_per_sample.cpu().tolist(),
            "student_state_key": student_info.get("student_state_key"),
        },
        "fixed_gate": {
            "saved": any(label in args.experiments for label in ("A_fixed", "B_fixed")),
            "source": "student checkpoint gate computed once before optimization",
            "support_fraction": float(torch.cat([
                value.expand(-1, -1, 3, -1, -1).reshape(value.shape[0], -1)
                for value in fixed_gate.values()
            ]).gt(0).float().mean()),
            "mean": float(torch.cat([
                value.reshape(value.shape[0], -1) for value in fixed_gate.values()
            ]).mean()),
        },
        "target": "teacher final context Z(W); direct RGB path uses frozen teacher obs_encoder.encode_image_context",
        "determinism": "teacher and student eval mode; dropout disabled; teacher weights frozen; FP32 target and loss",
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    _save_plot(results, output / "curves.png")
    print(f"Report: {output / 'report.json'}", flush=True)
    for result in results:
        print(f"{result['name']}: initial={result['initial_mse']:.8f} final={result['final_mse']:.8f} "
              f"raw_reduction={(1 - result['final_mse'] / result['initial_mse']):.2%} "
              f"student_initial={result.get('initial_student_mse')}", flush=True)


if __name__ == "__main__":
    main(parse_args())
