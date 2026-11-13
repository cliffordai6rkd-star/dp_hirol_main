"""Small-contact-set RGB fitting probes, independent of production training.

Compare a checkpoint's frozen-condition RGB path with per-example free-pixel
optimization. This tests optimization feasibility, not a mathematical RGB
representation bound and not held-out generalization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


def select_contact_windows(records, delta, count=5):
    """Prefer one verified training window from each of five episodes."""
    selected, seen = [], set()
    for index, record in enumerate(records):
        if record["split"] != "train" or record["episode"] in seen:
            continue
        phases = torch.as_tensor(record["phase"])
        energy = float(delta[index].float().square().mean())
        if not bool((phases == 2).all()) or energy <= 1e-4:
            continue
        selected.append(index)
        seen.add(record["episode"])
        if len(selected) == count:
            return selected
    raise ValueError(f"Need {count} distinct training episodes with nonzero contact targets; got {len(selected)}")


def effect_weights(delta):
    """Detached, bounded teacher weights; all patches retain positive weight."""
    energy = delta.detach().float().square().mean(-1)
    weights = .5 + .5 * (energy / energy.mean(-1, keepdim=True).clamp_min(1e-12)).clamp(max=10)
    return weights / weights.mean(-1, keepdim=True)


def metric_record(prediction, base, delta, weights):
    prediction, base, delta = prediction.float(), base.float(), delta.float()
    error = (prediction - base - delta).square().mean(-1)
    effect = delta.square().mean(-1)
    top = effect.topk(max(1, math.ceil(effect.shape[-1] * .1)), dim=-1).indices
    top_error, top_energy = error.gather(-1, top), effect.gather(-1, top)
    predicted_delta = prediction - base
    # One cosine per complete observation, so zero-effect token norms do not
    # destabilize a channel-wise average of tiny vectors.
    cosine = F.cosine_similarity(predicted_delta.flatten(-2), delta.flatten(-2), dim=-1)
    baseline = float(effect.mean())
    return {
        "mse": float(error.mean()),
        "zero_residual_mse": baseline,
        "relative_mse": float(error.mean()) / baseline,
        "effect_weighted_mse": float((weights * error).mean()),
        "top10_mse": float(top_error.mean()),
        "top10_zero_residual_mse": float(top_energy.mean()),
        "top10_relative_mse": float(top_error.mean() / top_energy.mean()),
        "delta_cosine": float(cosine.mean()),
        "per_window_mse": error.flatten(1).mean(-1).cpu().tolist(),
    }


def encode(teacher, images, shape):
    return teacher.obs_encoder.encode_image_context(
        images, input_grad=True, apply_fusion_ffn=True,
    ).float().reshape(shape)


def run_mode(mode, teacher, args, indices, cache, output):
    device = next(teacher.parameters()).device
    base = cache["base"][indices].clone().to(device)
    delta = cache["delta"][indices].clone().to(device)
    target = base + delta
    tau = cache["tau"][indices].clone().to(device)
    images = {key: cache["obs"][key][indices, :base.shape[1]].clone().to(device)
              for key in teacher.obs_encoder.image_keys}
    weights = effect_weights(delta)
    count = len(indices)
    model = pixels = None
    frozen = {}
    info = {}
    if mode == "rgb":
        model, preprocessor, cfg, _, _, info = load_render_net(args.student_checkpoint, "cpu")
        del preprocessor
        if not bool(cfg.distillation.get("use_teacher_ffn", False)):
            raise ValueError("Probe requires a student using teacher fusion FFN supervision")
        model.to(device)
        model.set_training_stage("rgb")
        model.set_stage_mode(True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("RGB stage has no trainable parameters")
        frozen = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                  if not value.requires_grad}
        optimizer = torch.optim.AdamW(parameters, lr=args.rgb_lr, weight_decay=1e-5)
    else:
        # Unconstrained per-example pixel deltas. Clamp gives RGB in [0,1].
        # This has no gate/alpha constraints and is intentionally an optimistic
        # optimization probe, not a deployable predictor.
        pixels = nn.ParameterDict({key: nn.Parameter(torch.zeros_like(image))
                                   for key, image in images.items()})
        parameters = list(pixels.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.pixel_lr)

    def render(index):
        current = {key: value[index:index+1] for key, value in images.items()}
        if mode == "rgb":
            # Conditions are model predictions, exactly as at inference. The
            # entire condition path is frozen and left in eval mode.
            return model(current, tau[index:index+1])
        return {key: (value + pixels[key][index:index+1]).clamp(0, 1)
                for key, value in current.items()}

    @torch.no_grad()
    def evaluate(step):
        prediction = torch.cat([encode(teacher, render(i), base[i:i+1].shape)
                                for i in range(count)], dim=0)
        record = {"step": step, "seconds": time.monotonic()-started,
                  **metric_record(prediction, base, delta, weights)}
        print(json.dumps({"mode": mode, **record}), flush=True)
        history.append(record)
        (output / f"{mode}_progress.json").write_text(json.dumps(history, indent=2)+"\n")
        return prediction

    history = []
    started = time.monotonic()
    evaluate(0)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for i in range(count):
            prediction = encode(teacher, render(i), base[i:i+1].shape)
            error = (prediction - target[i:i+1]).square().mean(-1)
            loss = (weights[i:i+1] * error).mean() / count
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {mode} loss at step {step}")
            loss.backward()
            del prediction, loss
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            evaluate(step)
    frozen_unchanged = None
    if model is not None:
        frozen_unchanged = all(torch.equal(value.detach().cpu(), frozen[name])
                               for name, value in model.named_parameters() if name in frozen)
        if not frozen_unchanged:
            raise AssertionError("Frozen condition parameters changed")
    result = {"mode": mode, "history": history,
              "initial": history[0], "final": history[-1],
              "trainable_parameters": sum(p.numel() for p in parameters),
              "frozen_condition_parameters_unchanged": frozen_unchanged,
              "student_state": info,
              "interpretation": "Training-set optimization probe only; no claim about RGB mathematical bound or generalization"}
    (output / f"{mode}_result.json").write_text(json.dumps(result, indent=2)+"\n")
    del optimizer, parameters, model, pixels, frozen, images, base, target, delta, tau
    gc.collect()
    torch.cuda.empty_cache()
    return result


def plot(results, output):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, result in results.items():
        history = result["history"]
        for axis, key in zip(axes, ("relative_mse", "top10_relative_mse", "delta_cosine")):
            axis.plot([row["step"] for row in history], [row[key] for row in history], label=name)
    for axis in axes[:2]:
        axis.axhline(1, color="black", linestyle="--", label="Zero residual")
        axis.set_yscale("log")
    for axis, title in zip(axes, ("Residual relative MSE", "Top 10% effect relative MSE", "Delta direction cosine")):
        axis.set(title=title, xlabel="Optimization step")
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle("Five verified contact windows | frozen teacher FFN image path")
    figure.tight_layout()
    figure.savefig(output / "rgb_capacity.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/diagnostics/tau_capacity_v2/paired_targets.pt"))
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("rgb", "pixels"), default=["rgb", "pixels"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--rgb-lr", type=float, default=2e-4)
    parser.add_argument("--pixel-lr", type=float, default=.01)
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1:
        parser.error("steps and log-every must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.25)
    cache = torch.load(args.data, map_location="cpu", weights_only=False)
    manifest = json.loads(args.data.with_name("data_manifest.json").read_text())
    indices = select_contact_windows(cache["records"], cache["delta"])
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"indices": indices, "records": [cache["records"][i] for i in indices],
                "teacher_checkpoint": manifest["checkpoint"],
                "student_checkpoint": str(args.student_checkpoint.resolve()),
                "student_initialization": "existing EMA checkpoint; RGB-only parameters unfrozen",
                "pixel_initialization": "original image plus zero pixel delta, no gate/alpha constraints",
                "objective": "effect-weighted token MSE; per-observation w=normalize(.5+.5*clip(effect/mean(effect),max=10))",
                "teacher_image_path": "encode_image_context(input_grad=True,apply_fusion_ffn=True)",
                "batch_size": 1, "accumulation": len(indices), "steps": args.steps,
                "lr": {"rgb": args.rgb_lr, "pixels": args.pixel_lr}}
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    payload, cfg = common.load_checkpoint_payload(Path(manifest["checkpoint"]))
    teacher = common.instantiate_policy(payload, cfg, torch.device("cpu"), capture_attention=False)
    del payload
    teacher.eval().requires_grad_(False).to("cuda:0")
    # Verify that cached baseline still matches the current frozen image path.
    baseline_numeric_gaps = []
    with torch.no_grad():
        for index in indices:
            images = {key: cache["obs"][key][index:index+1].to("cuda:0")
                      for key in teacher.obs_encoder.image_keys}
            current = encode(teacher, images, cache["base"][index:index+1].shape)
            expected = cache["base"][index:index+1]
            gap = current.cpu() - expected
            baseline_numeric_gaps.append({"max_abs": float(gap.abs().max()),
                                          "mse": float(gap.square().mean())})
            # Different per-call batch sizes can change FP32 accumulation.
            # This tolerance is far below the verified ~0.02+ target MSE.
            torch.testing.assert_close(current.cpu(), expected, rtol=1e-4, atol=5e-5)
    protocol['baseline_numeric_gaps'] = baseline_numeric_gaps
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    results = {}
    for mode in args.modes:
        results[mode] = run_mode(mode, teacher, args, indices, cache, args.output)
        plot(results, args.output)
    (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
