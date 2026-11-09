"""Exact single-patch CA-update interventions on a frozen teacher's epsilon.

For each observation/camera/patch, replace U_p(W) by U_p(W_ref), preserving
the contact alpha attached to each branch.
Replay residual, FFN and final LayerNorm, then the complete DiT condition
encoder and action decoder at identical normal-trajectory (x_t, t) inputs.
The video shows the mean of per-DDIM-step epsilon L2 differences. All step
maps are cached. This measures local denoiser sensitivity, not an independently
resampled action trajectory or a physical contact location.
"""

from __future__ import annotations

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diffusion_policy.scripts import visualize_wrench_effect_episode as effects


@torch.inference_mode()
def capture_fusion_pair(policy, obs, baseline):
    """Cache the fixed query and both U branches; check exact fusion replay."""
    encoder = policy.obs_encoder
    queries = []
    handle = encoder.image_norm.register_forward_pre_hook(
        lambda module, args: queries.append(args[0].detach().clone())
    )
    try:
        with effects.WrenchEffectCapture(policy) as capture:
            normal = capture.encode(obs)
            reference_obs = effects.make_zero_wrench_obs(policy, obs) if baseline == "raw_zero" else obs
            reference = capture.encode(reference_obs, normalized_zero=baseline == "normalized_zero")
    finally:
        handle.remove()
    if len(queries) != 2:
        raise ValueError("Expected one image-query normalization per encoder call")
    torch.testing.assert_close(queries[0], queries[1], rtol=0, atol=0)
    query = queries[0]
    normal = {key: value.to(device=query.device) for key, value in normal.items()}
    reference = {key: value.to(device=query.device) for key, value in reference.items()}
    for branch in (normal, reference):
        replay = encoder.finish_cross_attention(encoder, query, branch["ca_update"])
        replay = encoder.context_norm(replay)
        torch.testing.assert_close(replay, branch["context"], rtol=1e-5, atol=1e-6)
    return query, normal, reference


@torch.inference_mode()
def replacement_contexts(encoder, query, normal, reference, token_indices):
    """One different (observation, camera, patch) token per batch sample.

    In the supported single-CA/no-SA structure, residual, FFN and LayerNorm
    act independently on each token. Recompute these layers at the changed
    token and reuse exact unchanged tokens; this equals a full encoder replay.
    The DiT condition encoder still recomputes ALL tokens after intervention.
    Alpha is applied inside the same attended-output path as the encoder.
    """
    channels = query.shape[-1]
    flat_query = query.reshape(-1, channels)
    flat_reference = reference["ca_update"].reshape(-1, channels)
    changed = encoder.finish_cross_attention(
        encoder, flat_query[token_indices, None], flat_reference[token_indices, None],
    )
    changed = encoder.context_norm(changed)[:, 0]
    context = normal["context"].reshape(1, -1, channels).expand(len(token_indices), -1, -1).clone()
    context[torch.arange(len(token_indices), device=context.device), token_indices] = changed
    return context


@torch.inference_mode()
def normal_denoising_trace(policy, context, seed):
    """Record the policy's actual normal trajectory without changing kernels."""
    trace = []

    def capture(module, args, output):
        trace.append((args[0].detach().clone(), args[1].detach().clone(), output.detach().clone()))

    handle = policy.model.register_forward_hook(capture)
    try:
        generator = torch.Generator(device=context.device).manual_seed(int(seed))
        policy.conditional_sample((1, policy.horizon, policy.action_dim), context=context, generator=generator)
    finally:
        handle.remove()
    if len(trace) != policy.num_inference_steps:
        raise ValueError("Unexpected number of normal denoising calls")
    return trace


@torch.inference_mode()
def patch_action_sensitivity(policy, obs, *, baseline="raw_zero", seed=42, batch_size=32):
    """Return [DDIM, observation, all-camera-patches] exact intervention norms."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if baseline not in {"raw_zero", "normalized_zero"}:
        raise ValueError("Unsupported baseline")
    if getattr(policy.noise_scheduler.config, "prediction_type", None) != "epsilon":
        raise ValueError("Action sensitivity requires an epsilon-predicting teacher")
    query, normal, reference = capture_fusion_pair(policy, obs, baseline)
    steps, patches, channels = query.shape
    context = normal["context"].reshape(1, steps * patches, channels)
    trace = normal_denoising_trace(policy, context, seed)
    result = torch.zeros((len(trace), steps * patches), dtype=torch.float32, device=context.device)
    # Skip exact no-ops by comparing the final alpha-scaled context.
    changed_mask = (normal["ca_update"] != reference["ca_update"]).any(dim=-1)
    changed_mask = changed_mask | ((normal["context"] != reference["context"]).any(dim=-1))
    changed = changed_mask.flatten().nonzero().flatten()
    noop_max, batch_normal_max = 0., 0.
    for start in range(0, len(changed), batch_size):
        indices = changed[start:start + batch_size]
        replaced = replacement_contexts(policy.obs_encoder, query, normal, reference, indices)
        # Two identical normal samples measure the numerical no-op floor.
        batch_context = torch.cat((context, context, replaced), dim=0)
        for step_index, (trajectory, timestep, serial_prediction) in enumerate(trace):
            prediction = policy.model(trajectory.expand(len(indices) + 2, -1, -1), timestep, batch_context)
            norms = torch.linalg.vector_norm((prediction[2:] - prediction[:1]).float().flatten(1), dim=-1)
            if not torch.isfinite(norms).all():
                raise ValueError("Non-finite action sensitivity")
            result[step_index, indices] = norms
            noop_max = max(noop_max, float(torch.linalg.vector_norm(prediction[1] - prediction[0])))
            batch_normal_max = max(batch_normal_max, float(torch.linalg.vector_norm(prediction[:1] - serial_prediction)))
    return {
        "per_step": result.reshape(len(trace), steps, patches).cpu().numpy(),
        "timesteps": np.asarray([int(timestep) for _, timestep, _ in trace], dtype=np.int64),
        "normal_epsilon": torch.stack([prediction[0] for _, _, prediction in trace]).float().cpu().numpy(),
        "normal_noisy_action": torch.stack([trajectory[0] for trajectory, _, _ in trace]).float().cpu().numpy(),
        "noop_epsilon_l2_max": noop_max,
        "batch_vs_serial_normal_epsilon_l2_max": batch_normal_max,
        "intervened_tokens": len(changed),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True, help="Figure B's *_raw.npz; infer checkpoint, reference and frames.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("batch-size and max-frames must be positive")
    return args


def run(args):
    source = args.source_cache.expanduser().resolve()
    source_stem = source.name.removesuffix("_raw.npz")
    source_metadata = source.with_name(source_stem + "_metadata.json")
    metadata = json.loads(source_metadata.read_text())
    directory = args.output_dir.expanduser().resolve() if args.output_dir else source.parent
    stem = f"wrench_action_sensitivity_episode_{metadata['episode']}"
    if metadata["baseline"] != "raw_zero":
        stem += "_" + metadata["baseline"]
    cache = directory / (stem + "_raw.npz")
    output_metadata = directory / (stem + "_metadata.json")
    if not args.overwrite and any(path.exists() for path in (cache, output_metadata, directory / (stem + "_action_sensitivity.mp4"))):
        raise FileExistsError("Action sensitivity output already exists; use --overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(metadata["checkpoint"])
    payload, cfg = effects.common.load_checkpoint_payload(checkpoint)
    device = effects.common.resolve_device(args.device, cfg)
    policy = effects.common.instantiate_policy(payload, cfg, device, capture_attention=False).float()
    policy.requires_grad_(False)
    del payload
    dataset = effects.common.instantiate_episode_dataset(cfg, args.dataset_path, video_backend="pyav")
    started = time.monotonic()
    results = []
    try:
        with np.load(source, allow_pickle=False) as saved:
            anchors = saved["anchors"][:args.max_frames]
            arrays = {key: saved[key] for key in (
                "recorded_wrench", "wrench_time_sec", "wrench_contact", "wrench_causal_contact", "contact_intervals_sec")}
            arrays["anchors"] = anchors
            arrays["episode_time_sec"] = saved["episode_time_sec"][:len(anchors)]
        lookup = dict(effects.common.episode_sample_indices(dataset, metadata["episode"]))
        for index, anchor in enumerate(anchors):
            sample = dataset[lookup[int(anchor)]]
            obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()}
            result = patch_action_sensitivity(policy, obs, baseline=metadata["baseline"], seed=args.seed, batch_size=args.batch_size)
            results.append(result)
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(anchors):
                elapsed = time.monotonic() - started
                print(f"Intervened {index + 1}/{len(anchors)} frames | {elapsed:.1f}s | "
                      f"ETA {elapsed / (index + 1) * (len(anchors) - index - 1):.0f}s | "
                      f"no-op {result['noop_epsilon_l2_max']:.3g}", flush=True)
        all_keys = tuple(policy.obs_encoder.image_keys)
        offsets = np.cumsum([0] + [int(policy.obs_encoder.patch_counts[key]) for key in all_keys])
        for camera_index, key in enumerate(metadata["raw_camera_keys"]):
            position = all_keys.index(key)
            maps = np.stack([result["per_step"][..., offsets[position]:offsets[position + 1]] for result in results])
            arrays[f"camera_{camera_index}_action_sensitivity_per_step"] = maps
            arrays[f"camera_{camera_index}_action_sensitivity"] = maps.mean(axis=1)
        arrays["ddim_timesteps"] = results[0]["timesteps"]
        for field in ("normal_epsilon", "normal_noisy_action", "noop_epsilon_l2_max", "batch_vs_serial_normal_epsilon_l2_max", "intervened_tokens"):
            arrays[field] = np.stack([result[field] for result in results])
        np.savez_compressed(cache, **arrays)
        output = {key: metadata[key] for key in (
            "checkpoint", "episode", "fps", "dataset_fps", "stride", "baseline", "baseline_definition",
            "raw_camera_keys", "display_camera_keys", "camera_patch_grids", "observation_display", "wrench_plot")}
        output.update({
            "source_cache": str(source), "source_metadata": str(source_metadata),
            "video_frames": len(anchors), "precision": "float32", "seed": args.seed,
            "batch_size": args.batch_size, "elapsed_seconds": time.monotonic() - started,
            "definition": "D_A(t,o,c,p)=L2(epsilon_normal(x_t,t)-epsilon_U[o,c,p]<-U_ref[o,c,p](x_t,t)); full action horizon and channels",
            "intervention": "One observation slot, one camera, one patch per intervention. Replace its CA update and replay fusion; a closed reference uses its pure visual token. Closed normal observations remain unchanged. RGB, other context tokens and x_t/t are fixed.",
            "fusion_replay": "Exact tokenwise residual, FFN, output LayerNorm and context LayerNorm; full and reference replay checked against ordinary encoder outputs. Unchanged tokens are reused. Full DiT condition encoder and all decoder layers rerun for each intervention/timestep.",
            "trajectory": "All original DDIM timesteps from one normal policy rollout; each comparison uses the same normal x_t and t. Interventions do not advance a separate sampler trajectory.",
            "aggregation": "Arithmetic mean of per-timestep L2 norms; NOT L2 of timestep-averaged epsilon differences",
            "ddim_timesteps": arrays["ddim_timesteps"].tolist(),
            "units": "Noise-prediction L2 over normalized action horizon/channels; not physical action units",
            "noop_epsilon_l2_max": float(arrays["noop_epsilon_l2_max"].max()),
            "batch_vs_serial_normal_epsilon_l2_max": float(arrays["batch_vs_serial_normal_epsilon_l2_max"].max()),
            "normal_control": "Normal epsilon is recomputed in the SAME batch as replacements; two identical normal controls measure the no-op floor.",
        })
        output_metadata.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    finally:
        dataset.lerobot_dataset.close()
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    render_args = SimpleNamespace(
        episode=metadata["episode"], output_dir=directory, dataset_path=args.dataset_path,
        video_backend="pyav", metrics=("action_sensitivity",), stride=1, max_frames=None,
        fps=None, camera="both", observation=metadata["observation_display"],
        baseline=metadata["baseline"], color_scope=metadata.get("color_scope", "image"),
        attention_percentiles=metadata.get("color_percentiles", [2., 98.]), alpha=.45,
        video_codec="h264", save_frames=args.save_frames, overwrite=args.overwrite,
    )
    return effects.render_cached_effects(cache, render_args)


if __name__ == "__main__":
    run(parse_args())
