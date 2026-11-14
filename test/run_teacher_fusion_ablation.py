"""Controlled short teacher-only ablation, with clear held-out intervention tests.

python test/run_teacher_fusion_ablation.py --variant baseline --output-dir RUN/baseline
"""

from pathlib import Path
import argparse
import contextlib
import copy
import hashlib
import json
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def common_initialization_hash(policy):
    return tensor_hash({name: value for name, value in policy.state_dict().items()
                        if not name.startswith("obs_encoder.backbone.")
                        and not any(key in name for key in (
                            "cross_attention.", "image_norm.", "force_norm.",
                            "torque_pool.", "film_layers.",
                        ))})


def episode_ids(dataset):
    anchors = np.array([dataset._anchor_index_from_sampler_row(row) for row in dataset.indices])
    return np.unique(dataset.episode_index[anchors]).tolist()


def prepare_data(cfg, policy, image_cache=None):
    if image_cache is not None:
        cfg.task.dataset.preload_images = False
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    cache_metadata = None
    if image_cache is not None:
        if (image_cache / "metadata.json").exists():
            cache_metadata = json.loads((image_cache / "metadata.json").read_text())
            root = {key: np.load(image_cache / f"{key}.npy", mmap_mode="r") for key in dataset.rgb_keys}
        else:
            import zarr
            root = zarr.open(str(image_cache), mode="r")
            cache_metadata = root.attrs["metadata"]
        assert cache_metadata["dataset_length"] == dataset.dataset_length
        assert cache_metadata["rgb_keys"] == list(dataset.rgb_keys)
        assert cache_metadata["extra"]["image_feature_map"] == dict(dataset.image_feature_map)
        for key in dataset.rgb_keys:
            assert tuple(root[key].shape) == (dataset.dataset_length, *cfg.shape_meta.obs[key].shape)
            assert root[key].dtype == np.uint8
        # Verify actual decoded pixels, rather than trusting a cache name or
        # its old machine-specific absolute dataset path.
        indices = [0, dataset.dataset_length // 2, dataset.dataset_length - 1]
        decoded = dataset._build_image_chunk(indices, {key: tuple(cfg.shape_meta.obs[key].shape)
                                                      for key in dataset.rgb_keys})
        for key in dataset.rgb_keys:
            actual = np.rint(np.clip(decoded[key], 0., 1.) * 255).astype(np.uint8)
            np.testing.assert_array_equal(actual, np.asarray(root[key][indices]))
        dataset.image_data = {key: root[key] for key in dataset.rgb_keys}
        dataset.lerobot_dataset.close()
        dataset._video_decoder = None
    dataset.set_contact_label_config(WrenchPreprocessor.from_policy(policy).get_config())
    validation = dataset.get_validation_dataset()
    train_ids, val_ids = episode_ids(dataset), episode_ids(validation)
    if len(train_ids) != cfg.ablation.train_episodes or len(val_ids) != cfg.ablation.validation_episodes:
        raise ValueError(f"Unexpected episode split: training={train_ids}, validation={val_ids}")
    assert not set(train_ids) & set(val_ids)
    normalizer = dataset.get_normalizer(**cfg.task.normalizer)
    policy.set_normalizer(normalizer)
    # Materialize only cheap causal torque windows. Donors are a deterministic
    # derangement within each anchor phase, shared by all three variants.
    original_rgb = validation.rgb_keys
    validation.rgb_keys = ()
    torque = torch.stack([torch.as_tensor(validation[i]["obs"][policy.wrench_key])
                          for i in range(len(validation))])
    validation.rgb_keys = original_rgb
    phases = validation.phase_labels.copy()
    permutation = np.arange(len(validation))
    rng = np.random.default_rng(cfg.ablation.seed + 1103)
    for phase in (0, 1, 2):
        ids = rng.permutation(np.flatnonzero(phases == phase))
        if len(ids) > 1:
            permutation[ids] = np.roll(ids, 1)
    manifest = dict(
        train_episode_ids=train_ids, validation_episode_ids=val_ids,
        train_windows=len(dataset), validation_windows=len(validation),
        train_phase_counts=np.bincount(dataset.phase_labels, minlength=3).tolist(),
        validation_phase_counts=np.bincount(phases, minlength=3).tolist(),
        split_sha256=hashlib.sha256(dataset.indices.tobytes() + validation.indices.tobytes()).hexdigest(),
        normalizer_sha256=tensor_hash(normalizer.state_dict()),
        shuffle_sha256=hashlib.sha256(permutation.tobytes()).hexdigest(),
        shuffle="Fixed derangement within anchor phase; image, alpha and phase held fixed",
        decoded_cache_verified_against_source_pixels=image_cache is not None,
        decoded_cache_metadata=cache_metadata,
    )
    return dataset, validation, torque[permutation], manifest


def to_device(batch, device):
    return {key: to_device(value, device) if isinstance(value, dict)
            else torch.as_tensor(value).to(device, non_blocking=True) for key, value in batch.items()}


def grad_statistics(encoder):
    result = {}
    for group, selectors in {
        "cross_attention": ("cross_attention.",),
        "film": ("film_layers.",),
        "torque_pool": ("torque_pool.",),
        "lstm": ("force_encoder.",),
    }.items():
        gradients = [p.grad.detach().float() for name, p in encoder.named_parameters()
                     if p.grad is not None and any(s in name for s in selectors)]
        if gradients:
            square_sum = sum(g.square().sum() for g in gradients)
            result[group + "/gradient_norm"] = float(square_sum.sqrt())
            result[group + "/gradient_rms"] = float((square_sum / sum(g.numel() for g in gradients)).sqrt())
        else:
            result[group + "/gradient_norm"] = 0.
            result[group + "/gradient_rms"] = 0.
    return result


def denoising_inputs(policy, action, seed):
    generator = torch.Generator(device=action.device).manual_seed(seed)
    normalized = policy.normalizer["action"].normalize(action)
    noise = torch.randn(normalized.shape, device=action.device, generator=generator)
    timesteps = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps,
                              (len(action),), device=action.device, generator=generator)
    return policy.noise_scheduler.add_noise(normalized, noise, timesteps), timesteps, noise


@torch.no_grad()
def evaluate(policy, dataset, shuffled_torque, cfg, device, dino_cache, *, sample_actions=False):
    policy.eval()
    encoder = policy.obs_encoder
    encoder.capture_fusion_diagnostics = False
    loader = DataLoader(dataset, batch_size=cfg.val_dataloader.batch_size, shuffle=False, num_workers=0)
    rows = []
    visual_probes = []
    anchor_phases = dataset.phase_labels
    offset = 0
    for batch_index, raw in enumerate(loader):
        batch = to_device(raw, device)
        obs, action = batch["obs"], batch["action"]
        count = len(action)
        images, _, _, _ = policy.prepare_context_inputs(obs)
        # Frozen DINO outputs are identical across torque interventions and
        # validation steps. Cache only clear images, before trainable projection.
        if batch_index not in dino_cache:
            flat = torch.cat([images[key].flatten(0, 1) for key in encoder.image_keys])
            cls, patches = encoder._encode_dino(flat)
            dino_cache[batch_index] = (cls.cpu(), patches.cpu())
        cached = tuple(value.to(device, non_blocking=True) for value in dino_cache[batch_index])
        original_encode = encoder._encode_dino
        encoder._encode_dino = lambda image, **kwargs: cached
        zero_obs = {**obs, policy.wrench_key: torch.zeros_like(obs[policy.wrench_key])}
        shuffled_obs = {**obs, policy.wrench_key: shuffled_torque[offset:offset + count].to(device)}
        # Keep alpha fixed: isolate torque content, rather than the opening
        # of a gate. A separate alpha=0 test is part of the preflight.
        try:
            contexts = [policy._encode_context(item, apply_curriculum_mask=False)
                        for item in (obs, zero_obs, shuffled_obs)]
        finally:
            encoder._encode_dino = original_encode
        noisy, timesteps, noise = denoising_inputs(policy, action, cfg.ablation.seed + 100000 + batch_index)
        predictions = [policy.model(noisy, timesteps, context) for context in contexts]
        errors = [(prediction.float() - noise).square().flatten(1).mean(1) for prediction in predictions]
        response_sq = (contexts[0].float() - contexts[1].float()).square().flatten(1).mean(1)
        action_values = None
        if sample_actions:
            samples = []
            for context in (contexts[0], contexts[2]):
                generator = torch.Generator(device=device).manual_seed(cfg.ablation.seed + 200000 + batch_index)
                samples.append(policy.conditional_sample(action.shape, context=context, generator=generator))
            start = policy.action_start_index
            stop = start + policy.n_action_steps
            actual = policy.normalizer["action"].unnormalize(samples[0])[:, start:stop]
            changed = policy.normalizer["action"].unnormalize(samples[1])[:, start:stop]
            ground_truth = action[:, start:stop]
            action_values = [value.flatten(1).mean(1) for value in (
                (actual - changed).square(), (actual - ground_truth).square(), (changed - ground_truth).square(),
            )]
        for index in range(count):
            row = dict(index=offset + index, phase=int(anchor_phases[offset + index]),
                       episode=int(dataset.episode_index[dataset._anchor_index_from_sampler_row(
                           dataset.indices[offset + index])]),
                       denoising_mse=float(errors[0][index]), zero_denoising_mse=float(errors[1][index]),
                       shuffled_denoising_mse=float(errors[2][index]), context_response_square=float(response_sq[index]))
            if action_values is not None:
                row.update(action_difference_square=float(action_values[0][index]),
                           action_mse=float(action_values[1][index]), shuffled_action_mse=float(action_values[2][index]))
            rows.append(row)
        if sample_actions and len(visual_probes) < 6:
            for index in range(count):
                if anchor_phases[offset + index] != 2 or any(p["episode"] == int(dataset.episode_index[
                        dataset._anchor_index_from_sampler_row(dataset.indices[offset + index])]) for p in visual_probes):
                    continue
                delta = (contexts[0][index].float() - contexts[1][index].float()).square().mean(-1).sqrt()
                visual_probes.append(dict(index=offset + index, episode=int(dataset.episode_index[
                    dataset._anchor_index_from_sampler_row(dataset.indices[offset + index])]),
                    images={key: value[index].cpu() for key, value in images.items()},
                    response=delta.cpu(), action=actual[index].cpu(), shuffled_action=changed[index].cpu(),
                    ground_truth=ground_truth[index].cpu()))
        offset += count
    def aggregate(selected):
        result = {key: float(np.mean([row[key] for row in selected])) for key in selected[0]
                  if key not in {"index", "phase", "episode"}}
        result["context_response_rms"] = math.sqrt(result.pop("context_response_square"))
        result["shuffle_denoising_penalty"] = result["shuffled_denoising_mse"] - result["denoising_mse"]
        result["zero_denoising_penalty"] = result["zero_denoising_mse"] - result["denoising_mse"]
        if "action_difference_square" in result:
            result["action_prediction_difference_rms"] = math.sqrt(result.pop("action_difference_square"))
            result["shuffle_action_penalty"] = result["shuffled_action_mse"] - result["action_mse"]
        result["samples"] = len(selected)
        return result
    result = {"all": aggregate(rows)}
    for phase, name in enumerate(("free", "precontact", "contact")):
        selected = [row for row in rows if row["phase"] == phase]
        if selected:
            result[name] = aggregate(selected)
    return result, rows, visual_probes


def checkpoint(path, policy, ema, optimizer, cfg, step, metrics):
    # Frozen DINO is referenced by the pinned local model path; saving it on
    # every short-run checkpoint would duplicate hundreds of MB unchanged.
    def trainable_state(model):
        return {name: value.cpu() for name, value in model.state_dict().items()
                if not name.startswith("obs_encoder.backbone.")}
    torch.save(dict(model=trainable_state(policy), ema=trainable_state(ema), optimizer=optimizer.state_dict(),
                    config=OmegaConf.to_container(cfg, resolve=True), optimizer_step=step, metrics=metrics), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "film_only", "hybrid"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="Validation only: load EMA, run the same five episodes, never optimize")
    parser.add_argument("--insert-zero-film", action="store_true",
                        help="Allow only new, zero-initialized Hybrid FiLM parameters when loading Baseline")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    overrides = [] if args.steps is None else [f"ablation.optimizer_steps={args.steps}"]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "diffusion_policy/config")):
        cfg = compose(config_name=f"teacher_fusion_ablation_{args.variant}", overrides=overrides)
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, output / "config.yaml")
    seed_all(cfg.ablation.seed)
    policy = hydra.utils.instantiate(cfg.policy)
    common_hash = common_initialization_hash(policy)
    dataset, validation, shuffled_torque, manifest = prepare_data(cfg, policy, args.image_cache)
    manifest.update(variant=args.variant, common_initialization_sha256=common_hash,
                    seed=cfg.ablation.seed, optimizer_steps=cfg.ablation.optimizer_steps,
                    action_head="8-layer DiT, epsilon prediction, 8-step DDIM sampling",
                    alpha_zero_semantics="Original image-only FFN path, torque-independent",
                    validation="All valid windows from exactly five held-out complete episodes; clear RGB",
                    trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    policy.to(device)
    if args.evaluate_checkpoint is not None:
        payload = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=False)
        loaded = policy.load_state_dict(payload["ema"], strict=False)
        allowed = ["obs_encoder.backbone."]
        if args.insert_zero_film:
            assert args.variant == "hybrid"
            allowed += ["obs_encoder.film_layers.", "obs_encoder.torque_pool."]
            assert all(torch.count_nonzero(layer.mlp[-1].weight) == 0
                       and torch.count_nonzero(layer.mlp[-1].bias) == 0 for layer in policy.obs_encoder.film_layers)
        assert not loaded.unexpected_keys, loaded.unexpected_keys
        assert all(any(name.startswith(prefix) for prefix in allowed) for name in loaded.missing_keys), loaded.missing_keys
        step = int(payload["optimizer_step"])
        del payload
        started = time.monotonic()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            metrics, rows, probes = evaluate(policy, validation, shuffled_torque, cfg, device, {}, sample_actions=True)
        record = dict(kind="validation", step=step, metrics=metrics, elapsed_seconds=time.monotonic() - started)
        write_json(output / "summary.json", dict(
            variant=args.variant, optimizer_steps=step, best_validation_denoising_mse=metrics["all"]["denoising_mse"],
            final=metrics, elapsed_seconds=time.monotonic() - started, manifest=manifest,
            validation_only=True, source_checkpoint=str(args.evaluate_checkpoint),
            film_training_steps=0 if args.insert_zero_film else (step if args.variant != "baseline" else None),
            comparison_limit="Hybrid FiLM is an untrained zero-initialized insertion, not a trained ablation" if args.insert_zero_film else None,
        ))
        write_json(output / "per_sample_diagnostics.json", rows)
        torch.save(probes, output / "visual_probes.pt")
        (output / "log.jsonl").write_text(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return
    # EMA shares only the immutable frozen DINO; trainable parameters remain independent.
    ema = copy.deepcopy(policy, memo={id(policy.obs_encoder.backbone): policy.obs_encoder.backbone})
    ema.eval().requires_grad_(False)
    optimizer = policy.get_optimizer(**OmegaConf.to_container(cfg.optimizer, resolve=True))
    parameters = [p for p in policy.parameters() if p.requires_grad]
    train_loader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=True,
                              num_workers=cfg.dataloader.num_workers, pin_memory=True,
                              persistent_workers=cfg.dataloader.num_workers > 0,
                              generator=torch.Generator().manual_seed(cfg.ablation.seed))
    iterator = iter(train_loader)
    amp = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    first_batch = to_device(next(iterator), device)
    encoder = policy.obs_encoder
    with torch.no_grad(), amp():
        obs = first_batch["obs"]
        closed_obs = {**obs, "contact_alpha": torch.zeros_like(obs["contact_alpha"])}
        policy.eval()
        closed = policy._encode_context(closed_obs, apply_curriculum_mask=False)
        visual = encoder.encode_image_context({key: obs[key] for key in encoder.image_keys}, apply_fusion_ffn=True)
        torch.testing.assert_close(closed, visual, rtol=0, atol=0)
        changed_obs = {**closed_obs, policy.wrench_key: torch.randn_like(obs[policy.wrench_key])}
        changed = policy._encode_context(changed_obs, apply_curriculum_mask=False)
        torch.testing.assert_close(closed, changed, rtol=0, atol=0)
    # Real-image check that stage labels select exactly the intended RGB frames.
    seen = []
    handle = encoder.register_forward_pre_hook(
        lambda module, a, k: seen.append({key: value.clone() for key, value in k["images"].items()}), with_kwargs=True)
    try:
        policy.train()
        with amp():
            loss = policy.compute_loss(first_batch, optimizer_step=0,
                                       generator=torch.Generator(device=device).manual_seed(4242))
        loss.backward()
    finally:
        handle.remove()
    phase = first_batch["obs"]["contact_phase"]
    selected = (phase == 1) | (phase == 2)
    for key in encoder.image_keys:
        torch.testing.assert_close(seen[0][key][~selected], first_batch["obs"][key][~selected], rtol=0, atol=0)
        if selected.any():
            assert not torch.equal(seen[0][key][selected], first_batch["obs"][key][selected])
    preflight = dict(alpha_zero_matches_visual_ffn_exactly=True,
                     alpha_zero_torque_independent_exactly=True, phase_blur_selection_correct=True,
                     precontact_samples=int((phase == 1).sum()), contact_samples=int((phase == 2).sum()),
                     initial_loss=float(loss.detach()), gradients=grad_statistics(encoder))
    write_json(output / "preflight.json", preflight)
    optimizer.zero_grad(set_to_none=True)
    if args.preflight_only:
        print("PREFLIGHT PASSED", flush=True)
        return
    # Restart all random streams so the preflight consumes no training data/noise.
    seed_all(cfg.ablation.seed)
    del iterator
    train_loader.generator.manual_seed(cfg.ablation.seed)
    iterator = iter(train_loader)
    dino_cache = {}
    best = float("inf")
    ema_parameters = dict(ema.named_parameters())
    started = time.monotonic()
    records = []
    def validation_at(step, final=False):
        nonlocal best
        with amp():
            metrics, rows, probes = evaluate(ema, validation, shuffled_torque, cfg, device, dino_cache,
                                             sample_actions=final)
        record = dict(kind="validation", step=step, metrics=metrics, elapsed_seconds=time.monotonic() - started)
        records.append(record)
        with (output / "log.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if metrics["all"]["denoising_mse"] < best:
            best = metrics["all"]["denoising_mse"]
            checkpoint(output / "best.ckpt", policy, ema, optimizer, cfg, step, metrics)
        write_json(output / "latest_validation.json", record)
        if final:
            write_json(output / "per_sample_diagnostics.json", rows)
            torch.save(probes, output / "visual_probes.pt")
            write_json(output / "summary.json", dict(variant=args.variant, optimizer_steps=step,
                       best_validation_denoising_mse=best, final=metrics,
                       elapsed_seconds=time.monotonic() - started, manifest=manifest))
        print(json.dumps(record), flush=True)
        return metrics
    validation_at(0)
    for step in range(1, cfg.ablation.optimizer_steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        batch = to_device(raw, device)
        policy.train()
        encoder.capture_fusion_diagnostics = step == 1 or step % cfg.ablation.log_every == 0
        optimizer.zero_grad(set_to_none=True)
        warmup = cfg.training.lr_warmup_steps
        multiplier = min(1., step / warmup) if step <= warmup else .5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, cfg.ablation.optimizer_steps - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = cfg.optimizer.learning_rate * multiplier
        with amp():
            loss = policy.compute_loss(batch, optimizer_step=step - 1,
                generator=torch.Generator(device=device).manual_seed(cfg.ablation.seed + step))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        diagnostics = grad_statistics(encoder) if encoder.capture_fusion_diagnostics else {}
        if diagnostics:
            diagnostics.update({key: float(value) for key, value in encoder.last_fusion_diagnostics.items()})
        torch.nn.utils.clip_grad_norm_(parameters, cfg.training.max_grad_norm)
        optimizer.step()
        decay = min(.9999, 1 - (1 + max(0, step - 1)) ** (-.75))
        with torch.no_grad():
            for name, value in policy.named_parameters():
                if value.requires_grad:
                    ema_parameters[name].lerp_(value, 1 - decay)
        if diagnostics:
            record = dict(kind="train", step=step, loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"],
                          blur_sigma=policy.blur_scheduler.sigma(step - 1), diagnostics=diagnostics,
                          elapsed_seconds=time.monotonic() - started)
            records.append(record)
            with (output / "log.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % cfg.ablation.validation_every == 0 or step == cfg.ablation.optimizer_steps:
            metrics = validation_at(step, final=step == cfg.ablation.optimizer_steps)
            checkpoint(output / "latest.ckpt", policy, ema, optimizer, cfg, step, metrics)
    write_json(output / "log.json", records)


if __name__ == "__main__":
    main()
