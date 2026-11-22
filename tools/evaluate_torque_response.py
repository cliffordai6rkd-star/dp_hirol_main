"""Evaluate paired teacher/student torque responses with fixed images and gates.

No optimization runs here. Checkpoints are from the completed FFN experiment.
All interventions replace preprocessed torque histories only; the teacher's
phase gate and the student's causal phase/spatial gates remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net
from tools.quick_validate_student_ffn import (
    ExperimentArm, PhaseGatedFFN, apply_readout, encode_rendered,
)
from tools.quick_validate_target_renderer import choose_samples, emit, select_batch, write_json


def contact_donors(phases, episodes):
    """Choose deterministic contact histories from a different episode."""
    flat_phase = phases.cpu().flatten()
    flat_episode = episodes.cpu()[:, None].expand_as(phases).flatten()
    pools = {int(e): torch.where((flat_phase == 2) & (flat_episode == e))[0]
             for e in flat_episode.unique()}
    pools = {e: ids for e, ids in pools.items() if len(ids)}
    if len(pools) < 2:
        raise ValueError("Cross-episode contact interventions need two contact episodes")
    chosen = []
    for i, e in enumerate(flat_episode.tolist()):
        eligible = sorted(k for k in pools if k != e)
        donor_episode = eligible[i % len(eligible)]
        pool = pools[donor_episode]
        chosen.append(int(pool[(37 * i + 17) % len(pool)]))
    return torch.tensor(chosen).reshape_as(phases)


def response_values(teacher_delta, student_delta):
    """One vector per observation, with features/patches flattened last."""
    t, s = teacher_delta.float(), student_delta.float()
    te, se = t.square().mean(-1), s.square().mean(-1)
    dot = (t * s).mean(-1)
    cosine = (dot / (te * se).sqrt().clamp_min(1e-20)).clamp(-1, 1)
    return dict(teacher_energy=te, student_energy=se,
                error_energy=(s - t).square().mean(-1), cosine=cosine,
                amplitude_ratio=(se / te.clamp_min(1e-20)).sqrt())


def aggregate_response(values, selected, *, noise_floor=1e-7):
    count = int(selected.sum())
    if not count:
        return dict(observations=0)
    t = values["teacher_energy"][selected]
    s = values["student_energy"][selected]
    err = values["error_energy"][selected]
    responsive = t > noise_floor ** 2
    return dict(
        observations=count, responsive_observations=int(responsive.sum()),
        teacher_rms=float(t.mean().sqrt()), student_rms=float(s.mean().sqrt()),
        difference_mse=float(err.mean()),
        # A zero-response student scores NMSE=1; perfect agreement scores 0.
        relative_difference_mse=float(err.sum() / t.sum()) if t.sum() > noise_floor ** 2 else None,
        amplitude_rms_ratio=float((s.sum() / t.sum()).sqrt()) if t.sum() > noise_floor ** 2 else None,
        cosine_mean=float(values["cosine"][selected][responsive].mean()) if responsive.any() else None,
        amplitude_ratio_median=float(values["amplitude_ratio"][selected][responsive].median())
        if responsive.any() else None,
    )


@torch.no_grad()
def make_cache(teacher, gate_model, dataset, records, device, batch_size, workers):
    chunks = {}
    loader = DataLoader(Subset(dataset, [r["dataset_index"] for r in records]),
                        batch_size=batch_size, num_workers=workers, shuffle=False)
    for index, raw in enumerate(loader):
        obs = dict_apply(raw["obs"], lambda x: x.to(device))
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
        _, masks, condition = gate_model(images, wrench, phase_inputs=obs["phase_inputs"],
                                        intensity_only=True, return_condition=True)
        phases = obs["contact_phase"][:, :teacher.n_obs_steps]
        gate = (phases == 1) | (phases == 2)
        target = teacher.obs_encoder(images=images, wrench_history=wrench, fusion_gate=gate)
        if index == 0:
            torch.testing.assert_close(target, teacher._encode_context(obs, apply_curriculum_mask=False),
                                       rtol=1e-5, atol=1e-6)
        values = dict(phase=phases, phase_gate=condition["contact_alpha"].float(),
                      predicted_phase=condition["phase_prediction"], wrench=wrench.float(),
                      target=target.float())
        for key in teacher.obs_encoder.image_keys:
            values["image_" + key] = images[key].float()
            values["gate_" + key] = gate_model._mask_gate(masks[key].flatten(0, 1), images[key]).reshape_as(masks[key])
        for key, value in values.items():
            chunks.setdefault(key, []).append(value.detach())
        if index % 10 == 0:
            emit("cache", completed=min((index + 1) * batch_size, len(records)), total=len(records))
    return {key: torch.cat(parts) for key, parts in chunks.items()}


def subsets(cache):
    p, opened = cache["phase"].cpu(), cache["phase_gate"].cpu() > 0
    result = dict(contact=p == 2, contact_correct_open=(p == 2) & opened,
                  precontact=p == 1, precontact_correct_open=(p == 1) & opened,
                  true_open=(p == 1) | (p == 2),
                  true_and_predicted_open=((p == 1) | (p == 2)) & opened,
                  free=p == 0)
    for name, mask in list(result.items()):
        result["original_probe_" + name] = mask & cache["original_probe"].cpu()[:, None]
    return result


@torch.no_grad()
def evaluate_arm(arm, encoder, cache, teacher_contexts, batch_size):
    collected = {name: [] for name in teacher_contexts}
    for start in range(0, len(cache["phase"]), batch_size):
        batch = select_batch(cache, slice(start, start + batch_size))
        for mode in collected:
            wrench = (torch.zeros_like(batch["wrench"]) if mode == "zero" else
                      batch["other_wrench"] if mode == "other_contact" else batch["wrench"])
            counterfactual = {**batch, "wrench": wrench}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rendered = arm(counterfactual)
            for key in arm.image_keys:
                support = batch["gate_" + key] * batch["phase_gate"][..., None, None, None]
                difference = rendered[key] - batch["image_" + key]
                if torch.where(support == 0, difference, 0).abs().max() != 0:
                    raise AssertionError("Intervention changed RGB outside the fixed support")
            z = encode_rendered(encoder, arm, rendered, batch["phase_gate"]).float()
            collected[mode].append(z.cpu())
    contexts = {key: torch.cat(value) for key, value in collected.items()}
    b, steps = cache["phase"].shape
    results, details = {}, {}
    for pair, (a, baseline) in {"real_vs_zero": ("real", "zero"),
                               "other_vs_zero": ("other_contact", "zero"),
                               "real_vs_other": ("real", "other_contact")}.items():
        t = (teacher_contexts[a] - teacher_contexts[baseline]).reshape(b, steps, -1)
        s = (contexts[a] - contexts[baseline]).reshape_as(t)
        values = response_values(t, s)
        details[pair] = values
        results[pair] = {name: aggregate_response(values, mask) for name, mask in subsets(cache).items()}
    return results, details


def plot_results(output, results, *, split="validation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["arm"] for r in results))
    labels = ["No FFN", "Frozen FFN", "Learnable FFN", "FFN only"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for axes_row, pair in zip(axes, ("real_vs_zero", "real_vs_other")):
        for ax, (metric, title) in zip(axes_row, (("cosine_mean", "Response cosine (higher better)"),
                                               ("amplitude_rms_ratio", "Student / teacher RMS (ideal=1)"),
                                               ("relative_difference_mse", "Response NMSE (zero student=1)"))):
            vals = [[r["responses"][pair]["contact_correct_open"][metric] for r in results if r["arm"] == name]
                    for name in names]
            means = [np.mean(v) for v in vals]
            ax.bar(labels, means, yerr=[np.std(v) for v in vals], capsize=3,
                   color=["#78909c", "#e69f00", "#009e73", "#cc79a7"])
            for x, y in enumerate(means):
                ax.annotate(f"{y:.3g}", (x, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
            ax.axhline(1 if metric != "cosine_mean" else 0, color="black", lw=.7, ls="--")
            ax.set_title(pair.replace("_", " ") + "\n" + title, fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=.2)
    split_label = "Original training" if split == "train" else "Held-out"
    fig.suptitle(f"{split_label} contact observations with correctly open gates; two checkpoint seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "response_comparison.png", dpi=180)
    fig.savefig(output / "response_comparison.pdf")
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("data/debug/20261119_student_ffn_quick_01"))
    parser.add_argument("--output", type=Path, default=Path("data/debug") / time.strftime("%Y%m%d_%H%M%S_torque_response"))
    parser.add_argument("--per-phase", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("train", "validation"), default="validation",
                        help="train reuses the exact windows optimized in the original experiment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    previous = json.loads((args.experiment_dir / "results.json").read_text())
    manifest = previous["manifest"]
    gate_model, _, config, teacher_cfg, device, _ = load_render_net(Path(manifest["arguments"]["student_checkpoint"]), "cuda:0")
    payload, teacher_cfg = common.load_checkpoint_payload(Path(manifest["teacher_checkpoint"]))
    teacher = common.instantiate_policy(payload, teacher_cfg, device, capture_attention=False).eval().requires_grad_(False)
    del payload
    encoder = teacher.obs_encoder
    if teacher.fusion_gate_mode != "phase_binary" or not teacher.preserve_zero_wrench or teacher.contact_fusion_gate:
        raise ValueError("This paired probe expects phase-binary teacher and preserved physical zeros")
    ds_cfg = copy.deepcopy(teacher_cfg.task.dataset)
    with open_dict(ds_cfg):
        ds_cfg.preload_images = False
        ds_cfg.image_randomer_config = None
        ds_cfg.video_backend = "pyav"
        ds_cfg.phase_predictor = config.render_net.phase_predictor
    dataset = hydra.utils.instantiate(ds_cfg)
    validation = dataset.get_validation_dataset()
    if args.split == "train":
        records = manifest["train"]
        selected_dataset = dataset
    else:
        _, expanded = choose_samples(validation, 100000, args.per_phase)
        records = sorted({r["dataset_index"]: r for r in [*expanded, *manifest["validation"]]}.values(),
                         key=lambda r: r["dataset_index"])
        selected_dataset = validation
    original = {r["dataset_index"] for r in manifest["train" if args.split == "train" else "validation"]}
    cache = make_cache(teacher, gate_model, selected_dataset, records, device, args.batch_size, args.workers)
    del gate_model, dataset, validation
    cache["original_probe"] = torch.tensor([r["dataset_index"] in original for r in records], device=device)
    episodes = torch.tensor([r["episode"] for r in records])
    donors = contact_donors(cache["phase"], episodes)
    cache["other_wrench"] = cache["wrench"].flatten(0, 1)[donors.flatten().to(device)].reshape_as(cache["wrench"])
    teacher_contexts = {"real": cache["target"].cpu()}
    for mode in ("zero", "other_contact"):
        parts = []
        for start in range(0, len(records), args.batch_size):
            b = select_batch(cache, slice(start, start + args.batch_size))
            images = {key: b["image_" + key] for key in encoder.image_keys}
            wrench = torch.zeros_like(b["wrench"]) if mode == "zero" else b["other_wrench"]
            phases = b["phase"]
            parts.append(encoder(images=images, wrench_history=wrench,
                                 fusion_gate=(phases == 1) | (phases == 2)).float().cpu())
        teacher_contexts[mode] = torch.cat(parts)
    probe_manifest = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        split=args.split,
        teacher_checkpoint=manifest["teacher_checkpoint"], student_gate_checkpoint=manifest["arguments"]["student_checkpoint"],
        history_steps=encoder.wrench_history_steps, records=records, donor_flat_observation_indices=donors.tolist(),
        conditions=["real", "zero", "other_contact"],
        fixed="Images, teacher true phase gate, student predicted phase gate and original spatial support",
        intervention="Entire preprocessed torque history; physical zero remains normalized zero; donors are contact from another episode",
        note="Teacher zero torque keeps phase fusion OPEN; no retraining; checkpoints precede phase-masked loss change",
        samples={name: int(mask.sum()) for name, mask in subsets(cache).items()},
        teacher_context_rms={name: float(teacher_contexts['real'].reshape(len(records), teacher.n_obs_steps, -1)[mask].square().mean().sqrt())
                             for name, mask in subsets(cache).items() if mask.any()},
    )
    write_json(args.output / "manifest.json", probe_manifest)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for file in [Path(__file__), Path(__file__).with_name("quick_validate_student_ffn.py"),
                 Path(__file__).with_name("quick_validate_target_renderer.py")]:
        shutil.copy2(file, snapshot / file.name)
    results, observations = [], []
    for selected in previous["results"]:
        name, seed = selected["arm"], selected["seed"]
        renderer = None
        if name != "ffn_only_no_render":
            renderer = build_render_net(encoder.wrench_projection.in_features,
                                        OmegaConf.to_container(config.render_net, resolve=True)).to(device)
        readout = None if name == "no_ffn" else PhaseGatedFFN(encoder, trainable=name != "frozen_teacher_ffn")
        arm = ExperimentArm(renderer, readout, encoder.image_keys).to(device)
        checkpoint = args.experiment_dir / f"{name}_seed{seed}_best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        arm.load_state_dict(state["model"], strict=True)
        arm.eval().requires_grad_(False)
        responses, details = evaluate_arm(arm, encoder, cache, teacher_contexts, args.batch_size)
        result = dict(arm=name, seed=seed, checkpoint=str(checkpoint), best_step=state["step"], responses=responses)
        results.append(result)
        for pair, values in details.items():
            for i, record in enumerate(records):
                for slot in range(teacher.n_obs_steps):
                    observations.append(dict(arm=name, seed=seed, pair=pair, episode=record["episode"],
                        anchor=record["anchor"], observation_slot=slot, original_probe=record["dataset_index"] in original,
                        phase=int(cache["phase"][i, slot]), predicted_phase=int(cache["predicted_phase"][i, slot]),
                        **{key: float(value[i, slot]) for key, value in values.items()}))
        write_json(args.output / "results.json", dict(manifest=probe_manifest, results=results,
                                                      elapsed_seconds=time.monotonic() - started))
        emit("evaluated", arm=name, seed=seed, contact=responses["real_vs_zero"]["contact_correct_open"],
             elapsed_seconds=time.monotonic() - started)
        del arm, state, renderer, readout
    with (args.output / "observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
        writer.writeheader(); writer.writerows(observations)
    assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    plot_results(args.output, results, split=args.split)
    emit("complete", output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
