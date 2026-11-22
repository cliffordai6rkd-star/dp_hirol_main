"""Train a small image + tau_ext GRU phase gate and re-score frozen CIVI students.

The gate has no access to phase_inputs, joint position, or velocity.  Each
camera is downsampled by a shared CNN; the checkpoint-configured teacher-preprocessed
tau_ext history is encoded by a GRU.  Their embeddings are concatenated and an
MLP emits free/precontact/contact logits for every observation step.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, open_dict

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net
from tools.full_data_paired_distill import evaluate as evaluate_student
from tools.quick_validate_student_ffn import ExperimentArm, PhaseGatedFFN
from tools.quick_validate_target_renderer import choose_samples, emit, select_batch, write_json
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.image_tau_phase_gate import ImageTauPhaseGate, load_phase_gate


def move_obs(raw_obs, device):
    return dict_apply(raw_obs, lambda value: value.to(device, non_blocking=True))


def teacher_batch(teacher, raw_obs, device):
    obs = move_obs(raw_obs, device)
    images, tau_ext, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
    labels = obs["contact_phase"][:, :teacher.n_obs_steps].long()
    return images, tau_ext.float(), labels


def phase_metrics(logits, labels):
    prediction = logits.argmax(dim=-1)
    confusion = torch.zeros(3, 3, dtype=torch.long, device=logits.device)
    confusion.index_put_((labels.reshape(-1), prediction.reshape(-1)),
                         torch.ones_like(labels.reshape(-1), dtype=torch.long), accumulate=True)
    recalls = []
    for phase in range(3):
        recalls.append(float(confusion[phase, phase] / confusion[phase].sum().clamp_min(1)))
    free_total = int(confusion[0].sum())
    opened_free = int(confusion[0, 1:].sum())
    open_total = int(confusion[1:].sum())
    opened_open = int(confusion[1:, 1:].sum())
    return {
        "accuracy": float((prediction == labels).float().mean()),
        "phase_recall": recalls,
        "free_false_open_rate": opened_free / max(free_total, 1),
        "precontact_open_rate": float((prediction[labels == 1] != 0).float().mean())
        if (labels == 1).any() else None,
        "contact_open_rate": float((prediction[labels == 2] != 0).float().mean())
        if (labels == 2).any() else None,
        "open_recall": opened_open / max(open_total, 1),
        "observations": int(labels.numel()),
        "confusion": confusion.cpu().tolist(),
    }


def run_gate_epoch(model, teacher, loader, device, optimizer=None, class_weights=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    confusion = torch.zeros(3, 3, dtype=torch.long, device=device)
    for raw in loader:
        images, tau_ext, labels = teacher_batch(teacher, raw["obs"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images, tau_ext)
            loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1),
                                   weight=class_weights)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.detach()) * count
        total_count += count
        prediction = logits.detach().argmax(dim=-1)
        confusion.index_put_((labels.reshape(-1), prediction.reshape(-1)),
                             torch.ones_like(labels.reshape(-1), dtype=torch.long), accumulate=True)
    metrics = phase_metrics_from_confusion(confusion)
    metrics["ce_loss"] = total_loss / max(total_count, 1)
    return metrics


def phase_metrics_from_confusion(confusion):
    recalls = [float(confusion[p, p] / confusion[p].sum().clamp_min(1)) for p in range(3)]
    free_total = int(confusion[0].sum())
    opened_free = int(confusion[0, 1:].sum())
    open_total = int(confusion[1:].sum())
    opened_open = int(confusion[1:, 1:].sum())
    return {
        "accuracy": float(torch.diag(confusion).sum() / confusion.sum().clamp_min(1)),
        "phase_recall": recalls,
        "free_false_open_rate": opened_free / max(free_total, 1),
        "precontact_open_rate": float(confusion[1, 1:].sum() / confusion[1].sum().clamp_min(1)),
        "contact_open_rate": float(confusion[2, 1:].sum() / confusion[2].sum().clamp_min(1)),
        "open_recall": opened_open / max(open_total, 1),
        "observations": int(confusion.sum()),
        "confusion": confusion.cpu().tolist(),
    }


@torch.no_grad()
def replace_gate(model, teacher, cache, device, batch_size=32):
    model.eval()
    phases = []
    probabilities = []
    for start in range(0, len(cache["phase"]), batch_size):
        batch = select_batch(cache, slice(start, start + batch_size))
        images = {key: batch["image_" + key].to(device, dtype=torch.float32,
                                                 non_blocking=True)
                  for key in teacher.obs_encoder.image_keys}
        tau = batch["wrench"].to(device, dtype=torch.float32, non_blocking=True)
        logits = model(images, tau)
        probabilities.append(logits.softmax(dim=-1).cpu())
        phases.append(logits.argmax(dim=-1).cpu())
    predicted = torch.cat(phases)
    cache["new_phase_prediction"] = predicted
    cache["new_phase_probabilities"] = torch.cat(probabilities)
    # precontact/contact both open; keep the new gate fixed for all torque
    # counterfactuals in the paired student evaluation.
    cache["phase_gate"] = (predicted != 0).to(torch.float16)
    return cache


def evaluate_gate_effect(student_checkpoint_dir, teacher, student_model, phase_model,
                         teacher_cfg, cfg,
                         device, train_dataset, validation_dataset, output, seed=42):
    from tools.full_data_paired_distill import cache_split as build_cache

    all_results = {}
    for split, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        _, records = choose_samples(dataset, 100000, len(dataset))
        records = [r for r in records if r["phase"] in (1, 2)]
        cache = build_cache(teacher, student_model, dataset, records, device,
                            batch_size=2, workers=4, split=split)
        original_gate = cache["phase_gate"].clone()
        cache = replace_gate(phase_model, teacher, cache, device)
        labels = cache["phase"].long()
        prediction = cache["new_phase_prediction"].long()
        all_results.setdefault("gate", {})[split] = phase_metrics_from_confusion(
            torch.zeros(3, 3, dtype=torch.long).index_put_(
                (labels.reshape(-1), prediction.reshape(-1)),
                torch.ones_like(labels.reshape(-1), dtype=torch.long), accumulate=True,
            )
        )
        all_results.setdefault("students", {})[split] = {}
        gate_predictions = {
            "original_phase_gate": original_gate.bool(),
            "new_image_tau_gate": cache["phase_gate"].bool(),
            "oracle_phase_gate": labels != 0,
        }
        for arm_name in ("absolute_only", "absolute_plus_response"):
            state = torch.load(student_checkpoint_dir / (arm_name + "_endpoint.pt"),
                               map_location="cpu", weights_only=False)
            renderer = build_render_net(teacher.obs_encoder.wrench_projection.in_features,
                                        OmegaConf.to_container(cfg.render_net, resolve=True))
            arm = ExperimentArm(renderer, PhaseGatedFFN(teacher.obs_encoder),
                                teacher.obs_encoder.image_keys).to(device)
            arm.load_state_dict(state["model"], strict=True)
            arm.eval().requires_grad_(False)
            all_results["students"][split][arm_name] = {}
            for gate_name, gate in gate_predictions.items():
                emit("student_evaluation_start", split=split, arm=arm_name, gate=gate_name)
                cache["phase_gate"] = gate.to(dtype=torch.float16)
                all_results["students"][split][arm_name][gate_name] = evaluate_student(
                    arm, teacher.obs_encoder, cache, batch_size=2,
                )
                write_json(output / "new_gate_loss_metrics.json", all_results)
                emit("student_evaluation_complete", split=split, arm=arm_name,
                     gate=gate_name,
                     absolute_mse=all_results["students"][split][arm_name][gate_name]["absolute_mse"])
            del arm, renderer, state
        # Save per-split gate predictions for later inspection without images.
        np.savez_compressed(output / (split + "_new_gate_predictions.npz"),
                            phase=labels.numpy(), prediction=prediction.numpy(),
                            p_open=cache["new_phase_probabilities"][..., 1:].sum(-1).numpy())
        del cache
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-model-dir", type=Path,
                        default=Path("model/civi_image_tau_phase_gate"))
    parser.add_argument("--gate-checkpoint", type=Path,
                        help="Reuse an already trained gate and only re-score the frozen students")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.gate_checkpoint is not None)
    model_dir = args.output / "model"
    model_dir.mkdir(parents=True, exist_ok=args.gate_checkpoint is not None)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The gate experiment requires CUDA")

    student_model, preprocessor, cfg, teacher_cfg, device, _ = load_render_net(args.student_checkpoint, device)
    payload, teacher_cfg = common.load_checkpoint_payload(Path(cfg.teacher_checkpoint))
    teacher = common.instantiate_policy(payload, teacher_cfg, device, capture_attention=False)
    del payload
    teacher.eval().requires_grad_(False)
    ds_cfg = copy.deepcopy(teacher_cfg.task.dataset)
    with open_dict(ds_cfg):
        ds_cfg.preload_images = False
        ds_cfg.image_randomer_config = None
        ds_cfg.video_backend = "pyav"
        ds_cfg.phase_predictor = cfg.render_net.phase_predictor
    train_dataset = hydra.utils.instantiate(ds_cfg)
    validation_dataset = train_dataset.get_validation_dataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)
    model = ImageTauPhaseGate(teacher.obs_encoder.image_keys,
                              wrench_dim=teacher.obs_encoder.wrench_projection.in_features,
                              hidden_dim=args.hidden_dim,
                              image_size=args.image_size).to(device)
    # The requested training objective is ordinary three-class cross entropy;
    # no class reweighting is silently applied.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    preprocessing = OmegaConf.to_container(cfg.wrench_preprocessing, resolve=True)
    preprocessing_state = {key: value.detach().cpu()
                           for key, value in preprocessor.state_dict().items()}
    if args.gate_checkpoint is not None:
        checkpoint = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
        model = load_phase_gate(args.gate_checkpoint, device)
        best = float(checkpoint["best_val_ce"])
        best_epoch = int(checkpoint["epoch"])
        history_path = args.gate_checkpoint.parent.parent / "phase_training_metrics.json"
        if history_path.exists():
            import json
            history = json.loads(history_path.read_text())
            bad_epochs = len(history) - best_epoch
        checkpoint.update(wrench_preprocessing=preprocessing,
                          wrench_preprocessor=preprocessing_state,
                          history_steps=int(teacher.obs_encoder.wrench_history_steps))
        torch.save(checkpoint, model_dir / "phase_gate_best.pt")
    started = time.monotonic()
    for epoch in range(1, 1 if args.gate_checkpoint is not None else args.epochs + 1):
        train_metrics = run_gate_epoch(model, teacher, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_gate_epoch(model, teacher, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
               "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        emit("phase_epoch", **row)
        write_json(args.output / "phase_training_metrics.json", history)
        if val_metrics["ce_loss"] < best - args.min_delta:
            best = val_metrics["ce_loss"]
            best_epoch = epoch
            bad_epochs = 0
            state = {key: value.detach().cpu()
                     for key, value in model.state_dict().items()}
            checkpoint = {"model": state, "epoch": epoch,
                          "best_val_ce": best,
                          "config": {key: str(value) if isinstance(value, Path) else value
                                     for key, value in vars(args).items()},
                          "architecture": {"image_keys": list(teacher.obs_encoder.image_keys),
                                           "wrench_dim": int(teacher.obs_encoder.wrench_projection.in_features),
                                           "hidden_dim": int(args.hidden_dim),
                                           "image_size": int(args.image_size),
                                           "gru_layers": 1},
                          "history_steps": int(teacher.obs_encoder.wrench_history_steps),
                          "wrench_preprocessing": preprocessing,
                          "wrench_preprocessor": preprocessing_state}
            torch.save(checkpoint, model_dir / "phase_gate_best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    payload = torch.load(model_dir / "phase_gate_best.pt", map_location=device,
                         weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    args.reuse_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.reuse_model_dir / "phase_gate_best.pt")
    write_json(args.reuse_model_dir / "config.json", {
        "architecture": payload["architecture"],
        "labels": {"0": "free", "1": "precontact", "2": "contact"},
        "gate_rule": "open renderer iff argmax(phase_logits) != 0",
        "preprocessing": "teacher prepare_context_inputs_with_future_contact; image resized to image_size x image_size with area interpolation",
        "history_steps": int(teacher.obs_encoder.wrench_history_steps),
        "wrench_preprocessing": preprocessing,
        "best_epoch": int(best_epoch), "best_validation_ce": float(best),
        "stopped_epoch": len(history),
        "early_stopped": bad_epochs >= args.patience,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    })
    with torch.no_grad():
        final_train = run_gate_epoch(model, teacher, train_loader, device)
        final_validation = run_gate_epoch(model, teacher, val_loader, device)
    gate_summary = {
        "architecture": "shared per-camera 4/8/12-channel CNN at 64x64 + tau_ext GRU + MLP -> 3-class CE",
        "inputs": {"images": list(teacher.obs_encoder.image_keys),
                   "tau_ext_history_steps": int(teacher.obs_encoder.wrench_history_steps),
                   "tau_ext_dim": int(teacher.obs_encoder.wrench_projection.in_features),
                   "tau_ext_source": "teacher-preprocessed normalized tau_ext with physical zeros preserved"},
        "labels": "0=free, 1=precontact, 2=contact; open iff prediction != 0",
        "objective": "unweighted cross_entropy(logits, contact_phase)",
        "best_epoch": int(best_epoch), "best_validation_ce": float(best),
        "stopped_epoch": len(history), "early_stopped": bad_epochs >= args.patience,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_train": final_train, "final_validation": final_validation,
        "history": history, "elapsed_seconds": time.monotonic() - started,
    }
    write_json(model_dir / "phase_gate_summary.json", gate_summary)
    shutil.copy2(model_dir / "phase_gate_summary.json", args.reuse_model_dir / "phase_gate_summary.json")
    # The old renderer supplies the fixed spatial supports; only its phase
    # switch is replaced by the newly trained gate.
    effects = evaluate_gate_effect(args.student_run, teacher, student_model, model,
                                   teacher_cfg, cfg, device, train_dataset,
                                   validation_dataset, args.output, seed=args.seed)
    write_json(args.output / "new_gate_loss_metrics.json", effects)
    manifest = {"arguments": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                "gate": gate_summary, "loss_recalculation": effects,
                "fixed": "teacher, DINO, RenderNet RGB, teacher FFN and spatial supports",
                "counterfactual_gate": "new predicted phase on actual history, held fixed for zero/other torque conditions",
                "reference": str(args.student_run)}
    write_json(args.output / "manifest.json", manifest)
    final_payload = {**payload, "summary": gate_summary, "manifest": manifest}
    torch.save(final_payload, model_dir / "phase_gate_final.pt")
    torch.save(final_payload, args.reuse_model_dir / "phase_gate_final.pt")
    train_dataset.lerobot_dataset.close()
    validation_dataset.lerobot_dataset.close()
    emit("complete", output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
