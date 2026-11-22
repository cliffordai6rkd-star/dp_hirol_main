"""Teacher torque-reliance diagnostic (paired action MSE under torque interventions).

Loads a ForceAwareDiffusionTransformerPolicy teacher (EMA weights) from a workspace
checkpoint, evaluates the policy's own ``predict_action`` path on fixed windows,
and intervenes only on the wrench tensor / fusion gate passed to ``obs_encoder``:

  real              unchanged inputs
  zero_vals         normalized wrench history -> 0, fusion gate kept as computed
                    from the real inputs (A/C: real torque; B: phase labels)
  zero_raw          raw wrench observations -> 0 before preprocessing; the gate is
                    recomputed (closes for torque-derived gates, i.e. A/C)
  shuffle           wrench history deranged across the batch, gate kept
  bypass            fusion gate forced closed (pre-fusion visual tokens)

Per-window inference noise is paired: every condition re-seeds the same per-batch
generator, and batches are deterministic (no shuffle).

Run with PYTHONPATH pointing at the code version that trained the teacher, and
cwd at the repo root so relative dataset / DINO paths resolve.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from diffusion_policy.common.pytorch_util import dict_apply

CONDITIONS = ("real", "zero_vals", "zero_raw", "shuffle", "bypass")
PHASES = ("free", "pre", "contact")


# ----------------------------------------------------------------- phase labels
def debounce(above, enter, exit_):
    state, a, b = False, 0, 0
    out = np.zeros(len(above), dtype=bool)
    for i, x in enumerate(above):
        a = a + 1 if x else 0
        b = b + 1 if not x else 0
        if not state and a >= enter:
            state, a = True, 0
        elif state and b >= exit_:
            state, b = False, 0
        out[i] = state
    return out


def frame_phase_labels(ds, key, dims, norm, threshold, fps, pre_s=0.5, enter=5, exit_=5):
    values = np.asarray(ds._load_column(key, dtype=np.float32), dtype=np.float32)
    values = values[:, list(dims)]
    mag = np.abs(values).sum(-1) if norm == "l1" else np.linalg.norm(values, axis=-1)
    ep = np.asarray(ds.episode_index)
    labels = np.zeros(len(mag), dtype=np.int8)
    pre_n = int(round(pre_s * fps))
    for e in np.unique(ep):
        rows = np.flatnonzero(ep == e)
        c = debounce(mag[rows] > threshold, enter, exit_)
        lab = np.where(c, 2, 0).astype(np.int8)
        onsets = np.flatnonzero(c & ~np.r_[False, c[:-1]])
        for o in onsets:
            s = max(0, o - pre_n)
            seg = lab[s:o]
            seg[seg == 0] = 1
        labels[rows] = lab
    return labels


# ----------------------------------------------------------------- encoder hook
class EncoderHook:
    """Wraps obs_encoder.forward; intervenes on wrench_history and the fusion gate."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.orig = encoder.forward
        self.mode = "real"
        self.perm = None
        self.last_context = None
        self.last_gate = None
        self.gate_kw = None
        encoder.forward = self

    def __call__(self, *args, **kwargs):
        wh = kwargs.get("wrench_history")
        gate_kw = next((k for k in ("contact_gate", "fusion_gate") if kwargs.get(k) is not None), None)
        if gate_kw is None and any(kwargs.get(k) is not None for k in ("future_contact", "contact_alpha")):
            raise NotImplementedError("attention_alpha gating is not supported by this diagnostic")
        self.gate_kw = gate_kw or self.gate_kw
        gate = kwargs.get(gate_kw) if gate_kw else None
        if self.mode == "zero_vals":
            kwargs["wrench_history"] = torch.zeros_like(wh)
        elif self.mode == "shuffle":
            kwargs["wrench_history"] = wh[self.perm]
        elif self.mode == "bypass":
            if gate_kw is None:
                raise RuntimeError("teacher exposes no binary fusion gate; cannot bypass")
            kwargs[gate_kw] = torch.zeros_like(gate)
        self.last_gate = (kwargs.get(gate_kw) if gate_kw else None)
        out = self.orig(*args, **kwargs)
        self.last_context = out.detach()
        return out


# ----------------------------------------------------------------- main
def load_teacher(ckpt, device):
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    sd = payload["state_dicts"]["ema_model"]
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if unexpected or [m for m in missing if not m.startswith("obs_encoder.backbone")]:
        raise RuntimeError(f"state_dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    policy.to(device).eval()
    return cfg, policy, {"missing_backbone_keys": len(missing), "unexpected_keys": len(unexpected)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("data/debug") / time.strftime("%Y%m%d_%H%M%S_teacher_torque_reliance"),
                    help="Result directory (default: data/debug/<timestamp>_teacher_torque_reliance)")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--max-windows", type=int, default=3000)
    ap.add_argument("--max-batches", type=int, default=None, help="smoke test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase-key", required=True)
    ap.add_argument("--phase-dims", required=True, help="comma list")
    ap.add_argument("--phase-norm", choices=["l1", "l2"], required=True)
    ap.add_argument("--phase-threshold", type=float, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--preload-images", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    cfg, policy, load_info = load_teacher(args.ckpt, device)
    amp = cfg.training.get("amp", {})
    amp_enabled = bool(amp.get("enabled", False))
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(amp.get("dtype", "bfloat16")), torch.bfloat16)

    ds_cfg = cfg.task.dataset
    with open_dict(ds_cfg):
        if "preload_images" in ds_cfg:
            ds_cfg.preload_images = bool(args.preload_images)
        if "image_randomer_config" in ds_cfg:
            ds_cfg.image_randomer_config = None
    dataset = hydra.utils.instantiate(ds_cfg)
    split_note = "train episodes (val_ratio=%s)" % ds_cfg.get("val_ratio")
    if args.split == "val":
        dataset = dataset.get_validation_dataset()
        split_note = "held-out val episodes (val_ratio=%s)" % ds_cfg.get("val_ratio")
    n_all = len(dataset)
    if n_all == 0:
        raise RuntimeError("selected split is empty")
    stride = max(1, math.ceil(n_all / args.max_windows))
    subset = np.arange(0, n_all, stride)

    frame_labels = frame_phase_labels(
        dataset, args.phase_key, [int(x) for x in args.phase_dims.split(",")],
        args.phase_norm, args.phase_threshold, args.fps)
    anchors = np.array([dataset._anchor_index_from_sampler_row(dataset.indices[i]) for i in subset])
    window_phase = frame_labels[anchors]
    episodes = np.unique(np.asarray(dataset.episode_index)[anchors])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, subset.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=False)

    hook = EncoderHook(policy.obs_encoder)
    wrench_keys = {k for k in (policy.wrench_key, policy.phase_wrench_key,
                               getattr(policy, "future_phase_wrench_key", None)) if k}
    start, end = policy.action_start_index, policy.action_start_index + policy.n_action_steps
    To = policy.n_obs_steps

    acc = {c: {"norm": np.zeros(len(subset)), "raw": np.zeros(len(subset))} for c in CONDITIONS}
    gate_open = {c: np.zeros(len(subset)) for c in CONDITIONS}
    zdist = np.full(len(subset), np.nan)       # mean per-patch ||Z^T - Z^0||, last obs step
    zdist_rel = np.full(len(subset), np.nan)
    pos = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        for bi, batch in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            batch = dict_apply(batch, lambda v: v.to(device, non_blocking=True))
            bsz = int(batch["action"].shape[0])
            sl = slice(pos, pos + bsz)
            target = batch["action"][:, start:end].float()
            g = torch.Generator()
            g.manual_seed(args.seed * 100003 + bi)
            perm = torch.randperm(bsz, generator=g)
            src = torch.empty_like(perm)
            src[perm] = perm.roll(-1)  # derangement: every window gets another window's wrench
            hook.perm = src.to(device)
            contexts = {}
            for cond in CONDITIONS:
                obs = batch["obs"]
                if cond == "zero_raw":
                    obs = dict(obs)
                    for k in wrench_keys:
                        obs[k] = torch.zeros_like(obs[k])
                hook.mode = "real" if cond == "zero_raw" else cond
                gen = torch.Generator(device=device)
                gen.manual_seed(args.seed + bi)
                pred = policy.predict_action(obs, generator=gen)["model_action_pred"][:, start:end].float()
                nrm = policy.normalizer["action"]
                acc[cond]["norm"][sl] = ((nrm.normalize(pred) - nrm.normalize(target)) ** 2).flatten(1).mean(1).cpu().numpy()
                acc[cond]["raw"][sl] = ((pred - target) ** 2).flatten(1).mean(1).cpu().numpy()
                gate = hook.last_gate
                gate_open[cond][sl] = (gate[:, -1].float().cpu().numpy() if gate is not None else np.nan)
                if cond in ("real", "bypass"):
                    contexts[cond] = hook.last_context.float()
            zt, z0 = contexts["real"], contexts["bypass"]
            zt = zt.reshape(bsz, To, -1, zt.shape[-1])[:, -1]
            z0 = z0.reshape(bsz, To, -1, z0.shape[-1])[:, -1]
            d = (zt - z0).norm(dim=-1)
            zdist[sl] = d.mean(1).cpu().numpy()
            zdist_rel[sl] = (d / z0.norm(dim=-1).clamp_min(1e-6)).mean(1).cpu().numpy()
            pos += bsz
            if bi % 20 == 0:
                print(f"[{time.time()-t0:7.1f}s] batch {bi}/{len(loader)} real={acc['real']['norm'][:pos].mean():.5f} "
                      f"bypass={acc['bypass']['norm'][:pos].mean():.5f}", flush=True)

    n = pos
    ph = window_phase[:n]
    open_real = gate_open["real"][:n] > 0.5

    def summarize(mask):
        out = {"n": int(mask.sum())}
        if mask.sum() == 0:
            return out
        for c in CONDITIONS:
            out[c] = {"action_mse": float(acc[c]["norm"][:n][mask].mean()),
                      "action_mse_raw": float(acc[c]["raw"][:n][mask].mean())}
        out["gate_open_frac_real"] = float(np.nanmean(gate_open["real"][:n][mask]))
        om = mask & open_real
        out["z_dist_open"] = float(zdist[:n][om].mean()) if om.any() else None
        out["z_dist_rel_open"] = float(zdist_rel[:n][om].mean()) if om.any() else None
        return out

    metrics = {
        "ckpt": args.ckpt, "split": args.split, "split_note": split_note,
        "windows_evaluated": n, "windows_in_split": n_all, "stride": stride,
        "episodes_evaluated": int(len(episodes)),
        "amp": {"enabled": amp_enabled, "dtype": str(amp_dtype)},
        "num_inference_steps": int(getattr(policy, "num_inference_steps", -1)),
        "gate_kwarg": hook.gate_kw,
        "phase_rule": {"key": args.phase_key, "dims": args.phase_dims, "norm": args.phase_norm,
                       "threshold": args.phase_threshold, "enter_exit_frames": 5, "pre_seconds": 0.5,
                       "fps": args.fps},
        "load_info": load_info,
        "overall": summarize(np.ones(n, dtype=bool)),
        "by_phase": {p: summarize(ph == i) for i, p in enumerate(PHASES)},
        "by_gate_real": {"open": summarize(open_real), "closed": summarize(~open_real)},
        "elapsed_s": time.time() - t0,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(args.out / "per_window.npz", phase=ph, gate_open=gate_open["real"][:n], zdist=zdist[:n],
             **{f"{c}_{k}": acc[c][k][:n] for c in CONDITIONS for k in ("norm", "raw")})
    print(json.dumps(metrics["overall"], indent=1))


if __name__ == "__main__":
    main()
