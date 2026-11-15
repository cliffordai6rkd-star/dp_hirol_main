"""Independent A/B pilot on the 12 Oracle windows; production code is unchanged."""
from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from student_dual_path_common import (load_inputs, datasets_from_reference, to_device,
    tensor_hash, write_json, PreparedDataset)
from diagnose_student_renderer import load_model, encode, event
from diffusion_policy.model.vision.response_distillation import response_weighted_token_mse
from diffusion_policy.model.diffusion.ema_model import EMAModel


def direction_loss(student_delta, teacher_delta, threshold):
    target = teacher_delta.detach().float()
    amplitude = target.square().mean(-1).sqrt()
    weights = (amplitude*(amplitude > threshold)).detach()
    cosine = F.cosine_similarity(student_delta.float(), target, dim=-1, eps=1e-6)
    return (weights*(1-cosine)).sum()/weights.sum().clamp_min(1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group', choices=['a', 'b'], required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--microbatch', type=int, default=3)
    a = p.parse_args(); a.root = a.root.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda:0'); teacher, _, initial = load_inputs(a.root/'inputs', dev)
    model = load_model(a.root, initial, 'c_film_mlp', dev)
    initial_hash = tensor_hash(model.state_dict())
    model.set_training_stage('rgb'); model.set_stage_mode(True)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMAModel(ema_model, **dict(initial['cfg'].ema))
    frozen = {k:v.detach().clone() for k,v in model.state_dict().items()
              if k.startswith(('condition_network.', 'context_adapter.'))}
    _, val, _ = datasets_from_reference(a.root/'inputs', a.root/'decoded_cuda')
    dataset = PreparedDataset(val, a.root/'prepared_cuda/validation')
    oracle = json.loads((a.root/'oracle_direction_12/oracle_direction.json').read_text())
    selected = [r['index'] for r in oracle['selection']]
    threshold = oracle['thresholds'][0]
    ids = np.flatnonzero(val.phase_labels == 2).tolist()
    assert len(selected) == 12 and len(ids) == 468 and set(selected) <= set(ids)
    fit = to_device(next(iter(DataLoader(Subset(dataset, selected), batch_size=12))), dev)
    images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(fit['obs'])
    zero_wrench = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
    target, dt = fit['target'].float(), fit['target'].float()-fit['zero'].float()
    all_response_weight = (fit['response'].float()-1e-6).clamp_min(0).sum().clamp_min(1e-8)
    amplitude = dt.square().mean(-1).sqrt()
    all_direction_weight = (amplitude*(amplitude > threshold)).sum().clamp_min(1e-8)
    slices = [slice(i, min(i+a.microbatch, len(selected))) for i in range(0, len(selected), a.microbatch)]
    optimizer = torch.optim.AdamW(model.rgb_parameters(), **dict(initial['cfg'].optimizer))
    # Check the low-response contract before any model update.
    q = torch.randn(1, 3, 4, device=dev, requires_grad=True)
    quiet_teacher = torch.zeros_like(q, requires_grad=True)
    quiet_loss = direction_loss(q, quiet_teacher, threshold)
    quiet_loss.backward()
    assert quiet_loss == 0 and torch.count_nonzero(q.grad) == 0 and quiet_teacher.grad is None
    del q, quiet_teacher

    def objectives(sl):
        im = {k:v[sl] for k,v in images.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rendered, rendered0 = model(im, wrench[sl]), model(im, zero_wrench[sl])
        z = encode(teacher, model, rendered, grad=True)
        z0 = encode(teacher, model, rendered0, grad=True)
        global_mse, _, numerator, _ = response_weighted_token_mse(z, target[sl], fit['response'][sl], noise_floor=1e-6)
        part_amplitude = amplitude[sl]
        part_direction_weight = (part_amplitude*(part_amplitude > threshold)).sum()
        directional = direction_loss(z-z0, dt[sl], threshold)*part_direction_weight/all_direction_weight
        return global_mse*len(z)/len(selected)+numerator/all_response_weight, directional, z, z0

    # A fixed coefficient balances the two initial RGB-head gradient norms.
    # Both groups compute it from identical weights and inputs; no validation tuning.
    model.eval()
    gb, gd = torch.zeros_like(model.output.weight).float(), torch.zeros_like(model.output.weight).float()
    for sl in slices:
        with torch.backends.cudnn.flags(enabled=False):
            base, directional, z, z0 = objectives(sl)
            gb += torch.autograd.grad(base, model.output.weight, retain_graph=True)[0].float()
            gd += torch.autograd.grad(directional, model.output.weight)[0].float()
        del base, directional, z, z0
    proposed_lambda = float(gb.norm()/gd.norm().clamp_min(1e-12))
    initial_gradient_cosine = float(F.cosine_similarity(gb.flatten(), gd.flatten(), dim=0))
    del gb, gd
    coefficient = proposed_lambda if a.group == 'b' else 0.
    manifest = dict(group=a.group, steps=a.steps, seed=42, fit_indices=selected,
        effective_batch_size=len(selected), microbatch=a.microbatch,
        initial_state_sha256=initial_hash, starting_rgb_step=1000,
        architecture='D: interleaved SA/CA + decoder FiLM + frozen MLP128 adapter',
        threshold=threshold, direction_lambda=coefficient, gradient_balanced_lambda=proposed_lambda,
        initial_head_gradient_cosine=initial_gradient_cosine,
        objective='Global MSE + response-weighted MSE + lambda * teacher-thresholded direction loss',
        direction_weights='stop_gradient(teacher patch response RMS * (RMS > threshold))',
        validation='Full 468 contact windows, plus separate 456 windows excluding the fitted 12',
        optimizer=OmegaConf.to_container(initial['cfg'].optimizer, resolve=True),
        ema=OmegaConf.to_container(initial['cfg'].ema, resolve=True),
        low_teacher_response_zero_loss_and_gradient_verified=True)
    write_json(a.output/'manifest.json', manifest)
    history = []; started = time.monotonic()

    @torch.no_grad()
    def evaluate(step):
        ema_model.eval(); records = []
        for raw in DataLoader(Subset(dataset, ids), batch_size=16, num_workers=4, pin_memory=True):
            batch = to_device(raw, dev)
            im, wr, _, _ = teacher.prepare_context_inputs_with_future_contact(batch['obs'])
            wr0 = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wr))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rendered, rendered0 = ema_model(im, wr), ema_model(im, wr0)
            z, z0 = encode(teacher, ema_model, rendered), encode(teacher, ema_model, rendered0)
            s, t = z-z0, batch['target'].float()-batch['zero'].float()
            errors = (z-batch['target'].float()).square().mean(-1)
            response_weights = (batch['response'].float()-1e-6).clamp_min(0)
            amplitude = t.square().mean(-1).sqrt()
            cosine = F.cosine_similarity(s, t, dim=-1, eps=1e-6)
            delta_rgb = torch.cat([(rendered[k].float()-im[k].float()).flatten(1) for k in im], dim=1)
            for i in range(len(z)):
                valid = amplitude[i] > threshold
                sv, tv = s[i][valid], t[i][valid]
                row = dict(index=int(batch['index'][i]), token_mse=float(errors[i].mean()),
                    response_difference_mse=float((s[i]-t[i]).square().mean()),
                    weighted_numerator=float((errors[i]*response_weights[i]).sum()), weight_sum=float(response_weights[i].sum()),
                    valid_patches=int(valid.sum()), cosine_sum=float(cosine[i][valid].sum()),
                    cosine_weighted_sum=float((cosine[i][valid]*amplitude[i][valid]).sum()), cosine_weight_sum=float(amplitude[i][valid].sum()),
                    ss=float(sv.double().square().sum()), tt=float(tv.double().square().sum()),
                    st=float((sv.double()*tv.double()).sum()), rgb_mae=float(delta_rgb[i].abs().mean()),
                    rgb_mse=float(delta_rgb[i].square().mean()), rgb_linf=float(delta_rgb[i].abs().max()))
                records.append(row)
        metrics = {}
        for scope, rows in [('all_468', records), ('fitted_12', [r for r in records if r['index'] in selected]),
                            ('unseen_456', [r for r in records if r['index'] not in selected])]:
            ss, tt, st = [sum(r[k] for r in rows) for k in ['ss', 'tt', 'st']]
            metrics[scope] = dict(samples=len(rows), token_mse=float(np.mean([r['token_mse'] for r in rows])),
                response_weighted_mse=sum(r['weighted_numerator'] for r in rows)/sum(r['weight_sum'] for r in rows),
                response_difference_mse=float(np.mean([r['response_difference_mse'] for r in rows])),
                patch_cosine=sum(r['cosine_sum'] for r in rows)/max(sum(r['valid_patches'] for r in rows), 1),
                teacher_weighted_patch_cosine=sum(r['cosine_weighted_sum'] for r in rows)/max(sum(r['cosine_weight_sum'] for r in rows), 1e-12),
                joint_cosine=st/max((ss*tt)**.5, 1e-12),
                rgb_mae=float(np.mean([r['rgb_mae'] for r in rows])),
                rgb_rms=float(np.mean([r['rgb_mse'] for r in rows])**.5), rgb_linf=max(r['rgb_linf'] for r in rows),
                valid_patches=sum(r['valid_patches'] for r in rows))
        row = dict(kind='validation', step=step, weights='EMA', metrics=metrics, elapsed_seconds=time.monotonic()-started)
        history.append(row); write_json(a.output/'log.json', history)
        write_json(a.output/f'per_sample_{step}.json', records)
        event('validation', group=a.group, **row)
        return metrics

    initial_metrics = evaluate(0)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    for step in range(1, a.steps+1):
        model.set_stage_mode(True); optimizer.zero_grad(set_to_none=True)
        base_value = direction_value = loss_value = response_error = 0.
        for sl in slices:
            base, directional, z, z0 = objectives(sl)
            loss = base+coefficient*directional
            assert torch.isfinite(loss)
            loss.backward()
            base_value += float(base.detach()); direction_value += float(directional.detach())
            loss_value += float(loss.detach())
            response_error += float(((z-z0)-dt[sl]).detach().square().mean())*len(z)/len(selected)
            del base, directional, z, z0, loss
        before_clip = float(torch.nn.utils.clip_grad_norm_(model.rgb_parameters(), 1.))
        optimizer.step(); ema.step(model)
        if step == 1 or step % 25 == 0:
            row = dict(kind='train', step=step, base_loss=base_value, direction_loss=direction_value,
                total_loss=loss_value, response_difference_mse=response_error,
                gradient_norm_before_clip=before_clip, elapsed_seconds=time.monotonic()-started)
            history.append(row); write_json(a.output/'log.json', history); event('train', group=a.group, **row)
    for k, v in frozen.items(): assert torch.equal(v, model.state_dict()[k]), k
    assert all(p.grad is None for p in teacher.parameters())
    final = evaluate(a.steps)
    torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},
                    ema_model={k:v.cpu() for k,v in ema_model.state_dict().items()}, manifest=manifest),
               a.output/f'checkpoint_{a.steps}.pt')
    summary = dict(manifest=manifest, initial=initial_metrics, final=final, frozen_teacher_adapter_condition_verified=True)
    write_json(a.output/'summary.json', summary)
    event('complete', group=a.group, steps=a.steps)


if __name__ == '__main__': main()
