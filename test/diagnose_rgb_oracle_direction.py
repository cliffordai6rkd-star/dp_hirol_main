"""Frozen-encoder RGB oracles: paired response direction and L2-regularized fits."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import load_inputs, datasets_from_reference, to_device, write_json, tensor_hash
from diagnose_student_renderer import load_model, encode, event
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def direction_statistics(s, t, thresholds):
    s, t = s.detach().float(), t.detach().float()
    trms, srms = t.square().mean(-1).sqrt(), s.square().mean(-1).sqrt()
    cosine = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
    result = []
    for threshold in thresholds:
        valid = trms > threshold
        count = int(valid.sum())
        if not count:
            result.append(dict(threshold=threshold, patches=0)); continue
        sv, tv, cv, weights = s[valid], t[valid], cosine[valid], trms[valid]
        result.append(dict(threshold=float(threshold), patches=count, retained_fraction=float(valid.float().mean()),
            patch_cosine_mean=float(cv.mean()), patch_cosine_median=float(cv.median()),
            teacher_weighted_patch_cosine=float((cv*weights).sum()/weights.sum()),
            joint_cosine=float(F.cosine_similarity(sv.flatten(), tv.flatten(), dim=0)),
            student_response_rms=float(sv.square().mean().sqrt()), teacher_response_rms=float(tv.square().mean().sqrt()),
            response_difference_mse=float((sv-tv).square().mean()),
            student_near_zero_fraction=float((srms[valid] < 1e-4).float().mean())))
    valid = trms > thresholds[0]
    bins = []
    bounds = torch.quantile(trms[valid], trms.new_tensor([0, .25, .5, .75, 1]))
    for i in range(4):
        mask = valid & (trms >= bounds[i]) & ((trms <= bounds[i+1]) if i == 3 else (trms < bounds[i+1]))
        bins.append(dict(teacher_rms_range=[float(bounds[i]), float(bounds[i+1])], patches=int(mask.sum()),
                         patch_cosine_mean=float(cosine[mask].mean()) if mask.any() else None))
    return dict(thresholds=result, amplitude_quartiles=bins,
                all_response_difference_mse=float((s-t).square().mean()),
                no_student_response_mse=float(t.square().mean()))


def fit_pixels(teacher, model, images, target, *, steps, lr=.003, bounds=None, rgb_lambda=0.):
    """Project RGB to [0,1]; optimize only image parameters, using mean L2."""
    pixels = {k: torch.nn.Parameter(v.detach().clone()) for k, v in images.items()}
    lower = {k: (v-bounds[k]).clamp(0, 1) if bounds is not None else torch.zeros_like(v) for k, v in images.items()}
    upper = {k: (v+bounds[k]).clamp(0, 1) if bounds is not None else torch.ones_like(v) for k, v in images.items()}
    optimizer = torch.optim.Adam(list(pixels.values()), lr=lr)
    best = torch.full((len(target),), float('inf'), device=target.device)
    best_images = {k: v.detach().clone() for k, v in images.items()}
    history = []; start = time.monotonic()
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        z = encode(teacher, model, pixels, grad=True)
        token = (z-target).square().mean((1, 2))
        rgb_mse = torch.cat([(pixels[k]-v).flatten(1) for k, v in images.items()], dim=1).square().mean(1)
        objective = token+rgb_lambda*rgb_mse
        with torch.no_grad():
            improved = objective < best; best[improved] = objective[improved]
            for k in pixels:
                best_images[k][improved] = pixels[k][improved]
        if step % 50 == 0 or step == steps:
            row = dict(step=step, token_mse=float(token.mean()), rgb_mse=float(rgb_mse.mean()),
                       objective=float(objective.mean()), elapsed_seconds=time.monotonic()-start)
            history.append(row); event('oracle_fit', rgb_lambda=rgb_lambda, **row)
        if step == steps: break
        objective.mean().backward()
        optimizer.param_groups[0]['lr'] = lr*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        optimizer.step()
        with torch.no_grad():
            for k, value in pixels.items():
                value.copy_(torch.maximum(lower[k], torch.minimum(upper[k], value)))
    with torch.no_grad():
        z = encode(teacher, model, best_images)
    changes = torch.cat([(best_images[k]-v).flatten(1) for k, v in images.items()], dim=1)
    rgb = dict(mae=float(changes.abs().mean()), rms=float(changes.square().mean().sqrt()),
               linf=float(changes.abs().max()), l2_per_sample_mean=float(changes.norm(dim=1).mean()))
    return best_images, z, rgb, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['paired', 'l2'], default='paired')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--cases', nargs='+', default=['full_0.02', 'full_0.05', 'full_0.10', 'gated_0.10'])
    a = p.parse_args(); a.root = a.root.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda:0'); teacher, _, initial = load_inputs(a.root/'inputs', dev)
    model = load_model(a.root, initial, 'c_film_mlp', dev)
    model_hash = tensor_hash(model.state_dict())
    reference = a.root/'renderer_diagnostics_1000'
    artifact = torch.load(reference/'full_0.02_images.pt', map_location='cpu', weights_only=False)
    _, val, _ = datasets_from_reference(a.root/'inputs', a.root/'decoded_cuda')
    selection = artifact['selection']
    obs = to_device(next(iter(DataLoader(Subset(val, [s['index'] for s in selection]),
                                        batch_size=len(selection))))['obs'], dev)
    images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
    assert all(torch.equal(images[k].cpu(), artifact['original'][k]) for k in images)
    target, zero = artifact['target'].to(dev), artifact['zero_target'].to(dev)
    dt = target-zero
    zero_wrench = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
    with torch.no_grad():
        raw_z = encode(teacher, model, images)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            render = model(images, wrench)
            render0, masks0, cond0 = model(images, zero_wrench, return_condition=True)
        rz, rz0 = encode(teacher, model, render), encode(teacher, model, render0)
    if a.mode == 'l2':
        # Preserve all six episodes while selecting exactly ten fixed windows.
        ids = np.linspace(0, len(selection)-1, 10).round().astype(int).tolist()
        selection = [selection[i] for i in ids]
        images = {k: v[ids] for k, v in images.items()}; target = target[ids]
        rows = []
        for weight in [0., 10., 100.]:
            optimized, z, rgb, history = fit_pixels(teacher, model, images, target,
                                                   steps=a.steps, rgb_lambda=weight)
            errors = (z-target).square().mean((1, 2))
            delta = torch.cat([(optimized[k]-v).flatten(1) for k, v in images.items()], dim=1)
            with torch.no_grad():
                fp32_errors = (encode(teacher, model, optimized, bf16=False)-target).square().mean((1, 2))
            row = dict(rgb_lambda=weight, steps=a.steps, samples=10, selection=selection,
                baseline_mse=float((raw_z[ids]-target).square().mean()),
                renderer_mse=float((rz[ids]-target).square().mean()), token_mse=float(errors.mean()),
                rgb=rgb, fp32_token_mse=float(fp32_errors.mean()), history=history,
                per_sample=[dict(**s, token_mse=float(errors[i]), rgb_rms=float(delta[i].square().mean().sqrt()),
                                 rgb_linf=float(delta[i].abs().max())) for i, s in enumerate(selection)])
            rows.append(row); write_json(a.output/'rgb_l2_oracle.json', rows)
            torch.save(dict(selection=selection, original={k:v.cpu() for k,v in images.items()},
                optimized={k:v.cpu() for k,v in optimized.items()}), a.output/f'l2_{weight:g}_images.pt')
            event('l2_done', **{k:v for k,v in row.items() if k not in ['history','per_sample','selection']})
    else:
        # Precision sensitivity is an empirical numerical reference, not sensor noise.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            alternate = teacher._encode_context(obs, apply_curriculum_mask=False).float()
            alternate0 = teacher._encode_context(zero_wrench_reference(teacher, obs), apply_curriculum_mask=False).float()
        noise = ((alternate-alternate0)-dt).square().mean(-1).sqrt()
        threshold = max(1e-3, 5*float(torch.quantile(noise.flatten(), .99)))
        thresholds = [threshold, 2*threshold, 4*threshold]
        records = dict(selection=selection, checkpoint_step=1000, zero_oracle_steps=a.steps,
            response_definition='Each Oracle branch independently fits its own absolute teacher context, with matched '
                'initial RGB, token loss, optimizer, budget and branch-specific constraints. Response = fitted normal - fitted zero.',
            teacher_precision_noise_p99=float(torch.quantile(noise.flatten(), .99)),
            threshold_policy='Patch RMS > max(0.001, 5 * p99 BF16-vs-FP32 teacher response discrepancy); sensitivity at 2x and 4x',
            thresholds=thresholds, rendernet=direction_statistics(rz-rz0, dt, thresholds), oracle={})
        tensors = dict(teacher_response=dt.cpu(), renderer_response=(rz-rz0).cpu())
        for case in a.cases:
            normal = torch.load(reference/f'{case}_images.pt', map_location='cpu', weights_only=False)
            assert normal['selection'] == selection
            epsilon = float(case.split('_')[1])
            if case.startswith('gated_'):
                gates = {k: model._mask_gate(masks0[k].flatten(0,1), images[k]).reshape_as(masks0[k]).float()
                         *cond0['contact_alpha'].float()[...,None,None,None] for k in images}
            else: gates = {k:torch.ones_like(v[:,:,:1]) for k,v in images.items()}
            bounds = {k:epsilon*g for k,g in gates.items()}
            zero_images, oz0, rgb0, history = fit_pixels(teacher, model, images, zero,
                                                        steps=a.steps, bounds=bounds)
            with torch.no_grad():
                oz = encode(teacher, model, {k:v.to(dev) for k,v in normal['optimized'].items()})
                oz_fp32 = encode(teacher, model, {k:v.to(dev) for k,v in normal['optimized'].items()}, bf16=False)
                oz0_fp32 = encode(teacher, model, zero_images, bf16=False)
            row = direction_statistics(oz-oz0, dt, thresholds)
            row.update(normal_token_mse=float((oz-target).square().mean()), zero_token_mse=float((oz0-zero).square().mean()),
                       zero_rgb=rgb0, zero_fit_history=history,
                       fp32=direction_statistics(oz_fp32-oz0_fp32, dt, thresholds))
            records['oracle'][case] = row; tensors[case] = (oz-oz0).cpu()
            write_json(a.output/'oracle_direction.json', records)
            event('paired_done', case=case, normal_mse=row['normal_token_mse'], zero_mse=row['zero_token_mse'],
                  primary=row['thresholds'][0], quartiles=row['amplitude_quartiles'])
        torch.save(tensors, a.output/'response_tensors.pt')
    assert model_hash == tensor_hash(model.state_dict())
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in model.parameters())
    event('complete', mode=a.mode, no_network_weights_updated=True)


if __name__ == '__main__': main()
