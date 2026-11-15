"""Sample-specific RGB reachability and read-only renderer response diagnostics.

Uses the pinned inputs of the student ablation. No model optimizer is created;
the only optimized tensors are independent RGB images for selected windows.
"""
from pathlib import Path
import argparse
import contextlib
import json
import time

import dill
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import (
    load_inputs, model_for, datasets_from_reference, to_device, write_json,
    tensor_hash, PreparedDataset,
)
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def checkpoint_for(root, variant):
    names = dict(a_baseline='a_checkpoint_1000.ckpt', c_film_mlp='c_mlp_checkpoint_1000.ckpt')
    return root / 'review_1000' / names.get(variant, variant + '/checkpoint_1000.ckpt')


def load_model(root, initial, variant, device):
    payload = torch.load(checkpoint_for(root, variant), map_location='cpu',
                         pickle_module=dill, weights_only=False)
    assert dill.loads(payload['pickles']['global_step']) == 1000
    model, _, _ = model_for(initial, variant)
    model.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def selected_contacts(dataset):
    """Two spread-out windows per held-out episode; no loss-based selection."""
    groups = {}
    for index in np.flatnonzero(dataset.phase_labels == 2):
        anchor = dataset._anchor_index_from_sampler_row(dataset.indices[index])
        episode = int(dataset.episode_index[anchor])
        groups.setdefault(episode, []).append(int(index))
    return [dict(index=ids[min(len(ids)-1, int((len(ids)-1)*fraction))], episode=episode)
            for episode, ids in sorted(groups.items()) for fraction in (1/3, 2/3)]


def encode(teacher, model, images, *, grad=False, bf16=True):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        return teacher.obs_encoder.encode_image_context(
            images, context_adapter=model.context_adapter, input_grad=grad).float()


def corr(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    x, y = x-x.mean(), y-y.mean()
    denominator = x.norm()*y.norm()
    return float((x*y).sum()/denominator) if denominator > 1e-12 else None


def response_metrics(student, teacher):
    s, t = student.float(), teacher.float()
    srms, trms = s.square().mean().sqrt(), t.square().mean().sqrt()
    return dict(teacher_response_rms=float(trms), student_response_rms=float(srms),
                amplitude_ratio=float(srms/trms) if trms > 1e-12 else None,
                response_cosine=float(F.cosine_similarity(s.flatten(), t.flatten(), dim=0)),
                patch_magnitude_pearson=corr(s.square().mean(-1).sqrt(), t.square().mean(-1).sqrt()),
                response_difference_mse=float((s-t).square().mean()),
                no_student_response_mse=float(t.square().mean()))


def pixel_probe(args, teacher, model, validation, selection, device):
    obs = to_device(next(iter(DataLoader(Subset(validation, [s['index'] for s in selection]),
                                        batch_size=len(selection))))['obs'], device)
    images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
    with torch.no_grad():
        # Match the formal cached FP32 teacher targets, without image curriculum.
        target = teacher._encode_context(obs, apply_curriculum_mask=False).float()
        zero_target = teacher._encode_context(zero_wrench_reference(teacher, obs),
                                              apply_curriculum_mask=False).float()
        baseline = encode(teacher, model, images)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            render, masks, condition = model(images, wrench, return_condition=True)
        rendered_tokens = encode(teacher, model, render)
        gates = {k: model._mask_gate(masks[k].flatten(0, 1), images[k]).reshape_as(masks[k]).float()
                 * condition['contact_alpha'].float()[..., None, None, None]
                 for k in images}
    original = {k: v.detach().float() for k, v in images.items()}
    baseline_errors = (baseline-target).square().mean((1, 2))
    renderer_errors = (rendered_tokens-target).square().mean((1, 2))
    write_json(args.output / 'selection.json', dict(samples=selection,
        phases='contact only; two windows per held-out episode', seed=42,
        checkpoint_step=1000, target='unchanged teacher final context_norm',
        adapter='same frozen MLP128 adapter as C MLP; student never replays teacher FFNs',
        image_range=[0, 1], precision='BF16 DINO/MLP; FP32 teacher targets, losses and pixel parameters',
        objective='unweighted final token MSE only; diagnostic pixel fit, not production loss',
        norm='per RGB channel L-infinity; gated case bound additionally multiplied by predicted soft gate and alpha',
        steps=args.steps, lr=args.lr))
    for case in args.cases:
        epsilon = float(case.split('_')[1])
        gated = case.startswith('gated_')
        bounds = {k: epsilon*(gates[k] if gated else torch.ones_like(v[:, :, :1]))
                  for k, v in original.items()}
        lower = {k: (v-bounds[k]).clamp(0, 1) for k, v in original.items()}
        upper = {k: (v+bounds[k]).clamp(0, 1) for k, v in original.items()}
        variables = {k: torch.nn.Parameter(v.clone()) for k, v in original.items()}
        optimizer = torch.optim.Adam(list(variables.values()), lr=args.lr)
        best = baseline_errors.clone()
        best_images = {k: v.clone() for k, v in original.items()}
        best_steps = torch.zeros(len(selection), dtype=torch.long, device=device)
        history = []
        started = time.monotonic()
        for step in range(args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            z = encode(teacher, model, variables, grad=True)
            errors = (z-target).square().mean((1, 2))
            loss = errors.mean()
            assert torch.isfinite(loss)
            with torch.no_grad():
                improved = errors < best
                best[improved] = errors[improved]
                best_steps[improved] = step
                for k in variables:
                    best_images[k][improved] = variables[k][improved]
            if step % 25 == 0 or step == args.steps:
                row = dict(step=step, token_mse=float(loss), best_token_mse=float(best.mean()),
                           per_sample=errors.detach().cpu().tolist(), elapsed_seconds=time.monotonic()-started)
                history.append(row)
                event('pixels', case=case, **{k: v for k, v in row.items() if k != 'per_sample'})
            if step == args.steps:
                break
            loss.backward()
            optimizer.param_groups[0]['lr'] = args.lr*(.1+.9*.5*(1+np.cos(np.pi*step/args.steps)))
            optimizer.step()
            with torch.no_grad():
                for k, value in variables.items():
                    value.copy_(torch.maximum(lower[k], torch.minimum(upper[k], value)))
        with torch.no_grad():
            best_z = encode(teacher, model, best_images)
            final_errors = (best_z-target).square().mean((1, 2))
            assert torch.allclose(final_errors, best, atol=2e-6, rtol=2e-5)
            baseline_fp32 = (encode(teacher, model, original, bf16=False)-target).square().mean((1, 2))
            best_fp32 = (encode(teacher, model, best_images, bf16=False)-target).square().mean((1, 2))
        changes = torch.cat([(best_images[k]-v).flatten(1) for k, v in original.items()], dim=1)
        per_sample = [dict(**sample, baseline_mse=float(baseline_errors[i]),
            renderer_mse=float(renderer_errors[i]), optimized_mse=float(best[i]),
            best_step=int(best_steps[i]), rgb_mae=float(changes[i].abs().mean()),
            rgb_rms=float(changes[i].square().mean().sqrt()), rgb_linf=float(changes[i].abs().max()),
            baseline_fp32_mse=float(baseline_fp32[i]), optimized_fp32_mse=float(best_fp32[i]))
            for i, sample in enumerate(selection)]
        for k in best_images:
            assert ((best_images[k]-original[k]).abs() <= bounds[k]+1e-7).all()
            assert ((best_images[k] >= 0) & (best_images[k] <= 1)).all()
        result = dict(case=case, samples=len(selection), steps=args.steps,
            baseline_token_mse=float(baseline_errors.mean()), renderer_token_mse=float(renderer_errors.mean()),
            optimized_token_mse=float(best.mean()), improvement_pct=100*float((baseline_errors.mean()-best.mean())/baseline_errors.mean()),
            optimized_rgb_mae=float(changes.abs().mean()), optimized_rgb_linf=float(changes.abs().max()),
            all_rgb_constraints_passed=True, per_sample=per_sample, history=history)
        write_json(args.output / f'{case}.json', result)
        torch.save(dict(selection=selection, original={k: v.cpu() for k, v in original.items()},
            renderer={k: v.cpu() for k, v in render.items()}, optimized={k: v.cpu() for k, v in best_images.items()},
            target=target.cpu(), zero_target=zero_target.cpu(), optimized_tokens=best_z.cpu()),
            args.output / f'{case}_images.pt')
        event('pixels_done', **{k: v for k, v in result.items() if k not in ['history', 'per_sample']})


@torch.no_grad()
def response_probe(args, teacher, initial, validation, device):
    ds = PreparedDataset(validation, args.root / 'prepared_cuda/validation')
    ids = np.flatnonzero(validation.phase_labels == 2).tolist()
    loader = DataLoader(Subset(ds, ids), batch_size=16, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    for variant in ['a_baseline', 'c_film_mlp']:
        model = load_model(args.root, initial, variant, device)
        rows, spatial = [], []
        sum_s2 = sum_t2 = sum_error = sum_dot = 0.
        for raw in loader:
            batch = to_device(raw, device)
            images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(batch['obs'])
            zero_wrench = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                render, masks, condition = model(images, wrench, return_condition=True)
                render0, _, condition0 = model(images, zero_wrench, return_condition=True)
            z, z0 = encode(teacher, model, render), encode(teacher, model, render0)
            s, t = z-z0, batch['target'].float()-batch['zero'].float()
            sum_s2 += float(s.double().square().sum()); sum_t2 += float(t.double().square().sum())
            sum_error += float((s-t).double().square().sum()); sum_dot += float((s.double()*t.double()).sum())
            for i in range(len(s)):
                row = dict(index=int(batch['index'][i]), **response_metrics(s[i], t[i]),
                           alpha=float(condition['contact_alpha'][i].mean()),
                           zero_alpha=float(condition0['contact_alpha'][i].mean()))
                sc = s[i].reshape(teacher.n_obs_steps, -1, s.shape[-1])
                tc = t[i].reshape_as(sc)
                offset, within = 0, []
                for camera in teacher.obs_encoder.image_keys:
                    count = teacher.obs_encoder.patch_counts[camera]
                    for j in range(teacher.n_obs_steps):
                        sm, tm = sc[j, offset:offset+count].norm(dim=-1), tc[j, offset:offset+count].norm(dim=-1)
                        value = corr(sm, tm)
                        if value is not None:
                            within.append(value)
                        spatial.append(dict(index=row['index'], camera=camera, obs_step=j,
                                            student=sm.cpu(), teacher=tm.cpu(), correlation=value))
                    offset += count
                row['within_camera_patch_pearson'] = float(np.mean(within)) if within else None
                rows.append(row)
        elements = len(rows)*s[0].numel()
        summary = dict(variant=variant, samples=len(rows), checkpoint_step=1000,
            teacher_response_rms=(sum_t2/elements)**.5, student_response_rms=(sum_s2/elements)**.5,
            amplitude_ratio=(sum_s2/sum_t2)**.5, joint_response_cosine=sum_dot/(sum_s2*sum_t2)**.5,
            mean_sample_cosine=float(np.mean([r['response_cosine'] for r in rows])),
            mean_within_camera_patch_pearson=float(np.mean([r['within_camera_patch_pearson'] for r in rows
                                                           if r['within_camera_patch_pearson'] is not None])),
            response_difference_mse=sum_error/elements, no_student_response_mse=sum_t2/elements,
            mean_alpha=float(np.mean([r['alpha'] for r in rows])), mean_zero_alpha=float(np.mean([r['zero_alpha'] for r in rows])),
            intervention='physical-zero torque; both teacher and student recompute contact gating', per_sample=rows)
        write_json(args.output / f'response_{variant}.json', summary)
        torch.save(spatial, args.output / f'response_maps_{variant}.pt')
        event('response_done', **{k: v for k, v in summary.items() if k != 'per_sample'})
        del model


def gradient_probe(args, teacher, initial, validation, selection, device):
    rows = []
    # FP32 and native eval-mode LSTM backward for numerical sensitivity checks.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for variant in ['a_baseline', 'c_film_mlp']:
        model = load_model(args.root, initial, variant, device)
        original_state = tensor_hash(model.state_dict())
        model.set_training_stage('rgb'); model.eval()
        for sample in [selection[0], selection[-1]]:
            obs = to_device(next(iter(DataLoader(Subset(validation, [sample['index']]), batch_size=1)))['obs'], device)
            images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
            with torch.no_grad():
                target = teacher._encode_context(obs, apply_curriculum_mask=False).float()
                zero_target = teacher._encode_context(zero_wrench_reference(teacher, obs), apply_curriculum_mask=False).float()
            zero_wrench = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
            model.zero_grad(set_to_none=True)
            captured = []
            original_forward = model.backbone.forward
            def capture(*inputs, **kw):
                result = original_forward(*inputs, **kw, return_decoder_features=True)
                raw, _, decoded, fused = result
                features = {'/8': fused, '/4': decoded[0], '/2': decoded[1], '/1': decoded[2], 'raw_rgb': raw}
                for value in features.values():
                    value.retain_grad()
                captured.append(features)
                return result[:2]
            model.backbone.forward = capture
            try:
                with torch.backends.cudnn.flags(enabled=False):
                    render = model(images, wrench)
                    render0 = model(images, zero_wrench)
                    for value in [*render.values(), *render0.values()]:
                        value.retain_grad()
                    z = encode(teacher, model, render, grad=True, bf16=False)
                    z0 = encode(teacher, model, render0, grad=True, bf16=False)
                    loss = ((z-z0)-(target-zero_target)).square().mean()
                    loss.backward()
            finally:
                model.backbone.forward = original_forward
            feature_rows = []
            for index, features in enumerate(captured):
                branch = 'normal' if index < len(images) else 'zero'
                camera = list(images)[index % len(images)]
                for scale, h in features.items():
                    g = h.grad.detach().float()
                    feature_rows.append(dict(branch=branch, camera=camera, scale=scale,
                        shape=list(h.shape), feature_rms=float(h.detach().float().square().mean().sqrt()),
                        gradient_l2=float(g.norm()), gradient_rms=float(g.square().mean().sqrt()),
                        gradient_times_feature_rms=float((g*h.detach().float()).square().mean().sqrt())))
            pixel_gradients = {k: dict(l2=float(v.grad.norm()), rms=float(v.grad.square().mean().sqrt()),
                                      zero_branch_l2=float(render0[k].grad.norm())) for k, v in render.items()}
            parameter_gradients = {}
            for name, prefix in [('decoder_4', 'backbone.decoders.0.'), ('decoder_2', 'backbone.decoders.1.'),
                                 ('decoder_1', 'backbone.decoders.2.'), ('film', 'backbone.decoder_film.'),
                                 ('ca', 'backbone.cross_attention.'), ('lstm', 'wrench_encoder.')]:
                grads = [p.grad.detach().float().square().sum() for k, p in model.named_parameters()
                         if k.startswith(prefix) and p.grad is not None]
                parameter_gradients[name] = float(torch.stack(grads).sum().sqrt()) if grads else 0.
            # Input-direction derivative: isolate RGB -> DINO -> adapter, keeping the zero branch fixed.
            camera = list(images)[0]
            current = {k: v.detach() for k, v in render.items()}
            g = render[camera].grad.detach().clone()
            direction = -g / g.norm().clamp_min(1e-20)
            direction *= current[camera].numel()**.5
            direction *= ((current[camera] > .01) & (current[camera] < .99)).to(direction)
            direction /= direction.abs().max().clamp_min(1e-20)
            predicted = float((g*direction).sum())
            finite = []
            with torch.no_grad():
                for epsilon in (1e-3, 3e-4):
                    values = []
                    for sign in (1, -1):
                        perturbed = dict(current)
                        perturbed[camera] = current[camera] + sign*epsilon*direction
                        pz = encode(teacher, model, perturbed, bf16=False)
                        values.append(float(((pz-z0.detach())-(target-zero_target)).square().mean()))
                    numerical = (values[0]-values[1])/(2*epsilon)
                    finite.append(dict(epsilon=epsilon, autograd=predicted, finite_difference=numerical,
                                       relative_error=abs(numerical-predicted)/max(abs(predicted), 1e-12)))
            assert all(p.grad is None for p in teacher.parameters())
            assert all(p.grad is None for p in model.context_adapter.parameters())
            assert original_state == tensor_hash(model.state_dict())
            row = dict(variant=variant, **sample, response_loss=float(loss.detach()),
                pixel_gradients=pixel_gradients, feature_gradients=feature_rows,
                parameter_gradient_l2=parameter_gradients, finite_difference=finite,
                teacher_and_adapter_frozen=True, no_weights_updated=True,
                precision='FP32, TF32 disabled, native eval LSTM')
            rows.append(row)
            write_json(args.output / 'gradient_diagnostics.json', rows)
            event('gradient_done', variant=variant, index=sample['index'], loss=row['response_loss'],
                  parameter_gradients=parameter_gradients, finite_difference=finite)
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['pixels', 'response', 'all'], default='all')
    parser.add_argument('--cases', nargs='+', default=['full_0.02', 'full_0.05', 'full_0.10', 'gated_0.10'])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=.003)
    args = parser.parse_args()
    args.root = args.root.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0')
    teacher, _, initial = load_inputs(args.root/'inputs', device)
    _, validation, _ = datasets_from_reference(args.root/'inputs', args.root/'decoded_cuda')
    selection = selected_contacts(validation)
    assert selection and len({s['index'] for s in selection}) == len(selection)
    event('selected', samples=selection)
    if args.task in ['pixels', 'all']:
        model = load_model(args.root, initial, 'c_film_mlp', device)
        before = tensor_hash(model.state_dict())
        pixel_probe(args, teacher, model, validation, selection, device)
        assert before == tensor_hash(model.state_dict())
        assert all(p.grad is None for p in teacher.parameters())
        del model
    if args.task in ['response', 'all']:
        response_probe(args, teacher, initial, validation, device)
        gradient_probe(args, teacher, initial, validation, selection, device)
    event('complete', task=args.task)


if __name__ == '__main__':
    main()
