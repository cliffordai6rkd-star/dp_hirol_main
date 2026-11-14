"""Inspect the real RGB-to-DINO-to-adapter gradient without changing training."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from diffusion_policy.model.vision.response_distillation import response_weighted_token_mse
from diffusion_policy.scripts.visualize_cross_attention_episode import load_checkpoint_payload, instantiate_policy
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


def norm(value):
    return None if value is None else float(value.detach().float().norm())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Native recurrent backward permits eval-mode LSTMs; disabling cuDNN here
    # keeps dropout off for the deterministic finite-difference probe only.
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda:0')
    boundary, _, cfg, _, _, _ = load_render_net(args.run_dir / 'checkpoints/best_condition.ckpt', str(device))
    trained, _, _, _, _, _ = load_render_net(args.run_dir / 'checkpoints/best.ckpt', str(device))
    payload, teacher_cfg = load_checkpoint_payload(Path(cfg.teacher_checkpoint))
    teacher = instantiate_policy(payload, teacher_cfg, device, capture_attention=False)
    teacher.eval().requires_grad_(False)
    del payload
    samples = torch.load(ROOT / 'data/diagnostics/2026.11.14/response_distillation/samples.pt', map_location='cpu', weights_only=False)
    selected = [next(s for s in samples if s['split'] == split and s['phase'] == 2)
                for split in ('train', 'validation')]
    images = {k: torch.cat([s['images'][k] for s in selected]).to(device) for k in selected[0]['images']}
    wrench = torch.cat([s['wrench'] for s in selected]).to(device)
    target = torch.cat([s['target'] for s in selected]).to(device)
    response = torch.cat([s['response'] for s in selected]).to(device)

    def loss_for(rendered, model):
        tokens = teacher.obs_encoder.encode_image_context(rendered, input_grad=True,
                                                          context_adapter=model.context_adapter)
        global_mse, weighted, _, _ = response_weighted_token_mse(tokens, target, response)
        return global_mse + weighted, tokens

    rows = []
    for name in ('zero_initial', 'zero_after_one_update', 'random_initial', 'trained_best'):
        if name == 'zero_initial':
            model = copy.deepcopy(boundary)
            model.set_training_stage('rgb')
            optimizer = torch.optim.AdamW(model.rgb_parameters(), lr=2e-4, weight_decay=1e-4)
        elif name == 'random_initial':
            model = copy.deepcopy(boundary)
            model.set_training_stage('rgb')
            torch.manual_seed(42)
            model.output.reset_parameters()
        elif name == 'trained_best':
            model = trained
            model.set_training_stage('rgb')
        model.eval()  # Deterministic probe; keep RGB parameters trainable.
        model.zero_grad(set_to_none=True)
        rendered, masks, condition = model(images, wrench, return_condition=True)
        for value in rendered.values():
            value.retain_grad()
        loss, tokens = loss_for(rendered, model)
        tokens.retain_grad()
        loss.backward()
        stem = next(model.backbone.stem.parameters())
        row = dict(variant=name, loss=float(loss.detach()),
                   token_gradient_norm=norm(tokens.grad),
                   output_weight_gradient_norm=norm(model.output.weight.grad),
                   rgb_stem_gradient_norm=norm(stem.grad),
                   wrench_lstm_gradient_norm=norm(model.wrench_encoder.weight_ih_l0.grad),
                   rendered_pixel_gradient_norm={k: norm(v.grad) for k, v in rendered.items()},
                   mask_open_fraction={k: float((v > model.region_threshold).float().mean()) for k, v in masks.items()},
                   predicted_alpha=condition['contact_alpha'].detach().cpu().tolist(),
                   render_mae={k: float((v.detach() - images[k]).abs().mean()) for k, v in rendered.items()},
                   saturated_fraction={k: float(((v.detach() <= 0) | (v.detach() >= 1)).float().mean()) for k, v in rendered.items()},
                   frozen_teacher_and_adapter_have_no_gradients=all(p.grad is None for p in teacher.parameters())
                      and all(p.grad is None for p in model.context_adapter.parameters()))
        assert row['token_gradient_norm'] > 0 and row['output_weight_gradient_norm'] > 0
        assert row['frozen_teacher_and_adapter_have_no_gradients']
        if name == 'zero_initial':
            assert row['rgb_stem_gradient_norm'] == 0
            torch.nn.utils.clip_grad_norm_(model.rgb_parameters(), 1.)
            optimizer.step()
        else:
            assert row['rgb_stem_gradient_norm'] > 0
        if name == 'trained_best':
            # Differentiate the complete rendering function with respect to a
            # head direction, then check it against symmetric finite differences.
            weight = model.output.weight
            direction = weight.grad.detach().clone()
            direction /= direction.norm()
            predicted = float((weight.grad.detach() * direction).sum())
            original = weight.detach().clone()
            checks = []
            with torch.no_grad():
                for epsilon in (1e-3, 3e-4):
                    losses = []
                    for sign in (1, -1):
                        weight.copy_(original + sign * epsilon * direction)
                        changed = model(images, wrench)
                        value, _ = loss_for(changed, model)
                        losses.append(float(value))
                    difference = (losses[0] - losses[1]) / (2 * epsilon)
                    checks.append(dict(epsilon=epsilon, autograd=predicted, finite_difference=difference,
                                       relative_error=abs(predicted - difference) / max(abs(predicted), 1e-12)))
                weight.copy_(original)
            row['directional_derivative_checks'] = checks
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output_dir / 'gradient_probe.json').write_text(json.dumps(dict(
        samples=[s['id'] for s in selected], precision='FP32, TF32 and cuDNN disabled for deterministic derivatives', results=rows,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
