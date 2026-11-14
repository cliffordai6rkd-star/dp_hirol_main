"""Compare engineering optimizations in the complete RGB distillation step.

Uses real prepared samples and separate checkpoint copies. Includes the frozen
condition predictor, DINO input gradients, response loss, AdamW, clipping and
EMA. Dataset decoding, validation and diagnostic-only metrics are excluded.
The production process is resumed by a pipe watcher even if the probe fails.
"""

import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
import torch
from omegaconf import OmegaConf

from benchmark_render_backbone_compile import pause_training
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.response_distillation import response_weighted_token_mse
from diffusion_policy.scripts.visualize_cross_attention_episode import (
    instantiate_policy, load_checkpoint_payload,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--pause-pid', type=int)
    parser.add_argument('--emulate-precision-casts', action='store_true',
                        help='Preserve BF16 rounding at fused intermediate operations')
    parser.add_argument('--preserve-layout', action='store_true',
                        help='Retain eager convolution layouts instead of compiler layout changes')
    parser.add_argument('--eager-control', action='store_true',
                        help='Repeat eager baseline to verify stochastic replay')
    parser.add_argument('--optimization', choices=('renderer', 'updates', 'cudnn'), default='renderer')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', pickle_module=dill,
                            weights_only=False)
    cfg = checkpoint['cfg']
    if cfg.distillation.use_teacher_ffn or cfg.render_net.get('context_adapter', {}).get('enabled', False):
        raise ValueError('This probe measures the response student without FFN or adapter')
    samples = torch.load(args.samples, map_location='cpu', weights_only=False)
    # Mix free, approaching-contact and contact training windows.
    candidates = [samples[index] for index in (0, 2, 4, 5)]
    selected = [candidates[index % len(candidates)] for index in range(args.batch_size)]
    images_cpu = {key: torch.cat([s['images'][key] for s in selected])
                  for key in selected[0]['images']}
    wrench_cpu = torch.cat([s['wrench'] for s in selected])
    target_cpu = torch.cat([s['target'] for s in selected]).half()
    response_cpu = torch.cat([s['response'] for s in selected])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    comparisons = []
    with pause_training(args.pause_pid):
        device = torch.device('cuda:0')
        payload, teacher_cfg = load_checkpoint_payload(Path(cfg.teacher_checkpoint))
        teacher = instantiate_policy(payload, teacher_cfg, device, capture_attention=False)
        teacher.eval().requires_grad_(False)
        del payload
        gc.collect()
        images = {k: v.to(device) for k, v in images_cpu.items()}
        wrench = wrench_cpu.to(device)
        target = target_cpu.to(device)
        response = response_cpu.to(device)
        encode = torch.compile(teacher.obs_encoder.encode_image_context,
                               backend='inductor', mode='default', fullgraph=True, dynamic=False)
        cpu_rng = torch.get_rng_state()
        reference_tokens = reference_grads = reference_loss = None
        for optimized in ((False, False, True) if args.eager_control else (False, True)):
            compile_renderer = optimized and args.optimization == 'renderer'
            fused_updates = optimized and args.optimization == 'updates'
            torch.backends.cudnn.benchmark = optimized and args.optimization == 'cudnn'
            torch.set_rng_state(cpu_rng)
            renderer = build_render_net(
                wrench.shape[-1], OmegaConf.to_container(cfg.render_net, resolve=True),
            ).to(device)
            renderer.load_state_dict(checkpoint['state_dicts']['model'])
            renderer.set_training_stage('rgb')
            renderer.set_stage_mode(True)
            ema_model = copy.deepcopy(renderer).eval().requires_grad_(False)
            ema_model.load_state_dict(checkpoint['state_dicts']['ema_model'])
            ema = EMAModel(ema_model, **OmegaConf.to_container(cfg.ema, resolve=True))
            ema.load_state_dict(checkpoint['state_dicts']['ema'])
            optimizer = torch.optim.AdamW(
                renderer.parameters(), **OmegaConf.to_container(cfg.optimizer, resolve=True),
            )
            optimizer_state = copy.deepcopy(checkpoint['state_dicts']['optimizer'])
            if fused_updates:
                for group in optimizer_state['param_groups']:
                    group['fused'] = True
                    group['foreach'] = None
                averaged = dict(ema_model.named_parameters())
                source_parameters = [p.detach() for p in renderer.parameters() if p.requires_grad]
                averaged_parameters = [averaged[k] for k, p in renderer.named_parameters() if p.requires_grad]
                assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                               for m in renderer.modules())
            optimizer.load_state_dict(optimizer_state)
            del optimizer_state

            def update_ema():
                if not fused_updates:
                    ema.step(renderer)
                    return
                # This renderer has no BatchNorm parameters. Match the original
                # two separate FP32 operations, including their rounding order.
                ema.decay = ema.get_decay(ema.optimization_step)
                torch._foreach_mul_(averaged_parameters, ema.decay)
                torch._foreach_add_(averaged_parameters, source_parameters, alpha=1 - ema.decay)
                ema.optimization_step += 1
            if compile_renderer:
                renderer.compile_backbone(backend='inductor',
                                           fullgraph=True, dynamic=False,
                                           options={'emulate_precision_casts': args.emulate_precision_casts,
                                                    'layout_optimization': not args.preserve_layout})

            def step(capture=False):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    rendered = renderer(images, wrench)
                    tokens = encode(rendered, input_grad=True)
                    global_mse, response_mse, _, _ = response_weighted_token_mse(
                        tokens, target, response,
                        noise_floor=float(cfg.distillation.response_noise_floor),
                    )
                    loss = global_mse + float(cfg.distillation.response_loss_weight) * response_mse
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(renderer.parameters(),
                                                           float(cfg.training.max_grad_norm),
                                                           error_if_nonfinite=True)
                if capture:
                    captured = (float(loss.detach()), tokens.detach().float().cpu(),
                                {k: p.grad.detach().float().cpu().clone()
                                 for k, p in renderer.named_parameters() if p.grad is not None})
                optimizer.step()
                update_ema()
                # Production logging reads loss and gradient norm each step.
                values = float(loss.detach()), float(grad_norm)
                return captured if capture else values

            torch.cuda.reset_peak_memory_stats()
            # manual_seed also resets cuDNN's separate recurrent dropout state;
            # set_rng_state alone does not replay the two-layer LSTM dropout.
            torch.manual_seed(42)
            started = time.perf_counter()
            loss, tokens, grads = step(capture=True)
            torch.cuda.synchronize()
            first_step = time.perf_counter() - started
            if reference_tokens is None:
                reference_loss, reference_tokens, reference_grads = loss, tokens, grads
            token_rmse = (tokens - reference_tokens).square().mean().sqrt().item()
            grad_error = (sum((g - reference_grads[k]).square().sum() for k, g in grads.items()).sqrt()
                          / sum(g.square().sum() for g in reference_grads.values()).sqrt()).item()
            comparison = dict(compile_renderer=compile_renderer, fused_updates=fused_updates,
                              first_step_seconds=first_step,
                              loss=loss, token_rmse=token_rmse,
                              loss_relative_error=abs(loss - reference_loss) / max(reference_loss, 1e-12),
                              all_gradient_relative_l2=grad_error)
            comparison['numerical_check_passed'] = grad_error < .1 and comparison['loss_relative_error'] < .01
            comparisons.append(comparison)
            args.output.with_suffix('.comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
            print(json.dumps(comparison), flush=True)
            if fused_updates:
                assert token_rmse == 0, token_rmse
                assert grad_error < .01, grad_error
            for _ in range(8):
                step()
            torch.cuda.synchronize()
            timings = []
            for _ in range(args.steps):
                started = time.perf_counter()
                step()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
            result = dict(compile_renderer=compile_renderer, compiled_dino=True,
                          cudnn_benchmark=torch.backends.cudnn.benchmark,
                          fused_updates=fused_updates, optimization=args.optimization,
                          emulate_precision_casts=args.emulate_precision_casts,
                          preserve_layout=args.preserve_layout,
                          batch_size=args.batch_size, steps=args.steps,
                          median_seconds=statistics.median(timings),
                          mean_seconds=statistics.mean(timings),
                          samples_per_second=args.batch_size / statistics.mean(timings),
                          first_step_seconds=first_step, initial_loss=loss,
                          numerical_check_passed=comparison['numerical_check_passed'],
                          loss_relative_error=comparison['loss_relative_error'],
                          token_rmse=token_rmse, all_gradient_relative_l2=grad_error,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          includes_optimizer_step=True, includes_ema=True,
                          includes_data_loading=False, includes_diagnostic_metrics=False)
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            del optimizer, ema, ema_model, renderer, tokens, grads, step, update_ema
            if fused_updates:
                del averaged, source_parameters, averaged_parameters
            gc.collect()
            torch.cuda.empty_cache()
        assert all(p.grad is None for p in teacher.parameters())
        torch.cuda.synchronize()
    print(json.dumps({'distillation_step_speedup': results[0]['median_seconds']
                      / results[-1]['median_seconds']}), flush=True)


if __name__ == '__main__':
    main()
