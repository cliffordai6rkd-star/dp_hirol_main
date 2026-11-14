"""Compare the saved RGB renderer's eager and compiled spatial decoder.

This benchmark does not update weights or write production checkpoints. A
running training PID can be briefly suspended while its GPU memory stays in
place. A separate pipe watcher resumes it even if this benchmark exits early.
"""

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.vision.render_net import build_render_net


@contextlib.contextmanager
def pause_training(pid):
    if pid is None:
        yield
        return
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'train_distill_render_net.py' not in command:
        raise ValueError('pause-pid must refer to the distillation training process')
    # The pipe closes if this process dies; the watcher then releases training.
    watcher = subprocess.Popen([
        sys.executable, '-c',
        'import os,signal,sys; sys.stdin.buffer.read(); '
        'os.kill(int(sys.argv[1]), signal.SIGCONT)', str(pid),
    ], stdin=subprocess.PIPE)
    try:
        os.kill(pid, signal.SIGSTOP)
        print(json.dumps({'training_paused': pid}), flush=True)
        yield
    finally:
        watcher.stdin.close()
        watcher.wait(timeout=10)
        print(json.dumps({'training_resumed': pid}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--pause-pid', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', pickle_module=dill,
                            weights_only=False)
    config = OmegaConf.to_container(checkpoint['cfg'].render_net, resolve=True)
    samples = torch.load(args.samples, map_location='cpu', weights_only=False)
    # Include free, precontact and contact windows in a fixed mini-batch.
    selected = [samples[index % len(samples)] for index in range(args.batch_size)]
    images_cpu = {key: torch.cat([s['images'][key] for s in selected])
                  for key in selected[0]['images']}
    wrench_cpu = torch.cat([s['wrench'] for s in selected])
    cpu_rng = torch.get_rng_state()
    results = []
    reference_output = reference_grad = reference_grads = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pause_training(args.pause_pid):
        try:
            device = torch.device('cuda:0')
            images = {k: v.to(device) for k, v in images_cpu.items()}
            wrench = wrench_cpu.to(device)
            cuda_rng = torch.cuda.get_rng_state()
            for compile_backbone in (False, True):
                torch.set_rng_state(cpu_rng)
                renderer = build_render_net(wrench.shape[-1], config).to(device)
                renderer.load_state_dict(checkpoint['state_dicts']['model'])
                renderer.set_training_stage('rgb')
                renderer.set_stage_mode(True)
                if compile_backbone:
                    renderer.compile_backbone(backend='inductor', mode='default',
                                               fullgraph=True, dynamic=False)
                optimizer = torch.optim.AdamW(renderer.parameters(), lr=2e-4)

                def step(capture=False):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = renderer(images, wrench)
                        loss = sum((v.float() - .4).square().mean() for v in output.values())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(renderer.parameters(), 1.0)
                    if capture:
                        grad = renderer.output.weight.grad.detach().cpu().clone()
                        grads = {k: p.grad.detach().cpu().clone() for k, p in renderer.named_parameters()
                                 if p.grad is not None}
                        return {k: v.detach().cpu().clone() for k, v in output.items()}, grad, grads

                started = time.perf_counter()
                torch.cuda.set_rng_state(cuda_rng)
                output, grad, grads = step(capture=True)
                torch.cuda.synchronize()
                first_step_seconds = time.perf_counter() - started
                if not compile_backbone:
                    reference_output, reference_grad, reference_grads = output, grad, grads
                difference = max((output[k] - reference_output[k]).abs().max().item() for k in output)
                output_rmse = max((output[k] - reference_output[k]).square().mean().sqrt().item()
                                  for k in output)
                gradient_relative_l2 = ((grad - reference_grad).norm()
                                         / reference_grad.norm().clamp_min(1e-12)).item()
                differences = {k: ((g - reference_grads[k]).norm()
                                    / reference_grads[k].norm().clamp_min(1e-12)).item()
                               for k, g in grads.items()}
                global_grad_error = (sum((g - reference_grads[k]).square().sum()
                                         for k, g in grads.items()).sqrt()
                                      / sum(g.square().sum() for g in reference_grads.values()).sqrt()).item()
                if compile_backbone:
                    assert output_rmse < 2e-3, (difference, output_rmse)
                    assert gradient_relative_l2 < .1, gradient_relative_l2
                    assert global_grad_error < .05, global_grad_error
                for _ in range(4):
                    step()
                torch.cuda.synchronize()
                timings = []
                for _ in range(args.steps):
                    started = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - started)
                result = dict(compile_backbone=compile_backbone, batch_size=args.batch_size,
                              median_seconds=statistics.median(timings),
                              mean_seconds=statistics.mean(timings), first_step_seconds=first_step_seconds,
                              output_max_abs=difference, output_rmse=output_rmse,
                              output_weight_gradient_relative_l2=gradient_relative_l2,
                              all_gradient_relative_l2=global_grad_error,
                              per_parameter_gradient_relative_l2=differences,
                              peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                              includes_dino=False, includes_optimizer_step=False)
                results.append(result)
                args.output.write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps({k: v for k, v in result.items()
                                  if k != 'per_parameter_gradient_relative_l2'}), flush=True)
                del optimizer, renderer, output, grad, grads
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            torch.cuda.synchronize()
    print(json.dumps({'renderer_speedup': results[0]['median_seconds'] / results[1]['median_seconds']}),
          flush=True)


if __name__ == '__main__':
    main()
