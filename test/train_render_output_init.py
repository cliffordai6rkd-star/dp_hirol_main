"""Isolated RGB initialization ablation using trained, frozen conditions/adapter.

Restores the exact pre-RGB EMA weights from best_condition.ckpt, starts a fresh
RGB optimizer/EMA schedule, and optionally resets only the final convolution.
Does not alter the production network constructor or any existing checkpoint.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
from omegaconf import OmegaConf, open_dict
import torch

from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--initialization', choices=('zero', 'random'), default='random')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname).1s | %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.source_run / 'checkpoints/best_condition.ckpt'
    payload = torch.load(source, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(payload['cfg'], resolve=True))
    with open_dict(cfg):
        cfg.output_dir = str(args.output_dir.resolve())
        cfg.name = 'rgb_output_initialization_' + args.initialization
        cfg.training.resume_from = None
        cfg.dataloader.num_workers = 8
        cfg.val_dataloader.num_workers = 4
        cfg.val_dataloader.persistent_workers = True
        cfg.logging.progress_bar = False
        cfg.logging.status_interval_sec = 30.0
        cfg.initialization_ablation = dict(mode=args.initialization, seed=42,
            source_checkpoint=str(source.resolve()), initialization='PyTorch Conv2d.reset_parameters: uniform weights AND bias')
    workspace = TrainDistillRenderNetWorkspace(cfg, output_dir=str(args.output_dir.resolve()))
    workspace.model.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    del payload
    assert workspace.model.output.weight.count_nonzero() == 0
    assert workspace.model.output.bias.count_nonzero() == 0
    frozen_before = {name: value.detach().cpu().clone() for name, value in workspace.model.state_dict().items()
                     if name.startswith(('context_adapter.', 'condition_network.'))}
    if args.initialization == 'random':
        torch.manual_seed(42)
        workspace.model.output.reset_parameters()
    workspace.training_stage = 'rgb'
    workspace.model.set_training_stage('rgb')
    workspace.model.set_stage_mode(True)
    workspace.stage_transition_reason = 'initialization_ablation:pretrained_conditions_to_rgb'
    workspace.optimizer.state.clear()
    if workspace.ema is not None:
        workspace.ema.sync_parameters(workspace.model)
        workspace.ema.optimization_step = 0
        workspace.ema.decay = 0.
    for name, value in frozen_before.items():
        assert torch.equal(workspace.model.state_dict()[name].cpu(), value), name
    assert all(not p.requires_grad for p in workspace.model.context_adapter.parameters())
    assert all(not p.requires_grad for p in workspace.model.condition_network.parameters())
    metadata = dict(source_checkpoint=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    output_initialization=args.initialization, head_seed=42,
                    weight_std=float(workspace.model.output.weight.detach().std()),
                    bias=workspace.model.output.bias.detach().cpu().tolist(),
                    region_threshold=workspace.model.region_threshold,
                    batch_size=cfg.dataloader.batch_size, teacher=str(cfg.teacher_checkpoint),
                    adapter_and_condition_reused_exactly=True, rgb_optimizer_and_ema_reset=True,
                    production_constructor_unchanged=True)
    (args.output_dir / 'initialization_ablation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    workspace.run()


if __name__ == '__main__':
    main()
