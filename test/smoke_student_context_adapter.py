"""Three short production-loop epochs with real images and the pinned teacher.

Runs in an isolated temporary directory, without decoding the full dataset or
touching an existing training run. Each epoch selects one training stage.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from diffusion_policy.scripts.visualize_render_net_episode import (
    encode_student_context, load_render_net,
)
from diffusion_policy.workspace.train_distill_render_net_workspace import (
    TrainDistillRenderNetWorkspace, student_context_tokens,
)


class TinyDataset(Dataset):
    def __init__(self, samples, teacher, split):
        self.rows = []
        self.lerobot_dataset = SimpleNamespace(close=lambda: None)
        for phase in range(3):
            sample = next(s for s in samples if s['split'] == split and s['phase'] == phase)
            obs = {k: v.squeeze(0) for k, v in sample['images'].items()}
            raw = teacher.normalizer[teacher.wrench_key].unnormalize(sample['wrench'].to(teacher.device))
            for key in dict.fromkeys((teacher.wrench_key, teacher.phase_wrench_key, teacher.future_phase_wrench_key)):
                if key is not None:
                    obs[key] = raw.squeeze(0).cpu()
            obs['contact_alpha'] = sample['true_alpha'].squeeze(0)
            self.rows.append({'obs': obs, 'action': sample['action'].squeeze(0)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def get_validation_dataset(self):
        return self.validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path,
                        default=ROOT / 'data/diagnostics/2026.11.14/response_distillation/samples.pt')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'data/diagnostics/2026.11.14/student_adapter_smoke')
    parser.add_argument('--compile-student-dino', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(ROOT / 'diffusion_policy/config'), version_base=None):
        cfg = compose(config_name='train_distill_render_net_student_response_adapter', overrides=[
            'teacher_cache.enabled=false',
            f'training.compile.enabled={str(args.compile_student_dino).lower()}',
            'training.num_epochs=3', 'training.render_stages.adapter_epochs=1',
            'training.render_stages.intensity_epochs=1', 'dataloader.batch_size=1',
            'val_dataloader.batch_size=1', 'dataloader.num_workers=0',
            'dataloader.persistent_workers=false', 'val_dataloader.num_workers=0',
            'logging.progress_bar=false', 'logging.plot_every_steps=10000',
        ])
    OmegaConf.resolve(cfg)
    with tempfile.TemporaryDirectory(prefix='production_loop_', dir=args.output_dir) as temporary:
        workspace = TrainDistillRenderNetWorkspace(cfg, output_dir=temporary)
        samples = torch.load(args.samples, map_location='cpu', weights_only=False)
        training = TinyDataset(samples, workspace.teacher_policy, 'train')
        training.validation = TinyDataset(samples, workspace.teacher_policy, 'validation')
        workspace._instantiate_dataset = lambda config: training
        workspace.run()
        summary = json.loads((Path(temporary) / 'training_summary.json').read_text())
        checkpoints = {}
        for tag in ('best_adapter', 'best_condition', 'latest'):
            path = workspace.get_checkpoint_path(tag)
            payload = torch.load(path, map_location='cpu', pickle_module=dill, weights_only=False)
            checkpoints[tag] = {
                'stage': dill.loads(payload['pickles']['training_stage']),
                'stage_epoch': dill.loads(payload['pickles']['stage_epoch']),
                'has_adapter': any(k.startswith('context_adapter.') for k in payload['state_dicts']['model']),
            }
        assert [checkpoints[tag]['stage'] for tag in checkpoints] == ['adapter', 'intensity', 'rgb']
        assert summary['global_step'] == 9 and summary['teacher_trainable_parameters'] == 0
        model, preprocessor, _, _, _, _ = load_render_net(workspace.get_checkpoint_path('latest'), 'cuda:0')
        obs = {k: v.unsqueeze(0).to(workspace.device) for k, v in training.validation[2]['obs'].items()}
        with torch.no_grad(), workspace._autocast():
            expected = student_context_tokens(workspace.teacher_policy, model, obs)
            actual = encode_student_context(model, preprocessor, obs, workspace.teacher_policy.obs_encoder, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(not p.requires_grad and p.grad is None for p in workspace.teacher_policy.parameters())
        summary.update(checkpoint_stages=checkpoints, inference_max_abs=(actual - expected).abs().max().item(),
                       adapter_parameters=sum(p.numel() for p in model.context_adapter.parameters()),
                       verification_only=True, full_training_started=False)
        for path in Path(temporary).glob('*.jsonl'):
            shutil.copy2(path, args.output_dir / path.name)
        log_path = Path(temporary) / 'logs.json.txt'
        if log_path.is_file():
            shutil.copy2(log_path, args.output_dir / log_path.name)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k: summary[k] for k in ('global_step', 'checkpoint_stages', 'inference_max_abs',
                                                 'adapter_parameters', 'gpu_peak_allocated_mib')}, indent=2))
    gc.collect()


if __name__ == '__main__':
    main()
