"""Regression for spawning the real local LeRobot reader and paired cache."""

from pathlib import Path
import copy
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_policy.common.lerobot_v3_io import LeRobotV3Dataset, LeRobotV3Writer
from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache
from diffusion_policy.model.vision.render_net import RenderNet


class ReaderSamples(Dataset):
    def __init__(self, reader):
        self.reader = reader

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        return {'obs': {'state': self.reader[index]['observation.state']}}


def test_spawn_keeps_local_reader_and_shuffled_paired_targets_identical(tmp_path):
    root = tmp_path / 'dataset'
    writer = LeRobotV3Writer(str(root), fps=25, video_keys=[], features={
        'observation.state': {'dtype': 'float32', 'shape': (2,)},
    })
    for index in range(6):
        writer.add_frame({'observation.state': np.array([index, index + 1], dtype=np.float32)})
    writer.save_episode('spawn test')
    writer.finalize()
    reader = LeRobotV3Dataset(str(root))
    restored = pickle.loads(pickle.dumps(reader))
    torch.testing.assert_close(restored[3]['observation.state'], reader[3]['observation.state'],
                               rtol=0, atol=0)
    cache = TeacherTargetCache(tmp_path / 'targets', {'fixed_teacher': True}, 6,
                               token_dtype='float16')
    tokens = torch.arange(6).float()[:, None, None].expand(-1, 2, 3).clone()
    cache.append(tokens, {'side': tokens[..., 0]}, zero_tokens=-tokens)
    samples = CachedTeacherDataset(ReaderSamples(reader), cache)
    for workers in (0, 2):
        options = dict(num_workers=workers)
        if workers:
            options['multiprocessing_context'] = 'spawn'
        loader = DataLoader(samples, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(42), **options)
        observed = []
        for batch in loader:
            indices = batch['obs']['state'][:, 0].long()
            observed.extend(indices.tolist())
            torch.testing.assert_close(batch['teacher_targets']['tokens'], tokens[indices].half(),
                                       rtol=0, atol=0)
            torch.testing.assert_close(batch['teacher_targets']['zero_tokens'], -tokens[indices].half(),
                                       rtol=0, atol=0)
        if workers == 0:
            reference_order = observed
        else:
            assert observed == reference_order


def test_compiled_spatial_decoder_matches_values_gradients_and_checkpoint_keys():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(34)
        eager = RenderNet(
            wrench_dim=6, base_channels=4, conditioning_dim=8,
            image_keys=['side', 'wrist'], wrench_history_steps=3, max_obs_steps=2,
            patch_size=8, embed_dim=16, num_heads=2, image_self_attn_layers=1,
            cross_attn_layers=1, decoder_layers=2, conv_block='bottleneck',
            separate_gate_path=True, spatial_modulation={'enabled': True},
            high_attention_render={'enabled': True, 'threshold': 0.0},
        )
        eager.set_training_stage('rgb')
        eager.set_stage_mode(True)
        with torch.no_grad():
            eager.output.weight.normal_(std=0.03)
        compiled = copy.deepcopy(eager)
        compiled.compile_backbone(backend='inductor', fullgraph=True, dynamic=False)
        implementation = compiled._compiled_backbone_forward
        calls = []

        def counted(*args, **kwargs):
            calls.append(kwargs)
            return implementation(*args, **kwargs)

        compiled._compiled_backbone_forward = counted
        images = {key: torch.rand(1, 2, 3, 32, 32) * .5 + .25
                  for key in ('side', 'wrist')}
        wrench = torch.randn(1, 2, 3, 6)
        expected = eager(images, wrench)
        actual = compiled(images, wrench)
        assert len(calls) == 2
        assert all('modulation_gate' in c for c in calls)
        torch.testing.assert_close(actual, expected)
        sum(v.square().mean() for v in expected.values()).backward()
        sum(v.square().mean() for v in actual.values()).backward()
        reference = dict(eager.named_parameters())
        for name, parameter in compiled.named_parameters():
            if reference[name].grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, reference[name].grad,
                                           atol=2e-7, rtol=3e-4, msg=name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in compiled.spatial_modulations.parameters())
        assert tuple(compiled.state_dict()) == tuple(eager.state_dict())
        compiled.eval()
        eager.eval()
        with torch.no_grad():
            torch.testing.assert_close(compiled(images, wrench), eager(images, wrench), rtol=0, atol=0)
        assert len(calls) == 2  # Evaluation retains the eager reference path.
    finally:
        torch.set_num_threads(previous_threads)
