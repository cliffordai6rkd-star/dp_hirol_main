"""Cache paired teacher targets for episode-disjoint capacity diagnostics."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.25)
    device = torch.device('cuda:0')
    payload, cfg = common.load_checkpoint_payload(args.checkpoint)
    teacher = common.instantiate_policy(payload, cfg, device, capture_attention=False)
    teacher.eval().requires_grad_(False)
    state_key = 'ema_model' if 'ema_model' in payload['state_dicts'] else 'model'
    del payload
    dataset = common.instantiate_episode_dataset(cfg, None, 'pyav')
    # Match the original student's split. Held-out here refers to probes, not
    # the teacher, which was trained on all episodes.
    val_eps = np.flatnonzero(get_val_mask(len(dataset.episode_ranges), .1, 42))
    train_eps = np.flatnonzero(~get_val_mask(len(dataset.episode_ranges), .1, 42))
    train_eps = train_eps[np.linspace(0, len(train_eps) - 1, 12, dtype=int)]
    splits = {'train': train_eps.tolist(), 'validation': val_eps[::2].tolist(),
              'test': val_eps[1::2].tolist()}
    records, samples = [], []
    for split, episodes in splits.items():
        for episode in episodes:
            candidates = []
            for anchor, index in common.episode_sample_indices(dataset, episode):
                seq = dataset._sample_indices_to_sequence(index)
                if dataset.window_sampling_strategy == 'timestamp':
                    seq = dataset._retime_sequence_indices(seq)
                obs_indices = seq[:teacher.n_obs_steps]
                phases = dataset._frame_contact_phase[obs_indices]
                # Both observation slots must be persistent contact.
                if (phases == 2).all():
                    candidates.append((anchor, index))
            if len(candidates) < 8:
                raise RuntimeError(f'episode {episode}: only {len(candidates)} contact windows')
            chosen = [candidates[i] for i in np.linspace(0, len(candidates)-1, 8, dtype=int)]
            for anchor, index in chosen:
                sample = dataset[index]
                assert (sample['obs']['contact_phase'] == 2).all()
                assert (sample['obs']['contact_alpha'] == 1).all()
                samples.append(sample)
                records.append({'split': split, 'episode': episode, 'anchor': anchor, 'index': index,
                                'phase': sample['obs']['contact_phase'].tolist()})
            print(f'selected {split} episode={episode} contact_windows=8', flush=True)
    targets, bases, taus, zero_gaps, images = [], [], [], [], []
    try:
        for offset in range(0, len(samples), 2):
            batch = default_collate(samples[offset:offset+2])
            obs = {k: v.to(device) for k, v in batch['obs'].items()}
            with torch.no_grad():
                target = teacher._encode_context(obs, apply_curriculum_mask=False)
                zero = teacher._encode_context(make_zero_wrench_obs(teacher, obs), apply_curriculum_mask=False)
                img, tau, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
                base = teacher.obs_encoder.encode_image_context(img, apply_fusion_ffn=True)
                torch.testing.assert_close(base, zero, rtol=1e-5, atol=1e-5)
                energy = (target - base).float().square().mean((1,2))
                assert (energy > 1e-4).all(), f'Unexpected zero target at offset={offset}: {energy}'
                for j, value in enumerate(energy.tolist()):
                    records[offset+j]['delta_mse'] = value
            targets.append(target.cpu()); bases.append(base.cpu()); taus.append(tau.cpu())
            zero_gaps.append(float((base-zero).abs().max()))
            if offset == 0 or offset % 16 == 0:
                print(f'encoded {offset+len(target)}/{len(samples)}; energy={energy.tolist()}', flush=True)
    finally:
        dataset.lerobot_dataset.close()
    args.output.mkdir(parents=True, exist_ok=True)
    # Cache full windows, retaining each observation's own wrench history.
    base = torch.cat(bases)
    target = torch.cat(targets)
    steps = teacher.n_obs_steps
    cache = {'base': base.reshape(len(samples), steps, -1, base.shape[-1]),
             'delta': (target-base).reshape(len(samples), steps, -1, base.shape[-1]),
             'tau': torch.cat(taus), 'records': records,
             'obs': default_collate(samples)['obs']}
    torch.save(cache, args.output/'paired_targets.pt')
    report = {'checkpoint': str(args.checkpoint.resolve()), 'teacher_state': state_key,
              'split_episodes': splits, 'windows': len(samples), 'observations': len(samples)*steps,
              'image_zero_path_max_abs_gap': max(zero_gaps),
              'teacher_saw_all_episodes': True, 'targets': 'FP32, no augmentation, phase2 only',
              'records': records}
    (args.output/'data_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('CACHE_READY', flush=True)


if __name__ == '__main__':
    main()
