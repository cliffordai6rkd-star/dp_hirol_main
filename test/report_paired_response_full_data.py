"""A/B tables, episode diagnostics, generalization gaps and contact visualizations."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from student_dual_path_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    summaries = {g: json.loads((root/g/'summary.json').read_text()) for g in ['a', 'b']}
    a, b = [s['manifest'] for s in summaries.values()]
    for key in ['starting_state_sha256', 'checkpoint_sha256', 'teacher_checkpoint_sha256',
                'frozen_hashes', 'splits', 'optimizer', 'response_rms_threshold']:
        assert a[key] == b[key], key
    assert all(s['frozen_teacher_dino_dit_adapter_condition_verified'] for s in summaries.values())
    phases = ['free', 'precontact', 'contact']
    keys = ['token_mse', 'zero_token_mse', 'response_difference_mse', 'response_cosine',
            'response_rms_ratio', 'rgb_rms', 'rgb_max']
    comparison = dict(steps={g: s['steps'] for g, s in summaries.items()},
        initialization_verified=True, teacher_checkpoint_sha256=a['teacher_checkpoint_sha256'],
        threshold=a['response_rms_threshold'], zero_branch={g: s['manifest']['zero_branch'] for g, s in summaries.items()},
        coefficients={g: s['manifest']['coefficients'] for g, s in summaries.items()},
        metrics={}, episodes={}, generalization_gap={})
    for split in ['train', 'validation']:
        comparison['metrics'][split] = {}
        for phase in phases:
            rows = dict(initial=summaries['a']['initial']['metrics'][split]['phases'][phase],
                **{g: s['final']['metrics'][split]['phases'][phase] for g, s in summaries.items()})
            comparison['metrics'][split][phase] = rows
            print(f'\n{split} / {phase}')
            print('| Group | Token MSE | Zero MSE | Delta MSE | Active Cosine | RMS Ratio | RGB RMS | RGB Max |')
            print('|---|---:|---:|---:|---:|---:|---:|---:|')
            for label, row in rows.items():
                values = ['n/a' if row[k] is None else f'{row[k]:.6f}' for k in keys]
                print('| '+label+' | '+' | '.join(values)+' |')
    for g, summary in summaries.items():
        comparison['episodes'][g] = summary['final']['metrics']['validation']['episodes']
        comparison['generalization_gap'][g] = summary['final']['generalization_gap']
    histories = {g: json.loads((root/g/'log.json').read_text()) for g in ['a', 'b']}
    best = {}
    for g, history in histories.items():
        candidates = [r for r in history if r['kind'] == 'validation']
        row = min(candidates, key=lambda r: r['metrics']['validation']['phases']['contact']['token_mse'])
        best[g] = dict(step=row['step'], validation=row['metrics']['validation'])
    comparison['best_validation_checkpoint'] = best
    comparison['limitations'] = ('One seed. Frozen MLP and condition checkpoint were trained with the previous teacher; '
        'no recalibration was performed. Training and validation episodes are disjoint for this student experiment. '
        'A/B differ only in additional loss terms. All phase/episode metrics are sample-weighted; '
        'active cosine/RMS use a TRAIN-derived frozen teacher threshold. No action rollout is evaluated.')
    write_json(root/'comparison.json', comparison)

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for g, history in histories.items():
        rows = [r for r in history if r['kind'] == 'validation']
        for i, phase in enumerate(phases):
            for j, key in enumerate(keys[:4]):
                axes[i, j].plot([r['step'] for r in rows],
                    [r['metrics']['validation']['phases'][phase][key] for r in rows], '-o', label=g.upper())
                axes[i, j].set(title=f'{phase}: {key}', xlabel='Additional optimizer steps')
                axes[i, j].grid(alpha=.2)
    axes[0, 0].legend()
    fig.tight_layout(); fig.savefig(root/'validation_curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for g, history in histories.items():
        rows = [r for r in history if r['kind'] == 'train']
        for ax, key in zip(axes, ['global_mse', 'response', 'delta']):
            if key in rows[-1]['losses']:
                ax.plot([r['step'] for r in rows], [r['losses'][key] for r in rows], label=g.upper())
            ax.set(title=key, xlabel='Optimizer steps'); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(root/'training_curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for g in histories:
        rows = [r for r in histories[g] if r['kind'] == 'train']
        for ax, key in zip(axes, ['cnn', 'ca', 'lstm', 'decoder_film']):
            ax.plot([r['step'] for r in rows], [r['gradient_norms'][key] for r in rows], label=g.upper())
            ax.set(title=key+' gradient norm', xlabel='Optimizer steps', yscale='log'); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(root/'module_gradients.png', dpi=160); plt.close(fig)

    snapshots = {g: torch.load(root/g/f"visuals_{s['steps']}.pt", map_location='cpu', weights_only=False)
                 for g, s in summaries.items()}
    threshold = a['response_rms_threshold']
    for first in snapshots['a']:
        ep = first['episode']
        samples = {g: next(v for v in values if v['episode'] == ep) for g, values in snapshots.items()}
        assert samples['a']['index'] == samples['b']['index']
        cameras = list(first['original'])
        times = first['original'][cameras[0]].shape[0]
        patches = first['teacher_response'].shape[0]//times//len(cameras)
        side = int(patches**.5)
        assert side*side == patches
        target = first['teacher_response'].reshape(times, len(cameras), patches, -1)
        amplitude = target.square().mean(-1).sqrt()
        fig, axes = plt.subplots(len(cameras), 3, figsize=(11, 6), squeeze=False, constrained_layout=True)
        for i, camera in enumerate(cameras):
            axes[i, 0].imshow(amplitude[-1, i].reshape(side, side), cmap='magma')
            axes[i, 0].set_title(camera+' teacher response RMS')
            for j, g in enumerate(['a', 'b'], 1):
                response = samples[g]['student_response'].reshape_as(target)
                cosine = F.cosine_similarity(response[-1, i], target[-1, i], dim=-1)
                values = np.ma.masked_where((amplitude[-1, i] <= threshold).numpy(), cosine.numpy()).reshape(side, side)
                cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('lightgray')
                heat = axes[i, j].imshow(values, cmap=cmap, vmin=-1, vmax=1)
                axes[i, j].set_title(g.upper()+' response cosine')
        for ax in axes.flat: ax.axis('off')
        fig.colorbar(heat, ax=axes[:, 1:], shrink=.65)
        fig.suptitle(f'Validation episode {ep}; gray = below TRAIN-derived noise threshold')
        fig.savefig(root/f'contact_response_episode_{ep}.png', dpi=160); plt.close(fig)
        fig, axes = plt.subplots(2*len(cameras), 3, figsize=(10, 10), squeeze=False, constrained_layout=True)
        maximum = max(float((s['rendered'][k][-1]-s['original'][k][-1]).abs().mean(0).max())
                      for s in samples.values() for k in cameras)
        for i, camera in enumerate(cameras):
            original = first['original'][camera][-1].float()
            for j, g in enumerate(['original', 'a', 'b']):
                image = original if g == 'original' else samples[g]['rendered'][camera][-1].float()
                axes[2*i, j].imshow(image.permute(1, 2, 0).clamp(0, 1)); axes[2*i, j].set_title(camera+' '+g.upper())
                heat = axes[2*i+1, j].imshow((image-original).abs().mean(0), cmap='magma', vmin=0, vmax=maximum)
        for ax in axes.flat: ax.axis('off')
        fig.colorbar(heat, ax=axes, shrink=.6)
        fig.savefig(root/f'contact_rgb_episode_{ep}.png', dpi=160); plt.close(fig)
    print('\nIndependent episode contact results:')
    for episode in comparison['episodes']['a']:
        print(json.dumps(dict(episode=episode, **{g: comparison['episodes'][g][episode]['contact'] for g in ['a', 'b']})))


if __name__ == '__main__':
    main()
