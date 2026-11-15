"""Summarize paired Oracle direction diagnostics and the independent A/B pilot."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args(); root = args.root
    oracle_root = root/'oracle_direction_12'
    oracle = json.loads((oracle_root/'oracle_direction.json').read_text())
    rgb = json.loads((root/'rgb_l2_oracle_10/rgb_l2_oracle.json').read_text())
    destination = root/'response_direction_300'
    results = {g:json.loads((destination/g/'summary.json').read_text()) for g in ['a','b']}
    ma, mb = [r['manifest'] for r in results.values()]
    for key in ['initial_state_sha256', 'fit_indices', 'seed', 'optimizer', 'ema', 'microbatch',
                'threshold', 'steps', 'starting_rgb_step', 'gradient_balanced_lambda']:
        assert ma[key] == mb[key], key
    for key in results['a']['initial']:
        for name, value in results['a']['initial'][key].items():
            assert np.isclose(value, results['b']['initial'][key][name], rtol=1e-6, atol=1e-9), (key,name)
    comparison = {}
    for scope in ['all_468', 'fitted_12', 'unseen_456']:
        initial = results['a']['initial'][scope]
        left, right = results['a']['final'][scope], results['b']['final'][scope]
        comparison[scope] = dict(initial=initial, A=left, B=right,
            B_vs_A=dict(patch_cosine_absolute_change=right['patch_cosine']-left['patch_cosine'],
                response_mse_reduction_pct=100*(1-right['response_difference_mse']/left['response_difference_mse']),
                token_mse_reduction_pct=100*(1-right['token_mse']/left['token_mse']),
                rgb_rms_change_pct=100*(right['rgb_rms']/left['rgb_rms']-1)))
    summary = dict(experiment='Frozen D architecture, 300 additional RGB updates on the 12 Oracle windows',
        direction_lambda=mb['direction_lambda'], direction_threshold=mb['threshold'],
        initial_head_gradient_cosine=mb['initial_head_gradient_cosine'], comparison=comparison,
        interpretation_limits='The fitted 12 belong to the original validation split; full 468 metrics include them. '
            'The other 456 were excluded from this pilot optimization, but share episodes and can contain nearby '
            'overlapping temporal windows. They are not an independent episode-level generalization test. '
            'The teacher precision discrepancy is an empirical numerical reference, not measured sensor noise.',
        fixed_inputs_verified=True, both_frozen_branches_verified=all(r['frozen_teacher_adapter_condition_verified'] for r in results.values()))
    (destination/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['RenderNet', 'Oracle full .02', 'Oracle full .10', 'Oracle gated .10']
    values = [oracle['rendernet'], oracle['oracle']['full_0.02'], oracle['oracle']['full_0.10'], oracle['oracle']['gated_0.10']]
    for label, value in zip(labels, values):
        axes[0].plot([r['threshold'] for r in value['thresholds']],
                     [r.get('patch_cosine_mean', np.nan) for r in value['thresholds']], '-o', label=label)
        axes[1].plot(range(1,5), [r['patch_cosine_mean'] for r in value['amplitude_quartiles']], '-o', label=label)
    axes[0].set(xlabel='Teacher patch response RMS threshold', ylabel='Mean valid-patch cosine', title='Noise threshold sensitivity')
    axes[1].set(xlabel='Teacher response amplitude quartile (low to high)', ylabel='Mean valid-patch cosine', title='Strong teacher responses')
    for row in rgb:
        axes[2].scatter(row['rgb']['rms'], row['token_mse'], label=f"RGB L2 lambda={row['rgb_lambda']:g}")
    axes[2].axhline(rgb[0]['baseline_mse'], linestyle='--', color='black', label='Original + MLP')
    axes[2].set(xlabel='RGB modification RMS [0,1]', ylabel='Token MSE', title='10 windows, 300 pixel updates')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(destination/'oracle_direction_and_l2.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for group in ['a', 'b']:
        history = json.loads((destination/group/'log.json').read_text())
        trains = [r for r in history if r['kind']=='train']
        for ax, key in zip(axes[0], ['base_loss', 'direction_loss', 'response_difference_mse']):
            ax.plot([r['step'] for r in trains], [r[key] for r in trains], label=group.upper())
            ax.set(title='Fit 12: '+key, xlabel='Additional RGB optimizer updates')
        validations = [r for r in history if r['kind']=='validation']
        for ax, key in zip(axes[1], ['patch_cosine', 'response_difference_mse', 'token_mse']):
            ax.plot([r['step'] for r in validations], [r['metrics']['unseen_456'][key] for r in validations],
                     '-o', label=group.upper())
            ax.set(title='Other 456: '+key, xlabel='Additional RGB optimizer updates')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(destination/'ab_curves.png', dpi=170); plt.close(fig)
    tensors = torch.load(oracle_root/'response_tensors.pt', map_location='cpu', weights_only=False)
    target = tensors['teacher_response']; amplitude = target.square().mean(-1).sqrt()
    threshold = oracle['thresholds'][0]
    valid = amplitude > threshold
    probe_rows = []
    for label, key in [('RenderNet', 'renderer_response'), ('Oracle full .02', 'full_0.02'),
                       ('Oracle full .10', 'full_0.10'), ('Oracle gated .10', 'gated_0.10')]:
        cosine = F.cosine_similarity(tensors[key], target, dim=-1)
        for i, sample in enumerate(oracle['selection']):
            probe_rows.append(dict(method=label, **sample, valid_patches=int(valid[i].sum()),
                patch_cosine=float(cosine[i][valid[i]].mean())))
    (destination/'oracle_per_sample_direction.json').write_text(json.dumps(probe_rows, indent=2)+'\n')
    # Show individual patch directions, with weak teacher patches hidden in gray.
    reference = torch.load(root/'renderer_diagnostics_1000/full_0.02_images.pt', map_location='cpu', weights_only=False)
    cameras = list(reference['original'])
    obs_steps = reference['original'][cameras[0]].shape[1]
    patch_count = target.shape[1]//(obs_steps*len(cameras))
    side = int(patch_count**.5)
    assert side*side == patch_count
    for index in [0, len(oracle['selection'])-1]:
        fig, axes = plt.subplots(len(cameras), 4, figsize=(12, 3*len(cameras)), squeeze=False, constrained_layout=True)
        response = target[index].reshape(obs_steps, len(cameras), patch_count, -1)
        rms = response.square().mean(-1).sqrt()
        for camera_index, camera in enumerate(cameras):
            valid_camera = rms[-1,camera_index] > threshold
            axes[camera_index,0].imshow(rms[-1,camera_index].reshape(side,side), cmap='magma')
            axes[camera_index,0].set_title('Teacher response RMS / '+camera, fontsize=9)
            for column, (label,key) in enumerate([('RenderNet','renderer_response'),('Oracle full .02','full_0.02'),
                                                ('Oracle gated .10','gated_0.10')], start=1):
                student = tensors[key][index].reshape_as(response)
                cosine = F.cosine_similarity(student[-1,camera_index], response[-1,camera_index], dim=-1)
                values = np.ma.masked_where(~valid_camera.numpy(), cosine.numpy()).reshape(side,side)
                cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('lightgray')
                heat = axes[camera_index,column].imshow(values, cmap=cmap, vmin=-1, vmax=1)
                axes[camera_index,column].set_title(label+' patch cosine', fontsize=9)
        for ax in axes.flat: ax.axis('off')
        fig.colorbar(heat, ax=axes[:,1:], shrink=.65)
        fig.suptitle('Same contact window; gray = below teacher response threshold')
        fig.savefig(destination/f'patch_direction_{index}.png', dpi=170); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
