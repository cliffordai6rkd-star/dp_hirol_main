"""Plot the independent RGB/response probes; never load a training model."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args(); root = args.root
    cases = ['full_0.02', 'full_0.05', 'full_0.10', 'gated_0.10']
    pixels = {c: json.loads((root/f'{c}.json').read_text()) for c in cases}
    responses = {v: json.loads((root/f'response_{v}.json').read_text())
                 for v in ['a_baseline', 'c_film_mlp']}
    gradients = json.loads((root/'gradient_diagnostics.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for case, row in pixels.items():
        history = row['history']
        axes[0].plot([r['step'] for r in history], [r['best_token_mse'] for r in history],
                     marker='.', label=case)
        axes[1].plot([r['baseline_mse'] for r in row['per_sample']],
                     [r['optimized_mse'] for r in row['per_sample']], '.', label=case)
    baseline = pixels[cases[0]]['baseline_token_mse']
    render = pixels[cases[0]]['renderer_token_mse']
    axes[0].axhline(baseline, color='black', linestyle='--', label='Original RGB + frozen MLP')
    axes[0].axhline(render, color='gray', linestyle=':', label='RenderNet + same MLP (step 1000)')
    axes[0].set(xlabel='Per-sample RGB optimization updates', ylabel='Contact token MSE',
                title='12 fixed held-out windows: sample-specific reachability')
    upper = max(r['baseline_mse'] for r in pixels[cases[0]]['per_sample'])*1.05
    axes[1].plot([0, upper], [0, upper], '--', color='black')
    axes[1].set(xlabel='Original RGB token MSE', ylabel='Optimized RGB token MSE',
                title='Each point is one fitted window')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'rgb_reachability.png', dpi=170); plt.close(fig)
    artifacts = {c: torch.load(root/f'{c}_images.pt', map_location='cpu', weights_only=False)
                 for c in cases}
    selection = artifacts[cases[0]]['selection']
    for index in [0, len(selection)-1]:
        first = artifacts[cases[0]]
        cameras = list(first['original'])
        labels = ['original', 'RenderNet (1k)'] + cases
        images = [first['original'], first['renderer']] + [artifacts[c]['optimized'] for c in cases]
        fig, axes = plt.subplots(len(cameras)*2, len(labels), figsize=(17, 10), constrained_layout=True)
        vmax = max(float((value[k][index, -1].float()-first['original'][k][index, -1]).abs().mean(0).max())
                   for value in images for k in cameras)
        for camera_index, camera in enumerate(cameras):
            original = first['original'][camera][index, -1].float()
            for j, (label, value) in enumerate(zip(labels, images)):
                rendered = value[camera][index, -1].float()
                axes[2*camera_index, j].imshow(rendered.permute(1, 2, 0).clamp(0, 1))
                axes[2*camera_index, j].set_title(camera+': '+label, fontsize=8)
                heat = axes[2*camera_index+1, j].imshow((rendered-original).abs().mean(0),
                                                      cmap='magma', vmin=0, vmax=vmax)
                axes[2*camera_index+1, j].set_title('|RGB delta|', fontsize=8)
        for ax in axes.flat:
            ax.axis('off')
        fig.colorbar(heat, ax=axes, shrink=.6)
        fig.suptitle(f"Contact episode {selection[index]['episode']}, index {selection[index]['index']}")
        fig.savefig(root/f'pixel_render_{index}.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for variant, result in responses.items():
        rows = result['per_sample']
        axes[0].scatter([r['teacher_response_rms'] for r in rows],
                        [r['student_response_rms'] for r in rows], s=7, alpha=.45, label=variant)
        axes[1].hist([r['response_cosine'] for r in rows], bins=25, alpha=.45, label=variant)
        axes[2].hist([r['within_camera_patch_pearson'] for r in rows
                      if r['within_camera_patch_pearson'] is not None], bins=25, alpha=.45, label=variant)
    upper = max(r['teacher_response_rms'] for v in responses.values() for r in v['per_sample'])
    axes[0].plot([0, upper], [0, upper], '--', color='black')
    axes[0].set(xlabel='Teacher response RMS', ylabel='Student response RMS', title='Amplitude')
    axes[1].set(xlabel='Response vector cosine', title='Direction')
    axes[2].set(xlabel='Within-camera patch response Pearson', title='Spatial response magnitude')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'response_alignment.png', dpi=170); plt.close(fig)
    maps = {v: torch.load(root/f'response_maps_{v}.pt', map_location='cpu', weights_only=False)
            for v in responses}
    first_index = selection[0]['index']
    displayed = [r for r in maps['a_baseline'] if r['index']==first_index and r['obs_step']==1]
    fig, axes = plt.subplots(len(displayed), 3, figsize=(10, 3*len(displayed)), squeeze=False)
    for i, entry in enumerate(displayed):
        teacher = entry['teacher']; side = int(len(teacher)**.5)
        students = [next(r['student'] for r in maps[v] if r['index']==entry['index']
                         and r['camera']==entry['camera'] and r['obs_step']==1) for v in responses]
        maximum = max(float(x.max()) for x in [teacher, *students])
        for ax, label, values in zip(axes[i], ['Teacher', *responses], [teacher, *students]):
            heat = ax.imshow(values.reshape(side, side), vmin=0, vmax=maximum, cmap='magma')
            ax.set_title(label+' / '+entry['camera'], fontsize=8); ax.axis('off')
        fig.colorbar(heat, ax=list(axes[i]), shrink=.7)
    fig.savefig(root/'patch_response.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant in responses:
        rows = [r for r in gradients if r['variant']==variant]
        for branch, style in [('normal', '-o'), ('zero', '--o')]:
            xs = ['/8', '/4', '/2', '/1']
            for ax, key in zip(axes, ['gradient_l2', 'gradient_times_feature_rms']):
                ys = [np.mean([h[key] for row in rows for h in row['feature_gradients']
                               if h['branch']==branch and h['scale']==scale]) for scale in xs]
                ax.plot(xs, ys, style, label=variant+' / '+branch)
                ax.set(xlabel='Actual feature scale', yscale='log', ylabel=key)
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'decoder_gradient_scales.png', dpi=170); plt.close(fig)
    summary = dict(pixel_results=[{k: v for k, v in pixels[c].items() if k not in ['history', 'per_sample']} for c in cases],
                   responses={k: {name: value for name, value in row.items() if name != 'per_sample'}
                              for k, row in responses.items()},
                   limitations='RGB fits optimize each evaluated window, not held-out image generalization or a mathematical lower bound. '
                               'Response spatial correlations measure magnitude alignment, not vector direction. '
                               'Feature gradient norms depend on tensor sizes; include gradient-times-feature RMS.')
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
