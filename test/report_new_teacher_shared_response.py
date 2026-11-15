"""Tables, per-episode metrics, calibrated baselines and render diagnostics."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from student_dual_path_common import write_json
from paired_response_full_common import file_hash, PHASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', type=Path)
    a = p.parse_args()
    root = a.experiment
    adapter = json.loads((root/'adapter/summary.json').read_text())
    groups = {name: json.loads((root/name/'summary.json').read_text()) for name in ['cnn', 'd']}
    left, right = [groups[k]['manifest'] for k in ['cnn', 'd']]
    for key in ['checkpoints', 'shared_torque_state_sha256', 'frozen', 'sample_plan_sha256', 'cache_targets', 'coefficients']:
        assert left[key] == right[key], key
    assert groups['cnn']['steps'] == groups['d']['steps']
    assert groups['cnn']['initial']['metrics']['validation'] == groups['d']['initial']['metrics']['validation'], 'Identical initial rendered RGB must yield identical metrics'
    result = dict(adapter=adapter, groups=groups, episode_split=json.loads((root/'episode_split.json').read_text()),
        sampling=json.loads((root/'sampling.json').read_text()),
        extension_decision=dict(extend_both=groups['cnn']['steps'] == 1000 and any(s['eligible_for_extension']['extend'] for s in groups.values()),
            policy='If either arm has sustained internal-validation response improvement and preserves absolute/Free quality, extend BOTH to3000 for equal budgets; never exceed3000.',
            groups={name: s['eligible_for_extension'] for name, s in groups.items()}),
        checkpoints={name: {k: s[k] for k in ['checkpoint_sha256', 'best_checkpoint_sha256', 'initial_checkpoint_sha256']} for name, s in groups.items()},
        limitations=['One task and one seed;1000-step screen, not final optimization or architecture upper bounds.',
            'Both generators start fresh with zero RGB heads; historical pretrained RGB weights are not reused.',
            'Adapter and all RGB selection use only38/44/53/60;5/26 are evaluation-only in this experiment.',
            'Teacher itself selected its checkpoint using teacher validation, including5/26; these are held out from parameter training but not untouched final test episodes.',
            'Intermediate training curves use a fixed episode/phase probe; endpoint Generalization Gap uses ALL training windows.'])
    # Labels already used by the paired teacher targets; no new model forward
    # or training. Quantify whether the fixed old condition net closes the
    # permitted RGB update on the truly teacher-held-out contact episodes.
    cache_manifest = json.loads((root/'paired_cache/manifest.json').read_text())
    study = Path(cache_manifest['reused_from'])
    labels = json.loads((study/'validation/samples.json').read_text())
    alphas = np.load(study/'validation/alpha.npy', mmap_mode='r')
    samples = {name: {r['index']: r for r in json.loads((root/name/f"validation_samples_{groups[name]['steps']}.json").read_text())}
               for name in groups}
    alpha_rows = []
    for i, label in enumerate(labels):
        if label['phase'] != 2:
            continue
        index = label['index']
        assert samples['cnn'][index]['alpha'] == samples['d'][index]['alpha']
        alpha_rows.append(dict(episode=label['episode'], target=float(alphas[i].mean()), predicted=samples['cnn'][index]['alpha']))
    assert len(alpha_rows) == groups['cnn']['final']['metrics']['validation']['phases']['contact']['samples']
    result['condition_contact_alpha'] = {str(ep): dict(samples=sum(r['episode'] == ep for r in alpha_rows),
        teacher_alpha=float(np.mean([r['target'] for r in alpha_rows if r['episode'] == ep])),
        student_alpha=float(np.mean([r['predicted'] for r in alpha_rows if r['episode'] == ep])),
        mean_absolute_error=float(np.mean([abs(r['target']-r['predicted']) for r in alpha_rows if r['episode'] == ep])))
        for ep in sorted({r['episode'] for r in alpha_rows})}
    if (root/'extension_decision.json').exists():
        result['budget_extension_provenance'] = json.loads((root/'extension_decision.json').read_text())
    write_json(root/'comparison.json', result)
    columns = ['group', 'split', 'episode', 'phase', 'samples', 'zero_token_mse', 'response_difference_mse',
               'response_cosine', 'response_rms_ratio', 'token_mse', 'rgb_rms', 'rgb_max', 'zero_rgb_rms', 'zero_rgb_max']
    with (root/'per_episode_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, summary in groups.items():
            for split in ['train', 'validation']:
                for ep, values in summary['final']['metrics'][split]['episodes'].items():
                    for phase in ['all', *PHASES]:
                        row = dict(group=name, split=split, episode=ep, phase=phase)
                        row.update({k: v for k, v in values[phase].items() if k in columns})
                        writer.writerow(row)
    calibration = json.loads((root/'adapter/log.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # The last evaluation restores the best adapter; omit its duplicate step.
    progress = calibration[:-1]
    for ax, phase in zip(axes, PHASES):
        for split, label in [('selection', 'Internal validation'), ('external', 'External5/26')]:
            ax.plot([r['step'] for r in progress], [r['metrics'][split]['phases'][phase]['zero_token_mse'] for r in progress], label=label)
        ax.set(title=phase+' zero-torque MSE', xlabel='Adapter optimizer steps')
        ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(root/'adapter_calibration.png', dpi=160);plt.close(fig)
    keys = ['response_difference_mse', 'response_cosine', 'response_rms_ratio', 'token_mse', 'zero_token_mse', 'rgb_rms']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for name in groups:
        history = json.loads((root/name/'log.json').read_text())
        # Extension preserves the1000-step evaluation and does not insert duplicate initial metrics.
        for ax, key in zip(axes.flat, keys):
            ax.plot([r['step'] for r in history], [r['selection']['contact'][key] for r in history], '-o', label=name)
            ax.set(title='Internal validation contact: '+key, xlabel='RGB optimizer steps');ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            if key == 'zero_token_mse':
                baseline = history[0]['selection']['contact'][key]
                ax.set_ylim(.99*baseline, 1.01*baseline)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(root/'shared_learning_curves.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for name, color in [('cnn', 'tab:blue'), ('d', 'tab:orange')]:
        history = json.loads((root/name/'log.json').read_text())
        endpoints = [r for r in history if r['full_train']]
        for ax, key in zip(axes.flat, keys):
            ax.plot([r['step'] for r in history], [r['metrics']['validation']['phases']['contact'][key] for r in history],
                    '-o', color=color, label=name+' validation')
            ax.plot([r['step'] for r in endpoints], [r['metrics']['train']['phases']['contact'][key] for r in endpoints],
                    '--s', color=color, label=name+' all-train endpoints')
            ax.set(title='Contact: '+key, xlabel='RGB optimizer steps');ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Validation every200 steps; full training statistics only at labeled endpoints')
    fig.tight_layout();fig.savefig(root/'generalization_curves.png', dpi=160);plt.close(fig)
    last = groups['cnn']['steps']
    snapshots = {name: torch.load(root/name/f'visuals_{last}.pt', map_location='cpu', weights_only=False) for name in groups}
    threshold = groups['cnn']['manifest']['threshold']
    for first in snapshots['cnn']:
        ep = first['episode']
        pair = {name: next(r for r in rows if r['episode'] == ep) for name, rows in snapshots.items()}
        assert pair['cnn']['index'] == pair['d']['index']
        fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
        fig2, image_axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
        for camera, key in enumerate(first['original']):
            target = first['teacher_response'].reshape(2, 2, 196, 256)[-1, camera]
            amplitude = target.square().mean(-1).sqrt()
            amplitude_max = max(float(amplitude.max()), *[float(pair[name]['student_response'].reshape(2, 2, 196, 256)[-1, camera].square().mean(-1).sqrt().max()) for name in pair])
            axes[camera, 0].imshow(amplitude.reshape(14, 14), cmap='magma', vmin=0, vmax=amplitude_max);axes[camera, 0].set_title(key+' teacher RMS')
            original = first['original'][key][-1].float()
            image_axes[camera, 0].imshow(original.permute(1, 2, 0));image_axes[camera, 0].set_title(key+' original')
            for j, name in enumerate(['cnn', 'd']):
                response = pair[name]['student_response'].reshape(2, 2, 196, 256)[-1, camera]
                cosine = F.cosine_similarity(response, target, dim=-1)
                filtered = np.ma.masked_where((amplitude <= threshold).numpy(), cosine.numpy()).reshape(14, 14)
                heat = axes[camera, 1+2*j].imshow(filtered, cmap='coolwarm', vmin=-1, vmax=1)
                axes[camera, 1+2*j].set_title(name+' response cosine')
                rms_heat = axes[camera, 2+2*j].imshow(response.square().mean(-1).sqrt().reshape(14, 14), cmap='magma', vmin=0, vmax=amplitude_max)
                axes[camera, 2+2*j].set_title(name+' response RMS')
                rendered = pair[name]['rendered'][key][-1].float()
                image_axes[camera, 1+2*j].imshow(rendered.permute(1, 2, 0));image_axes[camera, 1+2*j].set_title(name+' RGB')
                change = (rendered-original).square().mean(0).sqrt()
                image_axes[camera, 2+2*j].imshow(change, cmap='magma', vmin=0, vmax=.05);image_axes[camera, 2+2*j].set_title(name+' residual RMS')
            fig.colorbar(rms_heat, ax=axes[camera, [0, 2, 4]], shrink=.7, label='Absolute response RMS')
        for ax in [*axes.flat, *image_axes.flat]:
            ax.axis('off')
        fig.colorbar(heat, ax=axes[:, [1, 3]], shrink=.7)
        fig.suptitle(f'Episode {ep} contact response; step{last}');fig.savefig(root/f'response_episode_{ep}.png', dpi=150);plt.close(fig)
        fig2.suptitle(f'Episode {ep} RGB; step{last}');fig2.savefig(root/f'rgb_episode_{ep}.png', dpi=150);plt.close(fig2)
    write_json(root/'code_provenance.json', {str(p.relative_to(Path(__file__).parent.parent)): file_hash(p)
        for p in [Path(__file__), Path(__file__).with_name('new_teacher_response_common.py'),
                  Path(__file__).with_name('prepare_new_teacher_shared_response.py'), Path(__file__).with_name('run_new_teacher_shared_response.py'),
                  Path(__file__).parent/'config/new_teacher_shared_response.yaml']})
    for name, summary in groups.items():
        print(name, 'train', json.dumps(summary['final']['metrics']['train']['phases']))
        print(name, 'validation', json.dumps(summary['final']['metrics']['validation']['phases']))
    print('EXTENSION', json.dumps(result['extension_decision']), flush=True)


if __name__ == '__main__':
    main()
