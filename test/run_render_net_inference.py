"""Run a saved student on one episode and record both subnetworks' execution.

Uses the existing deployment/rendering entry point. Stores a checkpoint
snapshot, original/rendered/difference videos, and the condition network's
predicted region/alpha video and arrays. No offline alpha labels are supplied.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import dill
import numpy as np
import torch

from diffusion_policy.scripts import visualize_render_net_episode as visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=5)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = args.checkpoint.resolve()
    payload, _ = visual.common.load_checkpoint_payload(source)
    info = {key: dill.loads(payload['pickles'][key]) for key in ('epoch', 'global_step', 'best_val_loss')}
    output = args.output_dir or source.parent.parent / 'inference' / (
        f"best_epoch{info['epoch']:04d}_step{info['global_step']:08d}_episode{args.episode}_"
        + datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    output = output.resolve()
    checkpoint = output / 'checkpoints' / 'best.ckpt'
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(payload, checkpoint, pickle_module=dill)
    del payload
    teacher_config = source.parent.parent / 'teacher_config.yaml'
    if teacher_config.exists():
        shutil.copyfile(teacher_config, output / 'teacher_config.yaml')
    counts = Counter()
    alphas, masks, latest = [], {}, {}
    handles = []
    original_load, original_paths, original_frames = visual.load_render_net, visual.output_paths, visual.video_frames

    def count(name):
        def hook(module, inputs, outputs):
            counts[name] += 1
        return hook

    def condition_output(module, inputs, outputs):
        region, condition = outputs
        counts['condition_network'] += 1
        alpha = condition['contact_alpha'].detach().float().cpu().numpy()
        if not np.isfinite(alpha).all():
            raise FloatingPointError('Non-finite predicted alpha')
        alphas.append(alpha[0])
        latest['alpha'] = float(alpha[0, -1])
        latest['masks'] = {}
        for key, value in region.items():
            value = value[0, -1, 0].detach().float().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError('Non-finite predicted region')
            latest['masks'][key] = value
            masks.setdefault(key, []).append(value.astype(np.float16))

    def load(*load_args, **kwargs):
        loaded = original_load(*load_args, **kwargs)
        model = loaded[0]
        if not model.condition_prediction_enabled or not model.high_attention_enabled:
            raise ValueError('This run requires the condition predictor and region-gated RGB renderer')
        handles.extend([
            model.condition_network.register_forward_hook(condition_output),
            model.condition_network.region_head.register_forward_hook(count('region_head')),
            model.condition_network.alpha_head.register_forward_hook(count('alpha_head')),
            model.backbone.register_forward_hook(count('rgb_backbone')),
        ])
        latest['camera_count'] = len(model.image_keys)
        latest['has_adapter'] = model.context_adapter is not None
        latest['region_threshold'] = model.region_threshold
        return loaded

    def paths(*path_args, **kwargs):
        result = original_paths(*path_args, **kwargs)
        result['condition'] = result['rendered'].with_name('predicted_region_alpha.mp4')
        return result

    def frames(images, rendered, camera_keys, **kwargs):
        result = original_frames(images, rendered, camera_keys, **kwargs)
        rows = []
        for key in camera_keys:
            region = latest['masks'][key]
            gray = np.uint8(np.clip(region, 0, 1) * 255)
            heatmap = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            original = visual._to_rgb(images[key][0, -1])
            overlay = np.uint8(.65 * original + .35 * heatmap)
            rows.append(visual._tile([
                visual._label(heatmap, f'{key} | predicted region [0,1]'),
                visual._label(overlay, f'{key} | region overlay'),
            ], horizontal=True))
        result['condition'] = visual._label(
            visual._tile(rows, horizontal=False),
            f"Predicted alpha={latest['alpha']:.4f} | condition network + RGB renderer | frame {len(alphas)}",
        )
        if len(alphas) == 1 or latest['alpha'] > latest.get('preview_alpha', -1):
            visual._save_rgb(output / 'condition_preview.png', result['condition'])
            visual._save_rgb(output / 'comparison_preview.png', result['comparison'])
            latest['preview_alpha'] = latest['alpha']
            latest['preview_frame'] = len(alphas) - 1
        return result

    visual.load_render_net, visual.output_paths, visual.video_frames = load, paths, frames
    try:
        paths = visual.run(visual.parse_args([
            '--checkpoint', str(checkpoint), '--episode', str(args.episode),
            '--camera', 'both', '--device', args.device,
        ]))
    finally:
        visual.load_render_net, visual.output_paths, visual.video_frames = original_load, original_paths, original_frames
        for handle in handles:
            handle.remove()
    metadata = json.loads(paths['metadata'].read_text())
    frames_count = metadata['video_frames']
    expected = dict(condition_network=frames_count, alpha_head=frames_count,
                    region_head=frames_count * latest['camera_count'],
                    rgb_backbone=frames_count * latest['camera_count'])
    if dict(counts) != expected:
        raise RuntimeError(f'Subnetwork calls do not cover every frame: {counts}, expected {expected}')
    arrays = dict(predicted_alpha=np.stack(alphas))
    arrays.update({f'predicted_region_{key}': np.stack(values) for key, values in masks.items()})
    arrays['anchors'] = np.load(paths['raw'])['anchors']
    np.savez_compressed(output / 'predicted_conditions.npz', **arrays)
    audit = dict(source_checkpoint=str(source), checkpoint=str(checkpoint), **info,
                 episode=args.episode, frames=frames_count, network_calls=dict(counts),
                 both_subnetworks_used=True, offline_alpha_labels_used=False,
                 context_adapter_enabled=latest['has_adapter'], region_threshold=latest['region_threshold'],
                 alpha_min=float(arrays['predicted_alpha'].min()),
                 alpha_max=float(arrays['predicted_alpha'].max()),
                 alpha_mean=float(arrays['predicted_alpha'].mean()),
                 preview_frame_index=latest['preview_frame'],
                 preview_predicted_alpha=latest['preview_alpha'],
                 inference_scope='condition_network_and_rgb_renderer',
                 videos={k: str(v) for k, v in paths.items() if v.suffix == '.mp4'})
    (output / 'inference_summary.json').write_text(json.dumps(audit, indent=2) + '\n')
    metadata.update(audit)
    paths['metadata'].write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
