"""Early-stopped phase-gate search with cached causal RGB/torque observations.

All selection uses unweighted CE on the original held-out episodes. Offline
future contact defines labels/diagnostic strata only, never model inputs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_policy.model.vision.image_tau_phase_gate import ImageTauPhaseGate, load_phase_gate
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from tools.train_image_tau_phase_gate import phase_metrics_from_confusion


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs), ensure_ascii=False), flush=True)


class FrameImages(Dataset):
    def __init__(self, dataset, sizes=(64, 96)):
        self.dataset, self.sizes = dataset, sizes

    def __len__(self):
        return self.dataset.dataset_length

    def __getitem__(self, index):
        frames, result = {}, {}
        for key in self.dataset.rgb_keys:
            image = self.dataset._load_frame_feature(
                frame_idx=index, feature_name=self.dataset.image_feature_map[key],
                expected_shape=self.dataset.shape_meta['obs'][key]['shape'], frame_cache=frames,
            )
            pixels = torch.from_numpy(np.array(image, copy=True)).float()[None]
            for size in self.sizes:
                result[f'{key}_{size}'] = F.interpolate(pixels, size=(size, size), mode='area')[0].half()
        return result


def build_cache(args, device):
    cfg = OmegaConf.load(args.teacher_config)
    ds_cfg = copy.deepcopy(cfg.task.dataset)
    with open_dict(ds_cfg):
        ds_cfg.preload_images = False
        ds_cfg.image_randomer_config = None
        ds_cfg.video_backend = 'pyav'
        ds_cfg.phase_predictor = None
    train_dataset = hydra.utils.instantiate(ds_cfg)
    val_dataset = train_dataset.get_validation_dataset()
    saved = torch.load(args.baseline, map_location='cpu', weights_only=False)
    preprocess = WrenchPreprocessor.from_config(saved['wrench_preprocessing'])
    preprocess.load_state_dict(saved['wrench_preprocessor'], strict=True)
    preprocess = preprocess.to(device).eval().requires_grad_(False)
    image_chunks = {}
    count = 0
    for batch in DataLoader(FrameImages(train_dataset), batch_size=128,
                            num_workers=args.workers, shuffle=False, pin_memory=True):
        for key, value in batch.items():
            image_chunks.setdefault(key, []).append(value)
        count += len(next(iter(batch.values())))
        if count % 1024 == 0 or count == train_dataset.dataset_length:
            emit('image_cache', frames=count, total=train_dataset.dataset_length)
    cache = {key: torch.cat(values).to(device) for key, values in image_chunks.items()}
    del image_chunks
    raw = train_dataset.windowed_lowdim_data['tau_ext']
    tau_chunks = []
    with torch.no_grad():
        for start in range(0, len(raw), 256):
            values = torch.as_tensor(raw[start:start + 256], device=device).float()
            duplicated = values[:, None].expand(-1, preprocess.n_obs_steps, -1, -1)
            normalized, _, _ = preprocess({'tau_ext': duplicated})
            tau_chunks.append(normalized[:, 0])
    cache['tau'] = torch.cat(tau_chunks)
    cache['labels'] = torch.as_tensor(train_dataset._frame_contact_phase, device=device).long()
    starts = np.empty(train_dataset.dataset_length, dtype=np.int64)
    leads = np.full(train_dataset.dataset_length, np.nan)
    times = train_dataset._timestamp_seconds(train_dataset.timestamps, key=train_dataset.timestamp_key)
    episodes = np.asarray(train_dataset.episode_index)
    for ep, rows in enumerate(train_dataset.episode_ranges):
        starts[rows] = rows.start
        local_phase = train_dataset._frame_contact_phase[rows]
        onsets = np.flatnonzero((local_phase == 2) & ~np.r_[False, local_phase[:-1] == 2]) + rows.start
        frame_ids = np.arange(rows.start, rows.stop)
        next_ids = np.searchsorted(onsets, frame_ids, side='left')
        valid = next_ids < len(onsets)
        leads[frame_ids[valid]] = times[onsets[next_ids[valid]]] - times[frame_ids[valid]]
    cache['episode_start'] = torch.as_tensor(starts, device=device)
    cache['episode'] = torch.as_tensor(episodes, device=device)
    cache['lead_seconds'] = torch.as_tensor(leads, device=device)
    cache['frame_seconds'] = torch.as_tensor(times, device=device)
    for split, dataset in [('train', train_dataset), ('validation', val_dataset)]:
        indices = dataset._sequence_indices_from_sampler_rows(dataset.indices)[:, :preprocess.n_obs_steps]
        cache[split] = torch.as_tensor(indices, device=device).long()
    train_episodes = torch.unique(cache['episode'][cache['train']]).tolist()
    val_episodes = torch.unique(cache['episode'][cache['validation']]).tolist()
    assert not set(train_episodes) & set(val_episodes)
    manifest = {
        'dataset': str(ds_cfg.dataset_path), 'teacher_config': str(args.teacher_config),
        'baseline_checkpoint': str(args.baseline), 'history_steps': int(cache['tau'].shape[1]),
        'image_keys': list(train_dataset.rgb_keys), 'image_cache_sizes': [64, 96],
        'preprocessing': saved['wrench_preprocessing'],
        'train_windows': len(cache['train']), 'validation_windows': len(cache['validation']),
        'train_episodes': train_episodes, 'validation_episodes': val_episodes,
        'objective_for_selection': 'unweighted validation CE; labels 0=free 1=precontact 2=contact',
        'input_rules': 'current/past RGB only; 50-frame normalized thresholded tau history; no phase/state/future inputs',
        'images': 'area-resized directly from original teacher RGB once, cached in fp16',
        'early_stopping': {'patience': args.patience, 'min_delta': args.min_delta, 'max_epochs': args.epochs},
        'seed': 42,
    }
    for split in ('train', 'validation'):
        labels = cache['labels'][cache[split]]
        manifest[split + '_phase_counts'] = torch.bincount(labels.flatten(), minlength=3).tolist()
    write_json(args.output / 'manifest.json', manifest)
    train_dataset.lerobot_dataset.close()
    val_dataset.lerobot_dataset.close()
    return cache, saved, manifest


def inputs(cache, observation_ids, architecture, modality='both'):
    offsets = torch.tensor(architecture.get('image_offsets', [0]), device=observation_ids.device)
    frames = torch.maximum(observation_ids[..., None] + offsets,
                           cache['episode_start'][observation_ids][..., None])
    size = architecture['image_size']
    images = {key: cache[f'{key}_{size}'][frames] for key in architecture['image_keys']}
    tau = cache['tau'][observation_ids]
    if modality == 'tau_only':
        images = {key: torch.zeros_like(value) for key, value in images.items()}
    elif modality == 'image_only':
        tau = torch.zeros_like(tau)
    return images, tau, cache['labels'][observation_ids]


@torch.no_grad()
def evaluate(model, cache, split, architecture, modality='both', return_logits=False):
    parts, counts, confusion, total = [], 0, torch.zeros(3, 3, device=cache['tau'].device, dtype=torch.long), 0.0
    model.eval()
    for ids in cache[split].split(256):
        images, tau, labels = inputs(cache, ids, architecture, modality)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images, tau).float()
        total += float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='sum'))
        prediction = logits.argmax(-1)
        confusion += torch.bincount((labels * 3 + prediction).flatten(), minlength=9).reshape(3, 3)
        counts += labels.numel()
        if return_logits:
            parts.append(logits.cpu())
    result = phase_metrics_from_confusion(confusion)
    result['ce_loss'] = total / counts
    return (result, torch.cat(parts)) if return_logits else result


def architecture(config, keys):
    return dict(image_keys=keys, wrench_dim=7, hidden_dim=config.get('hidden', 16),
                gru_layers=config.get('gru_layers', 1), image_size=config.get('size', 64),
                cnn_channels=config.get('channels', [4, 8, 12]), pool_size=config.get('pool', 1),
                dropout=config.get('dropout', 0.0), image_offsets=config.get('offsets', [0]),
                center_images=config.get('center', False), cnn_norm=config.get('cnn_norm', False))


def train_trial(config, cache, saved, manifest, args):
    name = config['name']
    trial_dir = args.output / name
    trial_dir.mkdir()
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    arch = architecture(config, manifest['image_keys'])
    model = ImageTauPhaseGate(**arch).to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-3),
                                  weight_decay=config.get('wd', 1e-4))
    train_ids = cache['train']
    batch_size = config.get('batch', 64)
    weights = torch.tensor(config.get('weights', [1., 1., 1.]), device='cuda:0')
    best, early_reference, bad, best_epoch, best_state = float('inf'), float('inf'), 0, 0, None
    history = []
    started = time.monotonic()
    emit('trial_start', name=name, config=config, parameters=sum(p.numel() for p in model.parameters()))
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_ids), device='cuda:0')
        total_ce, total_weight, total_count = 0., 0., 0
        for selected in order.split(batch_size):
            images, tau, labels = inputs(cache, train_ids[selected], arch, config.get('modality', 'both'))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images, tau).float()
                terms = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none')
                sample_weights = weights[labels.flatten()]
                loss = (terms * sample_weights).sum() / sample_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_ce += float(terms.detach().sum())
            total_weight += float(loss.detach()) * labels.numel()
            total_count += labels.numel()
        validation = evaluate(model, cache, 'validation', arch, config.get('modality', 'both'))
        row = dict(epoch=epoch, train_ce=total_ce / total_count,
                   train_objective=total_weight / total_count, validation=validation,
                   elapsed_seconds=time.monotonic() - started)
        history.append(row)
        if validation['ce_loss'] < best:
            best, best_epoch = validation['ce_loss'], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if validation['ce_loss'] < early_reference - args.min_delta:
            early_reference, bad = validation['ce_loss'], 0
        else:
            bad += 1
        write_json(trial_dir / 'history.json', history)
        emit('epoch', name=name, epoch=epoch, val_ce=validation['ce_loss'],
             pre_recall=validation['phase_recall'][1], bad_epochs=bad)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    train_metrics = evaluate(model, cache, 'train', arch, config.get('modality', 'both'))
    val_metrics, logits = evaluate(model, cache, 'validation', arch, config.get('modality', 'both'), True)
    summary = dict(name=name, config=config, architecture=arch,
                   parameters=sum(p.numel() for p in model.parameters()),
                   best_epoch=best_epoch, stopped_epoch=len(history), early_stopped=bad >= args.patience,
                   train=train_metrics, validation=val_metrics,
                   elapsed_seconds=time.monotonic() - started)
    checkpoint = dict(model=best_state, architecture=arch, epoch=best_epoch,
                      best_val_ce=best, config=config, summary=summary,
                      history_steps=manifest['history_steps'],
                      wrench_preprocessing=saved['wrench_preprocessing'],
                      wrench_preprocessor=saved['wrench_preprocessor'], manifest=manifest)
    torch.save(checkpoint, trial_dir / 'best.pt')
    np.savez_compressed(trial_dir / 'validation_predictions.npz', logits=logits.numpy(),
                        observation_frames=cache['validation'].cpu().numpy())
    write_json(trial_dir / 'summary.json', summary)
    emit('trial_complete', **summary)
    return summary


def write_leaderboard(results, output):
    write_json(output / 'results.json', sorted(results, key=lambda row: row['validation']['ce_loss']))
    rows = []
    for result in sorted(results, key=lambda row: row['validation']['ce_loss']):
        val = result['validation']
        rows.append(dict(name=result['name'], parameters=result['parameters'],
                         val_ce=val['ce_loss'], train_ce=result['train']['ce_loss'],
                         accuracy=val['accuracy'], precontact_recall=val['phase_recall'][1],
                         precontact_open=val['precontact_open_rate'],
                         contact_open=val['contact_open_rate'], free_false_open=val['free_false_open_rate'],
                         best_epoch=result['best_epoch'], stopped_epoch=result['stopped_epoch'],
                         seconds=result['elapsed_seconds']))
    with (output / 'leaderboard.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(model, cache, arch, output, name):
    results = {}
    for split in ('train', 'validation'):
        metrics, logits = evaluate(model, cache, split, arch, return_logits=True)
        frame_ids = cache[split].cpu()
        labels = cache['labels'][cache[split]].cpu()
        prediction = logits.argmax(-1)
        probs = logits.softmax(-1)
        losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').reshape_as(labels)
        tau = cache['tau'][cache[split]]
        tau_any = tau.ne(0).any(dim=(-1, -2)).cpu()
        tau_last = tau[..., -1, :].ne(0).any(-1).cpu()
        lead = cache['lead_seconds'][cache[split]].cpu()
        per_phase = {}
        for phase in range(3):
            selected = labels == phase
            per_phase[str(phase)] = dict(count=int(selected.sum()), fraction=float(selected.float().mean()),
                ce=float(losses[selected].mean()), contribution_to_total_ce=float(losses[selected].sum() / losses.numel()),
                history_all_zero_fraction=float((~tau_any[selected]).float().mean()),
                latest_tau_zero_fraction=float((~tau_last[selected]).float().mean()),
                recalls=[float((prediction[selected] == p).float().mean()) for p in range(3)])
        bins = []
        for low, high in zip([0, .1, .2, .3, .4], [.1, .2, .3, .4, .50001]):
            selected = (labels == 1) & (lead > low) & (lead <= high)
            if selected.any():
                bins.append(dict(lead_low_seconds=low, lead_high_seconds=high, count=int(selected.sum()),
                    ce=float(losses[selected].mean()), recall=float((prediction[selected] == 1).float().mean()),
                    open_rate=float((prediction[selected] != 0).float().mean()),
                    tau_history_zero=float((~tau_any[selected]).float().mean())))
        thresholds=[]
        for threshold in (.1,.2,.3,.4,.5,.6):
            opened = 1 - probs[..., 0] >= threshold
            thresholds.append(dict(threshold=threshold,
                free_false_open=float(opened[labels==0].float().mean()),
                precontact_open=float(opened[labels==1].float().mean()),
                contact_open=float(opened[labels==2].float().mean())))
        episodes = cache['episode'][cache[split]].cpu()
        per_episode = []
        for ep in torch.unique(episodes):
            selected = episodes == ep
            pre = selected & (labels == 1)
            per_episode.append(dict(episode=int(ep), observations=int(selected.sum()), ce=float(losses[selected].mean()),
                precontact_observations=int(pre.sum()),
                precontact_recall=float((prediction[pre] == 1).float().mean()) if pre.any() else None))
        results[split] = dict(metrics=metrics, phases=per_phase, precontact_by_lead=bins,
                             gate_thresholds=thresholds, by_episode=per_episode)
        np.savez_compressed(output / f'{name}_{split}_diagnostics.npz',
                            frame=frame_ids.numpy(), phase=labels.numpy(), logits=logits.numpy(),
                            lead_seconds=lead.numpy(), tau_history_nonzero=tau_any.numpy(),
                            tau_latest_nonzero=tau_last.numpy(), episode=episodes.numpy())
    write_json(output / f'{name}_diagnostics.json', results)
    return results


def trials():
    return [
        dict(name='tiny_lr1e3', lr=1e-3),
        dict(name='tiny_lr3e3', lr=3e-3),
        dict(name='tiny_lr3e4', lr=3e-4),
        dict(name='tiny_gru32', hidden=32, lr=1e-3),
        dict(name='tiny_spatial2', pool=2, lr=1e-3),
        dict(name='spatial2_h32', hidden=32, channels=[8,16,24], pool=2, lr=1e-3),
        dict(name='spatial4_h32', hidden=32, channels=[8,16,24], pool=4, lr=1e-3),
        dict(name='spatial2_h32_lr3e4', hidden=32, channels=[8,16,24], pool=2, lr=3e-4),
        dict(name='spatial2_h32_dropout', hidden=32, channels=[8,16,24], pool=2, dropout=.15, wd=1e-3, lr=1e-3),
        dict(name='spatial2_preweight3', hidden=32, channels=[8,16,24], pool=2, weights=[1,3,1], lr=1e-3),
        dict(name='spatial2_96px', hidden=32, channels=[8,16,24], pool=2, size=96, lr=1e-3),
        dict(name='temporal3_h16', hidden=16, pool=2, offsets=[-8,-4,0], lr=1e-3),
        dict(name='temporal3_h32', hidden=32, channels=[8,16,24], pool=2, offsets=[-8,-4,0], lr=1e-3),
        dict(name='temporal3_h32_preweight', hidden=32, channels=[8,16,24], pool=2, offsets=[-8,-4,0], weights=[1,3,1], lr=1e-3),
        dict(name='image_only', pool=2, hidden=32, channels=[8,16,24], modality='image_only', lr=1e-3),
        dict(name='tau_only', modality='tau_only', lr=1e-3),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('model/civi_image_tau_phase_gate/phase_gate_best.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-model-dir', type=Path, default=Path('model/civi_image_tau_phase_gate_ce_best'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--trials', type=Path, help='JSON list to override the first-round trials')
    parser.add_argument('--repeats', type=int, default=2, help='Repeat the top two both-modality configs with this many extra seeds')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()
    cache, saved, manifest = build_cache(args, torch.device('cuda:0'))
    baseline = load_phase_gate(args.baseline, 'cuda:0')
    baseline_arch = saved['architecture']
    baseline_diag = diagnose(baseline, cache, baseline_arch, args.output, 'baseline')
    emit('baseline_diagnostics', validation=baseline_diag['validation'])
    del baseline
    plans = json.loads(args.trials.read_text()) if args.trials else trials()
    write_json(args.output / 'trial_plan.json', plans)
    results = []
    for plan in plans:
        result = train_trial(plan, cache, saved, manifest, args)
        results.append(result)
        write_leaderboard(results, args.output)
    eligible = sorted([r for r in results if r['config'].get('modality','both') == 'both'],
                      key=lambda row: row['validation']['ce_loss'])
    for reference in eligible[:2]:
        for seed in range(43, 43 + args.repeats):
            plan = dict(reference['config'], name=reference['name'] + f'_seed{seed}', seed=seed)
            result = train_trial(plan, cache, saved, manifest, args)
            results.append(result)
            write_leaderboard(results, args.output)
    eligible = sorted([r for r in results if r['config'].get('modality','both') == 'both'],
                      key=lambda row: row['validation']['ce_loss'])
    winner = eligible[0]
    best_path = args.output / winner['name'] / 'best.pt'
    model = load_phase_gate(best_path, 'cuda:0')
    best_diag = diagnose(model, cache, winner['architecture'], args.output, 'best')
    args.reuse_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, args.reuse_model_dir / 'phase_gate_best.pt')
    write_json(args.reuse_model_dir / 'config.json', dict(architecture=winner['architecture'],
               history_steps=manifest['history_steps'], best_validation_ce=winner['validation']['ce_loss'],
               best_epoch=winner['best_epoch'], trial=winner['name'], source=str(best_path),
               input_mode='causal image history + thresholded normalized tau history',
               image_history_pad='edge, independently within each episode',
               wrench_preprocessing=saved['wrench_preprocessing']))
    write_json(args.reuse_model_dir / 'phase_gate_summary.json', winner)
    write_json(args.output / 'selection.json', dict(winner=winner, baseline=baseline_diag,
               best_diagnostics=best_diag, selection='minimum unweighted validation CE among both-modality trials',
               elapsed_seconds=time.monotonic() - started))
    snapshot=args.output/'source_snapshot'
    snapshot.mkdir()
    for path in (Path(__file__),Path('diffusion_policy/model/vision/image_tau_phase_gate.py')):
        shutil.copy2(path,snapshot/path.name)
    emit('complete', winner=winner['name'], validation_ce=winner['validation']['ce_loss'],
         output=str(args.output), elapsed_seconds=time.monotonic() - started)


if __name__ == '__main__':
    main()
