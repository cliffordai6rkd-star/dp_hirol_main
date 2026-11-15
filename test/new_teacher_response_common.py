"""Independent calibration/shared-response experiment; production models unchanged."""
from pathlib import Path
import copy
import hashlib
import json

import dill
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from student_dual_path_common import model_for, tensor_hash
from paired_response_full_common import episode_ids, PHASES
from diffusion_policy.model.vision.render_net import build_render_net


class SharedCNNLSTM(nn.Module):
    """Small spatial CNN, same torque encoder initialization as the D reference."""
    def __init__(self, reference):
        super().__init__()
        for key in ['wrench_projection', 'wrench_encoder', 'wrench_position',
                    'obs_position', 'force_modality']:
            setattr(self, key, copy.deepcopy(getattr(reference, key)))
        self.image_keys = reference.image_keys
        self.encoder = nn.Sequential(nn.Conv2d(5, 16, 5, 2, 2), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 32, 3, 1, 1), nn.GELU())
        self.torque_projection = nn.Sequential(nn.Linear(256, 32), nn.GELU())
        self.decoder = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 3, 3, 1, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, images, wrench):
        batch, steps, history, _ = wrench.shape
        tokens = (self.wrench_projection(wrench) + self.wrench_position
                  + self.obs_position[:, :steps] + self.force_modality)
        sequence, _ = self.wrench_encoder(tokens.reshape(batch * steps, history, -1))
        condition = self.torque_projection(sequence.mean(1))
        result = {}
        for camera, key in enumerate(self.image_keys):
            image = images[key]
            x = image.flatten(0, 1)
            labels = x.new_zeros(len(x), len(self.image_keys), *x.shape[-2:])
            labels[:, camera] = 1
            h = self.encoder(torch.cat([x, labels], 1))
            c = condition[..., None, None].expand(-1, -1, *h.shape[-2:])
            residual = self.decoder(torch.cat([h, c], 1))
            result[key] = F.interpolate(residual, x.shape[-2:], mode='bilinear', align_corners=False).tanh().reshape_as(image)
        return result


def fresh_reference(initial, teacher, condition_checkpoint, seed):
    # Resolve the exact D configuration, then instantiate a NEW RGB path instead
    # of importing pretrained RGB parameters from either historical checkpoint.
    _, cfg, _ = model_for(initial, 'c_film_mlp')
    torch.manual_seed(seed)
    model = build_render_net(7, dict(cfg.render_net))
    payload = torch.load(condition_checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
    state = payload['state_dicts']['ema_model']
    condition = {k.removeprefix('condition_network.'): v for k, v in state.items()
                 if k.startswith('condition_network.')}
    model.condition_network.load_state_dict(condition, strict=True)
    # Copy only the RGB torque path; do not overwrite the frozen condition net.
    enc = teacher.obs_encoder
    model.wrench_projection.load_state_dict(enc.wrench_projection.state_dict())
    model.wrench_encoder.load_state_dict(enc.force_encoder.state_dict())
    with torch.no_grad():
        for key in ['wrench_position', 'obs_position', 'force_modality']:
            getattr(model, key).copy_(getattr(enc, key))
    return model


def balanced_batches(phases, episodes, steps, batch_size, seed):
    """Equal Contact/Free counts, cyclic episode balance, no IDs as model inputs."""
    assert batch_size % 2 == 0
    phases, episodes = np.asarray(phases), np.asarray(episodes)
    rng = np.random.default_rng(seed)
    def stream(phase):
        eps = np.unique(episodes[phases == phase])
        queues = {}
        while True:
            for ep in rng.permutation(eps):
                if not queues.get(int(ep)):
                    queues[int(ep)] = rng.permutation(np.flatnonzero((phases == phase) & (episodes == ep))).tolist()
                yield queues[int(ep)].pop()
    contact, free = stream(2), stream(0)
    return np.asarray([[next(contact) for _ in range(batch_size // 2)]
                       + [next(free) for _ in range(batch_size // 2)]
                       for _ in range(steps)], dtype=np.int64)


def split_details(datasets, teacher_validation_episodes):
    result = {}
    for split, ds in datasets.items():
        episodes = episode_ids(ds)
        result[split] = dict(samples=len(ds), episodes=np.unique(episodes).tolist(),
            phase_counts=np.bincount(ds.phase_labels, minlength=3).tolist(),
            indices_sha256=hashlib.sha256(np.ascontiguousarray(ds.indices).tobytes()).hexdigest(),
            per_episode={str(e): {name: int(np.sum((episodes == e) & (ds.phase_labels == p)))
                                  for p, name in enumerate(PHASES)} for e in np.unique(episodes)})
    train_eps, val_eps = set(result['train']['episodes']), set(result['validation']['episodes'])
    assert not train_eps & val_eps
    all_eps = train_eps | val_eps
    teacher_train = all_eps - set(teacher_validation_episodes)
    result['overlap'] = dict(teacher_training_episodes=sorted(teacher_train),
        teacher_validation_episodes=sorted(teacher_validation_episodes),
        teacher_training_student_training=sorted(teacher_train & train_eps),
        teacher_training_student_validation=sorted(teacher_train & val_eps),
        both_parameter_training_held_out=sorted(val_eps - teacher_train),
        teacher_checkpoint_selection='Teacher checkpoint was selected using teacher validation, including episodes5/26; these are not untouched final test episodes.')
    return result


class FullPairedDataset(Dataset):
    def __init__(self, dataset, cache):
        self.dataset = dataset
        self.episodes = episode_ids(dataset)
        self.arrays = {k: np.load(Path(cache) / (k + '.npy'), mmap_mode='r') for k in ['target', 'zero']}
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        item.update({k: torch.from_numpy(np.array(v[index], copy=True)) for k, v in self.arrays.items()})
        item.update(index=index, episode=int(self.episodes[index]), phase=int(self.dataset.phase_labels[index]))
        return item


def module_norms(module):
    groups = dict(cnn=('encoder.', 'backbone.stem.', 'backbone.encoders.'),
        lstm=('wrench_encoder.',), ca=('backbone.cross_attention.',),
        decoder_film=('backbone.decoder_film.',), rgb_head=('decoder.2.', 'backbone.output.'))
    return {key: sum(float(p.grad.detach().float().square().sum())
                    for name, p in module.named_parameters()
                    if p.grad is not None and (key == 'all' or name.startswith(prefix))) ** .5
            for key, prefix in [('all', ()), *groups.items()]}


def summarize_zero(rows):
    def summary(selected):
        return dict(samples=len(selected), zero_token_mse=float(np.mean([r['zero_token_mse'] for r in selected]))) if selected else dict(samples=0)
    return dict(all=summary(rows), phases={name: summary([r for r in rows if r['phase'] == p])
        for p, name in enumerate(PHASES)}, episodes={str(ep): dict(all=summary([r for r in rows if r['episode'] == ep]),
            **{name: summary([r for r in rows if r['episode'] == ep and r['phase'] == p])
               for p, name in enumerate(PHASES)}) for ep in sorted({r['episode'] for r in rows})})


def continuation_decision(history, config):
    """Use only internal validation episodes; external5/26 never tune duration."""
    relevant = [r for r in history if r['step'] >= 600]
    values = [r['selection']['contact']['response_difference_mse'] for r in relevant]
    if len(values) < 3:
        return dict(extend=False, reason='Fewer than three late internal validation measurements')
    improvement = 1 - values[-1] / values[0]
    initial = history[0]['selection']
    final = relevant[-1]['selection']
    preservation = final['free']['token_mse'] <= initial['free']['token_mse'] * (1 + config.preservation_relative_tolerance)
    absolute = final['contact']['token_mse'] <= initial['contact']['token_mse'] * (1 + config.preservation_relative_tolerance)
    monotonic = all(b < a for a, b in zip(values, values[1:]))
    return dict(extend=bool(monotonic and improvement >= config.extension_min_relative_improvement and preservation and absolute),
        late_response_mse=values, relative_improvement=improvement,
        free_preserved=bool(preservation), contact_absolute_preserved=bool(absolute),
        criterion=f'Strictly improving internal validation at600/800/1000 plus >={100*config.extension_min_relative_improvement:g}% relative improvement and <=1% absolute/Free degradation')
