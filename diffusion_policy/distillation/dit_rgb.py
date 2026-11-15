"""Shared semantics for direct-torque RGB -> frozen DiT distillation.

No response predictor, target tokens, phase labels or action head are deployed.
"""
import contextlib
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

ARCHITECTURE = 'civi_direct_tau_rgb_dit_v1'
PHASES = ('free', 'precontact', 'contact')


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def move(value, device):
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def slice_batch(value, start, end):
    if isinstance(value, dict):
        return {k: slice_batch(v, start, end) for k, v in value.items()}
    return value[start:end]


def seed_for(sample_id, probe, seed=42):
    identity = f'{seed}:{sample_id}:{probe}'.encode()
    return int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), 'little') & ((1 << 63)-1)


def fixed_probe(action, sample_ids, probe, timesteps, seed=42):
    # Generate each sample separately: independent of loader order, rank and B.
    noises, times = [], []
    for sample in sample_ids:
        generator = torch.Generator(device=action.device).manual_seed(seed_for(int(sample), probe, seed))
        noises.append(torch.randn(action.shape[1:], device=action.device, generator=generator))
        times.append(torch.randint(timesteps, (1,), device=action.device, generator=generator))
    return torch.stack(noises), torch.cat(times)


def element_mse(prediction, target, valid_mask=None):
    error = (prediction.float() - target.float()).square()
    if valid_mask is None:
        return error.flatten(1).mean(1)
    mask = valid_mask.to(error)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error)
    count = mask.flatten(1).sum(1)
    if (count == 0).any():
        raise ValueError('Action sample has no valid elements')
    return (error * mask).flatten(1).sum(1) / count


def diffusion_pair(teacher, action, *, sample_ids=None, probe=0, seed=42):
    normalized = teacher.normalizer['action'].normalize(action.float())
    if normalized.shape[1:] != (teacher.horizon, teacher.action_dim):
        raise ValueError('Dataset action representation/horizon does not match teacher')
    steps = teacher.noise_scheduler.config.num_train_timesteps
    if sample_ids is None:
        noise = torch.randn_like(normalized)
        rho = torch.randint(steps, (len(action),), device=action.device)
    else:
        noise, rho = fixed_probe(normalized, sample_ids, probe, steps, seed)
    noisy = teacher.noise_scheduler.add_noise(normalized, noise, rho)
    kind = teacher.noise_scheduler.config.prediction_type
    if kind == 'epsilon':
        demo = noise
    elif kind == 'sample':
        demo = normalized
    elif kind == 'v_prediction':
        demo = teacher.noise_scheduler.get_velocity(normalized, noise, rho)
    else:
        raise ValueError(f'Unsupported checkpoint prediction_type={kind}')
    return noisy, rho, demo


@contextlib.contextmanager
def fp32_reference():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    # Disable the MHA inference fastpath too: use identical input-gradient and
    # target operators. It also avoids compiled/eager native-MHA divergence.
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.autocast('cuda', enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
        torch.backends.mha.set_fastpath_enabled(fast)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


class MixedPlan:
    """Checkpointable phase/episode balanced queues; complete coverage per queue.

    Each phase cycles episodes equally. Within an episode it visits a shuffled
    permutation of *all* its windows before repeating. A global batch is built
    once and sliced by rank; accumulation never changes its loss denominator.
    """
    def __init__(self, phases, episodes, counts=(2, 2, 4), seed=42):
        self.phases, self.episodes = np.asarray(phases), np.asarray(episodes)
        self.counts = tuple(counts)
        self.rng = np.random.default_rng(seed)
        self.episode_queues, self.window_queues = {}, {}
        self.position = 0
        for phase in range(3):
            if not (self.phases == phase).any():
                raise ValueError(f'No {PHASES[phase]} training windows')

    def next(self):
        batch = []
        for phase, count in enumerate(self.counts):
            for _ in range(count):
                if not self.episode_queues.get(phase):
                    self.episode_queues[phase] = self.rng.permutation(np.unique(self.episodes[self.phases == phase])).tolist()
                episode = self.episode_queues[phase].pop()
                key = (phase, episode)
                if not self.window_queues.get(key):
                    self.window_queues[key] = self.rng.permutation(np.flatnonzero((self.phases == phase) & (self.episodes == episode))).tolist()
                batch.append(self.window_queues[key].pop())
        self.rng.shuffle(batch)
        self.position += 1
        return batch

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, episode_queues=self.episode_queues,
                    window_queues=self.window_queues, position=self.position, counts=self.counts)

    def load_state_dict(self, state):
        if tuple(state['counts']) != self.counts:
            raise ValueError('Mixed batch composition changed on resume')
        self.rng.bit_generator.state = state['rng']
        self.episode_queues, self.window_queues = state['episode_queues'], state['window_queues']
        self.position = state['position']


def summarize(rows, retained=(5, 26)):
    episodes, scopes = [], {}
    for scope in ('internal', 'retained', 'all'):
        scopes[scope] = {}
        for phase in range(3):
            selected = [r for r in rows if r['phase'] == phase and
                        (scope == 'all' or ((r['episode'] in retained) == (scope == 'retained')))]
            values = []
            for ep in sorted({r['episode'] for r in selected}):
                group = [r for r in selected if r['episode'] == ep]
                item = dict(scope=scope, phase=PHASES[phase], episode=ep, samples=len(group))
                for key in group[0]:
                    if key not in ('index', 'episode', 'phase'):
                        item[key] = float(max(r[key] for r in group) if key=='rgb_linf' else np.mean([r[key] for r in group]))
                item['rgb_rms'] = item['rgb_mse']**.5 if 'rgb_mse' in item else item.get('rgb_rms',0.)
                item['actual_response_rms'] = item['actual_response_mse']**.5 if 'actual_response_mse' in item else item.get('actual_response_rms',0.)
                episodes.append(item); values.append(item)
            if values:
                scopes[scope][PHASES[phase]] = {k: float(max(r[k] for r in values) if k=='rgb_linf' else np.mean([r[k] for r in values]))
                    for k in values[0] if k not in ('scope', 'phase', 'episode', 'samples')}
    return scopes, episodes


@torch.no_grad()
def foreach_ema_step(ema, online):
    """Same mul/add order as EMAModel; frozen modules are never recopied."""
    ema.decay = ema.get_decay(ema.optimization_step)
    source, target = [], []
    for current, average in zip(online.parameters(),ema.averaged_model.parameters()):
        if current.requires_grad:
            source.append(current);target.append(average)
    torch._foreach_mul_(target,ema.decay)
    torch._foreach_add_(target,source,alpha=1-ema.decay)
    ema.optimization_step += 1
