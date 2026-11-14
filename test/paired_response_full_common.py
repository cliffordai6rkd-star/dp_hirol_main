"""Scientific contracts shared by the isolated full-data A/B experiment."""
from contextlib import contextmanager
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

PHASES = ('free', 'precontact', 'contact')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def episode_ids(dataset):
    return np.asarray([int(dataset.episode_index[
        dataset._anchor_index_from_sampler_row(row)]) for row in dataset.indices])


def spread_order(phases, episodes, seed, epoch):
    """Visit every row once; interleave episodes within proportional phase streams."""
    phases, episodes = np.asarray(phases), np.asarray(episodes)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    streams = []
    for phase in range(3):
        queues = {int(e): list(rng.permutation(np.flatnonzero((phases == phase) & (episodes == e))))
                  for e in np.unique(episodes[phases == phase])}
        ordered = []
        while queues:
            for episode in rng.permutation(list(queues)):
                ordered.append(queues[episode].pop())
                if not queues[episode]:
                    del queues[episode]
        streams.extend(((i + .5) / len(ordered), phase, int(index))
                       for i, index in enumerate(ordered))
    return np.asarray([index for _, _, index in sorted(streams)], dtype=np.int64)


def active_response_weights(delta, threshold):
    amplitude = delta.detach().float().square().mean(-1).sqrt()
    return (amplitude * (amplitude > threshold)).detach()


def weighted_difference(student_delta, teacher_delta, weights, denominator=None):
    errors = (student_delta.float() - teacher_delta.detach().float()).square().mean(-1)
    denominator = weights.sum() if denominator is None else denominator
    return (errors * weights.detach()).sum() / denominator.clamp_min(1e-12)


def deterministic_mode(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        elif isinstance(module, (torch.nn.MultiheadAttention, torch.nn.LSTM)):
            module.dropout = 0.


@contextmanager
def explicit_zero_alpha(model):
    original = model.condition_network.forward

    def closed(*args, **kwargs):
        masks, condition = original(*args, **kwargs)
        return masks, {**condition, 'contact_alpha': torch.zeros_like(condition['contact_alpha'])}

    model.condition_network.forward = closed
    try:
        yield
    finally:
        model.condition_network.forward = original


def gradient_norms(gradients, names):
    prefixes = dict(cnn=('backbone.stem.', 'backbone.encoders.'),
                    ca=('backbone.cross_attention.',), lstm=('wrench_encoder.',),
                    decoder_film=('backbone.decoder_film.',),
                    decoder=('backbone.decoders.',), rgb_head=('backbone.output.',))
    return {label: math.sqrt(sum(float(g.detach().float().square().sum())
                       for name, g in zip(names, gradients)
                       if g is not None and (label == 'all' or name.startswith(prefix))))
            for label, prefix in [('all', ()), *prefixes.items()]}


def sample_metrics(z, z0, target, zero, rendered, rendered0, images, alpha, alpha0, threshold):
    s, t = z.float() - z0.float(), target.float() - zero.float()
    errors, diff = (z.float() - target.float()).square().mean(-1), (s - t).square().mean(-1)
    valid = t.square().mean(-1).sqrt() > threshold
    cosine = F.cosine_similarity(s, t, dim=-1, eps=1e-6)
    change = torch.cat([(rendered[k].float() - images[k].float()).flatten(1) for k in images], 1)
    change0 = torch.cat([(rendered0[k].float() - images[k].float()).flatten(1) for k in images], 1)
    rows = []
    for i in range(len(z)):
        sv, tv = s[i][valid[i]], t[i][valid[i]]
        rows.append(dict(token_mse=float(errors[i].mean()),
            zero_token_mse=float((z0[i].float() - zero[i].float()).square().mean()),
            response_difference_mse=float(diff[i].mean()),
            valid_diff_sum=float(diff[i][valid[i]].sum()), valid_patches=int(valid[i].sum()),
            cosine_sum=float(cosine[i][valid[i]].sum()),
            ss=float(sv.double().square().sum()), tt=float(tv.double().square().sum()),
            rgb_mse=float(change[i].square().mean()), rgb_max=float(change[i].abs().max()),
            zero_rgb_mse=float(change0[i].square().mean()), zero_rgb_max=float(change0[i].abs().max()),
            alpha=float(alpha[i].float().mean()), zero_alpha=float(alpha0[i].float().mean())))
    return rows


def aggregate(rows):
    if not rows:
        return dict(samples=0)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    count = sum(r['valid_patches'] for r in rows)
    tt = sum(r['tt'] for r in rows)
    return dict(samples=len(rows), token_mse=mean('token_mse'), zero_token_mse=mean('zero_token_mse'),
        response_difference_mse=mean('response_difference_mse'),
        active_response_difference_mse=sum(r['valid_diff_sum'] for r in rows) / max(count, 1),
        response_cosine=sum(r['cosine_sum'] for r in rows) / count if count else None,
        response_rms_ratio=(sum(r['ss'] for r in rows) / tt)**.5 if tt else None,
        rgb_rms=mean('rgb_mse')**.5, rgb_max=max(r['rgb_max'] for r in rows),
        zero_rgb_rms=mean('zero_rgb_mse')**.5, zero_rgb_max=max(r['zero_rgb_max'] for r in rows),
        alpha=mean('alpha'), zero_alpha=mean('zero_alpha'), active_patches=count)


def grouped_metrics(rows):
    phases = dict(all=aggregate(rows), **{name: aggregate([r for r in rows if r['phase'] == i])
                                         for i, name in enumerate(PHASES)})
    episodes = {str(e): {name: aggregate([r for r in rows if r['episode'] == e and
                                         (name == 'all' or r['phase'] == i)])
                         for i, name in [(-1, 'all'), *enumerate(PHASES)]}
                for e in sorted({r['episode'] for r in rows})}
    return dict(phases=phases, episodes=episodes)


def extension_allowed(initial, current, tolerance=0.):
    """Extend BOTH arms only when neither loses contact accuracy/response or Free baseline."""
    checks = [('contact', 'token_mse'), ('contact', 'response_difference_mse'), ('free', 'zero_token_mse')]
    return all(current[p][k] <= initial[p][k] * (1 + tolerance) for p, k in checks)
