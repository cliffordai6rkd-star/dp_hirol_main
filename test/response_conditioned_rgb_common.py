"""Independent response-driven D wrapper; original renderer classes stay unchanged."""
from pathlib import Path
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paired_response_full_common import PHASES


class ResponseConditionedDecoder(nn.Module):
    def __init__(self, owner, epsilon):
        super().__init__()
        self.owner, self.epsilon = owner, float(epsilon)
        owner.requires_grad_(False)
        for module in [owner.backbone, owner.camera_embedding, owner.spatial_modulations]:
            module.requires_grad_(True)
        # A new residual interface leaves the serialized D modules untouched.
        self.response_projection = nn.Conv2d(256, owner.backbone.embed_dim, 1, bias=False)
        nn.init.zeros_(self.response_projection.weight)

    def forward(self, images, response, spatial_gates, alpha):
        b, steps = next(iter(images.values())).shape[:2]
        response = response.reshape(b, steps, len(self.owner.image_keys), 196, 256)
        # Frozen deployed alpha suppresses response conditioning without phase labels.
        response = response * alpha[..., None, None, None]
        result, backbone = {}, self.owner.backbone
        for camera, key in enumerate(self.owner.image_keys):
            image, gate = images[key], spatial_gates[key].flatten(0, 1)
            tokens = response[:, :, camera].reshape(b*steps, 196, 256)
            embedding = self.owner.camera_embedding.weight[camera].reshape(1, 1, -1)
            features, skips, height, width = backbone.encode(image.flatten(0, 1), tokens, embedding)
            update = self.response_projection(tokens.transpose(1, 2).reshape(b*steps, 256, 14, 14))
            features = features + F.interpolate(update, size=features.shape[-2:], mode='bilinear', align_corners=False)
            fused = features
            pooled = backbone.decoder_film.pool(tokens)
            for index, (decoder, skip) in enumerate(zip(backbone.decoders, reversed(skips))):
                if backbone.conv_block == 'bottleneck':
                    features = decoder(features, skip)
                else:
                    features = decoder(torch.cat([F.interpolate(features, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
                if index < len(self.owner.spatial_modulations):
                    features = self.owner.spatial_modulations[index](features, fused, gate)
                if index < 2:
                    features = backbone.decoder_film(features, pooled, index, gate, alpha.flatten())
            features = F.interpolate(features, size=(height, width), mode='bilinear', align_corners=False)
            raw = backbone.output(features).tanh().reshape_as(image).float()
            support = spatial_gates[key].float()*alpha[..., None, None, None]
            delta = self.owner.residual_scale*raw*support
            bound = self.epsilon*support
            delta = torch.maximum(-bound, torch.minimum(bound, delta))
            result[key] = torch.where(support > 0, (image.float()+delta).clamp(0, 1), image.float())
        return result


def metric_rows(student, predicted, target, original, rendered, ids, episodes, phases, threshold):
    s, p, t = student.float(), predicted.float(), target.float()
    active_t = t.square().mean(-1).sqrt() > threshold
    active_p = p.square().mean(-1).sqrt() > threshold
    cos_pt = F.cosine_similarity(p, t, dim=-1, eps=1e-6)
    cos_sp = F.cosine_similarity(s, p, dim=-1, eps=1e-6)
    cos_st = F.cosine_similarity(s, t, dim=-1, eps=1e-6)
    rgb = torch.cat([(rendered[k]-original[k]).flatten(1) for k in original], 1)
    rows = []
    for j, index in enumerate(ids):
        at, ap = active_t[j], active_p[j]
        rows.append(dict(index=int(index), episode=int(episodes[j]), phase=int(phases[j]),
            predictor_mse=float((p[j]-t[j]).square().mean()), realization_mse=float((s[j]-p[j]).square().mean()),
            end_to_end_mse=float((s[j]-t[j]).square().mean()), response_mse=float(s[j].square().mean()),
            predictor_energy=float(p[j].square().mean()), teacher_energy=float(t[j].square().mean()),
            teacher_active=int(at.sum()), predictor_active=int(ap.sum()),
            predictor_cosine_sum=float(cos_pt[j][at].sum()), realization_cosine_sum=float(cos_sp[j][ap].sum()),
            end_to_end_cosine_sum=float(cos_st[j][at].sum()),
            s_teacher_ss=float(s[j][at].double().square().sum()), teacher_ss=float(t[j][at].double().square().sum()),
            s_predictor_ss=float(s[j][ap].double().square().sum()), predictor_ss=float(p[j][ap].double().square().sum()),
            rgb_mse=float(rgb[j].square().mean()), rgb_max=float(rgb[j].abs().max())))
    return rows


def aggregate(rows):
    if not rows:
        return dict(samples=0)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    nt, np_ = sum(r['teacher_active'] for r in rows), sum(r['predictor_active'] for r in rows)
    tt, pp = sum(r['teacher_ss'] for r in rows), sum(r['predictor_ss'] for r in rows)
    return dict(samples=len(rows), **{k: mean(k) for k in ['predictor_mse', 'realization_mse', 'end_to_end_mse', 'response_mse', 'predictor_energy', 'teacher_energy']},
        predictor_cosine=sum(r['predictor_cosine_sum'] for r in rows)/nt if nt else None,
        realization_cosine=sum(r['realization_cosine_sum'] for r in rows)/np_ if np_ else None,
        end_to_end_cosine=sum(r['end_to_end_cosine_sum'] for r in rows)/nt if nt else None,
        response_rms_ratio_teacher=(sum(r['s_teacher_ss'] for r in rows)/tt)**.5 if tt else None,
        response_rms_ratio_predictor=(sum(r['s_predictor_ss'] for r in rows)/pp)**.5 if pp else None,
        response_rms=mean('response_mse')**.5, rgb_rms=mean('rgb_mse')**.5, rgb_max=max(r['rgb_max'] for r in rows))


def grouped(rows, config):
    return dict(phases={name: aggregate([r for r in rows if r['phase'] == p]) for p, name in enumerate(PHASES)},
        episodes={str(ep): {name: aggregate([r for r in rows if r['episode'] == ep and r['phase'] == p])
             for p, name in enumerate(PHASES)} for ep in sorted({r['episode'] for r in rows})},
        selection_contact=aggregate([r for r in rows if r['phase'] == 2 and r['episode'] in config.selection_episodes]),
        external_contact=aggregate([r for r in rows if r['phase'] == 2 and r['episode'] in config.external_episodes]))
