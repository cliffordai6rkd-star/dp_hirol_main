"""Lightweight causal CIVI RGB deployment: no teacher, DINO, Adapter or targets."""
import copy
from pathlib import Path

import torch
from torch import nn

from diffusion_policy.distillation.dit_rgb import ARCHITECTURE
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor


class CiviRGB(nn.Module):
    def __init__(self, renderer, preprocessor):
        super().__init__()
        self.renderer, self.preprocessor = renderer, preprocessor
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, tau_ext):
        """Chronological RGB [B,2,3,224,224] and raw-unit tau [B,2,8,7].

        Both RGB observations must be real causal frames. For each observation,
        supply its own trailing eight torque measurements. Do not repeat the
        current frame/history to synthesize the older observation.
        """
        if tuple(images) != tuple(self.renderer.image_keys):
            raise ValueError(f'Expected ordered cameras {self.renderer.image_keys}')
        if tau_ext.ndim != 4 or tau_ext.shape[1:] != (2,8,7):
            raise ValueError('Expected two causal observation histories [B,2,8,7]')
        for key,value in images.items():
            if value.shape != (len(tau_ext),2,3,224,224):
                raise ValueError(f'{key}: expected [B,2,3,224,224]')
        wrench = self.preprocessor({self.preprocessor.wrench_key:tau_ext})[0]
        # Deliberately no cached_condition/phase/alpha/sample_id input here.
        return self.renderer(images,wrench)

    @classmethod
    def load(cls, path, device='cpu'):
        payload = torch.load(path,map_location='cpu',weights_only=False)
        if payload['architecture'] != ARCHITECTURE or payload['interface'] != 'causal_rgb_only_v1':
            raise ValueError('Unsupported RGB artifact')
        renderer = build_render_net(7,payload['options'])
        renderer.load_state_dict(payload['renderer'],strict=True)
        renderer.set_training_stage('rgb')
        preprocessor = WrenchPreprocessor.from_config(payload['preprocessing'])
        preprocessor.load_state_dict(payload['preprocessing_state'],strict=True)
        return cls(renderer,preprocessor).to(device)


def export_rgb(checkpoint, destination, preprocessor):
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['architecture'] != ARCHITECTURE:
        raise ValueError('Export only accepts the audited direct torque architecture')
    if saved['options']['context_adapter']['enabled'] or saved['options']['dual_path']['enabled']:
        raise ValueError('Training-only or response-conditioned module in RGB export')
    artifact = dict(interface='causal_rgb_only_v1',architecture=ARCHITECTURE,
                    options=saved['options'],renderer=saved['ema_model'],
                    preprocessing=preprocessor.get_config(),preprocessing_state={k:v.detach().cpu().clone() for k,v in preprocessor.state_dict().items()
                        if not k.startswith('normalizer.') or k.startswith('normalizer.params_dict.'+preprocessor.wrench_key+'.')},
                    source_step=saved['step'],dependency=saved['dependency'],
                    causal_history='2 real RGB observations, each with its own trailing8 raw-unit torque samples')
    torch.save(artifact,Path(destination))
