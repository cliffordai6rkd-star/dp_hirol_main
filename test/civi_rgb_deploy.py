"""Inference-only causal interface for isolated CIVI renderer candidates.

No Teacher fusion/DiT/Adapter, cached targets, phases or sample IDs are evaluated.
Input observations are the two most recent causal RGB/torque-history observations,
matching the audited training context. A caller buffers past observations; there
is no future frame or torque. Do not silently replace this with one repeated frame.
"""
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from civi_semantic_patch_renderer import SemanticPatchRenderer


class FrozenVisualScene(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.backbone=encoder.backbone;self.projection=encoder.image_projection
        for name in ['obs_position','image_modality','camera_embedding']:
            value=getattr(encoder,name)
            self.register_buffer(name,value.detach().clone() if value is not None else None)
        self.register_buffer('mean',encoder.image_mean.detach().clone());self.register_buffer('std',encoder.image_std.detach().clone())
        self.keys=tuple(encoder.image_keys);self.register_tokens=encoder.num_register_tokens
        self.interpolate=encoder._interpolate_dino_positions
        self.requires_grad_(False).eval()

    def forward(self,images):
        b,t=images[self.keys[0]].shape[:2]
        flat=torch.cat([images[k].flatten(0,1) for k in self.keys],0)
        kwargs={'pixel_values':(flat-self.mean.to(flat))/self.std.to(flat)}
        if self.interpolate:kwargs['interpolate_pos_encoding']=True
        patches=self.backbone(**kwargs).last_hidden_state[:,1+self.register_tokens:]
        patches=self.projection(patches);pieces=patches.split(b*t)
        result=[]
        for camera,value in enumerate(pieces):
            value=value.reshape(b,t,196,256)+self.obs_position[:,:t]+self.image_modality
            if self.camera_embedding is not None:value=value+self.camera_embedding[:,:,camera]
            result.append(value)
        return torch.cat(result,2).reshape(b,-1,256)


class CIVISharedRGB(nn.Module):
    def __init__(self,teacher,predictor,renderer):
        super().__init__()
        if teacher.future_phase_wrench_key is not None:
            raise ValueError('This candidate preprocessing requires future data; not a causal deployment')
        self.scene=FrozenVisualScene(teacher.obs_encoder)
        self.preprocess=WrenchPreprocessor.from_policy(teacher)
        self.predictor=predictor;self.renderer=renderer
        self.image_keys=tuple(renderer.owner.image_keys);self.wrench_key=teacher.wrench_key
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self,obs):
        forbidden={'contact_phase','contact_alpha','episode','index','target','zero'}&set(obs)
        if forbidden:raise ValueError('Deployment input contains offline labels/targets: '+str(forbidden))
        images={k:obs[k][:,:2] for k in self.image_keys}
        if any(v.shape[1]!=2 for v in images.values()):raise ValueError('Requires the two most recent causal observations')
        wrench,_,_=self.preprocess(obs)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            visual=self.scene(images).float()
            masks,condition=self.renderer.owner.condition_network(images,wrench)
            spatial={k:self.renderer.owner._mask_gate(masks[k].flatten(0,1),images[k]).reshape_as(masks[k]).float() for k in images}
        pooled=visual.reshape(len(visual),2,392,256).mean(2)
        with torch.autocast('cuda',enabled=False):
            pred=self.predictor(visual,pooled,obs[self.wrench_key][:,:2].float(),torch.arange(784,device=visual.device))
        if isinstance(self.renderer,SemanticPatchRenderer):self.renderer.visual_tokens=visual
        with torch.autocast('cuda',dtype=torch.bfloat16):
            rendered=self.renderer(images,pred,spatial,condition['contact_alpha'].float())
        return rendered
