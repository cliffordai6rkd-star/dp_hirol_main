"""Spatial RGB realization using the same frozen scene features as Predictor."""
import torch
from torch import nn
from torch.nn import functional as F

from civi_patch_response_renderer import PatchResponseRenderer


class SemanticPatchRenderer(PatchResponseRenderer):
    def __init__(self,owner,epsilon=.1):
        super().__init__(owner,epsilon)
        self.scene=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.LayerNorm(128))
        self.visual_tokens=None

    def forward(self,images,response,spatial_gates,alpha,*,trace=False):
        b,steps=next(iter(images.values())).shape[:2];keys=self.owner.image_keys
        if self.visual_tokens is None or self.visual_tokens.shape!=(b,steps*len(keys)*196,256):
            raise ValueError('Requires frozen DINO pre-adapter scene tokens computed from the SAME RGB')
        visual=self.visual_tokens.detach().reshape(b,steps,len(keys),196,256)
        if response.shape!=(b,steps*len(keys)*196,256):raise ValueError('Response layout mismatch')
        response=response.reshape_as(visual);result={}
        counts={k:[] for k in ['allowed_channels','total_channels','hard_clipped_channels','range_clipped_channels']}
        for camera,key in enumerate(keys):
            image=images[key]
            scene=self.scene(visual[:,:,camera].reshape(b*steps,196,256))
            condition=self.response(response[:,:,camera].reshape(b*steps,196,256)/.1)
            patches=torch.arange(196,device=image.device);xy=torch.stack([(patches%14)/13,(patches//14)/13],-1).to(scene)
            times=torch.arange(steps,device=image.device).repeat(b)
            pos=torch.cat([xy[None].expand(b*steps,-1,-1),F.one_hot(times,2)[:,None].expand(-1,196,-1),
                F.one_hot(torch.full_like(times,camera),2)[:,None].expand(-1,196,-1)],-1).to(scene)
            prefix=torch.cat([scene,pos],-1)
            real=self.head(torch.cat([prefix,condition],-1));zero=self.head(torch.cat([prefix,torch.zeros_like(condition)],-1))
            raw=(real-zero).reshape(b*steps,14,14,3,16,16).permute(0,3,1,4,2,5).reshape_as(image).float()
            support=spatial_gates[key].float()*alpha[...,None,None,None]
            delta=self.epsilon*support*raw.tanh();before=image.float()+delta
            result[key]=torch.where(support>0,before.clamp(0,1),image.float())
            if trace:
                allowed=(support>0).expand_as(image);count=lambda x:x.flatten(1).sum(-1).float()
                counts['allowed_channels'].append(count(allowed));counts['total_channels'].append(torch.full((b,),image[0].numel(),device=image.device,dtype=torch.float32))
                counts['hard_clipped_channels'].append(torch.zeros(b,device=image.device));counts['range_clipped_channels'].append(count(((before<0)|(before>1))&allowed))
        if trace:return result,{k:torch.stack(v).mean(0) for k,v in counts.items()}
        return result
