"""Small spatially aligned RGB realization candidate, independent of formal D."""
import torch
from torch import nn
from torch.nn import functional as F


class PatchResponseRenderer(nn.Module):
    """Shared scene/response -> 16x16 RGB blocks; causal inputs, no sample state.

    The nonlinear head is evaluated at the actual and zero response using the
    same scene. Their difference makes a zero response exactly image-preserving.
    Final spatial*alpha support and the .1 pixel bound are unchanged.
    """
    def __init__(self,owner,epsilon=.1):
        super().__init__();self.owner=owner;self.epsilon=float(epsilon)
        owner.requires_grad_(False)
        self.scene=nn.Sequential(nn.Conv2d(3,128,16,stride=16),nn.GELU())
        self.response=nn.Linear(256,128,bias=False)
        self.head=nn.Sequential(nn.Linear(262,256),nn.GELU(),nn.Linear(256,768))
        nn.init.normal_(self.head[-1].weight,std=.005);nn.init.zeros_(self.head[-1].bias)
        self.use_projection,self.use_ca_response=False,False

    def forward(self,images,response,spatial_gates,alpha,*,trace=False):
        b,steps=next(iter(images.values())).shape[:2];keys=self.owner.image_keys
        if response.shape!=(b,steps*len(keys)*196,256):raise ValueError('Expected time,camera,14x14,channel layout')
        response=response.reshape(b,steps,len(keys),196,256)
        result={};counts={k:[] for k in ['allowed_channels','total_channels','hard_clipped_channels','range_clipped_channels']}
        for camera,key in enumerate(keys):
            image=images[key];height,width=image.shape[-2:]
            if (height,width)!=(224,224):raise ValueError('Candidate pins the audited224x224/14x14 geometry')
            scene=self.scene(image.flatten(0,1)).flatten(2).transpose(1,2)
            delta=response[:,:,camera].reshape(b*steps,196,256)
            condition=self.response(delta/.1)
            patches=torch.arange(196,device=image.device)
            xy=torch.stack([(patches%14)/13,(patches//14)/13],-1).to(scene)
            time=torch.arange(steps,device=image.device).repeat(b)
            pos=torch.cat([xy[None].expand(b*steps,-1,-1),F.one_hot(time,2)[:,None].expand(-1,196,-1),
                F.one_hot(torch.full_like(time,camera),2)[:,None].expand(-1,196,-1)],-1).to(scene)
            prefix=torch.cat([scene,pos],-1)
            real=self.head(torch.cat([prefix,condition],-1))
            zero=self.head(torch.cat([prefix,torch.zeros_like(condition)],-1))
            raw=(real-zero).reshape(b*steps,14,14,3,16,16).permute(0,3,1,4,2,5).reshape(b,steps,3,224,224).float()
            support=spatial_gates[key].float()*alpha[...,None,None,None]
            change=self.epsilon*support*raw.tanh()
            before=image.float()+change
            result[key]=torch.where(support>0,before.clamp(0,1),image.float())
            if trace:
                allowed=(support>0).expand_as(image)
                count=lambda x:x.flatten(1).sum(-1).float()
                counts['allowed_channels'].append(count(allowed))
                counts['total_channels'].append(torch.full((b,),image[0].numel(),device=image.device,dtype=torch.float32))
                counts['hard_clipped_channels'].append(torch.zeros(b,device=image.device))
                counts['range_clipped_channels'].append(count(((before<0)|(before>1))&allowed))
        if trace:return result,{k:torch.stack(v).mean(0) for k,v in counts.items()}
        return result
