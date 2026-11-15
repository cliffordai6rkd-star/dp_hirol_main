from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from civi_patch_response_renderer import PatchResponseRenderer
from civi_semantic_patch_renderer import SemanticPatchRenderer


def test_zero_conditions_bounds_gradients_and_camera_layout():
    torch.set_num_threads(1);torch.manual_seed(42)
    owner=nn.Module();owner.image_keys=('side','wrist');owner.wrench_encoder=nn.Linear(7,7)
    net=PatchResponseRenderer(owner)
    image={k:torch.full((1,2,3,224,224),.5) for k in owner.image_keys}
    gates={k:torch.ones(1,2,1,224,224) for k in image};alpha=torch.ones(1,2)
    delta=torch.zeros(1,784,256);delta[:,:196]=torch.randn(1,196,256)*.1
    rendered=net(image,delta,gates,alpha)
    assert (rendered['side'][:,0]-image['side'][:,0]).abs().sum()>0
    assert torch.equal(rendered['side'][:,1],image['side'][:,1])
    assert torch.equal(rendered['wrist'],image['wrist'])
    for args in [(torch.zeros_like(delta),alpha),(delta,torch.zeros_like(alpha))]:
        result=net(image,args[0],gates,args[1])
        assert all(torch.equal(result[k],image[k]) for k in image)
    loss=sum((rendered[k]-image[k]).square().mean() for k in image);loss.backward()
    assert net.response.weight.grad.norm()>0 and net.head[-1].weight.grad.norm()>0
    assert owner.wrench_encoder.weight.grad is None and not owner.wrench_encoder.weight.requires_grad
    assert max((rendered[k]-image[k]).abs().max() for k in image)<=.1


def test_semantic_scene_is_frozen_input_and_zero_response_is_identity():
    torch.set_num_threads(1);torch.manual_seed(42)
    owner=nn.Module();owner.image_keys=('side','wrist');owner.wrench_encoder=nn.Linear(7,7)
    net=SemanticPatchRenderer(owner)
    image={k:torch.full((1,2,3,224,224),.5) for k in owner.image_keys}
    gates={k:torch.ones(1,2,1,224,224) for k in image};alpha=torch.ones(1,2)
    net.visual_tokens=torch.randn(1,784,256,requires_grad=True)
    output=net(image,torch.randn(1,784,256)*.1,gates,alpha)
    sum((output[k]-image[k]).square().mean() for k in image).backward()
    assert net.visual_tokens.grad is None and net.scene[0].weight.grad.norm()>0
    zero=net(image,torch.zeros(1,784,256),gates,alpha)
    assert all(torch.equal(zero[k],image[k]) for k in image)
