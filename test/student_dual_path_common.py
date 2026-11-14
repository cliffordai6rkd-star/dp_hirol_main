"""Shared controlled inputs for the independent student experiments."""
from pathlib import Path
import copy
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.dual_path_render_net import load_common_initialization
from diffusion_policy.scripts.visualize_cross_attention_episode import instantiate_policy

VARIANTS = ('a_baseline','b_interleaved','c_film_linear','c_film_mlp')

def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_inputs(inputs,device):
    p=torch.load(inputs/'pinned_teacher_ema.ckpt',map_location='cpu',pickle_module=dill,weights_only=False)
    cfg=OmegaConf.create(OmegaConf.to_container(p['cfg'],resolve=True))
    teacher=instantiate_policy(p,cfg,device,capture_attention=False).eval().requires_grad_(False)
    del p
    initial=torch.load(inputs/'student_rgb_initial.ckpt',map_location='cpu',pickle_module=dill,weights_only=False)
    return teacher,cfg,initial

def model_for(initial,variant):
    cfg=OmegaConf.create(OmegaConf.to_container(initial['cfg'],resolve=True))
    options=OmegaConf.to_container(cfg.render_net,resolve=True)
    nested=options.pop('backbone',{})
    options.update(nested)
    options['high_attention_render']['threshold']=.2
    options['dual_path']=dict(enabled=variant!='a_baseline',decoder_film=variant.startswith('c_'))
    options['image_self_attn_layers']=6 if variant=='a_baseline' else 4
    options['context_adapter']=dict(enabled=True,kind='mlp' if variant=='c_film_mlp' else 'linear',hidden_dim=128)
    torch.manual_seed(42)
    model=build_render_net(7,options)
    source=initial['state_dicts']['ema_model']
    transfer=load_common_initialization(model,source)
    cfg.render_net=options
    return model,cfg,transfer

def datasets_from_reference(inputs,image_cache):
    meta=json.loads((inputs/'reference_train_cache.json').read_text())
    cfg=OmegaConf.create(meta['identity']['dataset'])
    cache_meta=json.loads((image_cache/'metadata.json').read_text())
    cfg.preload_images=False
    cfg.video_decode_device=('cuda:0' if cache_meta.get('decoder')=='TorchCodec CUDA' else 'cpu')
    dataset=hydra.utils.instantiate(cfg)
    if cache_meta.get('decoder')=='TorchCodec CUDA':
        dataset._video_decoder=dataset._make_video_decoder()
        assert dataset._video_decoder is not None, 'Historical CUDA preprocessing is required'
    assert cache_meta['dataset_length']==dataset.dataset_length
    assert cache_meta['extra']['image_feature_map']==dict(dataset.image_feature_map)
    arrays={k:np.load(image_cache/f'{k}.npy',mmap_mode='r') for k in dataset.rgb_keys}
    probes=[0,dataset.dataset_length//2,dataset.dataset_length-1]
    decoded=dataset._build_image_chunk(probes,{k:tuple(v.shape[1:]) for k,v in arrays.items()})
    for k,v in arrays.items():
        np.testing.assert_array_equal(np.rint(np.clip(decoded[k],0,1)*255).astype(np.uint8),v[probes])
    dataset.image_data=arrays
    dataset.lerobot_dataset.close();dataset._video_decoder=None
    validation=dataset.get_validation_dataset()
    manifests={}
    for name,split in [('train',dataset),('validation',validation)]:
        ref=json.loads((inputs/f'reference_{name}_cache.json').read_text())
        sha=hashlib.sha256(np.ascontiguousarray(split.indices).tobytes()).hexdigest()
        assert sha==ref['identity'][name]['indices_sha256'],name
        manifests[name]=dict(samples=len(split),indices_sha256=sha,
                             phase_counts=np.bincount(split.phase_labels,minlength=3).tolist())
    return dataset,validation,manifests

def to_device(value,device):
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    return value.to(device,non_blocking=True)

class PreparedDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,root):
        self.dataset=dataset
        meta=json.loads((root/'metadata.json').read_text())
        assert meta['complete'] and meta['samples']==len(dataset)
        self.arrays={k:np.load(root/f'{k}.npy',mmap_mode='r') for k in ['target','zero','response']}
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):
        item=self.dataset[i]
        item.update({k:torch.from_numpy(np.array(v[i],copy=True)) for k,v in self.arrays.items()})
        item['phase']=int(self.dataset.phase_labels[i]); item['index']=i
        return item
