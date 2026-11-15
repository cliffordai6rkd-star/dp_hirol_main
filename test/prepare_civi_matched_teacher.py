"""Build a fresh, hash-pinned paired cache and calibrate the existing MLP.

No old teacher targets or visual projections are reused. Historical RGB decoding
and sample-index references are reused and audited. The caller pins the exact
teacher checkpoint before invoking this script; a changing latest.ckpt is not input.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader,Subset

from student_dual_path_common import load_inputs,datasets_from_reference,to_device,write_json,tensor_hash
from paired_response_full_common import file_hash,episode_ids
from new_teacher_response_common import split_details
from prepare_new_teacher_shared_response import calibrate
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True,help='Existing70750 experiment for input/schema references only')
    p.add_argument('--old',type=Path,required=True,help='Historical RGB decode cache root')
    p.add_argument('--pinned-teacher',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert file_hash(a.pinned_teacher)==a.expected_sha256
    inputs=a.output/'inputs';inputs.mkdir()
    shutil.copy2(a.pinned_teacher,inputs/'pinned_teacher_ema.ckpt')
    for name in ['student_rgb_initial.ckpt','reference_train_cache.json','reference_validation_cache.json','reference_samples.pt']:
        shutil.copy2(a.base/'inputs'/name,inputs/name)
    teacher,tcfg,initial=load_inputs(inputs,torch.device('cuda:0'));before=tensor_hash(teacher.state_dict())
    train,val,identities=datasets_from_reference(inputs,a.old/'decoded_cuda')
    previous=json.loads((a.base/'episode_split.json').read_text())
    split=split_details(dict(train=train,validation=val),previous['overlap']['teacher_validation_episodes'])
    write_json(a.output/'episode_split.json',split)
    cache=a.output/'paired_cache';cache.mkdir()
    raw_reference=json.loads((a.base/'paired_cache/manifest.json').read_text())['reused_from']
    manifest=dict(teacher_checkpoint_sha256=a.expected_sha256,teacher_state_sha256=before,splits=identities,
        preprocessing='Unchanged audited historical CUDA RGB; physical raw zeros before normalizer; clear images',
        targets='FP32 teacher then FP16 storage; identical real/zero DINO tokens',
        visual='New teacher DINO/projection/embeddings BF16+FP32 additions; FP32 storage',
        reused_from=raw_reference,reuse_semantics='reused_from is ONLY the raw-history audit reference; NO old targets/visual tokens reused',
        arrays={},audit=[])
    enc=teacher.obs_encoder
    for name,ds in [('train',train),('validation',val)]:
        dest=cache/name;dest.mkdir();arrays={};offset=0
        np.save(dest/'episodes.npy',episode_ids(ds));np.save(dest/'phases.npy',ds.phase_labels)
        loader=DataLoader(ds,batch_size=32,num_workers=8,pin_memory=True)
        for raw in loader:
            obs=to_device(raw['obs'],torch.device('cuda:0'))
            original=enc._encode_dino;captured=[]
            def capture(image,**kw):
                out=original(image,**kw);captured.append(out);return out
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                enc._encode_dino=capture
                try:target=teacher._encode_context(obs,apply_curriculum_mask=False)
                finally:enc._encode_dino=original
                assert len(captured)==1
                enc._encode_dino=lambda image,**kw:captured[0]
                try:zero=teacher._encode_context(zero_wrench_reference(teacher,obs),apply_curriculum_mask=False)
                finally:enc._encode_dino=original
                images,_,_,_=teacher.prepare_context_inputs_with_future_contact(obs)
            seen=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                enc.encode_image_context(images,context_adapter=lambda x:(seen.append(x.detach()),x)[1])
            for key,value in dict(target=target,zero=zero,visual=seen[0]).items():
                arr=value.float().cpu().numpy().astype(np.float32 if key=='visual' else np.float16)
                assert np.isfinite(arr).all()
                if key not in arrays:arrays[key]=np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=arr.dtype,shape=(len(ds),*arr.shape[1:]))
                arrays[key][offset:offset+len(arr)]=arr
            offset+=len(target)
            if offset%512<len(target) or offset==len(ds):print(json.dumps(dict(event='new_teacher_cache',split=name,rows=offset,total=len(ds),elapsed_seconds=time.monotonic()-started)),flush=True)
        assert offset==len(ds)
        for v in arrays.values():v.flush()
        for phase in range(3):
            ids=np.flatnonzero(ds.phase_labels==phase);i=int(ids[len(ids)//2]);obs=to_device(next(iter(DataLoader(Subset(ds,[i]),batch_size=1)))['obs'],torch.device('cuda:0'))
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                actual=dict(target=teacher._encode_context(obs,apply_curriculum_mask=False),zero=teacher._encode_context(zero_wrench_reference(teacher,obs),apply_curriculum_mask=False))
            errors={k:float(np.mean((v.cpu().numpy()[0]-arrays[k][i].astype(np.float32))**2)) for k,v in actual.items()}
            assert max(errors.values())<1e-5,(name,phase,errors)
            manifest['audit'].append(dict(split=name,index=i,phase=phase,episode=int(episode_ids(ds)[i]),errors=errors,image_sha256={k:hashlib.sha256(obs[k].cpu().numpy().tobytes()).hexdigest() for k in enc.image_keys}))
        manifest['arrays'][name]={k:file_hash(dest/(k+'.npy')) for k in ['target','zero','visual','phases','episodes']}
        write_json(dest/'metadata.json',dict(complete=True,**identities[name]))
    assert tensor_hash(teacher.state_dict())==before
    write_json(cache/'manifest.json',manifest)
    free_noise=[]
    target=np.load(cache/'train/target.npy',mmap_mode='r');zero=np.load(cache/'train/zero.npy',mmap_mode='r')
    ids=np.flatnonzero(train.phase_labels==0)
    for start in range(0,len(ids),64):
        ix=ids[start:start+64];delta=target[ix].astype(np.float32)-zero[ix].astype(np.float32)
        free_noise.append(np.sqrt(np.mean(delta**2,axis=-1)).reshape(-1))
    threshold=max(.001,float(np.quantile(np.concatenate(free_noise),.99)))
    (a.output/'cnn').mkdir();write_json(a.output/'cnn/manifest.json',dict(threshold=threshold,source='New pinned teacher train-Free response99th percentile, minimum0.001'))
    (a.output/'d').mkdir();(a.output/'d/best.pt').symlink_to((a.base/'d/best.pt').resolve())
    config=OmegaConf.load(Path(__file__).parent/'config/new_teacher_shared_response.yaml')
    config.teacher_sha256=a.expected_sha256;config.teacher_state_sha256=before
    config.adapter_initial_checkpoint='adapter/best_adapter.pt'
    # Caller supplies a pinned-step provenance; never infer teacher version from a directory name.
    import dill
    payload=torch.load(inputs/'pinned_teacher_ema.ckpt',map_location='cpu',pickle_module=dill,weights_only=False)
    config.teacher_step=int(dill.loads(payload['pickles']['optimizer_step']))
    OmegaConf.save(config,a.output/'config.yaml')
    calibrate(argparse.Namespace(experiment=a.output,root=a.base),config,teacher,initial,cache,split,started)
    write_json(a.output/'matched_manifest.json',dict(teacher_step=config.teacher_step,teacher_sha256=a.expected_sha256,
        teacher_state_sha256=before,adapter_sha256=file_hash(a.output/'adapter/best_adapter.pt'),
        stage='Fresh cache and Adapter complete; condition/predictor matching required before RGB',
        elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
