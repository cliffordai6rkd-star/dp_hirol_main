"""Prepare full unchanged student splits and paired targets from the pinned teacher."""
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from student_dual_path_common import load_inputs,datasets_from_reference,to_device,write_json,tensor_hash
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--image-cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dev=torch.device('cuda:0');teacher,tcfg,initial=load_inputs(a.inputs,dev)
    train,val,manifest=datasets_from_reference(a.inputs,a.image_cache)
    manifest['teacher_state_sha256']=tensor_hash(teacher.state_dict())
    manifest['preparation']='FP32 paired targets; BF16 projected pre-adapter visual tokens; clear images'
    write_json(a.output_dir/'manifest.json',manifest)
    enc=teacher.obs_encoder;start=time.monotonic()
    for name,ds in [('train',train),('validation',val)]:
        dest=a.output_dir/name;dest.mkdir(exist_ok=True)
        if (dest/'metadata.json').exists():
            import json
            if json.loads((dest/'metadata.json').read_text()).get('complete'):
                print(dict(split=name,reuse_completed=True),flush=True)
                continue
        arrays={};offset=0
        for raw in DataLoader(ds,batch_size=a.batch_size,num_workers=8,pin_memory=True):
            obs=to_device(raw['obs'],dev)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                cached=[];original=enc._encode_dino
                def capture(image,**kw):
                    result=original(image,**kw);cached.append(result);return result
                enc._encode_dino=capture
                try: target=teacher._encode_context(obs,apply_curriculum_mask=False)
                finally: enc._encode_dino=original
                assert len(cached)==1
                enc._encode_dino=lambda image,**kw:cached[0]
                try: zero=teacher._encode_context(zero_wrench_reference(teacher,obs),apply_curriculum_mask=False)
                finally:enc._encode_dino=original
                response=(target-zero).float().norm(dim=-1)
                images={k:obs[k][:,:teacher.n_obs_steps] for k in enc.image_keys}
            seen=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                enc.encode_image_context(images,context_adapter=lambda x:(seen.append(x.detach()),x)[1])
            values=dict(target=target,zero=zero,response=response,visual=seen[0])
            count=len(target)
            for key,value in values.items():
                # Visual tokens include FP32 positional embeddings; keep them
                # exact for the adapter residual path, even under BF16 DINO.
                array=value.float().cpu().numpy().astype(np.float32 if key in ('response','visual') else np.float16)
                assert np.isfinite(array).all(),key
                if key not in arrays:
                    arrays[key]=np.lib.format.open_memmap(dest/f'{key}.npy',mode='w+',dtype=array.dtype,
                                                        shape=(len(ds),*array.shape[1:]))
                arrays[key][offset:offset+count]=array
            offset+=count
            if offset%512<count or offset==len(ds):
                print(dict(split=name,samples=offset,total=len(ds),elapsed_seconds=time.monotonic()-start),flush=True)
        for x in arrays.values():x.flush()
        assert offset==manifest[name]['samples']
        write_json(dest/'metadata.json',dict(complete=True,**manifest[name]))
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__':main()
