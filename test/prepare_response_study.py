"""Small, pinned current-teacher diagnostic cache; production data/model unchanged."""
from pathlib import Path
import argparse
import json
import sys

import dill
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import load_inputs, datasets_from_reference, to_device, write_json, tensor_hash
from paired_response_full_common import episode_ids, file_hash
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference
from diffusion_policy.common.sampler import get_val_mask


def selection(ds, eps, contact_per_episode=6, free_per_episode=2):
    rows=[]
    for episode in np.unique(eps):
        for phase,n in [(2,contact_per_episode),(0,free_per_episode)]:
            ids=np.flatnonzero((eps==episode)&(ds.phase_labels==phase))
            if len(ids): rows.extend(ids[np.linspace(0,len(ids)-1,min(n,len(ids))).round().astype(int)].tolist())
    return sorted(set(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    dev=torch.device('cuda:0');teacher,cfg,initial=load_inputs(a.study/'inputs',dev)
    train,val,splits=datasets_from_reference(a.study/'inputs',a.root/'decoded_cuda')
    datasets=dict(train=train,validation=val);eps={s:episode_ids(ds) for s,ds in datasets.items()}
    parquet=sorted(Path(train.dataset_path).glob('data/**/*.parquet'))
    poses=np.concatenate([np.asarray(pq.read_table(f,columns=['observation.joint']).column(0).to_pylist()) for f in parquet])
    teacher_val=get_val_mask(len(train.episode_ends),float(cfg.task.dataset.val_ratio),int(cfg.task.dataset.seed))
    manifest=dict(teacher_state_sha256=tensor_hash(teacher.state_dict()),
                  teacher_checkpoint_sha256=file_hash(a.study/'inputs/pinned_teacher_ema.ckpt'),
                  teacher_validation_episodes=np.flatnonzero(teacher_val).tolist(),splits=splits,
                  selection='six spread contact + two Free per TRAIN episode; all validation contact + two Free per episode',
                  student_checkpoint='review_1000/c_mlp_checkpoint_1000.ckpt',adapter='calibration_mlp_cuda/adapter.pt')
    enc=teacher.obs_encoder
    for split,ds in datasets.items():
        ids=selection(ds,eps[split]) if split=='train' else sorted(set(
            np.flatnonzero(ds.phase_labels==2).tolist()+[i for i in selection(ds,eps[split]) if ds.phase_labels[i]==0]))
        dest=a.study/split;dest.mkdir(parents=True,exist_ok=False)
        arrays={};offset=0;records=[]
        for raw in DataLoader(Subset(ds,ids),batch_size=16,num_workers=4,pin_memory=True):
            obs=to_device(raw['obs'],dev)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                captured=[];original=enc._encode_dino
                def capture(im,**kwargs):
                    out=original(im,**kwargs);captured.append(out);return out
                enc._encode_dino=capture
                try:target=teacher._encode_context(obs,apply_curriculum_mask=False).float()
                finally:enc._encode_dino=original
                assert len(captured)==1
                enc._encode_dino=lambda im,**kwargs:captured[0]
                try:zero=teacher._encode_context(zero_wrench_reference(teacher,obs),apply_curriculum_mask=False).float()
                finally:enc._encode_dino=original
                images,wr,contact,future=teacher.prepare_context_inputs_with_future_contact(obs)
                visual=[]
                # Exact FP32 matching features, before adapter/norm.
                enc.encode_image_context(images,context_adapter=lambda z:(visual.append(z),z)[1])
            values=dict(target=target,zero=zero,visual=visual[0],wrench=wr,
                        raw_wrench=obs[teacher.wrench_key],alpha=obs['contact_alpha'])
            for k,v in values.items():
                x=v.cpu().numpy().astype(np.float32 if k in ['wrench','raw_wrench','alpha'] else np.float16)
                if k not in arrays:arrays[k]=np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=x.dtype,shape=(len(ids),*x.shape[1:]))
                arrays[k][offset:offset+len(target)]=x
            for index in ids[offset:offset+len(target)]:
                anchor=ds._anchor_index_from_sampler_row(ds.indices[index]);episode=int(eps[split][index])
                sequence=ds._sample_indices_to_sequence(index)
                if ds.window_sampling_strategy=='timestamp':sequence=ds._retime_sequence_indices(sequence)
                records.append(dict(index=int(index),episode=episode,phase=int(ds.phase_labels[index]),
                    anchor=int(anchor),observation_frames=sequence[:ds.n_obs_steps].tolist(),
                    posture=poses[anchor].tolist(),teacher_held_out=bool(teacher_val[episode])))
            offset+=len(target)
            print(json.dumps(dict(split=split,done=offset,total=len(ids))),flush=True)
        for v in arrays.values():v.flush()
        write_json(dest/'samples.json',records)
        manifest[split]=dict(samples=len(ids),phase_counts=np.bincount([r['phase'] for r in records],minlength=3).tolist(),
                             episodes=sorted(set(eps[split].tolist())))
    write_json(a.study/'manifest.json',manifest)


if __name__=='__main__':main()
