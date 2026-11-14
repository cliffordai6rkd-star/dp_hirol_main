"""Check historical pixels and paired teacher targets before running ablations."""
from pathlib import Path
import argparse,hashlib
import numpy as np
import torch
from student_dual_path_common import datasets_from_reference,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    train,val,_=datasets_from_reference(root/'inputs',root/'decoded_cuda')
    refs=torch.load(root/'inputs/reference_samples.pt',weights_only=False)
    arrays={s:{k:np.load(root/'prepared_cuda'/s/f'{k}.npy',mmap_mode='r') for k in ['target','zero']}
            for s in ['train','validation']}
    rows=[]
    for ref in refs:
        ds=train if ref['split']=='train' else val;index=ref['dataset_index']
        obs=ds[index]['obs']
        same=all(hashlib.sha256(obs[k].unsqueeze(0).numpy().tobytes()).hexdigest()==v
                 for k,v in ref['image_sha256'].items())
        errors={k:float(np.square(arrays[ref['split']][k][index].astype(np.float32)-ref[rk].numpy()[0]).mean())
                for k,rk in [('target','target'),('zero','zero_target')]}
        rows.append(dict(sample=ref['id'],pixels_identical=same,**errors))
    write_json(root/'reference_reproduction_cuda.json',rows)
    assert all(r['pixels_identical'] and r['target']<1e-5 and r['zero']<1e-5 for r in rows),rows
    print('All 12 historical samples reproduce pixels and paired teacher targets.',flush=True)

if __name__=='__main__':main()
