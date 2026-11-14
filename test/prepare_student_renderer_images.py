"""Preserve the historical student's CUDA decode preprocessing in an independent cache."""
from pathlib import Path
import argparse,json
import hydra,numpy as np
from omegaconf import OmegaConf
from student_dual_path_common import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.inputs/'reference_train_cache.json').read_text())
    cfg=OmegaConf.create(meta['identity']['dataset']);cfg.preload_images=False;cfg.video_decode_device='cuda:0'
    ds=hydra.utils.instantiate(cfg);ds._video_decoder=ds._make_video_decoder()
    assert ds._video_decoder is not None
    shapes={k:tuple(cfg.shape_meta.obs[k].shape) for k in ds.rgb_keys}
    arrays={k:np.lib.format.open_memmap(a.output_dir/f'{k}.npy',mode='w+',dtype=np.uint8,
                                     shape=(ds.dataset_length,*shape)) for k,shape in shapes.items()}
    for start in range(0,ds.dataset_length,64):
        stop=min(start+64,ds.dataset_length)
        images=ds._build_image_chunk(list(range(start,stop)),shapes)
        for k,v in images.items():arrays[k][start:stop]=np.rint(np.clip(v,0,1)*255).astype(np.uint8)
        if start%2048==0:print(dict(frames=stop,total=ds.dataset_length),flush=True)
    for v in arrays.values():v.flush()
    write_json(a.output_dir/'metadata.json',dict(dataset_length=ds.dataset_length,rgb_keys=list(ds.rgb_keys),
        dtype='uint8',extra=dict(image_feature_map=dict(ds.image_feature_map)),decoder='TorchCodec CUDA'))

if __name__=='__main__':main()
