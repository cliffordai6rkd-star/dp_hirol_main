"""Finalize the shared 1k pilot using existing full-validation logs; no training."""
from pathlib import Path
import argparse,json,shutil
import dill,torch
from torch.utils.data import DataLoader,Subset
from omegaconf import OmegaConf
from student_dual_path_common import (VARIANTS,load_inputs,model_for,datasets_from_reference,to_device,write_json)
from report_student_dual_path import report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=root/'review_1000';step=1000
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    dev=torch.device('cuda:0');teacher,_,initial=load_inputs(root/'inputs',dev)
    _,val,_=datasets_from_reference(root/'inputs',root/'decoded_cuda')
    ref=json.loads((root/'experiments_cuda/a_baseline/per_sample_step0.json').read_text())
    ids=[r['index'] for phase in [0,1,2] for r in [x for x in ref if x['phase']==phase][:2]]
    raw=next(iter(DataLoader(Subset(val,ids),batch_size=len(ids))))
    obs=to_device(raw['obs'],dev);images,wrench,_,_=teacher.prepare_context_inputs_with_future_contact(obs)
    zero_wrench=teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
    for variant in VARIANTS:
        dest=out/variant;dest.mkdir(exist_ok=True)
        source={'a_baseline':out/'a_checkpoint_1000.ckpt','c_film_mlp':out/'c_mlp_checkpoint_1000.ckpt'}.get(
            variant,dest/'checkpoint_1000.ckpt')
        payload=torch.load(source,map_location='cpu',pickle_module=dill,weights_only=False)
        assert dill.loads(payload['pickles']['global_step'])==step
        model,cfg,_=model_for(initial,variant)
        model.load_state_dict(payload['state_dicts']['ema_model'],strict=True)
        model.to(dev).eval().requires_grad_(False)
        # Reuse all 1,893-window validation statistics already measured at 1k.
        history=[json.loads(s) for s in (root/'experiments_cuda'/variant/'log.jsonl').read_text().splitlines()]
        history=[r for r in history if r['step']<=step]
        metrics=next(r['metrics'] for r in history if r['kind']=='validation' and r['step']==step)
        manifest=json.loads((root/'experiments_cuda'/variant/'manifest.json').read_text())
        manifest['optimizer_steps']=step
        manifest['comparison_limit']='Uniform step 1000; A/C MLP later completed 5000, B/C Linear stopped near 1500'
        write_json(dest/'manifest.json',manifest);write_json(dest/'log.json',history)
        (dest/'log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        write_json(dest/'summary.json',dict(variant=variant,optimizer_steps=step,final=metrics,manifest=manifest,
            validation_reused_from_original_full_split=True,new_optimizer_updates=0))
        for key in ['preflight.json','per_sample_step0.json']:
            shutil.copy2(root/'experiments_cuda'/variant/key,dest/key)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            rendered,masks,condition=model(images,wrench,return_condition=True)
            zero_rendered=model(images,zero_wrench)
        probes=[dict(index=index,phase=int(val.phase_labels[index]),
            images={k:v[i].cpu() for k,v in images.items()},
            rendered={k:v[i].cpu() for k,v in rendered.items()},
            rendered_zero={k:v[i].cpu() for k,v in zero_rendered.items()},
            masks={k:v[i].cpu() for k,v in masks.items()},alpha=condition['contact_alpha'][i].cpu())
            for i,index in enumerate(ids)]
        torch.save(probes,dest/f'renders_step{step}.pt')
        print(variant,'step',step,'strict checkpoint reload passed; renders saved',flush=True)
        del model,payload
    report(out,root)

if __name__=='__main__':main()
