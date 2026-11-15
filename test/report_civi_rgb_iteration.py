"""Build the execution ledger and comparable plots from actual iteration outputs."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F


def oracle_visuals(root,output):
    """Same train images and gates, per-image Oracle vs frozen shared checkpoint."""
    directory=root/'final150k_train_oracle12'
    path=directory/'visuals.pt'
    if not path.exists():return
    samples=torch.load(path,map_location='cpu',weights_only=False)
    summary=json.loads((directory/'summary.json').read_text())
    threshold=summary['manifest']['threshold']
    dest=output/'oracle_train_only';dest.mkdir(exist_ok=True)
    for sample in samples[:3]:
        fig,axes=plt.subplots(2,7,figsize=(20,7))
        teacher=sample['teacher'].reshape(2,2,196,256)
        pred=sample['predicted'].reshape_as(teacher)
        student=sample['student'].reshape_as(teacher)
        shared=sample['shared_student'].reshape_as(teacher)
        for camera,key in enumerate(sample['original']):
            for col,(label,images) in enumerate([('Original',sample['original']),('Shared RGB',sample['shared_rendered']),('Oracle RGB',sample['rendered'])]):
                axes[camera,col].imshow(images[key][-1].float().permute(1,2,0));axes[camera,col].set_title(label,fontsize=10)
            vmax=float(pred[-1,camera].square().mean(-1).sqrt().max())
            for col,(label,value) in enumerate([('Predictor response',pred),('Shared response',shared),('Oracle response',student)],3):
                axes[camera,col].imshow(value[-1,camera].square().mean(-1).sqrt().reshape(14,14),vmin=0,vmax=vmax,cmap='magma')
                axes[camera,col].set_title(label+' RMS\nsame absolute scale',fontsize=9)
            mask=teacher[-1,camera].square().mean(-1).sqrt()>threshold
            cosine=F.cosine_similarity(student[-1,camera],pred[-1,camera],dim=-1)
            axes[camera,6].imshow(np.ma.masked_where(~mask.numpy(),cosine.numpy()).reshape(14,14),vmin=-1,vmax=1,cmap='coolwarm')
            axes[camera,6].set_title('Oracle vs Predictor cosine\nfixed teacher-active patches',fontsize=9)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f'TRAIN ONLY: episode{sample["episode"]}, window{sample["index"]}; Oracle has independent per-image pixels')
        fig.tight_layout();fig.savefig(dest/f'episode{sample["episode"]}_window{sample["index"]}.png',dpi=120);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--experiments',nargs='+',help='Report only a matched set; do not mix teacher targets')
    p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.root;output=a.output or root
    output.mkdir(parents=True,exist_ok=True)
    summaries={x.parent.name:d for x in sorted(root.glob('*/summary.json'))
        if x.parent.name!='smoke' and 'manifest' in (d:=json.loads(x.read_text())) and 'final' in d}
    if a.experiments:
        missing=set(a.experiments)-set(summaries)
        if missing:raise ValueError('Incomplete experiments: '+str(sorted(missing)))
        summaries={name:summaries[name] for name in a.experiments}
        hashes={s['manifest']['teacher_sha256'] for s in summaries.values()}
        if len(hashes)!=1:raise ValueError('A matched comparison must use one teacher checkpoint')
    fields=['experiment','teacher_sha256','seed','weights','split','episode','phase','samples','realization_mse','realization_cosine','end_to_end_mse',
        'end_to_end_cosine','predictor_mse','predictor_cosine','predictor_energy','teacher_energy','response_rms_ratio_predictor',
        'response_rms_ratio_teacher','absolute_token_mse','zero_token_mse','response_mse','rgb_rms','rgb_max','hard_clip_fraction','range_clip_fraction']
    with (output/'per_episode_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,s in summaries.items():
            seed=s['manifest'].get('iteration_config',{}).get('seed',42)
            weights=s['final'] if 'ema' in s['final'] else {'oracle':{'train':s['final']}}
            for weight,splits in weights.items():
                for split,m in splits.items():
                    for ep,phases in m['episodes'].items():
                        for phase,row in phases.items():
                            values=dict(experiment=name,teacher_sha256=s['manifest'].get('teacher_sha256'),seed=seed,
                                weights=weight,split=split,episode=ep,phase=phase,**row)
                            w.writerow({k:v for k,v in values.items() if k in fields})
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    keys=['realization_mse','realization_cosine','end_to_end_mse','response_rms_ratio_predictor','rgb_rms','response_mse']
    for name,s in summaries.items():
        path=root/name/'log.json'
        if not path.exists() or 'iteration_config' not in s['manifest']:continue
        rows=[x for x in json.loads(path.read_text()) if x['weights']=='ema']
        for ax,key in zip(axes.flat,keys):
            group='free' if key=='response_mse' else 'contact'
            for split,style in [('train','--'),('validation','-')]:
                def value(row):
                    metrics=row['metrics'][split]
                    return metrics['selection_macro_contact'][key] if split=='validation' and group=='contact' else metrics['phases'][group][key]
                ax.plot([x['step'] for x in rows],[value(x) for x in rows],style,marker='.',label=name+' '+split)
            ax.set(xlabel='Shared RGB optimizer updates',ylabel=key);ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for name in summaries:
        path=root/name/'training.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())
        for key in ['head','cnn_all','ca_attention','film','projection']:
            ax.plot([x['step'] for x in rows],[max(x['gradient_norms'].get(key,0),1e-12) for x in rows],label=name+' '+key)
    ax.set(xlabel='Shared RGB optimizer updates',ylabel='Gradient L2 norm',yscale='log');ax.grid(alpha=.2)
    ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(output/'gradient_norms.png',dpi=150);plt.close(fig)
    candidates=a.experiments or ['fit_baseline_s42_600','fit_semantic_patch_s42_600','full_continue_s42_600']
    artifacts={name:torch.load(root/name/'visuals_ema_endpoint.pt',map_location='cpu',weights_only=False)
        for name in candidates if (root/name/'visuals_ema_endpoint.pt').exists()}
    if artifacts:
        out=output/'visuals';out.mkdir(exist_ok=True)
        for ep in [5,26,38,44,53,60]:
            available={name:next((x for x in rows if x['episode']==ep),None) for name,rows in artifacts.items()}
            available={k:v for k,v in available.items() if v is not None}
            if not available:continue
            reference=next(iter(available.values()))
            assert len({x['index'] for x in available.values()})==1
            fig,axes=plt.subplots(2,1+2*len(available),figsize=(4+6*len(available),6),squeeze=False)
            fig2,maps=plt.subplots(2,2+2*len(available),figsize=(6+5*len(available),6),squeeze=False)
            teacher=reference['teacher'].reshape(2,2,196,256);pred=reference['predicted'].reshape_as(teacher)
            threshold=summaries[next(iter(available))]['manifest']['threshold']
            for camera,key in enumerate(reference['original']):
                original=reference['original'][key][-1].float()
                axes[camera,0].imshow(original.permute(1,2,0));axes[camera,0].set_title(key+' original')
                vmax=float(torch.maximum(teacher[-1,camera].square().mean(-1).sqrt(),pred[-1,camera].square().mean(-1).sqrt()).max())
                maps[camera,0].imshow(teacher[-1,camera].square().mean(-1).sqrt().reshape(14,14),vmin=0,vmax=vmax,cmap='magma');maps[camera,0].set_title('Teacher RMS')
                maps[camera,1].imshow(pred[-1,camera].square().mean(-1).sqrt().reshape(14,14),vmin=0,vmax=vmax,cmap='magma');maps[camera,1].set_title('Predictor RMS')
                maps[camera,0].set_title(f'Teacher RMS\nscale: 0 to {vmax:.3f}',fontsize=10)
                maps[camera,1].set_title('Predictor RMS\nsame scale',fontsize=10)
                mask=teacher[-1,camera].square().mean(-1).sqrt()>threshold
                for col,(name,item) in enumerate(available.items()):
                    label=name.removeprefix('final150k_').replace('teacher_response','Teacher loss').replace('baseline','Predictor loss').replace('_s','\nseed ').replace('_',' ')
                    rgb=item['rendered'][key][-1].float();delta=rgb-original
                    axes[camera,1+2*col].imshow(rgb.permute(1,2,0));axes[camera,1+2*col].set_title(label,fontsize=10)
                    axes[camera,2+2*col].imshow(delta.square().mean(0).sqrt(),vmin=0,vmax=.03,cmap='magma');axes[camera,2+2*col].set_title('RGB RMS per pixel')
                    student=item['student'].reshape_as(teacher)[-1,camera]
                    maps[camera,2+2*col].imshow(student.square().mean(-1).sqrt().reshape(14,14),vmin=0,vmax=vmax,cmap='magma');maps[camera,2+2*col].set_title(label+'\nresponse RMS',fontsize=9)
                    cosine=F.cosine_similarity(student,teacher[-1,camera],dim=-1)
                    maps[camera,3+2*col].imshow(np.ma.masked_where(~mask.numpy(),cosine.numpy()).reshape(14,14),vmin=-1,vmax=1,cmap='coolwarm');maps[camera,3+2*col].set_title('Cosine vs Teacher [-1, 1]\nfixed active mask',fontsize=9)
            for ax in [*axes.flat,*maps.flat]:ax.axis('off')
            for figure,suffix in [(fig,'rgb'),(fig2,'response')]:
                figure.suptitle(f'Episode{ep}; same window index{reference["index"]}; fixed absolute color scales')
                figure.tight_layout();figure.savefig(out/f'episode{ep}_{suffix}.png',dpi=120);plt.close(figure)
    comparison={}
    for name,s in summaries.items():
        if 'ema' not in s['final']:continue
        weights={}
        for label in ['online','ema']:
            train=s['final'][label]['train']['macro']['contact']
            validation=s['final'][label]['validation']['selection_macro_contact']
            weights[label]=dict(train_episode_probe=train,internal_validation_all_contact=validation,
                contact_mse_gap={key:validation[key]-train[key] for key in ['realization_mse','end_to_end_mse']},
                validation_phases=s['final'][label]['validation']['phases'])
        comparison[name]=dict(steps=s.get('steps'),teacher_sha256=s['manifest']['teacher_sha256'],
            seed=s['manifest']['iteration_config']['seed'],weights=weights,
            note='Train is an episode/phase midpoint probe for full-scope runs. Intermediate validation is a fixed internal panel; endpoint validation uses every validation window. Episodes5/26 are retained evaluation, not an untouched final test.')
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    oracle_visuals(root,output)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
