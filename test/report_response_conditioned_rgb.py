"""Matched Oracle/shared decoder tables, learning curves and response/RGB maps."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from student_dual_path_common import write_json
from paired_response_full_common import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    args=parser.parse_args()
    root=args.experiment
    oracle=json.loads((root/'oracle/summary.json').read_text())
    decoder=json.loads((root/'decoder/summary.json').read_text())
    for key in ['teacher_sha256','predictor_sha256','adapter_sha256','decoder_initial_sha256']:
        assert oracle['manifest'][key]==decoder['manifest'][key],key
    comparison=dict(oracle=oracle,decoder=decoder,
        matched=dict(oracle=oracle['final'],decoder=decoder['final']['oracle_matched_contact']),
        limits=['Oracle fits each of12 validation windows independently and is not a generalizing model.',
            'Predictor has residual error; successful realization does not eliminate predictor error.',
            'Fixed gate, alpha, pixel bound and one1000-step seed constrain this test; failure is not proof that RGB realization is impossible.',
            'Episodes5/26 are untouched by student/predictor fitting but were used in teacher checkpoint selection.'])
    write_json(root/'comparison.json',comparison)
    columns=['method','split','episode','phase','samples','predictor_mse','predictor_cosine','realization_mse',
        'realization_cosine','end_to_end_mse','end_to_end_cosine','response_rms_ratio_teacher',
        'response_rms_ratio_predictor','response_rms','rgb_rms','rgb_max']
    with (root/'per_episode_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for method,split,metrics in [('oracle','fixed_validation12',oracle['final']),
                ('decoder','matched_validation12',decoder['final']['oracle_matched_contact']),
                ('decoder','all_validation',decoder['final']['validation']),('decoder','all_train',decoder['final']['train'])]:
            for ep,phases in metrics['episodes'].items():
                for phase,row in phases.items():
                    writer.writerow({k:v for k,v in dict(method=method,split=split,episode=ep,phase=phase,**row).items() if k in columns})
    history=json.loads((root/'decoder/log.json').read_text())
    keys=['realization_mse','realization_cosine','end_to_end_mse','response_rms_ratio_predictor','response_rms','rgb_rms']
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,key in zip(axes.flat,keys):
        for group,label in [('selection_contact','Internal contact'),('external_contact','External5/26')]:
            ax.plot([r['step'] for r in history],[r['metrics']['validation'][group][key] for r in history],'-o',label=label)
        ax.set(title=key,xlabel='RGB optimizer steps');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(root/'decoder_learning_curves.png',dpi=160);plt.close(fig)
    log=json.loads((root/'oracle/log.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    for chunk in sorted({r['chunk'] for r in log}):
        rows=[r for r in log if r['chunk']==chunk]
        ax.plot([r['step'] for r in rows],[r['realization_loss'] for r in rows],label=f'Oracle pair{chunk//2+1}')
    ax.set(xlabel='Per-image optimizer steps',ylabel='Response realization MSE');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'oracle_learning_curves.png',dpi=160);plt.close(fig)
    oracle_visuals=torch.load(root/'oracle/visuals.pt',map_location='cpu',weights_only=False)
    shared=torch.load(root/'decoder/oracle_matched_visuals.pt',map_location='cpu',weights_only=False)
    for a in oracle_visuals:
        b=next(v for v in shared if v['index']==a['index'])
        torch.testing.assert_close(a['predicted'],b['predicted'],rtol=0,atol=0)
        fig,axes=plt.subplots(2,6,figsize=(17,6),constrained_layout=True)
        fig2,rgb_axes=plt.subplots(2,5,figsize=(15,6),constrained_layout=True)
        for camera,key in enumerate(a['original']):
            response={name:value.reshape(2,2,196,256)[-1,camera].float()
                for name,value in [('teacher',a['teacher']),('predictor',a['predicted']),('oracle',a['student']),('decoder',b['student'])]}
            vmax=max(float(v.square().mean(-1).sqrt().max()) for v in response.values())
            for col,name in enumerate(['teacher','predictor','oracle','decoder']):
                heat=axes[camera,col].imshow(response[name].square().mean(-1).sqrt().reshape(14,14),vmin=0,vmax=vmax,cmap='magma')
                axes[camera,col].set_title(key+' '+name+' RMS')
            for col,name in [(4,'oracle'),(5,'decoder')]:
                cos=F.cosine_similarity(response[name],response['predictor'],dim=-1)
                mask=response['predictor'].square().mean(-1).sqrt()<=.001
                axes[camera,col].imshow(np.ma.masked_where(mask.numpy(),cos.numpy()).reshape(14,14),vmin=-1,vmax=1,cmap='coolwarm')
                axes[camera,col].set_title(name+' vs predictor cosine')
            fig.colorbar(heat,ax=axes[camera,:4],shrink=.7,label='Absolute response RMS')
            image=a['original'][key][-1].float()
            rgb_axes[camera,0].imshow(image.permute(1,2,0));rgb_axes[camera,0].set_title(key+' original')
            for col,name,v in [(1,'Oracle',a),(3,'Shared',b)]:
                rgb=v['rendered'][key][-1].float()
                rgb_axes[camera,col].imshow(rgb.permute(1,2,0));rgb_axes[camera,col].set_title(name+' RGB')
                rgb_axes[camera,col+1].imshow((rgb-image).square().mean(0).sqrt(),vmin=0,vmax=.05,cmap='magma')
                rgb_axes[camera,col+1].set_title(name+' RGB residual RMS')
        for ax in [*axes.flat,*rgb_axes.flat]:ax.axis('off')
        suffix=f"episode_{a['episode']}_index_{a['index']}"
        fig.suptitle(suffix);fig.savefig(root/(suffix+'_response.png'),dpi=140);plt.close(fig)
        fig2.suptitle(suffix);fig2.savefig(root/(suffix+'_rgb.png'),dpi=140);plt.close(fig2)
    write_json(root/'code_provenance.json',{p.name:file_hash(p) for p in [Path(__file__),
        Path(__file__).with_name('run_response_conditioned_rgb.py'),Path(__file__).with_name('response_conditioned_rgb_common.py'),
        Path(__file__).parent/'config/response_conditioned_rgb.yaml']})
    print(json.dumps(dict(matched=comparison['matched'],validation=decoder['final']['validation']['phases']),indent=2))


if __name__=='__main__':main()
