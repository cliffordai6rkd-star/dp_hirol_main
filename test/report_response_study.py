"""Combine current-best teacher diagnostics, with fair matched-window oracle tables."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from paired_response_full_common import aggregate
from student_dual_path_common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);a=p.parse_args();root=a.study
    stability=json.loads((root/'stability/summary.json').read_text())
    names=['shared','rendernet','individual']
    summaries={k:json.loads((root/f'oracle_{k}_response_only/summary.json').read_text()) for k in names}
    assert len({s['manifest']['teacher_checkpoint_sha256'] for s in summaries.values()})==1
    assert len({s['manifest']['adapter_sha256'] for s in summaries.values()})==1
    chosen=json.loads((root/'oracle_individual_response_only/per_sample.json').read_text())
    comparison=dict(teacher=json.loads((root/'teacher_provenance.json').read_text()),stability=stability,
        full_contact={},matched_oracle={},episodes={},jacobian=None)
    for k in ['shared','rendernet']:
        s=summaries[k];comparison['full_contact'][k]={split:s['final'][split]['phases']['contact'] for split in ['train','validation']}
        comparison['full_contact'][k]['teacher_held_out_validation']=s['final']['validation']['teacher_held_out_contact']
        comparison['episodes'][k]=s['final']['validation']['episodes']
    for split in ['train','validation']:
        ids={r['index'] for r in chosen if r['split']==split}
        rows={'individual':[r for r in chosen if r['split']==split]}
        for k in ['shared','rendernet']:
            steps=summaries[k]['manifest']['steps']
            records=json.loads((root/f'oracle_{k}_response_only/{split}_samples_{steps}.json').read_text())
            rows[k]=[r for r in records if r['index'] in ids]
        comparison['matched_oracle'][split]={k:aggregate(r) for k,r in rows.items()}
        assert len({v['samples'] for v in comparison['matched_oracle'][split].values()})==1
        # Reference with strictly no torque response, using the same teacher targets.
        selected=np.asarray(sorted(ids));target=np.load(root/split/'target.npy',mmap_mode='r');zero=np.load(root/split/'zero.npy',mmap_mode='r')
        reference=float(np.mean((target[selected].astype(np.float32)-zero[selected].astype(np.float32))**2))
        comparison['matched_oracle'][split]['no_response_mse']=reference
    fixed=root/'oracle_individual_response_only_fixed_zero'
    if (fixed/'summary.json').exists():
        fixed_summary=json.loads((fixed/'summary.json').read_text())
        assert fixed_summary['manifest']['teacher_checkpoint_sha256']==summaries['individual']['manifest']['teacher_checkpoint_sha256']
        assert fixed_summary['manifest']['adapter_sha256']==summaries['individual']['manifest']['adapter_sha256']
        records=json.loads((fixed/'per_sample.json').read_text())
        for split in {r['split'] for r in records}:
            selected_rows=[r for r in records if r['split']==split]
            assert {r['index'] for r in selected_rows}=={r['index'] for r in chosen if r['split']==split}
            comparison['matched_oracle'][split]['individual_fixed_zero']=aggregate(selected_rows)
    for split in ['train','validation']:
        print(split,'matched windows',json.dumps(comparison['matched_oracle'][split]))
    jac=root/'local_jacobian_48/summary.json'
    if jac.exists():comparison['jacobian']=json.loads(jac.read_text())
    comparison['limitations']=['One current-best teacher at step70750; no final-teacher substitution.',
        'Old MLP/condition snapshot frozen. New teacher visual projection/context differs; zero-context mismatch is measured.',
        'Oracle sees targets on each evaluation image; shared networks never fit evaluation images.',
        'Shared CNN and D have different initialization and learning rates; negative CNN results are not a nonexistence proof.',
        '468 contact windows come from 6 student-held-out episodes, but only 96 windows in episodes5/26 are also teacher-held-out.',
        'CNN and D train300 steps on335 contacts/112 Free; individual fits150 steps on12 train and12 validation windows.',
        'Finite-iteration Jacobian ridge solutions are attainable directions, not exact upper bounds.']
    write_json(root/'comparison.json',comparison)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for k in ['shared','rendernet']:
        history=json.loads((root/f'oracle_{k}_response_only/log.json').read_text())
        for ax,key in zip(axes,['response_difference_mse','response_cosine','rgb_rms']):
            ax.plot([r['step'] for r in history],[r['metrics']['validation']['phases']['contact'][key] for r in history],'-o',label=k)
            ax.set(title='Validation contact: '+key,xlabel='Optimizer steps');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(root/'shared_response_curves.png',dpi=160);plt.close(fig)
    if comparison['jacobian'] is not None:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for row in comparison['jacobian']['samples']:
            scales=row['bounded_scale_sweep']
            axes[0].plot([v['scale'] for v in scales],[100*v['explained_energy'] for v in scales],'-o',label=str(row['episode']))
        axes[0].axhline(0,color='gray',linewidth=1);axes[0].set(xlabel='Gate-bounded perturbation scale',ylabel='Response MSE reduction (%)',title='Actual nonlinear RGB replay');axes[0].legend(title='Episode')
        modes=['linear','bounded_linear','bounded_nonlinear','bounded_best_scale']
        axes[1].bar(range(len(modes)),[100*comparison['jacobian']['mean'][k]['explained_energy'] for k in modes])
        axes[1].set_xticks(range(len(modes)),['Linear','Bounded linear','Actual bounded','Actual scaled'],rotation=15)
        axes[1].set(ylabel='Mean response MSE reduction (%)',title='Six individual local ridge solves')
        fig.tight_layout();fig.savefig(root/'jacobian_local_replay.png',dpi=160);plt.close(fig)
    if (fixed/'visuals.pt').exists():
        oracle_visuals=torch.load(fixed/'visuals.pt',map_location='cpu',weights_only=False)
        for ep in sorted({r['episode'] for r in oracle_visuals}):
            row=next(r for r in oracle_visuals if r['episode']==ep)
            fig,axes=plt.subplots(2,3,figsize=(10,7),constrained_layout=True)
            for i,key in enumerate(row['original']):
                image=row['original'][key][-1].float();rendered=row['rendered'][key][-1].float()
                axes[i,0].imshow(image.permute(1,2,0).clamp(0,1));axes[i,0].set_title(key+' original')
                axes[i,1].imshow(rendered.permute(1,2,0).clamp(0,1));axes[i,1].set_title('Individual Oracle')
                residual=(rendered-image).square().mean(0).sqrt()
                heat=axes[i,2].imshow(residual,cmap='magma',vmin=0,vmax=.05);axes[i,2].set_title('RGB residual RMS')
            for ax in axes.flat:ax.axis('off')
            fig.colorbar(heat,ax=axes[:,2],shrink=.7);fig.suptitle(f'Episode {ep}; zero-torque RGB fixed to original')
            fig.savefig(root/f'oracle_rgb_{ep}.png',dpi=150);plt.close(fig)
    snapshots={k:torch.load(root/f'oracle_{k}_response_only/visuals_300.pt',map_location='cpu',weights_only=False) for k in ['shared','rendernet']}
    threshold=summaries['shared']['manifest']['threshold']
    for first in snapshots['shared']:
        ep=first['episode'];samples={k:next(r for r in values if r['episode']==ep) for k,values in snapshots.items()}
        assert samples['shared']['index']==samples['rendernet']['index']
        cameras=list(first['original']);target=first['teacher_response'].reshape(2,2,196,256)[-1]
        fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
        for camera,key in enumerate(cameras):
            amplitude=target[camera].square().mean(-1).sqrt();axes[camera,0].imshow(amplitude.reshape(14,14),cmap='magma');axes[camera,0].set_title(key+' teacher RMS')
            for j,k in enumerate(['shared','rendernet'],1):
                response=samples[k]['student_response'].reshape(2,2,196,256)[-1,camera]
                cosine=F.cosine_similarity(response,target[camera],dim=-1)
                values=np.ma.masked_where((amplitude<=threshold).numpy(),cosine.numpy()).reshape(14,14)
                heat=axes[camera,j].imshow(values,cmap='coolwarm',vmin=-1,vmax=1);axes[camera,j].set_title(k+' response cosine')
        for ax in axes.flat:ax.axis('off')
        fig.colorbar(heat,ax=axes[:,1:],shrink=.65);fig.suptitle(f'Contact episode {ep}');fig.savefig(root/f'contact_direction_{ep}.png',dpi=150);plt.close(fig)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
