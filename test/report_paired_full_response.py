"""Tables, episode summaries, curves, RGB previews and response directions."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    groups=['a','b','c'];results={g:json.loads((root/g/'summary.json').read_text()) for g in groups}
    manifests={g:r['manifest'] for g,r in results.items()}
    ref=manifests['a']
    for g,m in manifests.items():
        for key in ['initial_sha256','teacher_state_sha256','seed','steps','fit_contact_indices','free_indices',
                    'fit_episodes','validation_episodes','config']:
            assert m[key]==ref[key],(g,key)
        assert m['zero_rgb_exact_identity'] and m['adapter_trainable']
        assert results[g]['teacher_and_condition_unchanged'] and results[g]['adapter_changed']
        assert not set(m['fit_episodes'])&set(m['validation_episodes'])
    summary=dict(steps=ref['steps'],fit_episodes=ref['fit_episodes'],validation_episodes=ref['validation_episodes'],
        initialization_verified=True,all_groups_adapter_trainable=True,
        coefficients={g:m['coefficients'] for g,m in manifests.items()},comparisons={})
    keys=['token_mse','zero_token_mse','response_difference_mse','valid_response_difference_mse',
          'response_cosine','response_rms_ratio','rgb_rms','rgb_max']
    for scope in ['train_contact','validation_contact','validation_free']:
        before=results['a']['initial'][scope]
        for g in groups:
            for k,v in before.items():assert np.isclose(v,results[g]['initial'][scope][k],rtol=1e-5,atol=1e-8),(scope,g,k)
        rows=dict(initial=before,**{g:results[g]['final'][scope] for g in groups})
        summary['comparisons'][scope]=rows
    histories={g:json.loads((root/g/'log.json').read_text()) for g in groups}
    summary['contact_episodes']={g:next(r['contact_episodes'] for r in reversed(histories[g]) if r['kind']=='validation') for g in groups}
    summary['episode_macro']={g:{k:float(np.mean([v[k] for v in eps.values()]))
                                for k in keys} for g,eps in summary['contact_episodes'].items()}
    summary['initial_gradients']=ref['initial_gradients']
    for left,right in [('a','b'),('b','c'),('a','c')]:
        x,y=[results[g]['final']['validation_contact'] for g in [left,right]]
        summary[f'{right}_vs_{left}']=dict(token_reduction_pct=100*(1-y['token_mse']/x['token_mse']),
            zero_reduction_pct=100*(1-y['zero_token_mse']/x['zero_token_mse']),
            response_reduction_pct=100*(1-y['response_difference_mse']/x['response_difference_mse']),
            cosine_change=y['response_cosine']-x['response_cosine'],
            rgb_rms_change_pct=100*(y['rgb_rms']/x['rgb_rms']-1))
    summary['limits']=('One seed, 12 repeated contact windows plus 6 Free anchors. Validation uses six disjoint episodes. '
        'All groups train the Adapter and share the Free zero-context anchor. '
        'Delta/cosine/RMS ratio use a fixed teacher response threshold; report unfiltered delta MSE separately. '
        'No closed-loop action evaluation is performed.')
    (root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    for scope in ['train_contact','validation_contact']:
        text='| Group | Token MSE | Zero token MSE | Delta MSE | Response cosine | RMS ratio | RGB RMS | RGB max |\n'
        text+='|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for name,row in summary['comparisons'][scope].items():
            selected=['token_mse','zero_token_mse','response_difference_mse','response_cosine','response_rms_ratio','rgb_rms','rgb_max']
            text+='| '+name+' | '+' | '.join(f'{row[k]:.6f}' for k in selected)+' |\n'
        (root/f'{scope}.md').write_text(text);print(scope+'\n'+text)
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for g in groups:
        rows=[r for r in histories[g] if r['kind']=='validation']
        for ax,key in zip(axes.flat,keys):
            ax.plot([r['step'] for r in rows],[r['metrics']['validation_contact'][key] for r in rows],'-o',label=g.upper())
            ax.set(title='Validation contact: '+key,xlabel='Additional updates');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(root/'validation_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for g in groups:
        rows=[r for r in histories[g] if r['kind']=='train']
        for ax,key in zip(axes.flat,['global_mse','response','delta','zero','free']):
            ax.plot([r['step'] for r in rows],[r['losses'][key] for r in rows],label=g.upper())
            ax.set(title='Training: '+key,xlabel='Additional updates');ax.grid(alpha=.2)
        rows=[r for r in histories[g] if r['kind']=='validation']
        axes[1,2].plot([r['step'] for r in rows],[r['metrics']['validation_free']['zero_token_mse'] for r in rows],'-o',label=g.upper())
    axes[1,2].set(title='Validation Free zero-token MSE',xlabel='Additional updates');axes[1,2].grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for g in groups:
        rows=json.loads((root/g/'gradients.json').read_text())
        for ax,branch in zip(axes,['all','renderer','adapter']):
            for key in ['base','delta','zero','free']:
                coefficient=1. if key=='base' else manifests[g]['coefficients'][key]
                ys=[coefficient*r['gradients'][key][branch] for r in rows]
                if max(ys)>0:ax.plot([r['step'] for r in rows],ys,label=g.upper()+' '+key)
            ax.set(title=branch+' weighted loss gradient L2',xlabel='Additional updates',yscale='log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(root/'gradient_norms.png',dpi=170);plt.close(fig)
    visuals={g:torch.load(root/g/'visuals.pt',map_location='cpu',weights_only=False) for g in groups}
    for episode in ref['validation_episodes']:
        samples={g:next(r for r in visuals[g] if r['episode']==episode) for g in groups}
        assert len({r['index'] for r in samples.values()})==1
        first=samples['a'];cameras=list(first['original']);steps=next(iter(first['original'].values())).shape[0]
        patches=first['teacher_response'].shape[0]//steps//len(cameras);side=int(patches**.5);assert side*side==patches
        fig,axes=plt.subplots(len(cameras),4,figsize=(12,3*len(cameras)),squeeze=False,constrained_layout=True)
        target=first['teacher_response'].reshape(steps,len(cameras),patches,-1)
        amplitude=target.square().mean(-1).sqrt();threshold=ref['config']['response_rms_threshold']
        for i,camera in enumerate(cameras):
            axes[i,0].imshow(amplitude[-1,i].reshape(side,side),cmap='magma');axes[i,0].set_title(camera+' teacher RMS')
            valid=amplitude[-1,i]>threshold
            for j,g in enumerate(groups,start=1):
                response=samples[g]['student_response'].reshape_as(target)
                cosine=F.cosine_similarity(response[-1,i],target[-1,i],dim=-1)
                values=np.ma.masked_where(~valid.numpy(),cosine.numpy()).reshape(side,side)
                cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('lightgray')
                heat=axes[i,j].imshow(values,cmap=cmap,vmin=-1,vmax=1);axes[i,j].set_title(g.upper()+' response cosine')
        for ax in axes.flat:ax.axis('off')
        fig.colorbar(heat,ax=axes[:,1:],shrink=.7);fig.suptitle(f'Validation episode {episode}; gray = weak teacher response')
        fig.savefig(root/f'response_direction_episode_{episode}.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(len(cameras)*2,4,figsize=(12,9),squeeze=False,constrained_layout=True)
        maximum=max(float((sample['rendered'][k][-1].float()-sample['original'][k][-1]).abs().mean(0).max())
                    for sample in samples.values() for k in cameras)
        for i,camera in enumerate(cameras):
            original=first['original'][camera][-1].float()
            for j,label in enumerate(['original',*groups]):
                image=original if j==0 else samples[label]['rendered'][camera][-1].float()
                axes[2*i,j].imshow(image.permute(1,2,0).clamp(0,1));axes[2*i,j].set_title(camera+' '+label.upper())
                heat=axes[2*i+1,j].imshow((image-original).abs().mean(0),cmap='magma',vmin=0,vmax=maximum)
        for ax in axes.flat:ax.axis('off')
        fig.colorbar(heat,ax=axes,shrink=.6);fig.suptitle(f'Validation episode {episode}: rendered RGB and absolute change')
        fig.savefig(root/f'rgb_episode_{episode}.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['comparisons','initial_gradients']},indent=2))


if __name__=='__main__':main()
