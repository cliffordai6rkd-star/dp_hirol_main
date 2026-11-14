"""Condition-matched cross-episode teacher-response diagnostics, no model training."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import torch
from torch.nn import functional as F

from student_dual_path_common import write_json


def summarize_pairs(rows):
    if not rows:return dict(pairs=0)
    keys=['full_context_response_cosine','response_cosine','centered_response_cosine','patch_response_cosine','visual_cosine','torque_cosine','posture_rms','log_magnitude_gap','alpha_gap']
    return dict(pairs=len(rows),strict_matched=sum(r['strict_matched'] for r in rows),
        **{k:float(np.nanmean([r[k] for r in rows])) for k in keys},
        episodes={str(e):dict(pairs=sum(r['episode_i']==e for r in rows),
            response_cosine=float(np.mean([r['response_cosine'] for r in rows if r['episode_i']==e])))
            for e in sorted({r['episode_i'] for r in rows})})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);a=p.parse_args()
    torch.set_num_threads(8);rng=np.random.default_rng(42)
    output=a.study/'stability';output.mkdir(exist_ok=True)
    arrays={s:{k:np.load(a.study/s/(k+'.npy'),mmap_mode='r') for k in ['target','zero','visual','raw_wrench','alpha']}
            for s in ['train','validation']}
    samples={s:json.loads((a.study/s/'samples.json').read_text()) for s in arrays}
    data={}
    for s,x in arrays.items():
        ids=np.flatnonzero(np.asarray([r['phase'] for r in samples[s]])==2)
        target=np.asarray(x['target'][ids],dtype=np.float32);zero=np.asarray(x['zero'][ids],dtype=np.float32)
        visual=np.asarray(x['visual'][ids],dtype=np.float32)
        n,tokens,dim=target.shape;delta=(target-zero).reshape(n,2,2,tokens//4,dim)
        # Last observation, matched camera. Pooled descriptors retain two distinct cameras.
        v=visual.reshape(n,2,2,tokens//4,dim)[:,-1]
        torque=np.asarray(x['raw_wrench'][ids],dtype=np.float32).reshape(n,-1)
        rows=[samples[s][int(i)] for i in ids]
        data[s]=dict(delta=delta[:,-1],delta_all=delta,visual=v,torque=torque,
            posture=np.asarray([r['posture'] for r in rows]),episodes=np.asarray([r['episode'] for r in rows]),
            anchors=np.asarray([r['anchor'] for r in rows]),alpha=np.asarray(x['alpha'][ids]).mean(1),rows=rows,
            pooled_delta=delta[:,-1].mean(2).reshape(n,-1),pooled_visual=v.mean(2).reshape(n,-1))
    train=data['train'];val=data['validation'];n=len(val['rows'])
    free=np.flatnonzero(np.asarray([r['phase'] for r in samples['train']])==0)
    quiet=np.asarray(arrays['train']['target'][free],dtype=np.float32)-np.asarray(arrays['train']['zero'][free],dtype=np.float32)
    threshold=max(.001,float(np.quantile(np.sqrt(np.mean(quiet**2,-1)),.99)))
    dt=torch.from_numpy(val['delta']);vis=torch.from_numpy(val['visual'])
    flat=F.normalize(dt.flatten(1),dim=1);response_cos=(flat@flat.T).numpy()
    full_flat=F.normalize(torch.from_numpy(val['delta_all']).flatten(1),dim=1)
    full_cos=(full_flat@full_flat.T).numpy()
    centered=F.normalize((dt-torch.from_numpy(train['delta'].mean(0))).flatten(1),dim=1)
    centered_cos=(centered@centered.T).numpy()
    visual=F.normalize(torch.from_numpy(val['pooled_visual']),dim=1);vc=(visual@visual.T).numpy()
    tau=F.normalize(torch.from_numpy(val['torque']),dim=1);tc=(tau@tau.T).numpy()
    magnitude=np.linalg.norm(val['torque'],axis=1).clip(1e-8)
    mg=np.abs(np.log(magnitude[:,None]/magnitude[None,:]))
    pose=np.sqrt(np.mean((val['posture'][:,None]-val['posture'][None,:])**2,-1))
    ag=np.abs(val['alpha'][:,None]-val['alpha'][None,:])
    same=val['episodes'][:,None]==val['episodes'][None,:]
    visual_pose=(1-vc)+pose+ag
    all_score=visual_pose+(1-tc)+.25*mg

    def pair(i,j):
        cosine=[]
        for camera in range(2):
            source=vis[i,camera];dest=vis[j,camera]
            semantic=F.normalize(source,dim=-1)@F.normalize(dest,dim=-1).T
            # Restrict the match to a coarse 2x2 visual region; never use response direction to select it.
            side=int(len(source)**.5);coords=torch.arange(len(source));region=(coords//side//7)*2+(coords%side//7)
            semantic[region[:,None]!=region[None,:]]=-2
            matched=semantic.argmax(-1)
            amp_i=dt[i,camera].square().mean(-1).sqrt();amp_j=dt[j,camera,matched].square().mean(-1).sqrt()
            valid=(amp_i>threshold)&(amp_j>threshold)&(semantic.max(-1).values>.8)
            if valid.any():cosine.extend(F.cosine_similarity(dt[i,camera][valid],dt[j,camera,matched][valid],dim=-1).tolist())
        return dict(i=i,j=j,episode_i=int(val['episodes'][i]),episode_j=int(val['episodes'][j]),
            response_cosine=float(response_cos[i,j]),patch_response_cosine=float(np.mean(cosine)) if cosine else float('nan'),
            full_context_response_cosine=float(full_cos[i,j]),
            centered_response_cosine=float(centered_cos[i,j]),
            visual_cosine=float(vc[i,j]),torque_cosine=float(tc[i,j]),posture_rms=float(pose[i,j]),
            log_magnitude_gap=float(mg[i,j]),alpha_gap=float(ag[i,j]),
            strict_matched=bool(vc[i,j]>.8 and tc[i,j]>.8 and pose[i,j]<.35 and mg[i,j]<.7 and ag[i,j]<.2))
    groups={k:[] for k in ['cross_episode_condition_matched','cross_episode_visual_pose_matched_torque_mismatch','within_episode_separated','cross_episode_uncontrolled']}
    for i in range(n):
        cross=np.flatnonzero(~same[i])
        j=int(cross[np.argmin(all_score[i,cross])]);groups['cross_episode_condition_matched'].append(pair(i,j))
        # Same visual/pose control, deliberately different torque; fixed quantile within candidate set.
        candidates=cross[tc[i,cross]<=np.quantile(tc[i,cross],.25)]
        j=int(candidates[np.argmin(visual_pose[i,candidates])]);groups['cross_episode_visual_pose_matched_torque_mismatch'].append(pair(i,j))
        within=np.flatnonzero(same[i]&(np.abs(val['anchors']-val['anchors'][i])>=25))
        if len(within):groups['within_episode_separated'].append(pair(i,int(within[np.argmin(all_score[i,within])])))
        groups['cross_episode_uncontrolled'].append(pair(i,int(rng.choice(cross))))
    write_json(output/'matched_pairs.json',groups)
    result=dict(teacher_manifest=json.loads((a.study/'manifest.json').read_text()),samples=n,task='cut_cucumber',
        response_rms_threshold=threshold,comparisons={k:summarize_pairs(v) for k,v in groups.items()},
        strict_only=summarize_pairs([r for r in groups['cross_episode_condition_matched'] if r['strict_matched']]))
    # Conditional predictability: TRAIN-only PCA/scalers/ridge, held-out episode evaluation.
    vscale=StandardScaler().fit(train['pooled_visual']);vp=PCA(n_components=16,random_state=42).fit(vscale.transform(train['pooled_visual']))
    tscale=StandardScaler().fit(train['torque']);tp=PCA(n_components=8,random_state=42).fit(tscale.transform(train['torque']))
    pscale=StandardScaler().fit(train['posture'])
    def features(x):
        v=vp.transform(vscale.transform(x['pooled_visual']));t=tp.transform(tscale.transform(x['torque']))
        posture=pscale.transform(x['posture']);base=np.concatenate([v,posture,x['alpha'][:,None]],1)
        return dict(visual=base,torque=t,joint=np.concatenate([base,t,(v[:,:,None]*t[:,None,:]).reshape(len(v),-1)],1))
    fx,fy=features(train),features(val);regression={}
    mean_prediction=np.broadcast_to(train['pooled_delta'].mean(0),val['pooled_delta'].shape)
    mean_mse=float(np.mean((mean_prediction-val['pooled_delta'])**2))
    regression['train_mean']=dict(validation_pooled_response_mse=mean_mse,
        mean_cosine=float(F.cosine_similarity(torch.from_numpy(mean_prediction.copy()),torch.from_numpy(val['pooled_delta']),dim=-1).mean()))
    for name in ['visual','torque','joint','joint_shuffled_train_torque']:
        x=fx['joint' if name.startswith('joint') else name]
        if name.endswith('torque') and name.startswith('joint_shuffled'):
            shuffled={**train,'torque':train['torque'][rng.permutation(len(train['torque']))]};x=features(shuffled)['joint']
        scale=StandardScaler().fit(x);reg=Ridge(alpha=100.).fit(scale.transform(x),train['pooled_delta'])
        prediction=reg.predict(scale.transform(fy['joint' if name.startswith('joint') else name]))
        mse=np.mean((prediction-val['pooled_delta'])**2);reference=np.mean(val['pooled_delta']**2)
        regression[name]=dict(validation_pooled_response_mse=float(mse),zero_predictor_mse=float(reference),
            explained_energy=1-float(mse/reference),mean_cosine=float(F.cosine_similarity(torch.from_numpy(prediction),torch.from_numpy(val['pooled_delta']),dim=-1).mean()),
            improvement_vs_train_mean=1-float(mse/mean_mse),
            per_episode={str(e):float(np.mean((prediction[val['episodes']==e]-val['pooled_delta'][val['episodes']==e])**2)) for e in np.unique(val['episodes'])})
    result['conditional_ridge']=regression
    matched=groups['cross_episode_condition_matched'];mismatch=groups['cross_episode_visual_pose_matched_torque_mismatch']
    macro_differences=[]
    for ep in np.unique(val['episodes']):
        macro_differences.append(float(np.mean([r['response_cosine'] for r in matched if r['episode_i']==ep]))-
                                 float(np.mean([r['response_cosine'] for r in mismatch if r['episode_i']==ep])))
    bootstrap=np.asarray([np.mean(rng.choice(macro_differences,size=len(macro_differences),replace=True)) for _ in range(2000)])
    result['episode_bootstrap_matched_minus_mismatch']=dict(episode_macro_mean=float(np.mean(macro_differences)),
        interval_95=np.quantile(bootstrap,[.025,.975]).tolist(),unit='query episode; only six episodes, not 468 independent samples')
    kmeans=KMeans(n_clusters=4,random_state=42,n_init=10).fit(tscale.transform(train['torque']))
    labels=kmeans.predict(tscale.transform(val['torque']))
    result['torque_modes']={str(k):dict(samples=int((labels==k).sum()),episodes=sorted(set(val['episodes'][labels==k].tolist())),
        mean_response_rms=float(dt[labels==k].square().mean().sqrt()) if (labels==k).any() else None,
        mean_torque_per_joint=val['torque'][labels==k].reshape(-1,2,8,7).mean((0,1,2)).tolist() if (labels==k).any() else None,
        cross_episode_response_cosine=float(np.mean(response_cos[(labels[:,None]==k)&(labels[None,:]==k)&~same]))
            if np.any((labels[:,None]==k)&(labels[None,:]==k)&~same) else None) for k in range(4)}
    centroids=torch.stack([dt[labels==k].mean(0).flatten() for k in range(4)])
    result['torque_mode_centroid_cosines']=(F.normalize(centroids,dim=-1)@F.normalize(centroids,dim=-1).T).tolist()
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    order=np.argsort(val['episodes'],kind='stable');heat=axes[0].imshow(response_cos[order][:,order],vmin=-1,vmax=1,cmap='coolwarm');axes[0].set_title('Teacher response cosine; ordered by episode');fig.colorbar(heat,ax=axes[0])
    axes[1].boxplot([[r['response_cosine'] for r in groups[k]] for k in groups],labels=['condition','torque mismatch','within far','uncontrolled']);axes[1].tick_params(axis='x',rotation=25);axes[1].set_title('Query-balanced pair response directions')
    fig.tight_layout();fig.savefig(output/'response_consistency.png',dpi=160);plt.close(fig)
    eps=np.unique(val['episodes']);fig,axes=plt.subplots(len(eps),2,figsize=(8,3*len(eps)))
    for i,e in enumerate(eps):
        amplitude=dt[val['episodes']==e].square().mean(-1).sqrt().mean(0)
        for c in range(2):axes[i,c].imshow(amplitude[c].reshape(14,14),cmap='magma');axes[i,c].set_title(f'episode {e}, camera {c}, mean patch RMS')
    fig.tight_layout();fig.savefig(output/'episode_patch_response.png',dpi=150);plt.close(fig)
    result['limitations']='6 episodes, one task; nearest matches can have poor overlap: inspect strict-only coverage. Low raw cosine does not disprove a conditional shared mapping. Ridge predicts camera-pooled responses, not all patches. No validation-based tuning.'
    write_json(output/'summary.json',result);print(json.dumps(result,allow_nan=True),flush=True)


if __name__=='__main__':main()
