"""CIVI paired-response A/B/C pilot. All mutations stay in diagnostic outputs."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from student_dual_path_common import (load_inputs, datasets_from_reference, to_device,
    tensor_hash, write_json, PreparedDataset)
from diagnose_student_renderer import load_model, encode, event
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference
from diffusion_policy.model.diffusion.ema_model import EMAModel


@contextmanager
def zero_alpha(model):
    """Override the predicted condition in this diagnostic, without changing weights."""
    original = model.condition_network.forward
    def closed(*args, **kwargs):
        masks, condition = original(*args, **kwargs)
        condition = dict(condition)
        condition['contact_alpha'] = torch.zeros_like(condition['contact_alpha'])
        return masks, condition
    model.condition_network.forward = closed
    try: yield
    finally: model.condition_network.forward = original


def episode_for(dataset, index):
    anchor = dataset._anchor_index_from_sampler_row(dataset.indices[index])
    return int(dataset.episode_index[anchor])


def choose_windows(dataset):
    groups = {}
    for index in np.flatnonzero(dataset.phase_labels == 2):
        sequence = dataset._sample_indices_to_sequence(int(index))
        if dataset.window_sampling_strategy == 'timestamp':
            sequence = dataset._retime_sequence_indices(sequence)
        if np.all(dataset._frame_contact_phase[sequence[:dataset.n_obs_steps]] == 2):
            groups.setdefault(episode_for(dataset, int(index)), []).append(int(index))
    episodes = sorted(groups)
    episodes = [episodes[i] for i in np.linspace(0, len(episodes)-1, 6).round().astype(int)]
    contact = [groups[e][int((len(groups[e])-1)*fraction)] for e in episodes for fraction in (1/3, 2/3)]
    free = []
    for episode in episodes:
        ids = [int(i) for i in np.flatnonzero(dataset.phase_labels == 0) if episode_for(dataset, int(i)) == episode]
        free.append(ids[len(ids)//2])
    assert len(set(contact)) == 12 and len(set(free)) == 6
    return contact, free


def load_batch(dataset, indices, device):
    return to_device(next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices)))), device)


@torch.no_grad()
def visual_tokens(teacher, images):
    seen = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        teacher.obs_encoder.encode_image_context(images, context_adapter=lambda z:(seen.append(z.detach()), z)[1])
    assert len(seen) == 1
    return seen[0]


def zero_tokens(teacher, model, visual):
    # The enforced alpha=0 renderer is exactly the identity. Reusing frozen
    # DINO visual tokens preserves Adapter and final Context Norm gradients.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return teacher.obs_encoder.context_norm(model.context_adapter(visual)).float()


def train_mode(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
        if isinstance(module, torch.nn.MultiheadAttention): module.dropout = 0.
        if isinstance(module, torch.nn.LSTM): module.dropout = 0.


def masked_response_weights(teacher_delta, threshold):
    amplitude = teacher_delta.detach().float().square().mean(-1).sqrt()
    valid = amplitude > threshold
    count = valid.sum()
    # Mean weight on valid patches is one; normalize by effective patch count.
    mean = (amplitude*valid).sum()/count.clamp_min(1)
    weights = torch.where(valid, amplitude/mean.clamp_min(1e-12), torch.zeros_like(amplitude)).detach()
    return weights, count.detach()


def aggregate(rows):
    if not rows: return dict(samples=0)
    n = len(rows)
    ss, tt, st = [sum(r[k] for r in rows) for k in ['ss','tt','st']]
    return dict(samples=n, token_mse=float(np.mean([r['token_mse'] for r in rows])),
        zero_token_mse=float(np.mean([r['zero_token_mse'] for r in rows])),
        response_difference_mse=float(np.mean([r['response_difference_mse'] for r in rows])),
        valid_response_difference_mse=sum(r['valid_diff_sum'] for r in rows)/max(sum(r['valid_count'] for r in rows),1),
        response_cosine=sum(r['cosine_sum'] for r in rows)/max(sum(r['valid_count'] for r in rows),1),
        joint_response_cosine=st/max((ss*tt)**.5,1e-12), response_rms_ratio=(ss/max(tt,1e-12))**.5,
        no_student_response_mse=float(np.mean([r['teacher_response_mse'] for r in rows])),
        rgb_rms=float(np.mean([r['rgb_mse'] for r in rows])**.5), rgb_max=max(r['rgb_max'] for r in rows),
        zero_rgb_max=max(r['zero_rgb_max'] for r in rows), valid_patches=sum(r['valid_count'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'config/civi_paired_full_response.yaml')
    p.add_argument('--group', choices=['a','b','c'], required=True)
    p.add_argument('--steps', type=int); p.add_argument('--check-only', action='store_true')
    args = p.parse_args(); cfg = OmegaConf.load(args.config)
    if args.steps is not None: cfg.steps = args.steps
    args.root = args.root.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists(): raise RuntimeError('Use a fresh output directory; existing results are preserved')
    torch.set_num_threads(4); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0'); teacher, _, initial = load_inputs(args.root/'inputs', device)
    model = load_model(args.root, initial, 'c_film_mlp', device)
    before = tensor_hash(model.state_dict())
    initial_adapter_hash = tensor_hash(model.context_adapter.state_dict())
    teacher_hash = tensor_hash(teacher.state_dict())
    model.set_training_stage('rgb'); model.context_adapter.requires_grad_(bool(cfg.train_adapter)); train_mode(model)
    assert cfg.train_adapter, 'User authorized training the same Adapter in all three groups'
    parameters = [p for p in model.parameters() if p.requires_grad]
    names = [name for name,p in model.named_parameters() if p.requires_grad]
    frozen = {k:v.detach().clone() for k,v in model.condition_network.state_dict().items()}
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMAModel(ema_model, **dict(initial['cfg'].ema))
    train, val, splits = datasets_from_reference(args.root/'inputs', args.root/'decoded_cuda')
    contact_ids, free_ids = choose_windows(train)
    train_episodes = sorted({episode_for(train,i) for i in contact_ids})
    val_episodes = sorted({episode_for(val,i) for i in range(len(val))})
    assert not set(train_episodes) & set(val_episodes)
    train_cache = PreparedDataset(train, args.root/'prepared_cuda/train')
    val_cache = PreparedDataset(val, args.root/'prepared_cuda/validation')
    fit, free = load_batch(train_cache,contact_ids,device), load_batch(train_cache,free_ids,device)
    images,wrench,_,_ = teacher.prepare_context_inputs_with_future_contact(fit['obs'])
    free_images,free_wrench,_,_ = teacher.prepare_context_inputs_with_future_contact(free['obs'])
    wr0 = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
    visual, free_visual = visual_tokens(teacher,images), visual_tokens(teacher,free_images)
    # Recompute BOTH teacher branches by the same function/precision and audit cache semantics.
    with torch.no_grad(), torch.autocast('cuda',enabled=False):
        target = teacher._encode_context(fit['obs'],apply_curriculum_mask=False).float()
        obs0 = zero_wrench_reference(teacher,fit['obs'])
        obs0['contact_alpha'] = torch.zeros_like(fit['obs']['contact_alpha'])
        zero = teacher._encode_context(obs0,apply_curriculum_mask=False).float()
    assert (target-fit['target'].float()).square().mean() < 1e-5
    assert (zero-fit['zero'].float()).square().mean() < 1e-5
    fit['target'],fit['zero'] = target,zero
    dt = target-zero
    rw = (dt.detach().norm(dim=-1)-1e-6).clamp_min(0)
    dw, valid_count = masked_response_weights(dt,cfg.response_rms_threshold)
    assert valid_count > 0
    response_den = rw.sum().clamp_min(1e-12)
    slices = [slice(i,min(i+cfg.microbatch,len(contact_ids))) for i in range(0,len(contact_ids),cfg.microbatch)]
    free_slices = [slice(i,min(i+cfg.microbatch,len(free_ids))) for i in range(0,len(free_ids),cfg.microbatch)]
    # Exercise the real renderer with explicit zero-alpha override before using its exact identity shortcut.
    with torch.no_grad(), zero_alpha(model), torch.autocast('cuda',dtype=torch.bfloat16):
        actual_zero,_,condition = model(images,wr0,return_condition=True)
    assert torch.count_nonzero(condition['contact_alpha']) == 0
    assert all(torch.equal(actual_zero[k],images[k]) for k in images)
    with torch.no_grad():
        reference_zero = encode(teacher,model,actual_zero)
        shortcut_zero = zero_tokens(teacher,model,visual)
    assert torch.allclose(reference_zero,shortcut_zero,atol=1e-6,rtol=1e-6)

    def terms(sl):
        im = {k:v[sl] for k,v in images.items()}
        with torch.backends.cudnn.flags(enabled=False), torch.autocast('cuda',dtype=torch.bfloat16):
            rendered = model(im,wrench[sl])
        z = encode(teacher,model,rendered,grad=True)
        z0 = zero_tokens(teacher,model,visual[sl])
        error = (z-target[sl]).square().mean(-1)
        difference = ((z-z0)-dt[sl]).square().mean(-1)
        return dict(global_mse=error.sum()/target.shape[0]/target.shape[1],
            response=(error*rw[sl]).sum()/response_den,
            delta=(difference*dw[sl]).sum()/valid_count,
            zero=(z0-zero[sl]).square().sum()/zero.numel())

    def free_term(sl):
        z0 = zero_tokens(teacher,model,free_visual[sl])
        return (z0-free['zero'][sl].float()).square().sum()/free['zero'].numel()

    def gradient_diagnostics():
        vectors = {k:[torch.zeros_like(p,dtype=torch.float32) for p in parameters]
                   for k in ['global_mse','response','delta','zero','free']}
        for sl in slices:
            losses = terms(sl)
            for key,value in losses.items():
                gradients = torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                for dest,g in zip(vectors[key],gradients):
                    if g is not None: dest.add_(g.detach().float())
            del losses,gradients
        for sl in free_slices:
            gradients = torch.autograd.grad(free_term(sl),parameters,allow_unused=True)
            for dest,g in zip(vectors['free'],gradients):
                if g is not None: dest.add_(g.detach().float())
        base = [g+cfg.lambda_response*r for g,r in zip(vectors['global_mse'],vectors['response'])]
        vectors['base'] = base
        result = {}
        for key,gs in vectors.items():
            result[key] = {label:float(sum((g.square().sum() for name,g in zip(names,gs)
                if (name.startswith('context_adapter.') if label=='adapter' else not name.startswith('context_adapter.')
                    if label=='renderer' else True)),torch.zeros((),device=device)).sqrt())
                for label in ['all','adapter','renderer']}
        return result

    gradient0 = gradient_diagnostics()
    assert gradient0['zero']['adapter'] > 0 and gradient0['zero']['renderer'] == 0
    assert gradient0['delta']['renderer'] > 0 and gradient0['free']['adapter'] > 0
    norm = gradient0['base']['all']
    coefficients = dict(global_mse=1.,response=float(cfg.lambda_response),
        delta=float(cfg.delta_gradient_fraction)*norm/max(gradient0['delta']['all'],1e-12),
        zero=float(cfg.zero_gradient_fraction)*norm/max(gradient0['zero']['all'],1e-12),
        free=float(cfg.free_gradient_fraction)*norm/max(gradient0['free']['all'],1e-12))
    for key,value in cfg.calibrated_coefficients.items():
        assert np.isclose(coefficients[key],value,rtol=1e-3), 'Initial-gradient calibration must reproduce'
        coefficients[key]=float(value)
    group = cfg.groups[args.group]
    if not group.paired_response: coefficients['delta'] = 0.
    if not group.contact_zero_protection: coefficients['zero'] = 0.
    manifest = dict(group=args.group,config=OmegaConf.to_container(cfg,resolve=True),
        initial_sha256=before,teacher_state_sha256=teacher_hash,seed=int(cfg.seed),steps=int(cfg.steps),
        fit_contact_indices=contact_ids,free_indices=free_ids,fit_episodes=train_episodes,validation_episodes=val_episodes,
        splits=splits,coefficients=coefficients,initial_gradients=gradient0,
        adapter_trainable=True,condition_frozen=True,zero_alpha_explicit=True,zero_rgb_exact_identity=True,
        zero_path_cache_equivalent=True,teacher_paired_same_readout_verified=True,
        delta_weights='teacher RMS / mean valid teacher RMS, stop gradient; denominator = number of valid patches',
        zero_and_free_weights='uniform all patches, including patches with zero response, to protect the visual baseline',
        baseline_response_weights='unchanged raw teacher L2 response, normalized by weight sum',
        dropout_disabled=True,starting_checkpoint=str(args.root/cfg.starting_checkpoint))
    write_json(args.output/'manifest.json',manifest); OmegaConf.save(cfg,args.output/'config.yaml')
    event('preflight',group=args.group,coefficients=coefficients,initial_gradients=gradient0,
          train_episodes=train_episodes,validation_episodes=val_episodes)
    if args.check_only:
        write_json(args.output/'preflight.json',manifest); return
    optimizer = torch.optim.AdamW(parameters,**OmegaConf.to_container(cfg.optimizer,resolve=True))
    started=time.monotonic(); history=[]; gradient_rows=[dict(step=0,gradients=gradient0)]
    write_json(args.output/'gradients.json',gradient_rows)

    @torch.no_grad()
    def evaluate(step):
        records=[]; visuals=[]; ema_model.eval()
        for split,dataset,ids in [('train_contact',train_cache,contact_ids),('train_free',train_cache,free_ids),
                                  ('validation',val_cache,list(range(len(val))))]:
            for raw in DataLoader(Subset(dataset,ids),batch_size=cfg.validation_batch,num_workers=4,pin_memory=True):
                batch=to_device(raw,device); im,wr,_,_=teacher.prepare_context_inputs_with_future_contact(batch['obs'])
                with torch.autocast('cuda',dtype=torch.bfloat16): rendered=ema_model(im,wr)
                z=encode(teacher,ema_model,rendered)
                z0=zero_tokens(teacher,ema_model,visual_tokens(teacher,im))
                t=batch['target'].float()-batch['zero'].float(); s=z-z0
                amp=t.square().mean(-1).sqrt(); valid=amp>cfg.response_rms_threshold
                cosine=F.cosine_similarity(s,t,dim=-1,eps=1e-6)
                pixels=torch.cat([(rendered[k].float()-im[k].float()).flatten(1) for k in im],dim=1)
                for i in range(len(z)):
                    index=int(batch['index'][i]); base_dataset=train if split.startswith('train') else val
                    phase=int(batch['phase'][i]); sv,tv=s[i][valid[i]],t[i][valid[i]]
                    row=dict(split=split,index=index,episode=episode_for(base_dataset,index),phase=phase,
                        token_mse=float((z[i]-batch['target'][i].float()).square().mean()),
                        zero_token_mse=float((z0[i]-batch['zero'][i].float()).square().mean()),
                        response_difference_mse=float((s[i]-t[i]).square().mean()),
                        valid_diff_sum=float((s[i]-t[i]).square().mean(-1)[valid[i]].sum()),valid_count=int(valid[i].sum()),
                        teacher_response_mse=float(t[i].square().mean()),cosine_sum=float(cosine[i][valid[i]].sum()),
                        ss=float(sv.double().square().sum()),tt=float(tv.double().square().sum()),st=float((sv.double()*tv.double()).sum()),
                        rgb_mse=float(pixels[i].square().mean()),rgb_max=float(pixels[i].abs().max()),zero_rgb_max=0.)
                    records.append(row)
                    if step==cfg.steps and split=='validation' and phase==2 and row['episode'] not in [x['episode'] for x in visuals]:
                        visuals.append(dict(episode=row['episode'],index=index,
                            original={k:v[i].cpu() for k,v in im.items()},rendered={k:v[i].cpu() for k,v in rendered.items()},
                            teacher_response=t[i].cpu(),student_response=s[i].cpu()))
        scopes=dict(train_contact=[r for r in records if r['split']=='train_contact'],
            train_free=[r for r in records if r['split']=='train_free'],
            validation_contact=[r for r in records if r['split']=='validation' and r['phase']==2],
            validation_free=[r for r in records if r['split']=='validation' and r['phase']==0],
            validation_all=[r for r in records if r['split']=='validation'])
        metrics={k:aggregate(rows) for k,rows in scopes.items()}
        per_episode={str(ep):aggregate([r for r in scopes['validation_contact'] if r['episode']==ep]) for ep in val_episodes}
        row=dict(kind='validation',step=step,metrics=metrics,contact_episodes=per_episode,elapsed_seconds=time.monotonic()-started)
        history.append(row);write_json(args.output/'log.json',history);write_json(args.output/f'per_sample_{step}.json',records)
        if visuals:torch.save(visuals,args.output/'visuals.pt')
        event('validation',group=args.group,step=step,train_contact=metrics['train_contact'],validation_contact=metrics['validation_contact'],
              free_zero_mse=metrics['validation_free']['zero_token_mse'])
        return metrics

    initial_metrics=evaluate(0)
    torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    for step in range(1,int(cfg.steps)+1):
        train_mode(model);optimizer.zero_grad(set_to_none=True)
        values={k:0. for k in coefficients}
        for sl in slices:
            losses=terms(sl);loss=sum(coefficients[k]*v for k,v in losses.items())
            assert torch.isfinite(loss);loss.backward()
            for k,v in losses.items():values[k]+=float(v.detach())
            del losses,loss
        for sl in free_slices:
            loss=free_term(sl);(coefficients['free']*loss).backward();values['free']+=float(loss.detach())
        grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg.max_gradient_norm))
        optimizer.step();ema.step(model)
        if step==1 or step%cfg.log_every==0:
            row=dict(kind='train',step=step,losses=values,total=sum(coefficients[k]*v for k,v in values.items()),
                gradient_before_clip=grad_norm,elapsed_seconds=time.monotonic()-started)
            history.append(row);write_json(args.output/'log.json',history);event('train',group=args.group,**row)
        if step%cfg.gradient_every==0:
            gradient_rows.append(dict(step=step,gradients=gradient_diagnostics()));write_json(args.output/'gradients.json',gradient_rows)
        if step%cfg.validation_every==0 or step==cfg.steps:final=evaluate(step)
    assert all(torch.equal(v,model.condition_network.state_dict()[k]) for k,v in frozen.items())
    assert teacher_hash==tensor_hash(teacher.state_dict()) and all(p.grad is None for p in teacher.parameters())
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},ema_model={k:v.detach().cpu() for k,v in ema_model.state_dict().items()},
        manifest=manifest),args.output/'checkpoint.pt')
    write_json(args.output/'summary.json',dict(manifest=manifest,initial=initial_metrics,final=final,
        teacher_and_condition_unchanged=True,adapter_changed=tensor_hash(model.context_adapter.state_dict())!=initial_adapter_hash))
    event('complete',group=args.group,steps=cfg.steps)


if __name__=='__main__':main()
