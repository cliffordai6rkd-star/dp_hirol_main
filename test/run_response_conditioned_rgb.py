"""Pinned predictor -> RGB Oracle or shared D decoder. All formal models are read-only."""
from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import load_inputs, datasets_from_reference, model_for, to_device, tensor_hash, write_json
from paired_response_full_common import file_hash, deterministic_mode
from new_teacher_response_common import FullPairedDataset
from run_scene_torque_response_predictor import SceneTorquePredictor, extract_wrench, phase_plan
from response_conditioned_rgb_common import ResponseConditionedDecoder, metric_rows, grouped
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def prepare(base, predictor_root, output, config, device):
    """Serve the exact frozen predictor from a new, read-only response cache."""
    assert file_hash(predictor_root/'best.pt') == config.predictor_sha256
    assert json.loads((base/'paired_cache/manifest.json').read_text())['teacher_checkpoint_sha256'] == config.teacher_sha256
    state = torch.load(predictor_root/'best.pt', map_location='cpu', weights_only=False)['model']
    predictor = SceneTorquePredictor(state['visual_scale'], state['torque_scale']).to(device).eval().requires_grad_(False)
    predictor.load_state_dict(state, strict=True)
    before = tensor_hash(predictor.state_dict())
    arrays = {s: {k: np.load(base/'paired_cache'/s/(k+'.npy'), mmap_mode='r') for k in ['visual', 'phases', 'episodes']}
              for s in ['train', 'validation']}
    wrench = extract_wrench(base, arrays)
    patches = torch.arange(784, device=device)
    for split, x in arrays.items():
        dest = np.lib.format.open_memmap(output/(split+'_predicted.npy'), mode='w+', dtype=np.float16, shape=x['visual'].shape)
        for start in range(0, len(dest), 32):
            ids = np.arange(start, min(start+32, len(dest)))
            visual = torch.from_numpy(np.array(x['visual'][ids], copy=True)).to(device)
            pooled = visual.reshape(len(ids), 2, 392, 256).mean(2)
            tau = torch.from_numpy(wrench[split][ids]).to(device)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                prediction = predictor(visual, pooled, tau, patches).float()
            dest[ids] = prediction.cpu().numpy().astype(np.float16)
        dest.flush()
        print(json.dumps(dict(event='predictor_cache', split=split, samples=len(dest))), flush=True)
    assert tensor_hash(predictor.state_dict()) == before
    record = dict(predictor_checkpoint_sha256=config.predictor_sha256, predictor_state_sha256=before,
        arrays={s: file_hash(output/(s+'_predicted.npy')) for s in arrays}, precision='Original predictor FP32, FP16 storage only',
        split=json.loads((base/'episode_split.json').read_text()), teacher_cache=json.loads((base/'paired_cache/manifest.json').read_text()))
    write_json(output/'prediction_manifest.json', record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['base', 'predictor', 'old', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=['prepare', 'oracle', 'decoder'], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'config/response_conditioned_rgb.yaml')
    a = p.parse_args()
    config = OmegaConf.load(a.config)
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    if a.mode == 'prepare':
        a.output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config, a.output/'config.yaml')
        prepare(a.base, a.predictor, a.output, config, device)
        return
    output = a.output/(a.mode+('_smoke' if a.smoke else ''))
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    assert file_hash(a.base/'inputs/pinned_teacher_ema.ckpt') == config.teacher_sha256
    prediction_manifest = json.loads((a.output/'prediction_manifest.json').read_text())
    assert prediction_manifest['predictor_checkpoint_sha256'] == config.predictor_sha256
    teacher, teacher_cfg, initial = load_inputs(a.base/'inputs', device)
    owner, _, _ = model_for(initial, 'c_film_mlp')
    source = a.base/'d/best.pt'
    assert file_hash(source) == config.decoder_initial_sha256
    saved = torch.load(source, map_location='cpu', weights_only=False)
    owner.load_state_dict(saved['ema_model'], strict=True)
    adapter = torch.load(a.base/'adapter/best_adapter.pt', map_location='cpu', weights_only=False)
    owner.context_adapter.load_state_dict(adapter['state'], strict=True)
    owner.to(device).eval().requires_grad_(False)
    deterministic_mode(owner)
    frozen = {name: tensor_hash(module.state_dict()) for name, module in [('teacher', teacher), ('adapter', owner.context_adapter), ('condition', owner.condition_network), ('torque', owner.wrench_encoder)]}
    train, val, identities = datasets_from_reference(a.base/'inputs', a.old/'decoded_cuda')
    cache_manifest = json.loads((a.base/'paired_cache/manifest.json').read_text())
    assert identities == cache_manifest['splits']
    datasets = {s: FullPairedDataset(ds, a.base/'paired_cache'/s) for s, ds in [('train', train), ('validation', val)]}
    predictions = {s: np.load(a.output/(s+'_predicted.npy'), mmap_mode='r') for s in datasets}
    threshold = json.loads((a.base/'cnn/manifest.json').read_text())['threshold']
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    def predicted(batch, split):
        return torch.from_numpy(np.array(predictions[split][batch['index'].cpu().numpy()], copy=True)).to(device).float()
    def inputs(batch):
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        return images, wrench
    @torch.no_grad()
    def gates(images, wrench):
        with amp():
            masks, condition = owner.condition_network(images, wrench)
            spatial = {k: owner._mask_gate(masks[k].flatten(0, 1), images[k]).reshape_as(masks[k]).float() for k in images}
        return spatial, condition['contact_alpha'].float()
    def encode(images, grad=False):
        with amp():
            return teacher.obs_encoder.encode_image_context(images, context_adapter=owner.context_adapter, input_grad=grad).float()
    def batch_for(split, ids):
        return to_device(next(iter(DataLoader(Subset(datasets[split], list(map(int, ids))), batch_size=len(ids)))), device)
    def rows_for(batch, pred, images, rendered, z0, z):
        return metric_rows(z-z0, pred, batch['target'].float()-batch['zero'].float(), images, rendered,
            batch['index'].cpu().tolist(), batch['episode'].cpu().tolist(), batch['phase'].cpu().tolist(), threshold)
    def snapshot(batch, pred, images, rendered, z, z0):
        return [dict(index=int(batch['index'][i]), episode=int(batch['episode'][i]),
            original={k:v[i].cpu() for k,v in images.items()}, rendered={k:v[i].cpu() for k,v in rendered.items()},
            predicted=pred[i].cpu(), teacher=(batch['target'][i].float()-batch['zero'][i].float()).cpu(),
            student=(z[i]-z0[i]).cpu()) for i in range(len(pred))]
    def check_frozen():
        for name, module in [('teacher', teacher), ('adapter', owner.context_adapter), ('condition', owner.condition_network), ('torque', owner.wrench_encoder)]:
            assert not any(p.requires_grad or p.grad is not None for p in module.parameters()), name
            assert tensor_hash(module.state_dict()) == frozen[name], name
    manifest = dict(source_commit=config.source_commit, config=OmegaConf.to_container(config),
        teacher_sha256=config.teacher_sha256, predictor_sha256=config.predictor_sha256,
        decoder_initial_sha256=file_hash(source), decoder_initial_step=saved['step'],
        adapter_sha256=file_hash(a.base/'adapter/best_adapter.pt'), frozen=frozen,
        predictor_cache=prediction_manifest, zero_branch='Original RGB exactly; same frozen DINO/Adapter/ContextNorm and encode batch for paired response.',
        support='abs RGB residual <=0.1*deployed spatial gate*contact alpha; clip[0,1]',
        response_target='Frozen RAW predictor output for Contact/Precontact, zero response for Free; phase labels used only in loss/sampling, never model input',
        conditioning='Predictor patch response multiplied by deployed frozen alpha; original RGB Torque LSTM unused/frozen; CA+FiLM consume predicted response tokens',
        precision='BF16 DINO/decoder; exact original online baseline at same batch shape; frozen predictor generated FP32')
    OmegaConf.save(config, output/'config.yaml')
    if a.mode == 'oracle':
        selected = []
        for ep in np.unique(datasets['validation'].episodes):
            ids = np.flatnonzero((datasets['validation'].episodes == ep) & (val.phase_labels == 2))
            selected.extend(ids[np.asarray([int((len(ids)-1)*f) for f in [1/3, 2/3]])].tolist())
        if a.smoke: selected = selected[:2]
        write_json(output/'selected_indices.json', selected)
        logs, records, visuals, initial_rows = [], [], [], []
        for offset in range(0, len(selected), config.oracle.batch_size):
            batch = batch_for('validation', selected[offset:offset+config.oracle.batch_size])
            images, wrench = inputs(batch)
            pred = predicted(batch, 'validation')
            spatial, alpha = gates(images, wrench)
            support = {k: spatial[k]*alpha[..., None, None, None] for k in images}
            z0 = encode(images).detach()
            variables = {k: nn.Parameter(torch.zeros_like(v)) for k,v in images.items()}
            optimizer = torch.optim.Adam(list(variables.values()), lr=config.oracle.lr)
            def render():
                return {k: torch.where(support[k] > 0, (images[k]+torch.maximum(-config.epsilon*support[k], torch.minimum(config.epsilon*support[k], variables[k]))).clamp(0,1), images[k]) for k in images}
            initial_rows.extend(rows_for(batch, pred, images, images, z0, z0))
            for step in range(1, (2 if a.smoke else config.oracle.steps)+1):
                optimizer.zero_grad(set_to_none=True)
                rendered = render()
                z = encode(rendered, True)
                loss = (z-z0-pred).square().mean()
                loss.backward()
                gradient = float(sum(v.grad.float().square().sum() for v in variables.values()).sqrt())
                assert gradient > 0 and torch.isfinite(loss)
                optimizer.step()
                with torch.no_grad():
                    for k,v in variables.items():
                        v.copy_(torch.maximum(-config.epsilon*support[k], torch.minimum(config.epsilon*support[k], v)))
                if step == 1 or step % 50 == 0:
                    logs.append(dict(chunk=offset, step=step, realization_loss=float(loss), rgb_gradient_norm=gradient))
                    write_json(output/'log.json', logs)
                    print(json.dumps(dict(event='oracle', **logs[-1])), flush=True)
            with torch.no_grad():
                rendered = render();z = encode(rendered)
                records.extend(rows_for(batch, pred, images, rendered, z0, z))
                visuals.extend(snapshot(batch, pred, images, rendered, z, z0))
                closed = {k: (images[k]+variables[k]*torch.zeros_like(support[k])).clamp(0,1) for k in images}
                assert all(torch.equal(closed[k], images[k]) for k in images)
        check_frozen()
        write_json(output/'samples.json', records)
        torch.save(visuals, output/'visuals.pt')
        write_json(output/'summary.json', dict(manifest=manifest, initial=grouped(initial_rows, config), final=grouped(records, config),
            frozen_verified=True, alpha0_identity=True, zero_image_original=True, elapsed_seconds=time.monotonic()-started))
        return
    network = ResponseConditionedDecoder(owner, config.epsilon).to(device).eval()
    parameters = [p for p in network.parameters() if p.requires_grad]
    manifest['trainable_parameters'] = sum(p.numel() for p in parameters)
    manifest['trainable_names'] = [n for n,p in network.named_parameters() if p.requires_grad]
    assert not any(n.startswith(('owner.wrench_', 'owner.obs_position', 'owner.force_modality',
        'owner.condition_network.', 'owner.context_adapter.')) for n in manifest['trainable_names'])
    frozen_config = config.decoder
    plan_config = OmegaConf.create(dict(seed=config.seed, steps=config.decoder.steps, phase_mix=list(config.decoder.phase_mix)))
    plan = phase_plan(train.phase_labels, datasets['train'].episodes, plan_config)
    np.save(output/'train_batches.npy', plan)
    manifest['sample_plan_sha256'] = file_hash(output/'train_batches.npy')
    manifest['sampling_coverage'] = {str(p): dict(episodes=sorted(set(datasets['train'].episodes[plan[:,sum(config.decoder.phase_mix[:p]):sum(config.decoder.phase_mix[:p+1])]].flatten().tolist())),
        unique_windows=len(np.unique(plan[:,sum(config.decoder.phase_mix[:p]):sum(config.decoder.phase_mix[:p+1])])) ) for p in range(3)}
    optimizer = torch.optim.AdamW(parameters, lr=config.decoder.lr, weight_decay=config.decoder.weight_decay)
    ema = copy.deepcopy(network).eval().requires_grad_(False)
    def update_ema():
        with torch.no_grad():
            for dest, src in zip(ema.parameters(), network.parameters()): dest.lerp_(src, .01)
    @torch.no_grad()
    def evaluate(model, step, train_probe=True):
        result, visual = {}, []
        for split, ds in datasets.items():
            ids = list(range(len(ds)))
            if split == 'train' and train_probe:
                ids = [int(c[len(c)//2]) for ep in np.unique(ds.episodes) for phase in range(3)
                    if len(c:=np.flatnonzero((ds.episodes == ep) & (ds.dataset.phase_labels == phase)))]
            if a.smoke: ids = ids[:16]
            records = []
            for raw in DataLoader(Subset(ds, ids), batch_size=config.decoder.validation_batch, num_workers=config.workers, pin_memory=True):
                batch = to_device(raw, device)
                images, wrench = inputs(batch);pred = predicted(batch, split)
                spatial, alpha = gates(images, wrench)
                z0 = encode(images)
                with amp(): rendered = model(images, pred, spatial, alpha)
                z = encode(rendered)
                records.extend(rows_for(batch, pred, images, rendered, z0, z))
                if split == 'validation':
                    for i in range(len(pred)):
                        if int(batch['phase'][i]) != 2 or any(v['episode'] == int(batch['episode'][i]) for v in visual):
                            continue
                        item = {**batch, 'index': batch['index'][i:i+1], 'episode': batch['episode'][i:i+1],
                                'target': batch['target'][i:i+1], 'zero': batch['zero'][i:i+1]}
                        visual.extend(snapshot(item, pred[i:i+1], {k:v[i:i+1] for k,v in images.items()},
                            {k:v[i:i+1] for k,v in rendered.items()}, z[i:i+1], z0[i:i+1]))
            result[split] = grouped(records, config)
            write_json(output/f'{split}_samples_{step}.json', records)
        torch.save(visual, output/f'visuals_{step}.pt')
        return result
    probe = batch_for('train', plan[0])
    im,wr = inputs(probe);pr = predicted(probe,'train');sp,al = gates(im,wr)
    with torch.no_grad(),amp():
        closed = network(im,pr,sp,torch.zeros_like(al))
        repeated = network(im,pr,sp,al);again = network(im,pr,sp,al)
    assert all(torch.equal(closed[k],im[k]) for k in im)
    assert all(torch.equal(repeated[k],again[k]) for k in im)
    zero_obs = zero_wrench_reference(teacher, probe['obs'])
    zero_im, zero_wr, _, _ = teacher.prepare_context_inputs_with_future_contact(zero_obs)
    assert all(torch.equal(zero_im[k],im[k]) for k in im)
    assert teacher.normalizer[teacher.wrench_key].unnormalize(zero_wr).abs().max() < 1e-5
    history, training, best = [], [], float('inf')
    first = evaluate(ema, 0)
    history.append(dict(step=0, metrics=first, elapsed_seconds=time.monotonic()-started))
    best = first['validation']['selection_contact']['realization_mse'] if not a.smoke else float('inf')
    initial_result = copy.deepcopy(first)
    torch.save(dict(model=network.state_dict(), ema_model=ema.state_dict(), step=0),output/'initial.pt')
    torch.save(dict(model=network.state_dict(), ema_model=ema.state_dict(), step=0),output/'best.pt')
    iterations = 2 if a.smoke else config.decoder.steps
    iterator = iter(DataLoader(datasets['train'], batch_sampler=plan[:iterations].tolist(), num_workers=config.workers, pin_memory=True))
    for step in range(1, iterations+1):
        batch = to_device(next(iterator),device)
        pred = predicted(batch,'train')
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for start in range(0,len(pred),config.decoder.microbatch):
            sl = slice(start,start+config.decoder.microbatch)
            part = {**batch,'obs':{k:v[sl] for k,v in batch['obs'].items()}}
            images,wr = inputs(part);spatial,alpha = gates(images,wr)
            z0 = encode(images).detach()
            with amp(): rendered = network(images,pred[sl],spatial,alpha)
            z = encode(rendered,True)
            target = pred[sl]* (batch['phase'][sl] != 0)[:,None,None]
            per_sample = (z-z0-target).square().mean((1,2))
            weights = torch.where(batch['phase'][sl] == 0,config.decoder.free_weight,1.)
            loss = (per_sample*weights).sum()/len(pred)
            assert torch.isfinite(loss)
            loss.backward();total += float(loss.detach())
        gradients = {label: float(sum(p.grad.float().square().sum() for n,p in network.named_parameters() if p.grad is not None and n.startswith(prefix)).sqrt())
            for label,prefix in [('projection','response_projection.'),('cnn','owner.backbone.stem.'),('ca','owner.backbone.cross_attention.'),('film','owner.backbone.decoder_film.'),('head','owner.backbone.output.')]}
        assert gradients['projection'] > 0 and gradients['head'] > 0
        torch.nn.utils.clip_grad_norm_(parameters,config.decoder.gradient_clip)
        optimizer.step();update_ema()
        if step == 1 or step%25 == 0 or a.smoke:
            training.append(dict(step=step,loss=total,gradient_norms=gradients,indices=batch['index'].cpu().tolist(),elapsed_seconds=time.monotonic()-started))
            write_json(output/'training.json',training)
            print(json.dumps(dict(event='train',**training[-1])),flush=True)
        if step%config.decoder.validate_every == 0 or step == iterations:
            check_frozen()
            scores = evaluate(ema,step)
            history.append(dict(step=step,metrics=scores,elapsed_seconds=time.monotonic()-started))
            write_json(output/'log.json',history)
            score = scores['validation']['selection_contact']['realization_mse'] if not a.smoke else total
            state = dict(model=network.state_dict(),ema_model=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,manifest=manifest)
            if score < best: best=score;torch.save(state,output/'best.pt')
            torch.save(state,output/'latest.pt')
            print(json.dumps(dict(event='validation',step=step,contact=scores['validation']['phases']['contact'],external=scores['validation']['external_contact'])),flush=True)
    best_state = torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    ema.load_state_dict(best_state['ema_model'],strict=True)
    final = evaluate(ema,'best',train_probe=a.smoke)
    # Same fixed images and encode batch size as Oracle, with no per-image fitting.
    matched_ids = []
    for ep in np.unique(datasets['validation'].episodes):
        ids = np.flatnonzero((datasets['validation'].episodes == ep) & (val.phase_labels == 2))
        matched_ids.extend(ids[np.asarray([int((len(ids)-1)*f) for f in [1/3,2/3]])].tolist())
    if a.smoke: matched_ids = matched_ids[:2]
    matched_rows, matched_visuals = [], []
    with torch.no_grad():
        for offset in range(0,len(matched_ids),config.oracle.batch_size):
            batch = batch_for('validation',matched_ids[offset:offset+config.oracle.batch_size])
            images,wr = inputs(batch);pred = predicted(batch,'validation');spatial,alpha = gates(images,wr)
            z0 = encode(images)
            with amp(): rendered = ema(images,pred,spatial,alpha)
            z = encode(rendered)
            matched_rows.extend(rows_for(batch,pred,images,rendered,z0,z))
            matched_visuals.extend(snapshot(batch,pred,images,rendered,z,z0))
    write_json(output/'oracle_matched_samples.json',matched_rows)
    torch.save(matched_visuals,output/'oracle_matched_visuals.pt')
    final['oracle_matched_contact'] = grouped(matched_rows,config)
    check_frozen()
    with torch.no_grad(),amp(): closed = ema(im,pr,sp,torch.zeros_like(al))
    assert all(torch.equal(closed[k],im[k]) for k in im)
    write_json(output/'summary.json',dict(manifest=manifest,initial=initial_result,final=final,best_step=best_state['step'],
        steps=iterations,best_checkpoint_sha256=file_hash(output/'best.pt'),frozen_verified=True,alpha0_identity=True,
        zero_image_original=True,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',mode=a.mode,steps=iterations)),flush=True)


if __name__ == '__main__':
    main()
