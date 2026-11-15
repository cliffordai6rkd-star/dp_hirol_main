"""Exactly 40 online AdamW updates: fixed-batch Mixed versus Contact/8.

Reuses the preceding FP32 probe's loader/readout/constraints, never its Oracle
or optimization routines. No model checkpoint or production file is written.
"""
from pathlib import Path
import argparse
import json
import os
import time
from collections.abc import Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

import diagnose_civi_local_updates as previous
from paired_response_full_common import file_hash, deterministic_mode
from student_dual_path_common import tensor_hash, write_json
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


STEPS = (0, 1, 5, 10, 20)


def norm(vector):
    return float(vector.double().norm())


def dot(a, b):
    return float(a.double().dot(b.double()))


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from tensors(v)


class MixedStudy(previous.Study):
    def __init__(self, args):
        # Only initialize the same six windows; none of the old optimization
        # methods (including pixel/Oracle methods) are called.
        super().__init__(args)
        old_protocol = json.loads((args.previous/'protocol.json').read_text())
        old_baseline = json.loads((args.previous/'baseline.json').read_text())
        assert self.selection == old_protocol['selection']
        assert self.initial_hash == old_protocol['initial_state_sha256']
        assert file_hash(Path(previous.__file__)) == old_protocol['source_sha256']
        for actual, reference in zip(self.baseline, old_baseline):
            assert actual == reference, (actual['label'], 'Previous FP32 baseline must reproduce exactly')
        config_path = args.checkpoint.parent/'config.yaml'
        config = OmegaConf.load(config_path)
        assert config.rgb_weight == 0 and config.free_weight == 1
        assert config.lr == args.network_lr and config.weight_decay == 1e-4 and config.gradient_clip == 1
        self.free_weight = float(config.free_weight)
        self.precision = {}
        self.mode = 'prepare'
        self.hooks = []
        modules = dict(renderer=self.r.network, dino=self.r.teacher.obs_encoder.backbone,
            projection=self.r.teacher.obs_encoder.image_projection,
            adapter=self.r.owner.context_adapter, context_norm=self.r.teacher.obs_encoder.context_norm,
            predictor=self.predictor, condition_network=self.r.owner.condition_network,
            teacher_context=self.r.teacher.obs_encoder)
        for name, module in modules.items():
            def forward_hook(_module, inputs, output, name=name):
                record = self.precision.setdefault(name, dict(input_float_dtypes=set(), output_float_dtypes=set(),
                    backward_float_dtypes=set(), grad_enabled_output_seen=False, modes=set()))
                record['modes'].add(self.mode)
                for value in tensors(inputs):
                    if value.is_floating_point():
                        record['input_float_dtypes'].add(str(value.dtype))
                        assert value.dtype == torch.float32, (name, value.dtype)
                for value in tensors(output):
                    if value.is_floating_point():
                        record['output_float_dtypes'].add(str(value.dtype))
                        assert value.dtype == torch.float32, (name, value.dtype)
                        if value.requires_grad:
                            record['grad_enabled_output_seen'] = True
                            def backward_hook(gradient, name=name):
                                self.precision[name]['backward_float_dtypes'].add(str(gradient.dtype))
                                assert gradient.dtype == torch.float32
                            value.register_hook(backward_hook)
            self.hooks.append(module.register_forward_hook(forward_hook))
        self.module_metadata = {name:dict(training=module.training,
            parameter_dtypes=sorted({str(p.dtype) for p in module.parameters()}),
            trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
            for name,module in modules.items()}
        for module in (self.r.teacher, self.predictor):
            deterministic_mode(module)
        dataset = self.r.datasets['train']
        used = {x['episode'] for x in self.selection}
        eligible = sorted(set(dataset.episodes[dataset.dataset.phase_labels == 1])-used)
        extra = []
        for ordinal, episode in enumerate((eligible[0], eligible[-1]), 1):
            indices = np.flatnonzero((dataset.episodes == episode) & (dataset.dataset.phase_labels == 1))
            extra.append(dict(label=f'P{ordinal}', index=int(indices[len(indices)//2]),
                              episode=int(episode), phase=1, split='train'))
        self.selection += extra
        assert len({x['episode'] for x in self.selection}) == 8
        write_json(args.output/'selection.json', self.selection)
        for meta in extra:
            self.samples.append(self.prepare_extra(meta))
        self.baseline = self.evaluate_rows()
        for actual, reference in zip(self.baseline[:6], old_baseline):
            assert all(actual[k] == v for k, v in reference.items())
        for sample in self.samples:
            if sample['phase'] == 0:
                assert torch.count_nonzero(sample['teacher']) == 0
        self.fixed_hash = self.input_hash()
        reference_config = json.loads((args.base/'inputs/reference_train_cache.json').read_text())
        assert reference_config['identity']['dataset']['image_randomer_config'] is None
        self.protocol = dict(name='fixed-eight Mixed versus Contact-only', selection=self.selection,
            steps_per_group=20, total_optimizer_updates=40, observed_steps=STEPS, microbatch=1,
            batch_denominator=8, free_weight=self.free_weight, rgb_weight=0,
            objectives=dict(A_Mixed='(sum C MSE(dS,dP) + sum P MSE(dS,dP) + free_weight*sum F MSE(dS,0))/8',
                B_Contact_only='sum C MSE(dS,dP)/8'),
            free_target='ZERO, as in original training; unlike previous auxiliary Free gradient with dP target',
            optimizer=old_protocol['network_optimizer'], optimizer_state='NEW AdamW in both groups; no checkpoint optimizer state loaded',
            model='Mutable online model initialized from the SAME prior EMA; no EMA creation or averaging',
            source_loss=str(Path(__file__).with_name('run_civi_rgb_iteration.py'))+':330-350',
            original_config_sha256=file_hash(config_path), original_config=OmegaConf.to_container(config),
            checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
            checkpoint_step=old_protocol['checkpoint_step'], initial_state_sha256=self.initial_hash,
            trainable_names=self.names, trainable_shapes=[list(p.shape) for p in self.params],
            trainable_parameters=sum(p.numel() for p in self.params), dependencies=self.dependencies,
            previous_protocol_sha256=file_hash(args.previous/'protocol.json'),
            previous_baseline_exact=True, previous_source_sha256=old_protocol['source_sha256'],
            source_sha256=file_hash(Path(__file__)), fixed_input_state_sha256=self.fixed_hash,
            precision='FP32 per module for all forward/backward; autocast and TF32 disabled; audited by tensor hooks',
            module_metadata=self.module_metadata, deterministic=True, random_augmentation=False, dropout=False,
            endpoint_scaling='NONE; no extra optimizer or scaled-step trials',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name())
        write_json(args.output/'protocol.json', self.protocol)
        write_json(args.output/'baseline.json', self.baseline)
        previous.event('batch_ready', selection=self.selection, previous_baseline_exact=True)

    @torch.no_grad()
    def prepare_extra(self, meta):
        r = self.r
        batch = r.batch('train', [meta['index']])
        images, wrench = r.inputs(batch)
        images = {k:v.float().detach() for k,v in images.items()}
        seen = []
        def capture(z):
            seen.append(z.detach())
            return r.owner.context_adapter(z)
        z0 = r.teacher.obs_encoder.encode_image_context(images, context_adapter=capture).float()
        visual = seen[0]
        pred = self.predictor(visual, visual.reshape(1,2,392,256).mean(2),
            batch['obs'][r.teacher.wrench_key][:,:2].float(), torch.arange(784,device=r.device)).detach()
        masks, condition = r.owner.condition_network(images,wrench)
        spatial = {k:r.owner._mask_gate(masks[k].flatten(0,1),images[k]).reshape_as(masks[k]).float().detach() for k in images}
        alpha = condition['contact_alpha'].float().detach()
        support = {k:spatial[k]*alpha[...,None,None,None] for k in images}
        target = r.teacher._encode_context(batch['obs'],apply_curriculum_mask=False).float()
        zero = r.teacher._encode_context(zero_wrench_reference(r.teacher,batch['obs']),apply_curriculum_mask=False).float()
        rendered, stats = r.network(images,pred,spatial,alpha,trace=True)
        rendered = {k:v.detach() for k,v in rendered.items()}
        return dict(**meta, images=images,spatial=spatial,alpha=alpha,support=support,
            lower={k:(v-.1*support[k]).clamp(0,1) for k,v in images.items()},
            upper={k:(v+.1*support[k]).clamp(0,1) for k,v in images.items()},
            pred=pred,teacher=(target-zero).detach(),z0=z0.detach(), initial_rgb=rendered,
            initial_response=(self.encode(rendered)-z0).detach(),initial_stats=stats)

    def input_hash(self):
        values = {}
        for s in self.samples:
            for field in ('images','spatial','alpha','support','pred','teacher','z0','initial_rgb','initial_response'):
                value = s[field]
                if isinstance(value,dict):
                    values.update({s['label']+'.'+field+'.'+k:v for k,v in value.items()})
                else:
                    values[s['label']+'.'+field] = value
        return tensor_hash(values)

    def phase_loss(self,s,rgb):
        response = self.encode(rgb,True)-s['z0']
        target = torch.zeros_like(s['pred']) if s['phase']==0 else s['pred']
        return (response-target).square().mean()

    def evaluate_rows(self):
        rows = []
        with torch.no_grad():
            for s in self.samples:
                rgb, stats = self.render(s,True)
                row = super().measure(s,rgb,stats)
                # Free Teacher response is exactly zero, so its Teacher error
                # is the exact directly evaluated actual response mean-square.
                row['actual_response_mse'] = row['teacher_mse'] if s['phase']==0 else row['response_rms']**2
                row['training_phase_loss'] = row['actual_response_mse'] if s['phase']==0 else row['predicted_mse']
                initial_mse = float(s['initial_response'].square().mean())
                initial_rms = float(s['initial_response'].square().mean().sqrt())
                row['response_mse_relative_change'] = row['actual_response_mse']/initial_mse-1 if initial_mse else None
                row['response_rms_relative_change'] = row['response_rms']/initial_rms-1 if initial_rms else None
                rows.append(row)
        return rows

    def summary_rows(self,rows):
        result = {}
        for phase,label in ((2,'contact'),(1,'precontact'),(0,'free')):
            subset = [r for r in rows if r['phase']==phase]
            agg = previous.mean_metrics(subset)
            agg['actual_response_mse'] = float(np.mean([r['actual_response_mse'] for r in subset]))
            baseline = [r for r in self.baseline if r['phase']==phase]
            initial_mse = float(np.mean([r['actual_response_mse'] for r in baseline]))
            initial_rms = previous.mean_metrics(baseline)['response_rms']
            agg['actual_response_mse_relative_change'] = agg['actual_response_mse']/initial_mse-1 if initial_mse else None
            agg['actual_response_rms_relative_change'] = agg['response_rms']/initial_rms-1 if initial_rms else None
            result[label] = agg
        result['mixed_objective'] = sum(r['training_phase_loss']*(self.free_weight if r['phase']==0 else 1) for r in rows)/8
        result['contact_only_objective'] = sum(r['predicted_mse'] for r in rows if r['phase']==2)/8
        result['all_rgb_update_rms'] = float(np.mean([r['update_rms']**2 for r in rows])**.5)
        result['all_rgb_total_rms'] = float(np.mean([r['total_rms']**2 for r in rows])**.5)
        return result

    def parameter_vector(self):
        return torch.cat([p.detach().flatten().cpu() for p in self.params])

    def gradient_vector(self):
        return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten().cpu() for p in self.params])

    def phase_gradients(self):
        self.restore()
        self.mode = 'initial_phase_gradients'
        means = {}
        for phase,label in ((2,'Contact'),(1,'Precontact'),(0,'Free')):
            samples = [s for s in self.samples if s['phase']==phase]
            self.r.network.zero_grad(set_to_none=True)
            for sample in samples:
                (self.phase_loss(sample,self.render(sample))/len(samples)).backward()
            means[label] = self.gradient_vector()
        weighted = dict(Contact=means['Contact']*.5,Precontact=means['Precontact']*.25,Free=means['Free']*(.25*self.free_weight))
        combined = dict(A_Mixed=weighted['Contact']+weighted['Precontact']+weighted['Free'], B_Contact_only=weighted['Contact'])
        result = dict(phase_order=list(means),phase_mean_norms={k:norm(v) for k,v in means.items()},
            phase_mean_cosines=[[previous.cosine(a,b) for b in means.values()] for a in means.values()],
            weighted_contribution_norms={k:norm(v) for k,v in weighted.items()},
            combined_gradient_norms={k:norm(v) for k,v in combined.items()},
            combined_cosine=previous.cosine(combined['A_Mixed'],combined['B_Contact_only']),
            weighted_contribution_dot_contact={k:dot(means['Contact'],v) for k,v in weighted.items()},
            denominator=8,free_weight=self.free_weight)
        self.restore()
        write_json(self.a.output/'phase_gradients.json',result)
        previous.event('phase_gradients',**result)
        return means,combined,result

    def run_group(self,name,means,combined):
        self.restore()
        assert tensor_hash(self.r.network.state_dict()) == self.initial_hash
        optimizer = torch.optim.AdamW(self.params,lr=self.a.network_lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
        assert not optimizer.state
        self.mode = name+'_online_evaluation'
        rows = self.evaluate_rows()
        assert rows == self.baseline
        history = [dict(step=0,rows=rows,summary=self.summary_rows(rows),parameter_total_displacement_norm=0)]
        updates = []
        initial_vector = self.parameter_vector()
        first = None
        vectors = {}
        for step in range(1,21):
            self.mode = name+'_forward_backward'
            optimizer.zero_grad(set_to_none=True)
            objective = 0.
            for sample in self.samples:
                # Both groups execute the SAME microbatch=1 forward/backward
                # shapes, including zero-weight non-Contact terms in B.
                weight = (self.free_weight if sample['phase']==0 else 1.) if name=='A_Mixed' else float(sample['phase']==2)
                term = self.phase_loss(sample,self.render(sample))*(weight/8)
                assert torch.isfinite(term)
                term.backward()
                objective += float(term.detach())
            raw_gradient = self.gradient_vector()
            before = self.parameter_vector()
            returned_norm = torch.nn.utils.clip_grad_norm_(self.params,1.)
            clipped_gradient = self.gradient_vector()
            optimizer.step()  # Exactly ONE call after all eight microbatches.
            after = self.parameter_vector()
            delta = after-before
            raw_norm,clipped_norm = norm(raw_gradient),norm(clipped_gradient)
            update = dict(step=step,objective_before=objective,gradient_norm_pre_clip=raw_norm,
                gradient_norm_post_clip=clipped_norm,clip_grad_reported_norm=float(returned_norm),
                gradient_clip_factor=clipped_norm/raw_norm,
                gradient_elements_changed_fraction=float((raw_gradient!=clipped_gradient).float().mean()),
                parameter_delta_norm=norm(delta),parameter_total_displacement_norm=norm(after-initial_vector),
                parameter_delta_linf=float(delta.abs().max()))
            updates.append(update)
            if step==1:
                expected = combined[name]
                mismatch = norm(raw_gradient-expected)/norm(expected)
                assert mismatch < 2e-6, (name,mismatch)
                first = dict(**update,weighted_gradient_reconstruction_relative_error=mismatch,
                    contact_gradient_dot_actual_delta=dot(means['Contact'],delta),
                    precontact_gradient_dot_actual_delta=dot(means['Precontact'],delta),
                    free_gradient_dot_actual_delta=dot(means['Free'],delta),
                    combined_gradient_dot_actual_delta=dot(raw_gradient,delta),
                    contact_descent_alignment=previous.cosine(-means['Contact'],delta),
                    nominal_weight_decay_delta_norm=norm(-self.a.network_lr*1e-4*before))
                vectors=dict(raw_gradient=raw_gradient.numpy(),clipped_gradient=clipped_gradient.numpy(),actual_parameter_delta=delta.numpy())
            if step in STEPS:
                self.mode = name+'_online_evaluation'
                rows = self.evaluate_rows()
                summary = self.summary_rows(rows)
                history.append(dict(step=step,rows=rows,summary=summary,**{k:v for k,v in update.items() if k!='step'}))
                previous.event('online',group=name,step=step,contact=summary['contact']['predicted_mse'],
                    precontact=summary['precontact']['predicted_mse'],free_response_mse=summary['free']['actual_response_mse'],
                    contact_rgb_update_rms=summary['contact']['update_rms'],parameter_delta_norm=norm(delta))
            if step==1:
                first['contact_loss_before'] = history[0]['summary']['contact']['predicted_mse']
                first['contact_loss_after'] = history[1]['summary']['contact']['predicted_mse']
                first['actual_contact_loss_change'] = first['contact_loss_after']-first['contact_loss_before']
                first['first_order_prediction_error'] = first['actual_contact_loss_change']-first['contact_gradient_dot_actual_delta']
        self.r.check_frozen()
        assert self.input_hash()==self.fixed_hash
        assert self.predictor_hash==tensor_hash(self.predictor.state_dict())
        result = dict(group=name,optimizer_steps=20,history=history,updates=updates,first_step=first)
        write_json(self.a.output/(name+'.json'),result)
        return result,vectors

    def run(self):
        means,combined,gradients = self.phase_gradients()
        a,av = self.run_group('A_Mixed',means,combined)
        b,bv = self.run_group('B_Contact_only',means,combined)
        # Numerical evidence only: gradients and measured deltas, NOT weights,
        # optimizer states, or checkpoints. Order/shapes are in protocol.json.
        np.savez_compressed(self.a.output/'first_step_vectors.npz',
            **{'phase_mean_'+k:v.numpy() for k,v in means.items()},
            **{'A_'+k:v for k,v in av.items()}, **{'B_'+k:v for k,v in bv.items()})
        comparison = []
        for ah,bh in zip(a['history'],b['history']):
            ar,br = ah['summary']['contact']['update_rms'],bh['summary']['contact']['update_rms']
            comparison.append(dict(step=ah['step'],contact_rgb_rms_ratio_A_over_B=ar/br if br else None,
                all_rgb_rms_ratio_A_over_B=ah['summary']['all_rgb_update_rms']/bh['summary']['all_rgb_update_rms'] if br else None))
        self.restore()
        assert tensor_hash(self.r.network.state_dict())==self.initial_hash
        self.r.check_frozen()
        assert file_hash(self.a.checkpoint)==self.protocol['checkpoint_sha256']
        precision = {k:{name:sorted(value) if isinstance(value,set) else value for name,value in record.items()} for k,record in self.precision.items()}
        for name in ('renderer','dino','projection','adapter','context_norm'):
            assert precision[name]['backward_float_dtypes']==['torch.float32'], (name,precision[name])
        write_json(self.a.output/'precision_audit.json',precision)
        result = dict(groups={a['group']:a,b['group']:b},phase_gradients=gradients,
            rgb_displacement_comparison=comparison,total_optimizer_steps=40,
            baseline_exact_previous=True,initial_hash=self.initial_hash,checkpoint_unchanged=True,
            frozen_verified=True,fixed_inputs_verified=True,network_restored=True,
            extra_updates=0,scaled_step_trials=0,elapsed_seconds=time.monotonic()-self.started,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(self.a.output/'summary.json',result)
        previous.event('complete',updates=40,elapsed_seconds=result['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('data/diagnostics/civi_student_iteration')
    matched = root/'final150k_match'
    for key,value in dict(base=matched/'matched',predictor=matched/'predictor',cache=matched/'predictor_cache',
        old=Path('data/diagnostics/2026.11.15/student_dual_path_ablation'),dependencies=matched/'dependencies.json',
        checkpoint=root/'final150k_baseline_s42_3000/best.pt',previous=root/'final150k_local_updates_20',
        output=root/'final150k_mixed_updates_20').items():
        parser.add_argument('--'+key,type=Path,default=value)
    args = parser.parse_args()
    args.steps,args.network_lr,args.pixel_lr = 20,2e-4,1e-3
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    assert torch.cuda.device_count()==1
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output/'previous_probe_source.py').write_bytes(Path(previous.__file__).read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    with torch.autocast('cuda',enabled=False):
        MixedStudy(args).run()


if __name__=='__main__':
    main()
