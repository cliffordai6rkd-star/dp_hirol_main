"""Matrix-free local RGB controllability: JVP/VJP ridge CG and nonlinear checks."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from run_shared_rgb_oracle import load_context
from student_dual_path_common import to_device,write_json,tensor_hash
from torch.utils.data import DataLoader,Subset


def conjugate_gradient(operator,b,iterations=12,tolerance=1e-5):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=torch.dot(r,r);history=[float(rr.sqrt())]
    for _ in range(iterations):
        ap=operator(p);den=torch.dot(p,ap)
        if den<=0:break
        alpha=rr/den;x=x+alpha*p;r=r-alpha*ap;next_rr=torch.dot(r,r)
        history.append(float(next_rr.sqrt()))
        if next_rr.sqrt()<=tolerance*history[0]:break
        p=r+(next_rr/rr)*p;rr=next_rr
    return x,history


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=12);p.add_argument('--samples',type=int,default=6);p.add_argument('--lambda-rgb',type=float,default=10.)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dev=torch.device('cuda:0');teacher,owner,datasets=load_context(a.root,a.study,dev)
    before=tensor_hash(teacher.state_dict())
    # Eager attention supports forward-mode derivatives and avoids flash-attention derivative limitations.
    backbone=teacher.obs_encoder.backbone
    if hasattr(backbone,'set_attn_implementation'):backbone.set_attn_implementation('eager')
    for module in backbone.modules():
        if hasattr(module,'config') and hasattr(module.config,'_attn_implementation'):module.config._attn_implementation='eager'
    output=a.study/('local_jacobian' if a.iterations==12 else f'local_jacobian_{a.iterations}');output.mkdir(exist_ok=False)
    selected=[];rows=datasets['validation'].rows
    for ep in sorted({r['episode'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['episode']==ep and r['phase']==2]
        if ids:selected.append(ids[len(ids)//2])
    selected=selected[:a.samples];result=[];start=time.monotonic()
    for index in selected:
        batch=to_device(next(iter(DataLoader(Subset(datasets['validation'],[index]),batch_size=1))),dev)
        images,wr,_,_=teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        keys=list(images);sizes=[v.numel() for v in images.values()];shapes=[v.shape for v in images.values()]
        pixels=torch.cat([v.flatten() for v in images.values()]).detach()
        def unpack(x):return {k:v.reshape(shape) for k,v,shape in zip(keys,x.split(sizes),shapes)}
        def phi(x):
            return teacher.obs_encoder.encode_image_context(unpack(x),context_adapter=owner.context_adapter,input_grad=True).float().flatten()
        y=(batch['target'].float()-batch['zero'].float()).flatten().detach()
        def jvp(v):return torch.func.jvp(phi,(pixels,),(v,))[1].detach()
        def vjp(v):
            x=pixels.detach().requires_grad_(True);z=phi(x)
            return torch.autograd.grad(z,x,v,create_graph=False)[0].detach()
        direction=torch.randn_like(pixels);direction/=direction.norm();direction*=pixels.numel()**.5
        tangent=jvp(direction)
        finite_differences=[]
        with torch.no_grad():
            for epsilon in [1e-3,3e-4,1e-4,3e-5]:
                fd=(phi(pixels+epsilon*direction)-phi(pixels-epsilon*direction))/(2*epsilon)
                finite_differences.append(dict(epsilon=epsilon,relative_error=float((tangent-fd).norm()/tangent.norm().clamp_min(1e-12))))
        relative=min(r['relative_error'] for r in finite_differences)
        cotangent=torch.randn_like(y)
        lhs=torch.dot(jvp(direction),cotangent);rhs=torch.dot(direction,vjp(cotangent))
        adjoint_error=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-8))
        assert adjoint_error<.001,('JVP/VJP adjoint failed',adjoint_error)
        print(json.dumps(dict(event='derivative_checks',index=index,finite_difference_sensitivity=finite_differences,
            adjoint_relative_error=adjoint_error)),flush=True)
        # Report finite-step nonlinearity/roundoff instead of declaring it an
        # autodiff failure. The exact JVP/VJP adjoint identity is checked above;
        # the nonlinear feature response is independently measured below.
        # Mean token MSE + lambda * mean pixel-square. No dense Jacobian is materialized.
        regularization=a.lambda_rgb*y.numel()/pixels.numel()
        b=vjp(y)
        delta,history=conjugate_gradient(lambda v:vjp(jvp(v))+regularization*v,b,a.iterations)
        linear=jvp(delta)
        with torch.no_grad():
            base=phi(pixels);actual=phi((pixels+delta).clamp(0,1))-base
            masks,cond=owner.condition_network(images,wr)
            gate={k:owner._mask_gate(masks[k].flatten(0,1),images[k]).reshape_as(masks[k]).float()*cond['contact_alpha'][...,None,None,None] for k in keys}
            bounds=torch.cat([(.1*gate[k]).expand_as(images[k]).flatten() for k in keys])
            bounded=torch.maximum(-bounds,torch.minimum(bounds,delta));bounded=(pixels+bounded).clamp(0,1)-pixels
            bounded_linear=jvp(bounded);bounded_actual=phi(pixels+bounded)-base
            rendered=owner(images,wr);rendered0=owner(images,teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wr)))
            student_delta=torch.cat([(rendered[k]-rendered0[k]).flatten() for k in keys])
        student_linear=jvp(student_delta)
        def metrics(v,pixel_delta):
            return dict(response_mse=float((v-y).square().mean()),explained_energy=1-float((v-y).square().sum()/y.square().sum()),
                response_cosine=float(F.cosine_similarity(v,y,dim=0)),response_rms_ratio=float(v.norm()/y.norm()),
                rgb_rms=float(pixel_delta.square().mean().sqrt()),rgb_max=float(pixel_delta.abs().max()))
        trust_region=[]
        with torch.no_grad():
            for scale in [.05,.1,.25,.5,1.]:
                change=scale*bounded
                measured=phi(pixels+change)-base
                trust_region.append(dict(scale=scale,**metrics(measured,change)))
        best_region=min(trust_region,key=lambda r:r['response_mse'])
        row=dict(index=index,episode=rows[index]['episode'],teacher_held_out=rows[index]['teacher_held_out'],
            jvp_finite_difference_relative_error=relative,finite_difference_sensitivity=finite_differences,adjoint_relative_error=adjoint_error,
            iterations=len(history)-1,cg_residual=history,lambda_rgb=a.lambda_rgb,
            no_response_mse=float(y.square().mean()),linear=metrics(linear,delta),nonlinear_clipped=metrics(actual,delta),
            bounded_linear=metrics(bounded_linear,bounded),bounded_nonlinear=metrics(bounded_actual,bounded),
            bounded_scale_sweep=trust_region,bounded_best_scale=best_region,
            rendernet_local_direction=metrics(student_linear,student_delta),elapsed_seconds=time.monotonic()-start)
        result.append(row);write_json(output/'samples.json',result);print(json.dumps(row),flush=True)
    assert before==tensor_hash(teacher.state_dict())
    summary=dict(samples=result,mean={kind:{key:float(np.mean([r[kind][key] for r in result]))
        for key in result[0][kind] if key!='scale'} for kind in ['linear','nonlinear_clipped','bounded_linear','bounded_nonlinear','bounded_best_scale','rendernet_local_direction']},
        limitations='Finite-iteration, regularized local solve: success shows a local direction; failure is not an exact reachability limit. Uses original-image response baseline, separate from absolute teacher-zero mismatch. Bounds applied post-solve, not an exact constrained optimum.')
    write_json(output/'summary.json',summary)


if __name__=='__main__':main()
