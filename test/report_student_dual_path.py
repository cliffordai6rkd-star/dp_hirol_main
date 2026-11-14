"""Tables, phase curves, gradient diagnostics and paired renderer previews."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from student_dual_path_common import VARIANTS,write_json

LABELS=['A: 6SA + 2CA / Linear','B: 2SA-CA-2SA-CA / Linear',
        'C: Interleaved + FiLM / Linear','C: Interleaved + FiLM / MLP128']
COLORS=['tab:blue','tab:orange','tab:green','tab:red']

def report(root,calibration_root):
    summaries={v:json.loads((root/v/'summary.json').read_text()) for v in VARIANTS}
    histories={v:json.loads((root/v/'log.json').read_text()) for v in VARIANTS}
    for key in ['shared_initialization_sha256','condition_sha256']:
        assert len({s['manifest'][key] for s in summaries.values()})==1,key
    assert len({s['optimizer_steps'] for s in summaries.values()})==1
    assert len({s['manifest']['adapter_sha256'] for v,s in summaries.items() if v!='c_film_mlp'})==1
    records=[]
    identity=json.loads((calibration_root/'identity_baseline.json').read_text())
    for v in VARIANTS:
        s=summaries[v];row=dict(variant=v,optimizer_steps=s['optimizer_steps'],**s['final']['all'])
        for phase in ['free','precontact','contact']:
            row.update({phase+'_'+k:x for k,x in s['final'][phase].items()})
        row['calibration_validation_mse']=json.loads((calibration_root/('calibration_mlp_cuda' if v=='c_film_mlp'
            else 'calibration_linear_cuda')/'summary.json').read_text())['final_validation_mse']
        initial=next(r for r in histories[v] if r['kind']=='validation' and r['step']==0)['metrics']
        row['initial_objective']=initial['all']['objective'];row['initial_contact_token_mse']=initial['contact']['token_mse']
        row['identity_objective']=identity['mlp' if v=='c_film_mlp' else 'linear']['all']['objective']
        row['renderer_gain_vs_identity_pct']=100*(1-row['objective']/row['identity_objective'])
        row['contact_teacher_response_mse']=identity['linear']['contact']['teacher_response_mse']
        row['contact_response_error_vs_no_student_response']=row['contact_response_difference_mse']/row['contact_teacher_response_mse']
        records.append(row)
    write_json(root/'comparison.json',records)
    keys=['calibration_validation_mse','free_token_mse','contact_token_mse','contact_response_difference_mse',
          'rgb_residual_mae','objective']
    table='| Variant | Zero calibration MSE | Free Token MSE | Contact Token MSE | Contact response difference MSE | RGB residual MAE | Global+weighted objective |\n'
    table+='|---|---:|---:|---:|---:|---:|---:|\n'
    for label,row in zip(LABELS,records):table+='| '+label+' | '+' | '.join(f'{row[k]:.6f}' for k in keys)+' |\n'
    (root/'comparison.md').write_text(table)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for v,label,color in zip(VARIANTS,LABELS,COLORS):
        rows=[r for r in histories[v] if r['kind']=='validation']
        for ax,(phase,key) in zip(axes.flat,[('all','objective'),('free','token_mse'),('contact','token_mse'),
            ('contact','response_difference_mse'),('all','rgb_residual_mae'),('precontact','token_mse')]):
            ax.plot([r['step'] for r in rows],[r['metrics'][phase][key] for r in rows],marker='o',label=label,color=color)
            ax.set(title=f'{phase}: {key}',xlabel='RGB optimizer updates');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for v,label,color in zip(VARIANTS,LABELS,COLORS):
        rows=[r for r in histories[v] if r['kind']=='train']
        ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],label=label,color=color,alpha=.8)
    ax.set(title='Training objective: logged batches (global + response-weighted MSE)',
           xlabel='RGB optimizer updates',ylabel='Loss');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'train_loss_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for kind,color in [('linear','tab:blue'),('mlp','tab:red')]:
        rows=json.loads((calibration_root/f'calibration_{kind}_cuda'/'log.json').read_text())
        for ax,split in zip(axes,['train','validation']):
            ax.plot([r['step'] for r in rows],[r[split+'_zero_mse'] for r in rows],label=kind,color=color)
            ax.set(title=split+' zero-torque calibration',xlabel='Adapter optimizer updates',ylabel='MSE');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'adapter_calibration.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for v,label,color in zip(VARIANTS,LABELS,COLORS):
        rows=[r for r in histories[v] if r['kind']=='train']
        for ax,name in zip(axes.flat,['cnn','sa','ca','film','lstm','spatial']):
            ys=[r['diagnostics'][name+'_gradient_norm'] for r in rows]
            if any(y>0 for y in ys):ax.plot([r['step'] for r in rows],ys,label=label,color=color)
            ax.set(title=name+' gradient L2 before clipping',xlabel='RGB updates',yscale='log');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'gradients.png',dpi=170);plt.close(fig)
    probes={v:torch.load(root/v/f"renders_step{summaries[v]['optimizer_steps']}.pt",map_location='cpu',weights_only=False)
            for v in VARIANTS}
    for phase,name in [(0,'free'),(1,'precontact'),(2,'contact')]:
        selected={v:next(p for p in ps if p['phase']==phase) for v,ps in probes.items()}
        assert len({p['index'] for p in selected.values()})==1
        vmax=max(float((p['rendered'][k]-p['images'][k]).abs().mean(1).max()) for p in selected.values() for k in p['images'])
        fig,axes=plt.subplots(4,6,figsize=(17,11),constrained_layout=True)
        for row,(v,label) in enumerate(zip(VARIANTS,LABELS)):
            p=selected[v]
            for col,k in enumerate(sorted(p['images'])):
                x=p['images'][k][-1].float();y=p['rendered'][k][-1].float()
                for j,img in enumerate([x.permute(1,2,0),y.permute(1,2,0),(y-x).abs().mean(0)]):
                    ax=axes[row,3*col+j]
                    if j==2:im=ax.imshow(img,cmap='magma',vmin=0,vmax=max(vmax,1e-8))
                    else:ax.imshow(img.clamp(0,1))
                    ax.set_title((label if col==0 and j==0 else '')+'\n'+k+': '+['original','rendered','|RGB delta|'][j],fontsize=8);ax.axis('off')
        fig.colorbar(im,ax=axes,shrink=.6);fig.suptitle(name+': identical sample, shared residual color scale')
        fig.savefig(root/f'render_{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    step=records[0]['optimizer_steps']
    text=f'四组学生统一比较第 {step:,} 步 RGB 训练的 EMA。完整验证覆盖原划分的 1,893 个窗口（Free 1,317、Precontact 108、Contact 468）。\n\n'
    if step!=5000:
        text+='按用户缩小实验预算的要求终止剩余训练。A/C MLP 此前已完成 5,000 步，B/C Linear 在约 1,500 步停止；本报告仅比较四组共同的第 1,000 步 checkpoint，复用已执行的完整验证数据，不执行额外优化。\n\n'
    text+='教师固定为原 2026.11.12 的 180,000 步 EMA；本机新训练教师未用于本次对照。原 Global Token MSE + Response-weighted MSE、region threshold=0.2、随机 RGB 输出初始权重、Condition Network、数据和随机种子保持一致。历史 CUDA 解码像素与成对目标在 12 个诊断样本上复现。\n\n'
    text+=table+'\n'
    linear=json.loads((calibration_root/'calibration_linear_cuda/summary.json').read_text())
    mlp=json.loads((calibration_root/'calibration_mlp_cuda/summary.json').read_text())
    text+=f"Adapter 均从零残差输出开始，在相同 2,000 步、batch=64 的零力矩校准后冻结。Linear 的完整验证 MSE 为 {linear['final_validation_mse']:.6f}，MLP128 为 {mlp['final_validation_mse']:.6f}，MLP 相对变化为 {(mlp['final_validation_mse']/linear['final_validation_mse']-1)*100:+.2f}%。两者参数量分别为 {linear['parameters']:,}、{mlp['parameters']:,}。这只衡量当前预算的校准结果，不代表两个函数族各自的最优拟合上限。\n\n"
    text+='为隔离 Adapter 自身的收益，还比较了相同冻结 Adapter 直接处理原图的基线：\n\n| Variant | Random-render initial objective | Original RGB + same adapter | Final objective | Renderer improvement over original RGB |\n|---|---:|---:|---:|---:|\n'
    for label,row in zip(LABELS,records):
        text+=f"| {label} | {row['initial_objective']:.6f} | {row['identity_objective']:.6f} | {row['objective']:.6f} | {row['renderer_gain_vs_identity_pct']:+.2f}% |\n"
    text+='\n从随机渲染初始值降下来的全部幅度，不能都解释为学会了力矩响应；上表最后一列才隔离了渲染器相对原图的额外收益。\n\n'
    by={r['variant']:r for r in records}
    for left,right,name in [('a_baseline','b_interleaved','A→B（减少两层 SA 并交错 CA）'),
                            ('b_interleaved','c_film_linear','B→C Linear（只增加 Decoder FiLM）'),
                            ('c_film_linear','c_film_mlp','C Linear→C MLP（只改变 Adapter）')]:
        x,y=by[left],by[right]
        text+=f"- {name}：Contact Token MSE 变化 {(y['contact_token_mse']/x['contact_token_mse']-1)*100:+.2f}%，Contact Response Difference MSE 变化 {(y['contact_response_difference_mse']/x['contact_response_difference_mse']-1)*100:+.2f}%，Free Token MSE 变化 {(y['free_token_mse']/x['free_token_mse']-1)*100:+.2f}%。\n"
    text+='\nTorque-response Difference 只用于诊断，不进入 Loss。它比较完整学生的 Normal/Physical-zero 输出差与教师的对应差，学生在干预时重新预测区域和 alpha。较大的 RGB 残差或较低的静态 Token MSE 本身不能证明力矩内容得到了更充分的利用。\n\n'
    text+=f"如果学生完全不响应力矩，该差分误差等于教师自身响应 MSE；本验证集 contact 的该基线为 {records[0]['contact_teacher_response_mse']:.6f}。四组最终误差相对这条基线分别为 "+' / '.join(f"{r['contact_response_error_vs_no_student_response']:.3f}×" for r in records)+'（A / B / C Linear / C MLP）。低于 1 才表示在该差分指标上优于完全无响应学生。\n\n'
    x,y=by['a_baseline'],by['c_film_mlp']
    promising=(y['contact_token_mse']<x['contact_token_mse'] and
                y['contact_response_difference_mse']<x['contact_response_difference_mse'] and
                y['renderer_gain_vs_identity_pct']>0 and y['contact_response_error_vs_no_student_response']<1)
    text+=('最终 C MLP 在接触 Token 拟合和响应差误差上均优于 A，且渲染相对同 Adapter 原图有额外收益，值得作为后续完整训练的候选。'
           if promising else '当前结果没有同时满足接触 Token 拟合、力矩响应拟合以及渲染额外收益三项改善，暂不建议直接据此启动完整训练。')
    text+='这是一轮单 seed 的短程对照，不能据微小差异认定稳定优势；A/B 同时改变 SA 数量与顺序，不能称为纯交错位置消融。\n\n'
    text+='[校准曲线](adapter_calibration.png) · [完整验证曲线（含初始点）](training_curves.png) · [训练日志曲线](train_loss_curves.png) · [梯度范数](gradients.png)\n\n'
    text+='[Free 渲染](render_free.png) · [Precontact 渲染](render_precontact.png) · [Contact 渲染](render_contact.png) · [机器可读对比表](comparison.json)\n'
    (root/'analysis.md').write_text(text)
    print(table)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--calibration-root',type=Path,required=True)
    a=p.parse_args();report(a.root,a.calibration_root)
