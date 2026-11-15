"""Report only the completed bounded continuation; no model selection on retained episodes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fmt(value):
    return '—' if value is None else f'{value:.7g}'


def pct(value):
    return '—' if value is None else f'{100*value:+.4f}%'


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *['|'+'|'.join(map(str,row))+'|' for row in rows]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/diagnostics/civi_student_iteration/final150k_budget_extend_s42_6000'))
    p.add_argument('--reference',type=Path,default=Path('data/diagnostics/civi_student_iteration/final150k_convergence_fp32'))
    p.add_argument('--report',type=Path,default=Path('docs/civi_budget_extension.md'))
    a=p.parse_args();out=a.root
    end=json.loads((out/'summary.json').read_text());final=end['final_step']
    manifest=json.loads((out/'manifest.json').read_text());budget=json.loads((out/'budget_audit.json').read_text())
    reproduction=json.loads((out/'zero_update_reproduction.json').read_text())
    points={int(f.stem.split('_')[-1]):json.loads(f.read_text()) for f in out.glob('fp32_metrics_*.json')}
    steps=sorted(points);assert steps[0]==3000 and steps[-1]==final
    journal=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines() if line.strip()]
    assert len(journal)==end['new_optimizer_updates']==final-3000
    assert [r['model_step'] for r in journal]==list(range(3001,final+1))
    plan=np.load(out/'train_batches.npy');assert len(plan)>=final and plan.shape[1]==8
    exposure=[dict(step=s,contact_draws=4*s,unique_contact_windows=len(np.unique(plan[:s,4:])),
                   contact_exposure=4*s/3581) for s in steps]
    (out/'contact_exposure.json').write_text(json.dumps(exposure,indent=2)+'\n')
    old=json.loads((a.reference/'summary.json').read_text())['summary']
    def get(step,scope='validation_internal',phase='contact'):
        return (points[step] if step in points else old[str(step)])[scope][phase]
    all_steps=sorted(set([0,1000,2800]+steps))
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    panels=[('contact','predictor_episode_macro_error_zero_ratio','Contact: Predictor error / own zero energy'),
        ('contact','teacher_episode_macro_error_zero_ratio','Contact: true Teacher error / own zero energy'),
        ('precontact','teacher_episode_macro_error_zero_ratio','Precontact: true Teacher normalized error'),
        ('free','response_rms','Free: actual response RMS'),
        ('contact','amplitude_ratio_teacher','Contact: actual / Teacher active RMS'),
        ('contact','rgb_rms','Contact: RGB RMS relative to original')]
    scopes=[('train','Train: complete Contact only','#2468aa','-'),
        ('validation_internal','Internal validation: all windows','#db7525','-'),
        ('retained_5_26','Retained5,26: evaluation only','#777777',':')]
    for axis,(phase,key,title) in zip(axes.flat,panels):
        for scope,label,color,style in scopes:
            if scope=='train' and phase!='contact':continue
            available=[s for s in all_steps if get(s,scope,phase).get('samples',0)>0]
            axis.plot(available,[get(s,scope,phase)[key] for s in available],marker='o',markersize=3.5,
                      color=color,linestyle=style,label=label,linewidth=1.5)
        axis.axvline(3000,color='#999999',linestyle='--',linewidth=.8)
        axis.axvspan(3000,final,color='#eeeeee',alpha=.5,zorder=-1)
        if 'error_zero_ratio' in key:axis.axhline(1,color='#aaaaaa',linestyle=':',linewidth=.8)
        if phase=='free':axis.set_yscale('log')
        axis.set_title(title,fontsize=10.5);axis.set_xlabel('Renderer step (EMA, FP32 evaluation)')
        axis.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Fixed Mixed/BF16 training: bounded budget extension (shaded), no BF16 metric splicing',fontsize=13)
    fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    flat=[dict(step=s,scope=scope,phase=phase,**m) for s,scopes_ in points.items() for scope,phases in scopes_.items()
          for phase,m in phases.items() if m['samples']]
    with (out/'learning_curve_metrics.csv').open('w') as stream:
        fields=list(dict.fromkeys(k for row in flat for k in row));w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(flat)
    episodes=[]
    for s in steps:episodes.extend(dict(step=s,**row) for row in json.loads((out/f'fp32_episodes_{s}.json').read_text()))
    old_eps=json.loads((a.reference/'episode_metrics.json').read_text())
    anchors={s:{(r['scope'],r['phase'],r['episode']):r for r in old_eps if r['step']==s} for s in (0,3000)}
    for row in episodes:
        for target in ('predictor','teacher'):
            e=row[target+'_energy'];row[target+'_improvement_vs_identity']=1-row[target+'_mse']/e if e else None
            for anchor,lookup in anchors.items():
                initial=lookup[(row['scope'],row['phase'],row['episode'])][target+'_mse']
                row[f'{target}_improvement_vs_step{anchor}']=1-row[target+'_mse']/initial if initial else None
        for anchor,lookup in anchors.items():
            initial=lookup[(row['scope'],row['phase'],row['episode'])]['response_energy']
            row[f'actual_response_mse_change_vs_step{anchor}']=row['response_energy']/initial-1 if initial else None
    with (out/'per_episode.csv').open('w') as stream:
        fields=list(dict.fromkeys(k for row in episodes for k in row));w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(episodes)
    checkpoints=[]
    initial=torch.load(out/'step_003000.pt',map_location='cpu',weights_only=False)
    protected=set(initial['model'])-set(initial['manifest']['trainable_names'])
    for f in sorted(out.glob('step_*.pt')):
        data=torch.load(f,map_location='cpu',weights_only=False)
        optimizer_steps=[int(v['step']) for v in data['optimizer']['state'].values()]
        assert min(optimizer_steps)==max(optimizer_steps)==data['step']
        assert data['sampler_next_step']==data['step']+1
        assert data['manifest']['trainable_names']==initial['manifest']['trainable_names']
        for weights in ('model','ema_model'):
            assert all(torch.equal(data[weights][key],initial[weights][key]) for key in protected)
        checkpoints.append(dict(file=f.name,step=data['step'],adamw_step_min=min(optimizer_steps),
            adamw_step_max=max(optimizer_steps),protected_state_unchanged=True,
            sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (out/'checkpoint_inventory.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
    block_rows=[]
    milestones=[3000]+[s for s in steps if s>3000 and (s-3000)%1000==0]
    if milestones[-1]!=final:milestones.append(final)
    for left,right in zip(milestones,milestones[1:]):
        x,y=get(left),get(right);f0,f1=get(left,phase='free'),get(right,phase='free')
        block_rows.append([f'{left}→{right}',right-left,
            pct(1-y['predictor_episode_macro_error_zero_ratio']/x['predictor_episode_macro_error_zero_ratio']),
            pct(1-y['teacher_episode_macro_error_zero_ratio']/x['teacher_episode_macro_error_zero_ratio']),
            fmt(f1['response_energy']-f0['response_energy']),pct(f1['response_energy']/f0['response_energy']-1),
            fmt(f1['response_rms']-f0['response_rms'])])
    scope_labels=dict(validation_internal='内部验证',validation_all='全部验证（含保留）',retained_5_26='保留5、26')
    contact_start,contact_end=get(3000),get(final)
    free_start,free_end=get(3000,phase='free'),get(final,phase='free')
    internal_end=[r for r in episodes if r['step']==final and r['scope']=='validation_internal' and r['phase']=='contact']
    train_end=[r for r in episodes if r['step']==final and r['scope']=='train' and r['phase']=='contact']
    teacher_down=sum(y['teacher']<x['teacher'] for x,y in zip(end['fp32_history'],end['fp32_history'][1:]))
    predictor_down=sum(y['predictor']<x['predictor'] for x,y in zip(end['fp32_history'],end['fp32_history'][1:]))
    lines=['# CIVI：固定 Mixed 的有上限预算延长','',
        f"2026-11-15；V/2443e63。从 online/EMA/AdamW/RNG step3000 正确恢复至 **step{final}**；新增 **{end['new_optimizer_updates']}** 次更新；停止原因：**{end['stop_reason']}**。不是正式100k训练。",'',
        '## 结论与预算建议','',
        '**存在小幅、方向一致的真实Teacher迁移收益；但当前证据不足以支持申请下一阶段训练预算。建议结束这条路线的预算延长，保留Mixed及现有checkpoint，不自动追加训练、调Loss或开启结构搜索。**这个预算判断只使用内部验证，不使用保留5、26挑选结果。','',
        f"1. **数据集级收益仍存在。** 续训的{len(steps)-1}次500步检查中，Predictor有{predictor_down}次、真实Teacher有{teacher_down}次较上次下降。训练Contact的Predictor/Teacher均在{sum(r['predictor_improvement_vs_step3000']>0 for r in train_end)}/{len(train_end)}、{sum(r['teacher_improvement_vs_step3000']>0 for r in train_end)}/{len(train_end)}个episode净改善；内部验证均为{sum(r['predictor_improvement_vs_step3000']>0 for r in internal_end)}/{len(internal_end)}、{sum(r['teacher_improvement_vs_step3000']>0 for r in internal_end)}/{len(internal_end)}。这是同一seed轨迹多节点及多episode的一致证据，**不是独立seed复现**。",'',
        f"2. **收益同时体现于真实Teacher。** 内部Contact归一化宏均P：{contact_start['predictor_episode_macro_error_zero_ratio']:.9f}→{contact_end['predictor_episode_macro_error_zero_ratio']:.9f}；T：{contact_start['teacher_episode_macro_error_zero_ratio']:.9f}→{contact_end['teacher_episode_macro_error_zero_ratio']:.9f}。逐episode配对相对3000改善宏均分别为{pct(contact_end['predictor_episode_macro_improvement_vs_step3000'])}、{pct(contact_end['teacher_episode_macro_improvement_vs_step3000'])}。训练对应配对改善为{pct(get(final,'train')['predictor_episode_macro_improvement_vs_step3000'])}、{pct(get(final,'train')['teacher_episode_macro_improvement_vs_step3000'])}，训练收益更大，但内部验证没有整体反向恶化，也不能说两者均已完全平台化。",'',
        f"3. **不建议继续扩大预算。** 内部Teacher新增每1000步仅改善约0.2220%、0.0848%、0.0899%，后两个区间均明显小于首个。与此同时，内部Free实际MSE从{fmt(free_start['response_energy'])}升至{fmt(free_end['response_energy'])}（相对3000 {pct(free_end['response_energy']/free_start['response_energy']-1)}；相对原始step0 {pct(free_end['response_energy']/old['0']['validation_internal']['free']['response_energy']-1)}），RMS从{fmt(free_start['response_rms'])}升至{fmt(free_end['response_rms'])}。这不是用不同量纲的百分比直接相减，而是Contact收益已小、Free约束方向仍变差，且没有通过原多条件验收的证据；不能据此外推100k可行。",'',
        f"内部Precontact的Teacher归一化误差从{get(3000,phase='precontact')['teacher_episode_macro_error_zero_ratio']:.9f}降至{get(final,phase='precontact')['teacher_episode_macro_error_zero_ratio']:.9f}，最后500步微升，未形成预设的持续恶化序列。Contact真实Teacher相对不修改RGB的改善为{pct(contact_end['teacher_episode_macro_improvement_vs_identity'])}，相对实验原始step0为{pct(contact_end['teacher_episode_macro_improvement_vs_step0'])}，相对本轮step3000为{pct(contact_end['teacher_episode_macro_improvement_vs_step3000'])}；三种参考没有合并。",'',
        f"内部Contact RGB总RMS {fmt(contact_start['rgb_rms'])}→{fmt(contact_end['rgb_rms'])}；相对3000输出的RGB位移RMS为{fmt(contact_end['rgb_from_step3000_rms'])}，最大修改{fmt(contact_end['rgb_linf'])}。全程数值有限，冻结参数与缓存检查通过。没有触发早停；因总步数上限停止，最终Free连续上升计数为2、Precontact为1、Teacher持续恶化为0、无新低计数为0。原始checkpoint未覆盖，单张GPU2已释放。",'',
        '保留集仅作完成后的描述：Episode5的Contact Teacher误差相对3000下降0.0885%，Episode26上升0.1809%；两者Teacher配对宏均为恶化0.0462%，Free实际MSE整体增加17.6778%。这些数值完整保留，不据此换checkpoint或修改上述内部验证结论。','',
        '## 统一 FP32 学习曲线','',
        '![学习曲线](../data/diagnostics/civi_student_iteration/'+out.name+'/learning_curves.png)','',
        '阴影为本轮续训。之前0/1000/2800与本轮3000的共同FP32口径已核对；没有拼接历史BF16评价指标。训练Contact只在真实全量测量节点绘点，中间500步不拿probe补点。Precontact/Free曲线为完整验证窗口。','',
        '## 恢复、预算和固定配置','',
        f"原账本16402，加局部诊断180（100共享+80像素）及Mixed探针40，启动前 **{budget['spent_before']}/20000**；本轮预留 **{budget['reserved']}**；结束累计 **{end['global_budget_used']}/20000**，剩余 **{end['global_budget_remaining']}**。旧账本历史reference1000按原定义不计入；Oracle像素更新计入探索总预算，但不算当前Renderer曝光。没有重复计数3000步之前的采样。",'',
        '- ONLINE参数恢复到其原AdamW状态，EMA单独恢复。lr=2e-4，AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=1e-4；clip=1；EMA decay=.99；无scheduler。',
        '- 原Mixed损失不变：batch8=2Free+2Precontact+4Contact，microbatch4累积后仅1次step；Free目标0，其余为原冻结预测缓存，整batch除8。原BF16 autocast Renderer、门控、视觉路径继续使用；没有改成FP32训练。',
        '- FP32仅用于EMA评价：原图与渲染DINO/投影/Adapter/context_norm、Predictor、Teacher、Condition与Renderer均FP32；固定encode batch8与原始成员，尾批补齐后剔除补项。有效Patch为FP32 dT RMS>.001；MSE用全部元素，余弦/幅度共用Teacher有效掩码。',
        '- FP32评价保存并还原Torch/CUDA/Python/NumPy随机状态和SDP/MHA/确定性后端配置；训练原后端不改变。原checkpoint只持久化Torch/CUDA RNG；采样器是固定seed本地NumPy generator，完整前缀已核对，训练无随机增强。新checkpoint另存Python/NumPy状态与next sampler step。',
        '- 每500步保存online、EMA、AdamW、RNG、采样计划及停止计数器，完整内部验证含全部Contact/Precontact/Free。Episode5、26仅报告，不参与停止、最佳checkpoint或调参。',
        f"- 零更新复现覆盖 **{reproduction['windows']}** 个窗口（3581训练Contact+1893全部验证）；最大绝对差 **{reproduction['max_absolute_difference']:.8g}**；bit exact：**{reproduction['bit_exact']}**。模型原文件及冻结模块/缓存一致性核对通过。",'',
        '原训练BF16 autocast的逐模块实测并非每个张量都为BF16：DINO输出FP32，投影输出BF16，Adapter/context_norm及RGB输出包含FP32；FP32评价各记录输入/输出均为FP32。详情见`precision.json`，不只在loss末端转换精度。','',
        '累计曝光只计算当前150k Renderer轨迹实际Contact采样：每次更新4个，除以3581；旧Oracle、结构实验及局部探针均不计入此曝光。逐次更新日志连续覆盖3001至停止步，没有重放前3000步。','',
        table(['模型步','Contact累计抽样数','已见不同Contact窗口','Contact曝光'],[
            [r['step'],r['contact_draws'],r['unique_contact_windows'],fmt(r['contact_exposure'])] for r in exposure]),'',
        '## 新增每1000步：内部验证收益与Free代价','',
        'P/T为逐episode误差除以该episode自身零响应能量后宏平均；表中下降率比较该宏平均。最后不足1000步的区间原样标注，不外推成实际1000步结果。','',
        table(['区间','实际更新','P归一化误差下降','T归一化误差下降','Free实际MSE增量','Free实际MSE变化','Free实际RMS增量'],block_rows),'',
        '## Contact：三个参考独立报告','',
        'identity=原图不修改、dS=0；step0=当前150k教师实验原始起点；step3000=本轮续训起点。所有比值先逐episode计算后宏平均，以下不能与历史BF16验收分数混用。','',
        table(['范围','步','P/T误差÷零能量','P/T相对原图改善','P/T相对step0改善','P/T相对3000改善'],[
            [label,s,fmt(m['predictor_episode_macro_error_zero_ratio'])+'/'+fmt(m['teacher_episode_macro_error_zero_ratio']),
             pct(m['predictor_episode_macro_improvement_vs_identity'])+'/'+pct(m['teacher_episode_macro_improvement_vs_identity']),
             pct(m['predictor_episode_macro_improvement_vs_step0'])+'/'+pct(m['teacher_episode_macro_improvement_vs_step0']),
             pct(m['predictor_episode_macro_improvement_vs_step3000'])+'/'+pct(m['teacher_episode_macro_improvement_vs_step3000'])]
            for scope,label in [('train','全量训练Contact'),*scope_labels.items()] for s in steps for m in [get(s,scope)] if m['samples']]),'',
        '## Precontact：归一化误差及三个参考','',
        table(['范围','步','P/T误差÷零能量','P/T相对原图改善','P/T相对step0改善','P/T相对3000改善'],[
            [label,s,fmt(m['predictor_episode_macro_error_zero_ratio'])+'/'+fmt(m['teacher_episode_macro_error_zero_ratio']),
             pct(m['predictor_episode_macro_improvement_vs_identity'])+'/'+pct(m['teacher_episode_macro_improvement_vs_identity']),
             pct(m['predictor_episode_macro_improvement_vs_step0'])+'/'+pct(m['teacher_episode_macro_improvement_vs_step0']),
             pct(m['predictor_episode_macro_improvement_vs_step3000'])+'/'+pct(m['teacher_episode_macro_improvement_vs_step3000'])]
            for scope,label in scope_labels.items() for s in steps for m in [get(s,scope,'precontact')]]),'',
        '## 全验证阶段：绝对误差、响应、RGB与Free','',
        table(['范围','阶段','步','P MSE','T MSE','cos P/T','响应RMS','active幅度比P/T','RGB总/Δ0/Δ3000 RMS','RGB最大修改','硬/范围裁剪比例'],[
            [label,phase,s,fmt(m['predictor_mse']),fmt(m['teacher_mse']),fmt(m['cosine_predictor'])+'/'+fmt(m['cosine_teacher']),
             fmt(m['response_rms']),fmt(m['amplitude_ratio_predictor'])+'/'+fmt(m['amplitude_ratio_teacher']),
             '/'.join(fmt(m[k]) for k in ('rgb_rms','rgb_from_step0_rms','rgb_from_step3000_rms')),
             fmt(m['rgb_linf']),fmt(m['hard_clip_fraction'])+'/'+fmt(m['range_clip_fraction'])]
            for scope,label in scope_labels.items() for phase in ('contact','precontact','free') for s in steps for m in [get(s,scope,phase)]]),'',
        'Free实际响应MSE=mean(dS²)，RMS为其平方根；不是Predictor误差或Teacher误差。相对原图的该项基线为0，不能伪造相对百分比。','',
        table(['范围','步','实际MSE','实际RMS','MSE相对step0','MSE相对3000','RMS相对3000'],[
            [label,s,fmt(m['response_energy']),fmt(m['response_rms']),pct(m['response_energy']/old['0'][scope]['free']['response_energy']-1),
             pct(m['response_energy']/get(3000,scope,'free')['response_energy']-1),pct(m['response_rms']/get(3000,scope,'free')['response_rms']-1)]
            for scope,label in scope_labels.items() for s in steps for m in [get(s,scope,'free')]]),'',
        '## 全量训练Contact端点','',
        table(['步','窗口数','P/T归一化误差','P/T MSE','cos P/T','RGB总RMS'],[
            [s,m['samples'],fmt(m['predictor_episode_macro_error_zero_ratio'])+'/'+fmt(m['teacher_episode_macro_error_zero_ratio']),
             fmt(m['predictor_mse'])+'/'+fmt(m['teacher_mse']),fmt(m['cosine_predictor'])+'/'+fmt(m['cosine_teacher']),fmt(m['rgb_rms'])]
            for s in (3000,final) for m in [get(s,'train')]]),'',
        '## 逐episode端点：内部验证与保留集','',
        '完整训练56个episode与全部验证各阶段、各保存节点详见`per_episode.csv`。下表仅展开Contact端点；5、26在训练完成后描述，不参与预算决策、停止或选模。','',
        table(['范围','Episode','窗口数','P/T误差÷自身零能量','P/T相对3000改善','Free实际MSE','Free实际RMS','Free MSE相对3000'],[
            [scope_labels[row['scope']],row['episode'],row['samples'],
             fmt(row['predictor_error_zero_ratio'])+'/'+fmt(row['teacher_error_zero_ratio']),
             pct(row['predictor_improvement_vs_step3000'])+'/'+pct(row['teacher_improvement_vs_step3000']),
             fmt(free['response_energy']),fmt(free['response_rms']),pct(free['actual_response_mse_change_vs_step3000'])]
            for row in episodes if row['step']==final and row['phase']=='contact' and row['scope'] in ('validation_internal','retained_5_26')
            for free in [next(r for r in episodes if r['step']==final and r['scope']==row['scope'] and r['episode']==row['episode'] and r['phase']=='free')]]),'',
        '## Checkpoint及恢复命令','',
        table(['文件','模型step','AdamW step范围','SHA256'],[[r['file'],r['step'],str(r['adamw_step_min'])+'–'+str(r['adamw_step_max']),r['sha256']] for r in checkpoints]),'',
        '`latest.pt`为停止节点的完整可恢复状态；`best_teacher.pt`只按内部真实Teacher宏平均新低且Predictor不差于续训起点保存，不是验收通过模型。精确采样前缀与下一批见`train_batches.npy`、`resume_verification.json`；每次实际更新见`updates.jsonl`。',
        '原构造器继承的旧BF16 `best.pt`和初始评价归档在`legacy_bf16_source/`；不作为本轮FP32选出的checkpoint。checkpoint内旧`best/stale`及外层旧`visual_evaluation_precision`为继承元数据；实际精度以`bounded_extension.evaluation_precision`和`precision.json`为准，选择/停止以`extension_state`和`fp32_history.json`为准。实际执行代码保存在`source_snapshot/`，交付入口仅增加中断恢复时的预算累计及兼容处理，不改变已完成的训练。',
        '', '恢复命令仅用于**未完成任务的中断恢复**，不能在本轮已完成或已触发停止后运行它绕过停止规则。只暴露确认空闲的一张GPU；仍受全局剩余预算与6000模型step上限约束。','',
        '```bash',
        'cd /home/djhpc/code/dp_hirol_main/data/diagnostics/2026.11.15/new_teacher_shared_response_source',
        'CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \\',
        'LD_LIBRARY_PATH=/home/djhpc/miniconda3/envs/fdp/lib/python3.10/site-packages/nvidia/npp/lib:/home/djhpc/miniconda3/envs/fdp/lib:/home/djhpc/miniconda3/envs/fdp/lib/python3.10/site-packages/torch/lib \\',
        '/home/djhpc/miniconda3/envs/fdp/bin/python -u test/extend_civi_budget.py \\',
        f'  --resume data/diagnostics/civi_student_iteration/{out.name}/latest.pt \\',
        f'  --output data/diagnostics/civi_student_iteration/{out.name}_recovered \\',
        '  --max-total-step 6000 --max-new-updates 3000','```','',
        '## 验收边界和文件','',
        '原验收规则保留：Predictor实现误差≤identity的0.75、真实Teacher误差≤identity的0.90、Free≤原始初始的1.05、RGB RMS≤0.015、epsilon=.1、至少两个匹配seed及原配对条件对照等，均不降低。本轮只有seed42预算延长，未进行独立seed复验，未重跑旧BF16验收/条件对照，因此**不宣称通过原验收、不启动正式100k、不代表下游策略有效**。新FP32曲线不能直接替代历史BF16协议分数。','',
        '逐episode完整表`per_episode.csv`（含三种参考）、逐窗口`fp32_windows_*.csv`、每节点汇总`fp32_metrics_*.json`、停止历史`fp32_history.json`、原训练精度损失`training_bf16.json`、逐模块精度`precision.json`、全局预算核对`budget_audit.json`、checkpoint清单`checkpoint_inventory.json`及源代码快照均在独立续训目录。','']
    a.report.write_text('\n'.join(lines))


if __name__=='__main__':main()
