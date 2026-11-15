"""Build plots/tables from completed read-only full-dataset reevaluation."""
from pathlib import Path
import argparse
import json
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


STEPS = (0,1000,2800,3000)
SCOPES = dict(train='训练全量', validation_internal='内部验证',
              validation_all='验证全量（含保留）', retained_5_26='保留 5、26')


def fmt(x):
    if x is None:
        return '—'
    return f'{x:.6g}'


def percent(x):
    return '—' if x is None else f'{100*x:+.3f}%'


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *['|'+'|'.join(map(str,row))+'|' for row in rows]])


def create_report(root, document):
    payload=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    summaries=payload['summary']
    def get(step,scope,phase='contact'):
        return summaries[str(step)][scope][phase]
    fig, axes=plt.subplots(2,3,figsize=(15,8.3),layout='constrained')
    curves=[('train','Train (56 episodes)','#2468aa','-'),
            ('validation_internal','Internal validation (4 episodes)','#db7525','-'),
            ('retained_5_26','Retained 5, 26 (evaluation only)','#6e6e6e',':')]
    specifications=[('contact','predictor_episode_macro_error_zero_ratio','Contact: Predictor error / own zero energy'),
        ('contact','teacher_episode_macro_error_zero_ratio','Contact: Teacher error / own zero energy'),
        ('precontact','predictor_episode_macro_error_zero_ratio','Precontact: Predictor error / own zero energy'),
        ('free','response_rms','Free: actual response RMS (absolute)'),
        ('contact','amplitude_ratio_teacher','Contact: actual / Teacher active RMS'),
        ('contact','rgb_rms','Contact: RGB RMS relative to original')]
    for ax,(phase,key,title) in zip(axes.flat,specifications):
        for scope,label,color,style in curves:
            ax.plot(STEPS,[get(s,scope,phase)[key] for s in STEPS],label=label,
                    color=color,linestyle=style,marker='o',markersize=4,linewidth=1.7)
        if 'error_zero_ratio' in key:
            ax.axhline(1,color='#aaaaaa',linewidth=.8,linestyle='--')
            ax.set_ylabel('Equal-episode macro error / zero-response energy')
        if phase=='free':
            ax.set_yscale('log')
        ax.set_title(title,fontsize=10.5)
        ax.set_xticks(STEPS,['0','1k','2.8k','3k'],rotation=40)
        ax.set_xlabel('Renderer step (EMA only)')
        ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
    fig.suptitle('Matched 150k Teacher: complete-dataset FP32 convergence reevaluation',fontsize=15)
    fig.savefig(root/'learning_curves.png',dpi=180)
    fig.savefig(root/'learning_curves.svg')
    plt.close(fig)
    # Flat summary is convenient for direct comparison without interpreting plots.
    flat=[dict(step=s,scope=scope,phase=phase,**metrics)
          for s in STEPS for scope,phases in summaries[str(s)].items() for phase,metrics in phases.items()]
    with (root/'learning_curve_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# CIVI：150k 匹配教师的全数据收敛复评','',
        '2026-11-15；`V / 2443e63e047aff9ba178bc11e0834e5f8205fe60`。仅已有 checkpoint 的 EMA 权重推理；**0 次 optimizer 更新，0 次 backward**。保留 Mixed、原网络和全部冻结依赖、门控及 epsilon=0.1 约束。','',
        '## 结论：A，训练与内部验证仍总体同步改善，末段明显放缓','',
        '**现有完整数据更支持 A；不支持 B，也还不足以将这条轨迹判为 C（两边均已基本平台化）。** Contact 的 Predictor 和真实 Teacher 归一化误差，在训练集与内部验证的四个现存节点均整体下降。这里的“同步”指两条汇总曲线方向一致，不代表相同速度，也不代表每个 episode 的 Teacher 误差都单调下降。','',
        '1,000→3,000 步，逐 episode 零能量归一化误差宏均的相对下降：Predictor 为训练 **4.815%**、内部验证 **1.521%**；Teacher 为训练 **2.458%**、内部验证 **0.694%**。内部验证仍改善，但速度明显低于训练；不能把这种速度差直接称为验证恶化。','',
        '2,800→3,000 步，逐 episode 配对误差相对下降的宏均：Predictor 为训练 **0.253%**、内部验证 **0.122%**；Teacher 为训练 **0.101%**、内部验证 **0.063%**。Predictor 改善覆盖训练 **53/56**、内部验证 **4/4** 个 episode；Teacher 为 **49/56**、**3/4**，内部 Episode 53 的 Teacher 误差微升 **0.0214%**。末段收益很小，因此不把仍有下降等同于未来会有显著收益。','',
        'Precontact 的 Predictor 和 Teacher 归一化误差在训练及内部验证也总体下降。Free 实际响应并非随训练全程下降：step0→3000，训练实际 MSE **6.70435e-6→9.82899e-6（+46.606%）**，内部验证 **2.53842e-6→3.59015e-6（+41.432%）**；最后 200 步分别回落 **2.557%／1.995%**。这项代价与 Contact 收益一并保留，不能称各阶段所有指标均改善，也不据此重启阶段 Loss 权重搜索。','',
        '末段 Contact RGB 总 RMS 反而小降：训练 **0.00400669→0.00393882（-1.694%）**，内部验证 **0.00328313→0.00326762（-0.472%）**；预测和 Teacher 误差仍下降。该末段收益不是由更大的总体 RGB 修改量直接解释，且本轮没有做学习率、步长或 RMS 追平。','',
        '**关于继续训练的证据：**当前仅有 **3.351 次 Contact 样本曝光**，完整训练/内部验证及多数 episode 仍存在小幅同向改善，支持“保留现有 Mixed 配置仍可能获得小幅收益”，不支持“已证明充分收敛”。但四个稀疏保存节点、4 个内部验证 episode 和很小的末段增益，不足以保证延长训练的收益或预算回报。本轮只提交这份证据，**没有延长训练，没有调 Loss，没有开启结构或新训练实验**。Episode 5、26 不参与上述 A/B/C 判断和模型选择。','',
        '## 统一学习曲线','',
        '![统一学习曲线](../data/diagnostics/civi_student_iteration/final150k_convergence_fp32/learning_curves.png)','',
        'Contact 和 Precontact 的主要比较采用逐 episode 的“误差 / 该 episode 自身零响应目标能量”，再对 episode 等权宏平均。虚线 1 表示不修改 RGB；训练与验证的原始 MSE 仅作绝对量报告，不用于跨分布判断泛化。保留 Episode 5、26 仅描述，不参与选择节点或主结论。','',
        '## 完整覆盖与可用节点','',
        table(['范围','Free','Precontact','Contact','Episode'],[
            [label,*[get(0,scope,p)['samples'] for p in ('free','precontact','contact')],get(0,scope)['episodes']]
            for scope,label in SCOPES.items()]),'',
        table(['Renderer 步','实际文件','Contact 累计抽样数','不同 Contact 窗口','Contact 曝光 = 抽样数 / 3581'],[
            [n['step'],'`'+n['path']+'`',e['contact_draws'],e['contact_unique'],f"{e['contact_exposure']:.6f}"]
            for n,e in zip(protocol['nodes'],protocol['exposure'])]),'',
        '只有 0、1000、2800、3000 步参数可复评。200、400、600、800 及 1200、1400、1600、1800、2000、2200、2400、2600 步只有历史指标，没有保存权重，未补训、未在曲线上伪造节点。3000-run 的 `initial.pt` 不存在，使用同一轨迹 1000-run 保存的真实 step0。1000 的 best/latest EMA 相同，selected_eval 是 step2800 副本，均不重复算节点。','',
        '曝光由两份实际 `train_batches.npy` 前缀与训练集 phase 标签交叉索引计数；1000-run 与 3000-run 前 1000 批完全一致，历史续跑记录和训练代码确认无重复回放。阶段累计抽样顺序 Free/Precontact/Contact：1000 步 2000/2000/4000，2800 步 5600/5600/11200，3000 步 6000/6000/12000。旧教师 RGB warm start、Oracle、其他种子或结构、Adapter/Condition 校准、180 次局部更新和 40 次 Mixed 探针均不计入该轨迹当前教师曝光；step0 定义为本轮 150k 教师实验的起点。','',
        '## 数值口径与参考定义','',
        '- FP32 覆盖 Renderer、门控/Condition、DINO、投影、Adapter、context_norm、Predictor、Teacher 原生上下文融合；禁用 autocast、TF32、随机增强、Dropout，使用确定性算法和 math SDP。模块参数与输出实测均为 FP32，位置参数输入的实测 dtype 见 `summary.json:precision`；DINO/Teacher 接口使用关键字参数，当前 hook 的输入栏为空，其 FP32 输入由显式 `.float()` 图像及 FP32 归一化路径保证，不把空栏伪称为实测。',
        '- 所有节点、原图读出、Teacher 非零/物理零扭矩读出均固定 encode batch=8（两时刻、两摄像头），最后一批复制末项补齐，仅统计真实窗口。完整训练集 17,740 项，完整验证集 1,893 项，每项恰好一次。',
        '- `E` 为冻结 DINO + 投影/嵌入 + 已校准 Adapter + context_norm；`dS=E(rendered)-E(original)`。从原图的 FP32 visual token 重新计算并共用 dP；原生 Teacher 两次 FP32 `_encode_context` 的差为 dT。历史 BF16 visual/cache 和 FP16 Teacher 缓存不参与指标。',
        '- MSE 覆盖全部 784×256 元素；不乘 alpha、不筛有效 Patch。Patch 余弦与 active 幅度统一用本批 FP32 dT 的 Patch RMS>0.001 掩码，对 dP/dT 都用同一掩码；余弦按有效 Patch 汇总。实际响应 RMS 同时报告全元素绝对量，避免只看掩码。',
        '- 逐 episode 的归一化比值为 `mean_window(MSE(dS,target)) / mean_window(MSE(0,target))`，然后对 episode 等权平均；P、T 各用自己的目标能量。能量为 0 的比值记未定义，不用 epsilon 伪造。',
        '- **相对原图/不修改 RGB**：基线 dS=0，误差为目标响应能量，改善 `1-error/energy`。**相对初始 EMA step0**：改善 `1-error/error_step0`；宏平均先逐 episode 做该配对比值，再等权平均。',
        '- 上一轮局部诊断以 step2800 为初始 checkpoint，因此另列 `vs step2800`，与本次收敛轨迹的实验起点 step0 分开。RGB 总 RMS 相对原图；更新 RMS 分别相对 step0 和 step2800 输出。',
        '- Free 实际响应直接报告 `mean(dS²)` 及其平方根；不把 Predictor 误差或 Teacher 误差当作实际响应。全量中并非每个 Free dT 都精确为 0，保留实测能量；初次评估因错误的全零断言停止，修正后完整重跑，未产生模型更新。','',
        '## Contact：逐 episode 零响应能量归一化（主判断）','',
        table(['范围','步','P 误差/零能量宏均','T 误差/零能量宏均','P 相对原图改善','T 相对原图改善'],[
            [label,s,fmt(m['predictor_episode_macro_error_zero_ratio']),fmt(m['teacher_episode_macro_error_zero_ratio']),
             percent(1-m['predictor_episode_macro_error_zero_ratio']),percent(1-m['teacher_episode_macro_error_zero_ratio'])]
            for scope,label in SCOPES.items() for s in STEPS for m in [get(s,scope)]]),'',
        '## Contact：绝对误差、余弦、幅度与 RGB','',
        table(['范围','步','P MSE','T MSE','cos P/T','实际响应 RMS','目标 RMS P/T','active 幅度比 P/T','RGB 总 RMS','RGB Δstep0 / Δ2800 RMS'],[
            [label,s,fmt(m['predictor_mse']),fmt(m['teacher_mse']),fmt(m['cosine_predictor'])+'/'+fmt(m['cosine_teacher']),
             fmt(m['response_rms']),fmt(m['predictor_rms'])+'/'+fmt(m['teacher_rms']),
             fmt(m['amplitude_ratio_predictor'])+'/'+fmt(m['amplitude_ratio_teacher']),fmt(m['rgb_rms']),
             fmt(m['rgb_from_step0_rms'])+'/'+fmt(m['rgb_from_step2800_rms'])]
            for scope,label in SCOPES.items() for s in STEPS for m in [get(s,scope)]]),'',
        '## Contact：相对初始 checkpoint 的改善（与相对原图分开）','',
        '以下为逐 episode 配对的改善率宏平均；正值改善，负值恶化。完整 CSV 另有按窗口加权的整体相对改善，两者定义不混用。','',
        table(['范围','步','P vs step0','T vs step0','P vs step2800','T vs step2800'],[
            [label,s,*[percent(m[f'{t}_episode_macro_improvement_vs_step{a}']) for a in (0,2800) for t in ('predictor','teacher')]]
            for scope,label in SCOPES.items() for s in STEPS for m in [get(s,scope)]]),'',
        '## Precontact（全部窗口）','',
        table(['范围','步','P MSE','T MSE','P/T 零能量比例宏均','余弦 P/T','实际 / P / T RMS','RGB 总 / Δstep0 RMS'],[
            [label,s,fmt(m['predictor_mse']),fmt(m['teacher_mse']),
             fmt(m['predictor_episode_macro_error_zero_ratio'])+'/'+fmt(m['teacher_episode_macro_error_zero_ratio']),
             fmt(m['cosine_predictor'])+'/'+fmt(m['cosine_teacher']),
             '/'.join(fmt(m[k]) for k in ('response_rms','predictor_rms','teacher_rms')),
             fmt(m['rgb_rms'])+'/'+fmt(m['rgb_from_step0_rms'])]
            for scope,label in SCOPES.items() for s in STEPS for m in [get(s,scope,'precontact')]]),'',
        '## Free（全部窗口，实际响应绝对值和相对变化）','',
        table(['范围','步','实际响应 MSE','实际响应 RMS','MSE vs step0','RMS vs step0','P MSE','T MSE','dT 能量','RGB 总 / Δstep0 RMS'],[
            [label,s,fmt(m['response_energy']),fmt(m['response_rms']),
             percent(m['response_energy']/get(0,scope,'free')['response_energy']-1),
             percent(m['response_rms']/get(0,scope,'free')['response_rms']-1),
             fmt(m['predictor_mse']),fmt(m['teacher_mse']),fmt(m['teacher_energy']),
             fmt(m['rgb_rms'])+'/'+fmt(m['rgb_from_step0_rms'])]
            for scope,label in SCOPES.items() for s in STEPS for m in [get(s,scope,'free')]]),'',
        '## 末段逐 episode 变化','',
        '下表将 2800→3000 的每个 episode 自身误差变化配对比较；未按保留集重新选模型。','']
    episodes=json.loads((root/'episode_metrics.json').read_text())
    look={(r['step'],r['scope'],r['phase'],r['episode']):r for r in episodes}
    rows=[]
    for scope in SCOPES:
        subset=[r for r in episodes if r['step']==3000 and r['scope']==scope and r['phase']=='contact']
        change={t:[r[t+'_mse']/look[(2800,scope,'contact',r['episode'])][t+'_mse']-1 for r in subset]
                for t in ('predictor','teacher')}
        rows.append([SCOPES[scope],len(subset),
                     *[f"{sum(v<0 for v in change[t])}/{len(subset)}；中位 {percent(float(np.median(change[t])))}" for t in ('predictor','teacher')]])
    lines.extend([table(['范围','Episode 数','P 下降数；误差相对变化中位','T 下降数；误差相对变化中位'],rows),'',
        '## 复现与原始文件','',
        '独立脚本：`test/reevaluate_civi_convergence.py`；制图/表脚本：`test/report_civi_convergence.py`。正式网络文件未修改。逐窗口 `windows_ema_{0,1000,2800,3000}.csv`；逐 episode `episode_metrics.json`；所有阶段/范围及两种初始参考 `learning_curve_metrics.csv`、`summary.json`；依赖 SHA、节点 EMA hash、采样账本和精度策略 `protocol.json`。','',
        f"完成用时 {payload['elapsed_seconds']:.1f} 秒；峰值 CUDA 已分配内存 {payload['peak_cuda_memory_bytes']/2**30:.3f} GiB。结束时重新核对全部 checkpoint 文件及 EMA tensor hash、Teacher/Adapter/Condition/torque/Predictor hash 未变；所有参数无梯度。没有保存或覆盖模型 checkpoint。",''])
    document.write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/diagnostics/civi_student_iteration/final150k_convergence_fp32'))
    p.add_argument('--report',type=Path,default=Path('docs/civi_convergence_fp32.md'))
    a=p.parse_args();create_report(a.root,a.report)


if __name__=='__main__':
    main()
