"""Plot and summarize the controlled torque-residual experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/diagnostics/tau_capacity_v2'))
    args=parser.parse_args();root=args.root
    models=('mlp','mlp_no_tau','transformer','dit_flow')
    labels={'mlp':'MLP + torque','mlp_no_tau':'MLP, images only',
            'transformer':'Transformer (regression)','dit_flow':'DiT (flow, 16 steps)'}
    results={key:[json.loads(p.read_text()) for p in sorted((root/'matched_width256').glob(f'{key}_*_result.json'))]
             for key in models}
    # Avoid mlp glob picking up mlp_no_tau files.
    results={key:[x for x in value if x['kind']==key] for key,value in results.items()}
    five=json.loads((root/'five_contact/five_contact_results.json').read_text())
    rgb=json.loads((root/'five_contact_rgb/summary.json').read_text())
    ext_path=root/'dit_extended/dit_flow_42_result.json'
    extended=json.loads(ext_path.read_text()) if ext_path.exists() else None
    summary={'baseline_test_mse':results['mlp'][0]['test']['energy'],
             'test_results':{},'five_contact_fits':{},'limits':[
                 'Held out from student probes only; teacher trained on all episodes.',
                 'Three held-out episodes, stable contact only; no action success-rate claim.',
                 'Direct token models bypass RGB; they are not deployable RGB replacements.',
                 'Equal 2000 updates for main models, not equal walltime or inference cost.',
                 'Five-contact RGB checkpoint and zero-pixel initializations/step budgets differ; feasibility only.',
                 'No result is a mathematical model capacity bound.'
             ]}
    for kind,runs in results.items():
        summary['test_results'][kind]={'params':runs[0]['params'],'seeds':[r['seed'] for r in runs],
            **{metric:{'mean':float(np.mean([r['test'][metric] for r in runs])),
                       'range':[min(r['test'][metric] for r in runs),max(r['test'][metric] for r in runs)]}
               for metric in ('mse','relative_mse','top10_relative_mse','top10_cosine')}}
    for kind,result in five['results'].items():
        summary['five_contact_fits'][kind]=result['history'][-1]
    for kind,result in rgb.items():
        summary['five_contact_fits'][kind]=result['final']
    if extended:
        summary['extended_dit_test']={**extended['test'],'steps':extended['steps'],
                                      'best_validation_step':extended['best_validation_step']}
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
    for kind,result in five['results'].items():
        h=result['history']
        axes[0,0].semilogy([r['step'] for r in h],[r['relative_mse'] for r in h],label=labels[kind])
    axes[0,0].set(title='A. Five REAL contact windows: fit capacity',xlabel='Optimizer updates',ylabel='Residual MSE / zero baseline')
    for kind,result in rgb.items():
        h=result['history']
        axes[0,1].plot([r['step'] for r in h],[r['relative_mse'] for r in h],
                       label='Frozen-condition RGB' if kind=='rgb' else 'Free RGB pixels (oracle)')
    axes[0,1].set(title='B. Same five windows: RGB feasibility',xlabel='Optimizer updates (5-window accumulation)',ylabel='Residual MSE / zero baseline')
    positions=np.arange(len(models));width=.34
    for shift,key,label in ((-width/2,'relative_mse','All patches'),(width/2,'top10_relative_mse','Top 10% effect patches')):
        values=np.array([[r['test'][key] for r in results[k]] for k in models])
        means=values.mean(1)
        axes[1,0].bar(positions+shift,means,width,label=label)
        axes[1,0].errorbar(positions+shift,means,yerr=np.stack([means-values.min(1),values.max(1)-means]),
                           fmt='none',ecolor='black',capsize=3)
    axes[1,0].set_xticks(positions,['MLP','MLP no torque','Transformer','Flow DiT'],rotation=12)
    axes[1,0].set(title='C. Held-out episodes (26,44,60), two seeds',ylabel='Error remaining / zero baseline')
    for kind in ('mlp','transformer','dit_flow'):
        r=results[kind][0]
        axes[1,1].plot([x['step'] for x in r['curve']],
                       [x['validation']['relative_mse'] for x in r['curve']],label=labels[kind]+', 2k')
    if extended:
        axes[1,1].plot([x['step'] for x in extended['curve']],
                       [x['validation']['relative_mse'] for x in extended['curve']],label='Flow DiT, 10k budget',linestyle='--')
    axes[1,1].set(title='D. Validation trajectory (seed 42)',xlabel='Optimizer updates',ylabel='Residual MSE / zero baseline')
    for ax in axes.flat:
        ax.axhline(1,color='gray',linestyle=':',linewidth=1,label='Zero residual' if ax==axes[0,0] else None)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Teacher torque residual: corrected controlled probes | lower error is better',fontsize=14)
    fig.savefig(root/'evidence.png',dpi=160)
    plt.close(fig)
    (root/'evidence_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
