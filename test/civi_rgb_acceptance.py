"""The user-agreed engineering gates; no data-dependent threshold adjustment."""
from pathlib import Path
import argparse
import json


def assess(summary,baseline=None):
    m=summary['final']['ema']['validation'];contact=m['selection_macro_contact']
    initial=summary.get('initial_full_ema') or summary.get('initial_full')
    free=m['phases']['free']['response_mse']
    initial_free=initial['validation']['phases']['free']['response_mse'] if initial else None
    controls=summary.get('controls',{})
    conditions={k:v['selection_macro_contact'] for k,v in controls.items()}
    correct=conditions.get('correct',{})
    conditional=bool(correct) and all(correct['realization_mse']<conditions[x]['realization_mse']
        and correct['end_to_end_mse']<conditions[x]['end_to_end_mse'] for x in ['shuffled','zero'] if x in conditions)
    total=sum(v['samples'] for v in m['phases'].values())
    overall_rgb=(sum(v['samples']*v['rgb_rms']**2 for v in m['phases'].values() if v['samples'])/total)**.5
    checks=dict(realization_25percent=contact['realization_mse']<=.75*contact['predictor_energy'],
        teacher_response_10percent=contact['end_to_end_mse']<=.9*contact['teacher_energy'],
        better_than_original_shared=baseline is not None and contact['end_to_end_mse']<baseline['end_to_end_mse'],
        free_within_5percent=initial_free is not None and free<=1.05*initial_free,
        rgb_rms_bound=overall_rgb<=.015,
        correct_beats_shuffled_and_zero=conditional and all(x in conditions for x in ['correct','shuffled','zero']),
        epsilon_fixed=summary['manifest'].get('iteration_config',{}).get('epsilon')==.1,
        actual_rgb_forward=summary.get('frozen_verified',False) and summary.get('alpha0_identity',False))
    eps=summary['manifest']['iteration_config']['selection_episodes']
    episode_reductions={str(ep):dict(
        realization=1-m['episodes'][str(ep)]['contact']['realization_mse']/m['episodes'][str(ep)]['contact']['predictor_energy'],
        end_to_end=1-m['episodes'][str(ep)]['contact']['end_to_end_mse']/m['episodes'][str(ep)]['contact']['teacher_energy']) for ep in eps}
    checks['not_one_episode_only']=sum(x['realization']>0 and x['end_to_end']>0 for x in episode_reductions.values())>=3
    return dict(checks=checks,all_single_seed_checks=all(checks.values()),requires_two_seeds=True,
        realization_reduction=1-contact['realization_mse']/contact['predictor_energy'],
        teacher_response_reduction=1-contact['end_to_end_mse']/contact['teacher_energy'],
        initial_free=initial_free,final_free=free,overall_rgb_rms=overall_rgb,episodes=episode_reductions)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summaries',type=Path,nargs='+');p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    baseline=None
    if a.baseline:
        s=json.loads(a.baseline.read_text());baseline=(s.get('initial_full_ema') or s['initial_full'])['validation']['selection_macro_contact']
    summaries=[json.loads(path.read_text()) for path in a.summaries]
    results={str(path):assess(s,baseline) for path,s in zip(a.summaries,summaries)}
    seeds={s['manifest']['iteration_config']['seed'] for s in summaries}
    keys=['teacher_sha256','predictor_sha256','adapter_sha256','condition_sha256','decoder_initial_sha256']
    def dependencies_of(s):
        m=s['manifest']
        return {key:m.get(key,m.get('config',{}).get(key)) for key in keys}
    dependencies={tuple(dependencies_of(s)[k] for k in keys) for s in summaries}
    variants={s['manifest']['iteration_variant'] for s in summaries}
    same_budget=len({s['steps'] for s in summaries})==1
    same_initialization=len({s['manifest']['initial_tensor_state_sha256'] for s in summaries})==1
    passed=len(seeds)>=2 and len(dependencies)==1 and len(variants)==1 and same_budget and same_initialization and all(v['all_single_seed_checks'] for v in results.values())
    result=dict(promote=passed,seeds=sorted(seeds),same_dependencies=len(dependencies)==1,same_variant=len(variants)==1,
        same_budget=same_budget,same_initialization=same_initialization,
        initial_tensor_state_sha256=summaries[0]['manifest']['initial_tensor_state_sha256'] if same_initialization else None,
        variant=next(iter(variants)) if len(variants)==1 else None,
        dependencies=dependencies_of(summaries[0]) if len(dependencies)==1 else None,results=results)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
