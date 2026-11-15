"""Match all frozen dependencies to ONE explicitly pinned teacher (no RGB training)."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

from omegaconf import OmegaConf
from student_dual_path_common import write_json
from paired_response_full_common import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','old','teacher','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--teacher-sha256',required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    target=a.output/'matched';predictor=a.output/'predictor';cache=a.output/'predictor_cache'
    def run(stage,args):
        write_json(a.output/'status.json',dict(stage=stage,state='running',command=args))
        with (a.output/(stage+'.console.log')).open('w') as f:
            subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        if not (target/'matched_manifest.json').exists():
            run('cache_adapter',['test/prepare_civi_matched_teacher.py','--base',str(a.base),'--old',str(a.old),
                '--pinned-teacher',str(a.teacher),'--expected-sha256',a.teacher_sha256,'--output',str(target)])
        assert file_hash(target/'inputs/pinned_teacher_ema.ckpt')==a.teacher_sha256
        c=OmegaConf.load(Path(__file__).parent/'config/scene_torque_response_predictor.yaml');c.teacher_sha256=a.teacher_sha256
        OmegaConf.save(c,a.output/'predictor.yaml')
        if not (predictor/'summary.json').exists():
            run('predictor',['test/run_scene_torque_response_predictor.py','--base',str(target),'--output',str(predictor),'--config',str(a.output/'predictor.yaml')])
        assert json.loads((predictor/'summary.json').read_text())['teacher_checkpoint_sha256']==a.teacher_sha256
        c=OmegaConf.load(Path(__file__).parent/'config/response_conditioned_rgb.yaml')
        c.teacher_sha256=a.teacher_sha256;c.predictor_sha256=file_hash(predictor/'best.pt')
        OmegaConf.save(c,a.output/'prediction_cache.yaml')
        if not (cache/'prediction_manifest.json').exists():
            run('prediction_cache',['test/run_response_conditioned_rgb.py','--base',str(target),'--predictor',str(predictor),
                '--old',str(a.old),'--output',str(cache),'--mode','prepare','--config',str(a.output/'prediction_cache.yaml')])
        assert json.loads((cache/'prediction_manifest.json').read_text())['predictor_checkpoint_sha256']==c.predictor_sha256
        if not (target/'condition/summary.json').exists():
            run('condition',['test/calibrate_civi_matched_condition.py','--base',str(target),'--old',str(a.old),'--source',str(a.base),'--steps','1000'])
        dependencies=dict(teacher_sha256=a.teacher_sha256,predictor_sha256=file_hash(predictor/'best.pt'),
            adapter_sha256=file_hash(target/'adapter/best_adapter.pt'),condition_sha256=file_hash(target/'condition/best.pt'),
            decoder_initial_sha256=file_hash(target/'d/best.pt'))
        dependencies['teacher_step']=json.loads((target/'matched_manifest.json').read_text())['teacher_step']
        write_json(a.output/'dependencies.json',dependencies)
        write_json(a.output/'status.json',dict(state='complete',stage='matched_dependencies_ready',dependencies=dependencies,
            base=str(target),predictor=str(predictor),cache=str(cache),rgb_training_started=False,
            next='Use explicit --dependencies for new-teacher RGB experiment; retain immutable acceptance gates'))
    except Exception as exc:
        write_json(a.output/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':main()
