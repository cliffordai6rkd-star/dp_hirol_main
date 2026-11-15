"""Pin exact final EMA, then optionally launch isolated matching on assigned GPU2."""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import time

import dill
import torch

from student_dual_path_common import write_json,tensor_hash
from paired_response_full_common import file_hash
from launch_paired_response_after_teacher import final_teacher_step


def remote_match_script(library,python,teacher,sha,output,source):
    values=dict(LIBRARY=library,PYTHON=python,TEACHER=teacher,SHA=sha,OUTPUT=output,SOURCE=source)
    preamble='\n'.join(f'{k}={v!r}' for k,v in values.items())+'\n'
    return preamble+'''import os,subprocess,json
from pathlib import Path
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','2'],text=True).strip()
assert int(gpu)<1000,'Assigned GPU2 busy; matching not launched'
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='2';env['LD_LIBRARY_PATH']=LIBRARY
args=[PYTHON,'-u','test/run_civi_teacher_match_suite.py','--base','data/diagnostics/2026.11.15/new_teacher70750_adapter_shared_response','--old','data/diagnostics/2026.11.15/student_dual_path_ablation','--teacher',TEACHER,'--teacher-sha256',SHA,'--output',OUTPUT]
out=Path(OUTPUT);assert not out.exists()
log=out.parent/'final150k_match.console.log'
p=subprocess.Popen(args,cwd=SOURCE,env=env,stdout=log.open('w'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(pid=p.pid,gpu=2,log=str(log),args=args)))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true');p.add_argument('--host');p.add_argument('--remote-source');p.add_argument('--remote-output')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    status=a.output/'status.json';previous=None
    while True:
        progress=final_teacher_step(a.teacher_dir)
        if progress!=previous:write_json(status,dict(state='waiting_for_exact_final_teacher',current_step=progress,required_step=150000));previous=progress
        if progress>=150000:
            latest=a.teacher_dir/'checkpoints/latest.ckpt'
            payload=torch.load(latest,map_location='cpu',pickle_module=dill,weights_only=False)
            actual=int(dill.loads(payload['pickles']['optimizer_step']))
            if actual==150000:break
            del payload
        if not a.wait:raise RuntimeError('Exact150000 checkpoint not yet saved; no earlier teacher substituted')
        time.sleep(30)
    pinned=a.output/'pinned_teacher_ema.ckpt';state=payload['state_dicts']['ema_model']
    torch.save(dict(cfg=payload['cfg'],state_dicts={'ema_model':state},pickles={'optimizer_step':dill.dumps(actual)}),pinned,pickle_module=dill)
    record=dict(source_checkpoint=str(latest.resolve()),source_sha256=file_hash(latest),pinned_ema_sha256=file_hash(pinned),
        teacher_state_sha256=tensor_hash(state),optimizer_step=actual,weights='ema_model')
    write_json(a.output/'provenance.json',record);write_json(status,dict(state='exact_final_teacher_pinned',**record))
    if a.host:
        assert a.remote_source and a.remote_output
        py='/home/djhpc/miniconda3/envs/fdp/bin/python';lib='/home/djhpc/miniconda3/envs/fdp/lib'
        library=f'{lib}/python3.10/site-packages/nvidia/npp/lib:{lib}:{lib}/python3.10/site-packages/torch/lib'
        def ssh(args):return subprocess.check_output(['ssh','-o','BatchMode=yes',a.host,shlex.join(args)],text=True)
        remote_parent=str(Path(a.remote_output).parent);ssh(['mkdir','-p',remote_parent+'/teacher_final150k'])
        remote_teacher=remote_parent+'/teacher_final150k/pinned_teacher_ema.ckpt'
        subprocess.run(['rsync','-az',str(pinned),a.host+':'+remote_teacher],check=True)
        script=remote_match_script(library,py,remote_teacher,record['pinned_ema_sha256'],a.remote_output,a.remote_source)
        process=json.loads(ssh(['python3','-c',script]))
        write_json(status,dict(state='final_teacher_matching_started',process=process,**record));print(json.dumps(process),flush=True)


if __name__=='__main__':main()
