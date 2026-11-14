"""Two-GPU student suite; fixed calibrated adapters and four equal RGB budgets."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from student_dual_path_common import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',nargs=2,default=['2','3']);a=p.parse_args();root=a.root.resolve()
    output=root/'experiments_cuda';output.mkdir(exist_ok=False)
    # Run the architecture and adapter endpoints first, then fill the ablations.
    pending=['a_baseline','c_film_mlp','b_interleaved','c_film_linear'];active={};finished={}
    start=time.monotonic()
    while pending or active:
        for gpu in a.gpus:
            if gpu in active or not pending:continue
            variant=pending.pop(0);kind='mlp' if variant=='c_film_mlp' else 'linear'
            calibration=root/f'calibration_{kind}_cuda'
            assert (calibration/'adapter.pt').is_file()
            cmd=[sys.executable,'test/run_student_dual_path.py','--variant',variant,
                 '--inputs',str(root/'inputs'),'--prepared',str(root/'prepared_cuda'),
                 '--image-cache',str(root/'decoded_cuda'),'--adapter',str(calibration/'adapter.pt'),
                 '--output-dir',str(output/variant),'--steps','5000']
            stream=(output/f'{variant}.console.log').open('w')
            child=subprocess.Popen(cmd,cwd=root/'source',env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,
                'PYTHONUNBUFFERED':'1'},stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active[gpu]=(variant,child,stream)
            print(json.dumps(dict(event='start',variant=variant,gpu=gpu,pid=child.pid)),flush=True)
        for gpu,(variant,child,stream) in list(active.items()):
            code=child.poll()
            if code is None:continue
            stream.close();finished[variant]=code;del active[gpu]
            print(json.dumps(dict(event='finish',variant=variant,returncode=code)),flush=True)
            if code:pending.clear()
        write_json(output/'suite_status.json',dict(active={g:dict(variant=v,pid=c.pid) for g,(v,c,_) in active.items()},
            pending=pending,finished=finished,elapsed_seconds=time.monotonic()-start))
        if active:time.sleep(5)
    if len(finished)!=4 or any(finished.values()):raise RuntimeError(f'Student suite incomplete: {finished}')
    subprocess.run([sys.executable,'test/report_student_dual_path.py',str(output),
                    '--calibration-root',str(root)],cwd=root/'source',check=True)

if __name__=='__main__':main()
