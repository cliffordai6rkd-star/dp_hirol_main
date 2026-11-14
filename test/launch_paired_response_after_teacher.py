"""Wait for the selected teacher's FINAL step, pin EMA, and launch remote A/B.

Run on the local teacher host. It consumes an already synchronized immutable
source snapshot on the GPU server. Status and logs remain in --output.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import dill
import torch

from paired_response_full_common import file_hash


def final_teacher_step(directory):
    """Read progress without deserializing a live 882MB checkpoint every poll."""
    path = directory/'logs.json.txt'
    if not path.exists():
        return 0
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-(1 << 18)))
        lines = stream.read().decode(errors='replace').splitlines()
    maximum = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        maximum = max(maximum, int(row.get('optimizer_step', 0)))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--teacher-pid', type=int)
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--remote-root', required=True, help='Existing student checkpoint/cache root')
    parser.add_argument('--remote-experiment', required=True)
    parser.add_argument('--remote-python', default='/home/djhpc/miniconda3/envs/fdp/bin/python')
    parser.add_argument('--gpus', nargs=2, default=['2', '3'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output/'status.json'
    started = time.monotonic()

    def status(state, **values):
        row = dict(state=state, elapsed_seconds=time.monotonic()-started, **values)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, indent=2)+'\n')
        temporary.replace(status_path)
        print(json.dumps(row), flush=True)

    def ssh(command):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', args.host, command], text=True).strip()

    def failed(kind, value, traceback):
        status('failed', error=str(value))
        sys.__excepthook__(kind, value, traceback)
    sys.excepthook = failed
    last_report = None
    while True:
        progress = final_teacher_step(args.teacher)
        if progress != last_report:
            status('waiting_for_final_teacher', current_step=progress, target_step=args.steps,
                   teacher_directory=str(args.teacher))
            last_report = progress
        if progress >= args.steps:
            latest = args.teacher/'checkpoints/latest.ckpt'
            if latest.exists():
                payload = torch.load(latest, map_location='cpu', pickle_module=dill, weights_only=False)
                actual = int(dill.loads(payload['pickles']['optimizer_step']))
                if actual == args.steps:
                    break
                del payload
        if args.teacher_pid:
            try:
                os.kill(args.teacher_pid, 0)
            except ProcessLookupError:
                raise RuntimeError('Teacher process ended before the required final checkpoint was available')
        time.sleep(30)
    assert 'ema_model' in payload['state_dicts']
    resolved = latest.resolve()
    status('pinning_final_teacher', checkpoint=str(resolved), optimizer_step=actual)
    pinned = args.output/'pinned_teacher_ema.ckpt'
    # Preserve the exact final EMA tensors and optimizer-step metadata; discard optimizer state.
    torch.save(dict(cfg=payload['cfg'], state_dicts={'ema_model': payload['state_dicts']['ema_model']},
        pickles={'optimizer_step': dill.dumps(actual)}), pinned, pickle_module=dill)
    provenance = dict(source_checkpoint=str(resolved), source_checkpoint_sha256=file_hash(resolved),
        pinned_ema_sha256=file_hash(pinned), optimizer_step=actual,
        teacher_weights='ema_model', adapter='existing calibrated MLP; frozen; not recalibrated')
    (args.output/'teacher_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    del payload
    remote = args.remote_experiment.rstrip('/')
    setup = (
        'from pathlib import Path; import shutil; '
        f'r=Path({remote!r}); old=Path({args.remote_root!r}); '
        'dest=r/"inputs"; dest.mkdir(parents=True,exist_ok=True); '
        '[shutil.copy2(old/"inputs"/name,dest/name) for name in '
        '["student_rgb_initial.ckpt","reference_train_cache.json","reference_validation_cache.json","reference_samples.pt"]]'
    )
    ssh(shlex.join([args.remote_python, '-c', setup]))
    subprocess.run(['rsync', '-az', str(pinned), str(args.output/'teacher_provenance.json'),
                    f'{args.host}:{remote}/inputs/'], check=True)
    source = remote+'/source'
    lib = '/home/djhpc/miniconda3/envs/fdp/lib'
    library_path = f'{lib}/python3.10/site-packages/nvidia/npp/lib:{lib}:{lib}/python3.10/site-packages/torch/lib'
    command = [args.remote_python, '-u', 'test/run_paired_response_full_suite.py',
               '--root', args.remote_root, '--experiment', remote, '--gpus', *args.gpus]
    shell = f'cd {shlex.quote(source)} && LD_LIBRARY_PATH={shlex.quote(library_path)} '
    shell += f'nohup {shlex.join(command)} >{shlex.quote(remote+"/suite.console.log")} 2>&1 </dev/null &'
    ssh(shell)
    status('remote_suite_started', experiment=remote, gpus=args.gpus, **provenance)
    while True:
        read = f'from pathlib import Path; p=Path({(remote+"/suite_status.json")!r}); print(p.read_text() if p.exists() else "{{}}")'
        state = json.loads(ssh(shlex.join([args.remote_python, '-c', read])))
        if state.get('state') in ['complete', 'failed']:
            break
        time.sleep(30)
    destination = args.output/'results'
    destination.mkdir(exist_ok=True)
    subprocess.run(['rsync', '-az', '--exclude=source/', '--exclude=inputs/', '--exclude=prepared_cuda/',
        '--exclude=*.ckpt', '--exclude=*.pt', f'{args.host}:{remote}/', str(destination)+'/'], check=True)
    status(state['state'], remote_status=state, results=str(destination))


if __name__ == '__main__':
    main()
