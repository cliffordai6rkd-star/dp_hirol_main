"""Prepare one shared paired cache, preflight, then run full-data A/B on two GPUs."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf
from student_dual_path_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('test/config/civi_paired_response_full_data.yaml'))
    parser.add_argument('--gpus', nargs=2, default=['2', '3'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.root, args.experiment = args.root.resolve(), args.experiment.resolve()
    config = OmegaConf.load(args.config)
    args.experiment.mkdir(parents=True, exist_ok=True)
    status_path = args.experiment/'suite_status.json'
    if status_path.exists():
        raise RuntimeError('Suite already exists; do not launch duplicate experiments')
    started = time.monotonic()

    def status(**values):
        write_json(status_path, dict(elapsed_seconds=time.monotonic()-started, **values))

    def failed(kind, value, traceback):
        status(state='failed', error=str(value))
        sys.__excepthook__(kind, value, traceback)
    sys.excepthook = failed

    def env(gpu):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1'}

    status(state='preparing_shared_cache')
    subprocess.run([sys.executable, 'test/prepare_student_dual_path.py', '--inputs', str(args.experiment/'inputs'),
        '--image-cache', str(args.root/'decoded_cuda'), '--output-dir', str(args.experiment/'prepared_cuda'),
        '--batch-size', '32'], env=env(args.gpus[0]), check=True)
    common = [sys.executable, 'test/run_paired_response_full_data.py', '--root', str(args.root),
              '--experiment', str(args.experiment), '--config', str(args.config.resolve())]
    preflight = args.experiment/'preflight'
    status(state='preflight')
    subprocess.run([*common, '--group', 'b', '--output', str(preflight), '--check-only'],
                   env=env(args.gpus[0]), check=True)
    common.extend(['--calibration-manifest', str(preflight/'manifest.json')])

    def run_pair(steps, resume=False):
        children = []
        for group, gpu in zip(['a', 'b'], args.gpus):
            output = args.experiment/group
            command = [*common, '--group', group, '--output', str(output), '--steps', str(steps)]
            if resume:
                command.extend(['--resume', str(output/'latest.ckpt')])
            if args.smoke:
                command.append('--smoke')
            stream = (args.experiment/f'{group}_{steps}.console.log').open('w')
            process = subprocess.Popen(command, env=env(gpu), stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            children.append((group, gpu, process, stream))
        while any(process.poll() is None for _, _, process, _ in children):
            status(state='running', budget=steps, children={g: dict(gpu=gpu, pid=p.pid, returncode=p.poll())
                   for g, gpu, p, _ in children})
            time.sleep(10)
        for group, _, process, stream in children:
            stream.close()
            if process.returncode:
                raise RuntimeError(f'{group} failed with exit code {process.returncode}; inspect its console log')
        summaries = {g: json.loads((args.experiment/g/'summary.json').read_text()) for g in ['a', 'b']}
        a, b = [summaries[g]['manifest'] for g in ['a', 'b']]
        for key in ['starting_state_sha256', 'checkpoint_sha256', 'teacher_checkpoint_sha256',
                    'frozen_hashes', 'splits', 'calibration_indices', 'optimizer', 'response_rms_threshold']:
            assert a[key] == b[key], key
        for filename in ['sampling_epochs.json']:
            assert json.loads((args.experiment/'a'/filename).read_text()) == json.loads((args.experiment/'b'/filename).read_text())
        return summaries

    budget = 1 if args.smoke else int(config.steps)
    summaries = run_pair(budget)
    extension = not args.smoke and all(s['eligible_for_extension'] for s in summaries.values())
    write_json(args.experiment/'extension_decision.json', dict(extend=extension, first_budget=budget,
        criteria='Both arms: contact token MSE, response difference MSE, Free zero MSE must not exceed step 0',
        groups={g: s['eligible_for_extension'] for g, s in summaries.items()}))
    if extension and config.maximum_steps > budget:
        for group in ['a', 'b']:
            for source, dest in [('summary.json', f'summary_{budget}.json'), ('latest.ckpt', f'checkpoint_{budget}.ckpt')]:
                shutil.copy2(args.experiment/group/source, args.experiment/group/dest)
        summaries = run_pair(int(config.maximum_steps), resume=True)
    status(state='reporting', extension=extension)
    subprocess.run([sys.executable, 'test/report_paired_response_full_data.py', str(args.experiment)], check=True)
    status(state='complete', extension=extension, steps={g: s['steps'] for g, s in summaries.items()})


if __name__ == '__main__':
    main()
