"""Run the three existing teacher ablations with a common 5k-step budget."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, default=["2", "3"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pending = ["baseline", "film_only", "hybrid"]
    active, finished = {}, {}
    started = time.monotonic()
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            variant = pending.pop(0)
            command = [sys.executable, "test/run_teacher_fusion_ablation.py",
                       "--variant", variant, "--steps", "5000", "--output-dir",
                       str(args.output_dir / variant), "--image-cache", str(args.image_cache)]
            stream = (args.output_dir / f"{variant}.console.log").open("w")
            process = subprocess.Popen(command, cwd=args.source,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
                stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = (variant, process, stream)
            print(json.dumps(dict(event="start", variant=variant, gpu=gpu, pid=process.pid)), flush=True)
        for gpu, (variant, process, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            finished[variant] = code
            del active[gpu]
            print(json.dumps(dict(event="finish", variant=variant, returncode=code)), flush=True)
            if code:
                pending.clear()
        status = dict(active={g: dict(variant=v, pid=p.pid) for g, (v, p, _) in active.items()},
                      pending=pending, finished=finished, elapsed_seconds=time.monotonic() - started)
        (args.output_dir / "suite_status.json").write_text(json.dumps(status, indent=2) + "\n")
        if active:
            time.sleep(5)
    if len(finished) != 3 or any(finished.values()):
        raise RuntimeError(f"Teacher suite incomplete: {finished}")
    subprocess.run([sys.executable, "test/report_teacher_fusion_ablation.py",
                    str(args.output_dir)], cwd=args.source, check=True)


if __name__ == "__main__":
    main()
