"""Rebuild or follow a RenderNet training PNG from its local JSON metrics."""

import argparse
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to RUN_DIRECTORY/training_metrics.png")
    parser.add_argument("--watch", action="store_true", help="Refresh when the metrics log changes")
    parser.add_argument("--interval", type=float, default=20.0, help="Refresh interval in seconds")
    parser.add_argument("--until-pid", type=int, help="Finish after this training process exits")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    # Limit only this plotting process; never change the trainer's environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from diffusion_policy.common.render_net_training_logger import MetricHistory, save_training_plot

    log_path = args.run_directory / "logs.json.txt"
    destination = args.output or args.run_directory / "training_metrics.png"
    last_mtime = None
    while True:
        finished = False
        if args.until_pid is not None:
            try:
                os.kill(args.until_pid, 0)
            except ProcessLookupError:
                finished = True
        mtime = log_path.stat().st_mtime_ns
        if mtime != last_mtime:
            history = MetricHistory.from_json(log_path)
            print(save_training_plot(history, destination), flush=True)
            last_mtime = mtime
        if not args.watch or finished:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
