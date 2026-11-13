"""Plot teacher training losses on CPU, optionally following a live JSON log."""

import argparse
import math
import numbers
import os
from pathlib import Path
import time


_METRICS = (
    "train_loss", "epoch_train_loss", "val_loss", "val_action_mse",
    "val_action_mse_raw", "curriculum/blur_sigma",
)


class TeacherLossHistory:
    """Keep actual measurements keyed by optimizer step, without filling gaps."""

    def __init__(self):
        self.series = {key: {} for key in _METRICS}
        self.step = self.epoch = 0

    def update(self, records, reset=False):
        if reset:
            self.__init__()
        for record in records:
            step = record.get("optimizer_step", record.get("global_step", record.get("step")))
            if not _finite(step):
                continue
            self.step = step
            if _finite(record.get("epoch")):
                self.epoch = record["epoch"]
            for key in _METRICS:
                value = record.get(key)
                if not _finite(value):
                    continue
                # New epoch summaries retain train_loss for compatibility, but
                # their globally reduced mean belongs to its own series.
                if key == "train_loss" and _finite(record.get("epoch_train_loss")):
                    continue
                self.series[key][step] = float(value)

    def points(self, key):
        values = self.series[key]
        steps = sorted(values)
        return steps, [values[step] for step in steps]

    def summary(self):
        fields = [f"step={self.step:g}", f"epoch={self.epoch:g}"]
        for key in ("train_loss", "epoch_train_loss", "val_loss", "curriculum/blur_sigma"):
            values = self.series[key]
            label = "blur_sigma" if key == "curriculum/blur_sigma" else key
            if values:
                step = max(values)
                fields.append(f"{label}={values[step]:.8g}@step{step:g}")
            elif key == "val_loss":
                fields.append("val_loss=not_recorded")
        return " | ".join(fields)


def _finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def infer_objective_label(run_directory):
    """Read plain saved YAML without instantiating a policy or importing Torch."""
    config_path = Path(run_directory) / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return "Training / validation objective"
    import yaml

    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return "Training / validation objective"
    policy = config.get("policy", {})
    head = config.get("action_head_backbone", policy.get("action_head_backbone"))
    if head == "act":
        loss = str(policy.get("act_loss_type", "l1")).upper()
        return f"Normalized action {loss} (ACT)"
    if head == "dit":
        return "Denoising MSE (DiT)"
    return "Training / validation objective"


def save_loss_plot(history, destination, smooth_window=20, max_points=4000,
                   objective_label="Training / validation objective"):
    """Save a four-panel figure; validation appears only when measured."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import numpy as np

    destination = Path(destination)
    figure = Figure(figsize=(13, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()

    def curve(axis, key, label, **options):
        x, y = history.points(key)
        if not x:
            return
        if len(x) > max_points:
            selected = np.linspace(0, len(x) - 1, max_points).astype(int)
            x, y = np.asarray(x)[selected], np.asarray(y)[selected]
        axis.plot(x, y, label=label, marker="o" if len(x) == 1 else ".",
                  markersize=4 if len(x) == 1 else 2, **options)

    x, y = history.points("train_loss")
    if x:
        curve(axes[0], "train_loss", "Train step (curriculum images)", color="tab:blue",
              linewidth=.6, alpha=.25)
        values = np.asarray(y, dtype=float)
        cumulative = np.concatenate(([0.], np.cumsum(values)))
        starts = np.maximum(0, np.arange(len(values)) + 1 - smooth_window)
        means = (cumulative[1:] - cumulative[starts]) / (np.arange(len(values)) + 1 - starts)
        selected = np.linspace(0, len(x) - 1, min(len(x), max_points)).astype(int)
        axes[0].plot(np.asarray(x)[selected], means[selected], color="tab:blue",
                     label=f"Train moving mean ({smooth_window} logged steps)",
                     marker="o" if len(x) == 1 else None, markersize=4, linewidth=1.3)
    curve(axes[0], "epoch_train_loss", "Train epoch mean (all ranks)", color="tab:red",
          linewidth=1.)
    curve(axes[0], "val_loss", "Validation objective (clear images)", color="tab:orange",
          linewidth=1.2)
    axes[0].set(title=objective_label, ylabel="Objective loss")
    if not history.series["val_loss"]:
        axes[0].text(.98, .95, "Validation not recorded yet", ha="right", va="top",
                     transform=axes[0].transAxes, color="gray", fontsize=9)

    curve(axes[1], "val_action_mse", "Validation action MSE", color="tab:green", linewidth=1.2)
    axes[1].set(title="Validation action error (clear images)", ylabel="Normalized action MSE")
    curve(axes[2], "val_action_mse_raw", "Validation action MSE", color="tab:purple", linewidth=1.2)
    axes[2].set(title="Validation action error (physical units)", ylabel="Raw action MSE")
    curve(axes[3], "curriculum/blur_sigma", "Training blur sigma", color="tab:gray", linewidth=1.2)
    axes[3].set(title="Pixel blur curriculum", ylabel="Gaussian sigma")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "No measurements yet", ha="center", va="center",
                      transform=axis.transAxes, color="gray")
    figure.suptitle(f"Teacher losses | epoch {history.epoch:g} | step {history.step:g}", fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.png")
    try:
        figure.savefig(temporary, format="png", dpi=120)
        temporary.replace(destination)
    finally:
        figure.clear()
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Teacher run directory or JSONL/JSON metrics file")
    parser.add_argument("--output", type=Path, help="PNG destination (default: training_losses.png)")
    parser.add_argument("--watch", action="store_true", help="Refresh as complete log records arrive")
    parser.add_argument("--quiet", action="store_true", help="Suppress normal progress output")
    parser.add_argument("--interval", type=float, default=10.)
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--pid", type=int, help="Stop watching when this training PID exits")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0 or args.smooth_window < 1 or args.max_points < 1:
        parser.error("interval, smooth-window, and max-points must be positive")
    if args.pid is not None and args.pid < 1:
        parser.error("pid must be positive")
    source = args.source
    if source.is_dir():
        source = next((source / name for name in ("logs.json.txt", "loss.json")
                       if (source / name).is_file()), source / "logs.json.txt")
    if not source.is_file() and not args.watch:
        parser.error(f"Metrics file does not exist: {source}")
    output = args.output or source.parent / "training_losses.png"

    # Apply process-local CPU limits only when running the CLI. Importing the
    # plot helpers must not change a training process's CUDA visibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    from diffusion_policy.scripts.plot_render_net_losses import MetricsReader, _pid_running

    reader, history = MetricsReader(source), TeacherLossHistory()
    label = infer_objective_label(source.parent)

    def refresh(first=False):
        records, reset = reader.read_new()
        if records or reset or first:
            history.update(records, reset=reset)
            save_loss_plot(history, output, args.smooth_window, args.max_points, label)
            if not args.quiet:
                print(f"Teacher loss plot | {history.summary()} | output={output}", flush=True)

    try:
        refresh(first=True)
        if args.watch:
            while True:
                if args.pid is not None and not _pid_running(args.pid):
                    refresh()
                    break
                time.sleep(args.interval)
                refresh()
    except KeyboardInterrupt:
        if not args.quiet:
            print("Teacher loss plot watcher stopped.", flush=True)


if __name__ == "__main__":
    main()
