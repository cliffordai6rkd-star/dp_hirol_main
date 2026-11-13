"""Plot saved RenderNet losses on CPU, optionally following a live metrics file."""

import os

# Set these before importing NumPy or Matplotlib. This process never imports
# PyTorch or the training logger/workspace.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
import math
import numbers
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


class MetricsReader:
    """Read complete new JSONL records, or replace a complete JSON snapshot.

    ``reset`` tells the consumer to discard previous records. Partial JSONL
    tails are retried from the same byte offset; incomplete snapshots preserve
    the last valid history. Replacing/truncating a JSONL file resets its history.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._identity = None
        self._stamp = None
        self._offset = 0
        self._prefix = self._tail = b""
        self._snapshot = None
        self._mode = None

    def read_new(self):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return [], False
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            stamp = (identity, stat.st_size, stat.st_mtime_ns)
            if stamp == self._stamp:
                return [], False
            head = stream.read(4096)
            if self._mode == "snapshot" and not head.strip():
                self._stamp = stamp
                return [], False
            snapshot = (head.lstrip().startswith(b"[")
                        or re.search(rb'"records"\s*:', head) is not None)
            if snapshot:
                stream.seek(0)
                content = stream.read()
                self._stamp = stamp
                try:
                    data = json.loads(content)
                except (ValueError, UnicodeError):
                    return [], False
                records = data.get("records") if isinstance(data, dict) else data
                if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
                    return [], False
                changed = content != self._snapshot or self._mode != "snapshot"
                self._identity, self._snapshot, self._mode = identity, content, "snapshot"
                return (records, True) if changed else ([], False)

            reset = self._identity is not None and (
                identity != self._identity or stat.st_size < self._offset or self._mode == "snapshot")
            if not reset and self._offset:
                # A same-inode rewrite can finish before the next poll and
                # already exceed the old size. Compare consumed boundaries.
                stream.seek(0)
                prefix = stream.read(len(self._prefix))
                stream.seek(self._offset - len(self._tail))
                reset = prefix != self._prefix or stream.read(len(self._tail)) != self._tail
            offset = 0 if reset else self._offset
            stream.seek(offset)
            content = stream.read()
            records = []
            consumed = 0
            for line in content.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    break
                if line.strip():
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        break
                    if not isinstance(record, dict):
                        break
                    records.append(record)
                consumed += len(line)
            self._stamp = stamp
            if not consumed and content:
                return [], False
            self._offset = offset + consumed
            self._identity, self._mode = identity, "jsonl"
            stream.seek(0)
            self._prefix = stream.read(min(256, self._offset))
            stream.seek(max(0, self._offset - 256))
            self._tail = stream.read(min(256, self._offset))
            return records, reset


_REGION_COMPONENTS = {"context_patch_mse", "mask_mse", "intensity_smoothness", "intensity_loss"}
_COMPONENTS = ("token_mse", "phase_ce", "alpha_mse", "context_patch_mse",
               "mask_mse", "intensity_smoothness", "intensity_loss", "loss",
               "force_delta_mse", "force_delta_loss", "context_region_loss")
_PREFIXES = ("train_", "val_", "epoch_train_", "epoch_val_", "initial_train_", "initial_val_")
_METRIC_KEYS = {prefix + component for prefix in _PREFIXES
                for component in (*_COMPONENTS, "mse", "open_token_mse", "closed_token_mse",
                                  "condition_loss")}


class LossHistory:
    """Measured losses keyed by optimizer step, including epoch/validation means."""

    def __init__(self):
        self.series = {}
        self.step = self.epoch = 0
        self.objective = "token_mse"
        self.metric_scope = "all"
        self.selection_metric = None

    def _put(self, key, step, value):
        if isinstance(value, numbers.Real) and math.isfinite(value):
            self.series.setdefault(key, {})[step] = float(value)

    def update(self, records, reset=False):
        if reset:
            self.__init__()
        for record in records:
            scope = record.get("metric_scope", self.metric_scope)
            if scope != self.metric_scope:
                for key in list(self.series):
                    if not key.endswith(("phase_ce", "alpha_mse")):
                        del self.series[key]
                self.metric_scope = scope
            warmup = record.get("objective") == "condition_warmup"
            if "objective" in record and not warmup:
                self.objective = record["objective"]
            if "selection_metric" in record:
                self.selection_metric = record["selection_metric"]
            step = record.get("global_step", record.get("step", self.step))
            if not isinstance(step, numbers.Real) or not math.isfinite(step):
                continue
            self.step, self.epoch = step, record.get("epoch", self.epoch)
            values = {key: value for key, value in record.items() if key in _METRIC_KEYS}
            if self.objective == "token_mse":
                for prefix in _PREFIXES:
                    measured_key = ("epoch_train_token_mse_measured" if prefix == "epoch_train_"
                                    else "train_token_mse_measured"
                                    if prefix in {"train_", "initial_train_"}
                                    else "val_token_mse_measured")
                    if prefix + "mse" in values and record.get(measured_key, not warmup):
                        component = ("loss" if self.selection_metric == "force context loss"
                                     else "token_mse")
                        values.setdefault(prefix + component, values[prefix + "mse"])
            if warmup and "train_mse" in values:
                values.setdefault("train_loss", values["train_mse"])
            if warmup:
                for prefix in _PREFIXES:
                    if prefix + "mse" not in values:
                        continue
                    validation = prefix in {"val_", "epoch_val_", "initial_val_"}
                    measured_key = ("val_token_mse_measured" if validation
                                    else "epoch_train_token_mse_measured" if prefix == "epoch_train_"
                                    else "train_token_mse_measured")
                    if not record.get(measured_key, False):
                        values.setdefault(prefix + "condition_loss", values[prefix + "mse"])
            for key, value in values.items():
                prefix = next((prefix for prefix in _PREFIXES if key.startswith(prefix)), None)
                component = key[len(prefix):]
                train = prefix in {"train_", "initial_train_"}
                epoch_train = prefix == "epoch_train_"
                token = component in {"token_mse", "mse", "open_token_mse", "closed_token_mse"}
                if token and train and not record.get("train_token_mse_measured", not warmup):
                    continue
                if token and epoch_train and not record.get("epoch_train_token_mse_measured", not warmup):
                    continue
                if token and not (train or epoch_train) and not record.get(
                        "val_token_mse_measured", not warmup):
                    continue
                if prefix.startswith("epoch_train"):
                    region_count = record.get("epoch_token_region_observation_count")
                elif train:
                    region_count = record.get("train_token_region_observation_count",
                                              record.get("token_region_observation_count"))
                else:
                    region_count = record.get("val_token_region_observation_count")
                if region_count == 0 and (component in _REGION_COMPONENTS or token):
                    continue
                if component in {"open_token_mse", "closed_token_mse"}:
                    window = component.split("_", 1)[0]
                    count_key = (f"epoch_{window}_token_count" if epoch_train
                                 else f"{prefix}{window}_token_count")
                    if record.get(count_key) == 0:
                        continue
                self._put(key, step, value)

    def points(self, key):
        values = self.series.get(key, {})
        steps = sorted(values)
        return np.asarray(steps, dtype=float), np.asarray([values[step] for step in steps], dtype=float)


def _thin(x, y, max_points):
    if len(x) > max_points:
        indices = np.linspace(0, len(x) - 1, max_points, dtype=int)
        return x[indices], y[indices]
    return x, y


def _rolling_mean(values, window):
    sums = np.concatenate(([0.], np.cumsum(values, dtype=float)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (sums[end] - sums[start]) / (end - start)


def save_loss_plot(history, output, smooth_window=100, max_points=4000):
    """Atomically replace loss panels; all horizontal axes use global steps."""
    if smooth_window < 1 or max_points < 1:
        raise ValueError("smooth_window and max_points must be positive")
    output = Path(output)
    force_panels = [
        ("force_delta_mse", "Force difference MSE (raw)"),
        ("force_delta_loss", "Force difference loss (normalized)"),
        ("context_region_loss", "Context region RGB loss"),
    ]
    if history.selection_metric == "force context loss":
        panels = [("loss", "Combined force-context loss"),
                  ("token_mse", "Context token MSE"), *force_panels]
    else:
        panels = list(zip(_COMPONENTS[:8], (
            "Context token MSE", "Contact phase CE", "Alpha MSE", "Region patch SmoothL1",
            "Region pixel MSE (diagnostic)", "Region smoothness",
            "Weighted region loss", "Total training loss",
        )))
        panels.extend(panel for panel in force_panels if any(
            prefix + panel[0] in history.series for prefix in _PREFIXES))
    rows = math.ceil(len(panels) / 2)
    figure = Figure(figsize=(16, 4 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, 2, squeeze=False).ravel()

    def curve(axis, key, label, **options):
        x, y = history.points(key)
        if len(x):
            x, y = _thin(x, y, max_points)
            axis.plot(x, y, label=label, **options)

    for axis, (component, title) in zip(axes, panels):
        if component == "token_mse" and history.metric_scope == "phases_1_2":
            title += "\npre-contact + contact (combined)"
        x, y = history.points("train_" + component)
        if len(x):
            raw_x, raw_y = _thin(x, y, max_points)
            axis.plot(raw_x, raw_y, color="tab:blue", linewidth=.6, alpha=.2, label="Train batch")
            mean_x, mean_y = _thin(x, _rolling_mean(y, smooth_window), max_points)
            axis.plot(mean_x, mean_y, color="tab:blue", linewidth=1.3,
                      label=f"Train mean ({smooth_window} measured batches)")
        curve(axis, "epoch_train_" + component, "Train epoch mean", color="tab:red",
              marker="^", markersize=4, linewidth=1)
        curve(axis, "initial_val_" + component, "Initial validation", color="tab:orange",
              marker="x", linestyle="none", markersize=6)
        curve(axis, "val_" + component, "Validation", color="tab:orange",
              marker="o", markersize=4, linewidth=1.3)
        curve(axis, "epoch_val_" + component, "Validation epoch mean", color="tab:green",
              marker="o", markersize=4, linewidth=1)
        if component == "token_mse":
            curve(axis, "train_closed_token_mse", "Closed window (diagnostic)",
                  color="gray", linestyle="--", linewidth=.7, alpha=.65)
            curve(axis, "val_closed_token_mse", "Validation closed window", color="gray",
                  marker="x", linestyle="--", linewidth=.9)
            if history.metric_scope == "all":
                curve(axis, "train_open_token_mse", "Open window (diagnostic)",
                      color="tab:green", linestyle="--", linewidth=.7, alpha=.65)
                curve(axis, "val_open_token_mse", "Validation open window", color="tab:green",
                      marker="x", linestyle="--", linewidth=.9)
        if component == "loss":
            curve(axis, "initial_val_condition_loss", "Initial condition validation",
                  color="tab:purple", marker="x", linestyle="none", markersize=6)
            curve(axis, "val_condition_loss", "Condition validation loss", color="tab:purple",
                  marker="o", markersize=4, linewidth=1.3)
            curve(axis, "epoch_val_condition_loss", "Condition validation epoch mean",
                  color="tab:purple", marker="o", markersize=4, linewidth=1)
        axis.set(title=title, xlabel="Optimizer step", ylabel="Loss")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(loc="upper right", fontsize=8)
        else:
            axis.text(.5, .5, "Waiting for measured losses", ha="center", va="center",
                      transform=axis.transAxes, color="gray")
    for axis in axes[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(f"RenderNet losses | epoch {history.epoch:g} | step {history.step:g}", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.png")
    try:
        figure.savefig(temporary, format="png", dpi=110)
        temporary.replace(output)
    finally:
        figure.clear()
    return output


def _pid_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return status != "Z"
    except (FileNotFoundError, IndexError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSONL/JSON metrics file or student run directory")
    parser.add_argument("--output", type=Path, help="PNG destination (default: training_losses.png)")
    parser.add_argument("--watch", action="store_true", help="Refresh when complete metrics change")
    parser.add_argument("--interval", type=float, default=10., help="Polling interval in seconds")
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--pid", type=int, help="Stop watching after this training PID exits")
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.smooth_window < 1 or args.max_points < 1:
        parser.error("interval, smooth-window, and max-points must be positive")
    if args.pid is not None and args.pid < 1:
        parser.error("pid must be positive")
    source = args.source
    if source.is_dir():
        source = next((source / name for name in ("logs.json.txt", "loss.json")
                       if (source / name).is_file()), None)
        if source is None:
            parser.error("Run directory contains neither logs.json.txt nor loss.json")
    if not source.is_file() and not args.watch:
        parser.error(f"Metrics file does not exist: {source}")
    output = args.output or source.parent / "training_losses.png"
    reader, history = MetricsReader(source), LossHistory()

    def refresh(first=False):
        records, reset = reader.read_new()
        if records or reset or first:
            history.update(records, reset=reset)
            save_loss_plot(history, output, args.smooth_window, args.max_points)
            print(f"CPU loss plot | step={history.step:g} epoch={history.epoch:g} "
                  f"new_records={len(records)} output={output}", flush=True)

    try:
        refresh(first=True)
        if args.watch:
            while True:
                if args.pid is not None and not _pid_running(args.pid):
                    refresh()
                    print(f"Training PID {args.pid} ended; loss plot watcher exited.", flush=True)
                    break
                time.sleep(args.interval)
                refresh()
    except KeyboardInterrupt:
        print("Loss plot watcher stopped.", flush=True)


if __name__ == "__main__":
    main()
