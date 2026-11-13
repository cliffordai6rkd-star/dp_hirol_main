"""Local JSON metrics and PNG curves for RenderNet distillation."""

from array import array
import json
import math
import logging
import multiprocessing as mp
import numbers
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from diffusion_policy.common.json_logger import JsonLogger


log = logging.getLogger(__name__)


class MetricHistory:
    """Keep only plotted numbers; full records remain in the JSON log."""

    step_keys = ("train_mse", "train_token_mse", "train_denoiser_mse", "train_demo_mse",
                 "train_mask_mse", "train_context_patch_mse", "train_intensity_loss",
                 "train_phase_ce", "train_alpha_mse",
                 "train_contact_open_ratio", "train_open_token_mse", "train_closed_token_mse",
                 "train_loss", "grad_norm", "lr",
                 "samples_per_second", "gpu_allocated_mib", "gpu_reserved_mib", "gpu_peak_allocated_mib",
                 "initial_train_mse")
    region_keys = {"train_mask_mse", "train_context_patch_mse", "train_intensity_loss"}
    scope_dependent_keys = {
        "train_mse", "train_token_mse", "train_denoiser_mse", "train_demo_mse",
        "train_open_token_mse", "train_closed_token_mse", "train_loss",
        "train_condition_loss", "val_condition_loss", "initial_train_mse", "epoch_train_mse",
        "val_mse", "epoch_val_mse", "epoch_best_val_mse",
    } | region_keys

    def __init__(self):
        self.series = {}
        self.step = self.epoch = 0
        self.objective = "token_mse"
        self.metric_scope = "all"

    def _append(self, name, x, y):
        if not math.isfinite(float(y)):
            return
        xs, ys = self.series.setdefault(name, (array("d"), array("d")))
        if xs and xs[-1] == x:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)

    def record(self, metrics):
        scope = metrics.get("metric_scope", self.metric_scope)
        if scope != self.metric_scope:
            # Keep the full JSON history, but plot only measurements with the
            # current selection rule. Auxiliary phase/alpha and runtime
            # histories remain comparable across this change.
            for key in self.scope_dependent_keys:
                self.series.pop(key, None)
            self.metric_scope = scope
        warmup = metrics.get("objective") == "condition_warmup"
        if "objective" in metrics and not warmup:
            self.objective = metrics["objective"]
        step, epoch = metrics.get("global_step", self.step), metrics.get("epoch", self.epoch)
        self.step, self.epoch = step, epoch
        token_measured = metrics.get("train_token_mse_measured", not warmup)
        region_count = metrics.get("train_token_region_observation_count",
                                   metrics.get("token_region_observation_count"))
        if warmup and "train_mse" in metrics:
            self._append("train_condition_loss", step, metrics["train_mse"])
        elif warmup and "initial_train_mse" in metrics and not token_measured:
            self._append("train_condition_loss", step, metrics["initial_train_mse"])
        for key in self.step_keys:
            if (not token_measured and key in {
                    "train_mse", "train_token_mse", "train_denoiser_mse", "train_demo_mse",
                    "train_open_token_mse", "train_closed_token_mse", "initial_train_mse"}):
                continue
            if region_count == 0 and key in self.region_keys:
                continue
            if (key == "train_open_token_mse" and metrics.get("train_open_token_count") == 0
                    or key == "train_closed_token_mse" and metrics.get("train_closed_token_count") == 0):
                continue
            if key in metrics:
                self._append(key, step, metrics[key])
        if ("epoch_train_mse" in metrics
                and metrics.get("epoch_train_token_mse_measured", not warmup)):
            self._append("epoch_train_mse", epoch, metrics["epoch_train_mse"])
        val = metrics.get("val_mse", metrics.get("initial_val_mse"))
        val_token_measured = metrics.get("val_token_mse_measured", not warmup)
        if val is not None and warmup and not val_token_measured:
            self._append("val_condition_loss", step, val)
        if (val is not None and val_token_measured
                and metrics.get("val_token_region_observation_count") != 0):
            self._append("val_mse", step, val)
            self._append("epoch_val_mse", epoch, val)
            best = metrics.get("best_val_mse", val)
            self._append("epoch_best_val_mse", epoch, best)

    def points(self, key, max_points=4000):
        xs, ys = self.series.get(key, (array("d"), array("d")))
        x, y = np.frombuffer(xs, dtype=np.float64), np.frombuffer(ys, dtype=np.float64)
        if len(x) > max_points:
            selected = np.linspace(0, len(x) - 1, max_points, dtype=int)
            x, y = x[selected], y[selected]
        return x, y

    @classmethod
    def from_json(cls, path):
        history = cls()
        with Path(path).open() as source:
            for line in source:
                # A live writer may not yet have finished its final record.
                if not line.endswith("\n"):
                    break
                if line.strip():
                    history.record(json.loads(line))
        return history


def save_training_plot(history, destination):
    """Render without a display server; atomically replace the current PNG."""
    destination = Path(destination)
    figure = Figure(figsize=(16, 9), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).ravel()

    def curve(axis, key, label, *, factor=1, **kwargs):
        x, y = history.points(key)
        if len(x):
            axis.plot(x, y * factor, label=label, **kwargs)

    curve(axes[0], "train_mse", "Train batch", alpha=0.65, linewidth=0.8)
    curve(axes[0], "val_mse", "Validation", marker="o", color="tab:orange")
    curve(axes[0], "initial_train_mse", "Initial train batch", marker="x", linestyle="none", color="black")
    curve(axes[1], "epoch_train_mse", "Train epoch mean", marker=".")
    curve(axes[1], "epoch_val_mse", "Validation", marker=".")
    curve(axes[1], "epoch_best_val_mse", "Best validation", linestyle="--", color="tab:green")
    x, y = history.points("epoch_val_mse")
    if len(x):
        best = int(np.argmin(y))
        axes[1].plot(x[best], y[best], "*", color="tab:red", markersize=13,
                     label=f"Best {y[best]:.6g} @ epoch {x[best]:g}")
    metric_label = "Denoiser output MSE" if history.objective == "denoiser_mse" else "Context token MSE"
    if history.metric_scope == "phases_1_2":
        metric_label += "\npre-contact / contact"
    curve(axes[2], "train_mse", metric_label, linewidth=0.9)
    if history.objective == "denoiser_mse":
        curve(axes[2], "train_token_mse", "Token MSE (diagnostic)", linewidth=0.8, alpha=0.65)
        curve(axes[2], "train_demo_mse", "Demonstration loss", linewidth=0.8, alpha=0.65)
    curve(axes[2], "train_mask_mse", "Region MSE", linewidth=0.9)
    curve(axes[2], "train_condition_loss", "Condition warmup loss", linewidth=0.9)
    curve(axes[2], "val_condition_loss", "Condition validation loss", linewidth=0.9,
          marker="o", color="tab:orange")
    curve(axes[2], "train_phase_ce", "Contact phase CE", linewidth=0.8, alpha=0.8)
    curve(axes[2], "train_alpha_mse", "Alpha MSE", linewidth=0.8, alpha=0.8)
    curve(axes[2], "train_loss", "Total loss", linewidth=0.9)
    curve(axes[2], "train_contact_open_ratio", "Contact open ratio", linewidth=0.8,
          color="tab:orange", alpha=0.8)
    curve(axes[2], "train_open_token_mse", "Open-window token MSE", linewidth=0.8,
          color="tab:green", alpha=0.8)
    curve(axes[2], "train_closed_token_mse", "Closed-window token MSE", linewidth=0.8,
          color="tab:red", alpha=0.8)
    curve(axes[3], "grad_norm", "Gradient norm before clipping", color="tab:purple", linewidth=0.8)
    curve(axes[4], "lr", "Learning rate", color="tab:green")
    curve(axes[5], "gpu_allocated_mib", "GPU allocated", factor=1 / 1024, color="tab:blue")
    curve(axes[5], "gpu_reserved_mib", "GPU reserved", factor=1 / 1024, color="tab:cyan")
    curve(axes[5], "gpu_peak_allocated_mib", "Peak allocated (run)", factor=1 / 1024,
          color="tab:blue", linestyle="--")
    speed_axis = axes[5].twinx()
    curve(speed_axis, "samples_per_second", "Throughput", color="tab:orange", alpha=0.65)
    speed_axis.set_ylabel("Samples / second")
    if speed_axis.lines:
        speed_axis.legend(loc="upper right", fontsize=8)

    titles = (metric_label, "Train / validation by epoch", "Training loss components",
              "Gradient norm", "Learning rate", "GPU memory and training throughput")
    labels = ("MSE", "MSE", "Loss", "L2 norm", "Learning rate", "GiB")
    for index, axis in enumerate(axes):
        axis.set(title=titles[index], xlabel="Epoch" if index == 1 else "Optimizer step",
                 ylabel=labels[index])
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(loc="upper left", fontsize=8)
        else:
            axis.text(0.5, 0.5, "Waiting for training metrics", ha="center", va="center",
                      transform=axis.transAxes, color="gray")
    figure.suptitle(f"RenderNet distillation | epoch {history.epoch:g} | step {history.step:g}", fontsize=15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=140)
    temporary.replace(destination)
    figure.clear()
    return destination


def _plot_worker(log_path, destination, request_event, stop_event, poll_seconds):
    """Render plots outside the training process and never block optimizer steps."""
    log_path = Path(log_path)
    destination = Path(destination)
    last_mtime = None
    while True:
        requested = request_event.wait(float(poll_seconds))
        request_event.clear()
        stopping = stop_event.is_set()
        try:
            mtime = log_path.stat().st_mtime_ns
            if requested or stopping or mtime != last_mtime:
                history = MetricHistory.from_json(log_path)
                save_training_plot(history, destination)
                last_mtime = mtime
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            # The logger may be opening or truncating the file during startup.
            pass
        except Exception:
            log.exception("Asynchronous RenderNet plot update failed")
        if stopping:
            return


class RenderNetTrainingLogger(JsonLogger):
    def __init__(self, directory, plot_every_steps=100, enable_plots=True,
                 async_plots=True, plot_interval_sec=20.0):
        self.directory = Path(directory)
        super().__init__(str(self.directory / "logs.json.txt"), filter_fn=lambda key, value:
                         isinstance(value, numbers.Number) or key in {"objective", "metric_name", "metric_scope"})
        self.history = MetricHistory()
        self.enable_plots = bool(enable_plots)
        self.async_plots = bool(async_plots)
        self.plot_interval_sec = float(plot_interval_sec)
        if self.plot_interval_sec <= 0:
            raise ValueError("logging.plot_interval_sec must be positive")
        self.plot_every_steps = int(plot_every_steps)
        if self.plot_every_steps < 0:
            raise ValueError("logging.plot_every_steps must be nonnegative (0 = epoch only)")
        self._plot_process = None
        self._plot_request = None
        self._plot_stop = None

    def start(self):
        super().start()
        if self.enable_plots:
            self.history = MetricHistory.from_json(self.path)
            if self.async_plots:
                context = mp.get_context("spawn")
                self._plot_request = context.Event()
                self._plot_stop = context.Event()
                self._plot_process = context.Process(
                    target=_plot_worker,
                    args=(self.path, self.directory / "training_metrics.png",
                          self._plot_request, self._plot_stop, self.plot_interval_sec),
                    name="render-net-plotter",
                    daemon=True,
                )
                self._plot_process.start()

    def stop(self):
        if self._plot_process is not None:
            # Request one final plot before closing the JSON file, then wait
            # briefly so the PNG is complete when the training run exits.
            self._plot_stop.set()
            self._plot_request.set()
            self._plot_process.join(timeout=30.0)
            if self._plot_process.is_alive():
                self._plot_process.terminate()
                self._plot_process.join(timeout=5.0)
            self._plot_process = None
            self._plot_request = None
            self._plot_stop = None
        super().stop()

    def log(self, data):
        super().log(data)
        if not self.enable_plots:
            return
        self.history.record(self.get_last_log())
        step = data.get("global_step", 0)
        if (self.plot_every_steps and "train_mse" in data
                and (step == 1 or step % self.plot_every_steps == 0)):
            self.save_plot()

    def save_plot(self):
        if not self.enable_plots:
            return None
        if self.async_plots:
            self._plot_request.set()
            return self.directory / "training_metrics.png"
        return save_training_plot(self.history, self.directory / "training_metrics.png")
