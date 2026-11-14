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
                 "train_force_delta_mse", "train_force_delta_loss", "train_context_region_loss",
                 "train_mask_mse", "train_context_patch_mse", "train_intensity_loss",
                 "train_intensity_smoothness",
                 "train_phase_ce", "train_alpha_mse",
                 "train_response_mse", "train_student_loss", "train_condition_loss", "train_adapter_mse",
                 "train_contact_open_ratio", "train_open_token_mse", "train_closed_token_mse",
                 "train_loss", "grad_norm", "lr",
                 "samples_per_second", "gpu_allocated_mib", "gpu_reserved_mib", "gpu_peak_allocated_mib",
                 "initial_train_mse")
    region_keys = {"train_mask_mse", "train_context_patch_mse", "train_intensity_loss",
                   "train_intensity_smoothness"}
    component_keys = ("token_mse", "denoiser_mse", "demo_mse", "mask_mse",
                      "force_delta_mse", "force_delta_loss", "context_region_loss",
                      "context_patch_mse", "intensity_smoothness", "intensity_loss",
                      "phase_ce", "alpha_mse", "contact_open_ratio", "open_token_mse",
                      "response_mse", "student_loss", "condition_loss", "adapter_mse",
                      "closed_token_mse", "loss")
    scope_dependent_keys = {
        "train_mse", "train_token_mse", "train_denoiser_mse", "train_demo_mse",
        "train_force_delta_mse", "train_force_delta_loss", "train_context_region_loss",
        "train_open_token_mse", "train_closed_token_mse", "train_loss",
        "train_condition_loss", "val_condition_loss", "initial_train_mse", "epoch_train_mse",
        "val_mse", "epoch_val_mse", "epoch_best_val_mse", "epoch_train_mse_step",
        "epoch_train_condition_loss",
    } | region_keys

    def __init__(self):
        self.series = {}
        self.step = self.epoch = 0
        self.objective = "token_mse"
        self.metric_scope = "all"
        self.selection_metric = None

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
            for key in list(self.series):
                if (key.startswith(("val_", "epoch_train_"))
                        and not key.endswith(("phase_ce", "alpha_mse", "contact_open_ratio"))):
                    self.series.pop(key)
            self.metric_scope = scope
        adapter_warmup = metrics.get("objective") == "adapter_warmup"
        warmup = metrics.get("objective") == "condition_warmup" or adapter_warmup
        warmup_component = "adapter_mse" if adapter_warmup else "condition_loss"
        if "objective" in metrics and not warmup:
            self.objective = metrics["objective"]
        if "selection_metric" in metrics:
            self.selection_metric = metrics["selection_metric"]
        step, epoch = metrics.get("global_step", self.step), metrics.get("epoch", self.epoch)
        self.step, self.epoch = step, epoch
        token_measured = metrics.get("train_token_mse_measured", not warmup)
        region_count = metrics.get("train_token_region_observation_count",
                                   metrics.get("token_region_observation_count"))
        if warmup and "train_mse" in metrics:
            self._append("train_" + warmup_component, step, metrics["train_mse"])
        elif warmup and "initial_train_mse" in metrics and not token_measured:
            self._append("train_" + warmup_component, step, metrics["initial_train_mse"])
        if warmup and "epoch_train_mse" in metrics:
            self._append("epoch_train_" + warmup_component, step, metrics["epoch_train_mse"])
        for key in self.step_keys:
            if (not token_measured and key in {
                    "train_mse", "train_token_mse", "train_denoiser_mse", "train_demo_mse",
                    "train_response_mse", "train_student_loss",
                    "train_open_token_mse", "train_closed_token_mse", "initial_train_mse"}):
                continue
            if region_count == 0 and key in self.region_keys:
                continue
            if (key == "train_open_token_mse" and metrics.get("train_open_token_count") == 0
                    or key == "train_closed_token_mse" and metrics.get("train_closed_token_count") == 0):
                continue
            if key in metrics:
                self._append(key, step, metrics[key])
        for prefix in ("epoch_train_", "val_"):
            measured = metrics.get("epoch_train_token_mse_measured" if prefix == "epoch_train_"
                                   else "val_token_mse_measured", not warmup)
            count_prefix = "epoch_" if prefix == "epoch_train_" else prefix
            count = metrics.get(count_prefix + "token_region_observation_count")
            for component in self.component_keys:
                key = prefix + component
                if key not in metrics:
                    continue
                if not measured and component in {
                        "token_mse", "denoiser_mse", "demo_mse", "open_token_mse", "closed_token_mse",
                        "response_mse", "student_loss"}:
                    continue
                if count == 0 and "train_" + component in self.region_keys:
                    continue
                if (component in {"open_token_mse", "closed_token_mse"}
                        and metrics.get(count_prefix + component.replace("mse", "count")) == 0):
                    continue
                self._append(key, step, metrics[key])
        if ("epoch_train_mse" in metrics
                and metrics.get("epoch_train_token_mse_measured", not warmup)):
            self._append("epoch_train_mse", epoch, metrics["epoch_train_mse"])
            self._append("epoch_train_mse_step", step, metrics["epoch_train_mse"])
        val = metrics.get("val_mse", metrics.get("initial_val_mse"))
        val_token_measured = metrics.get("val_token_mse_measured", not warmup)
        if val is not None and warmup and not val_token_measured:
            self._append("val_" + warmup_component, step, val)
        if (val is not None and val_token_measured
                and metrics.get("val_token_region_observation_count") != 0):
            self._append("val_mse", step, val)
            self._append("epoch_val_mse", epoch, val)
            best = metrics.get("best_val_mse", val)
            self._append("epoch_best_val_mse", epoch, best)

    def points(self, key, max_points=4000):
        xs, ys = self.series.get(key, (array("d"), array("d")))
        x, y = np.frombuffer(xs, dtype=np.float64), np.frombuffer(ys, dtype=np.float64)
        if max_points is not None and len(x) > max_points:
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
    """Plot each metric on its own scale, outside the training process."""
    destination = Path(destination)
    force_context = history.selection_metric == "force context loss"
    metric_label = ("Combined force-context loss" if force_context else
                    "Denoiser output MSE" if history.objective == "denoiser_mse" else
                    "Global + response-weighted token MSE" if "train_response_mse" in history.series
                    else "Context token MSE")
    if history.metric_scope == "phases_1_2":
        metric_label += "\npre-contact / contact"

    # A panel contains only one metric. Batch/epoch/validation curves share its
    # scale; unrelated losses, proportions and runtime measurements never do.
    panels = [(metric_label, "Loss" if force_context else "MSE",
               "train_mse", "epoch_train_mse_step", "val_mse", 1.)]
    if force_context:
        panels.append(("Context token MSE", "MSE", "train_token_mse",
                       "epoch_train_token_mse", "val_token_mse", 1.))
    if history.objective == "denoiser_mse":
        panels.extend([
            ("Token MSE (diagnostic)", "MSE", "train_token_mse", "epoch_train_token_mse", "val_token_mse", 1.),
            ("Demonstration loss", "Loss", "train_demo_mse", "epoch_train_demo_mse", "val_demo_mse", 1.),
        ])
    for title, component, ylabel in (
        ("Force difference MSE (raw)", "force_delta_mse", "MSE"),
        ("Force difference loss (normalized)", "force_delta_loss", "Loss"),
        ("Context region RGB loss", "context_region_loss", "Weighted L1"),
        ("Global context token MSE", "token_mse", "MSE"),
        ("Teacher response-weighted token MSE", "response_mse", "MSE"),
        ("Condition warmup loss", "condition_loss", "Loss"),
        ("Visual baseline calibration", "adapter_mse", "MSE"),
        ("Contact phase CE", "phase_ce", "Cross entropy"),
        ("Alpha MSE", "alpha_mse", "MSE"),
        ("Region patch SmoothL1", "context_patch_mse", "Loss"),
        ("Region pixel MSE (diagnostic)", "mask_mse", "MSE"),
        ("Region smoothness", "intensity_smoothness", "Loss"),
        ("Weighted region loss", "intensity_loss", "Loss"),
        ("Total training loss", "loss", "Loss"),
        ("Contact open ratio", "contact_open_ratio", "Fraction"),
        ("Open-window token MSE", "open_token_mse", "MSE"),
        ("Closed-window token MSE", "closed_token_mse", "MSE"),
    ):
        if force_context and component == "loss":
            continue
        if component == "token_mse" and "train_response_mse" not in history.series:
            continue
        panels.append((title, ylabel, "train_" + component,
                       "epoch_train_" + component, "val_" + component, 1.))
    for title, key, ylabel, factor in (
        ("Gradient norm before clipping", "grad_norm", "L2 norm", 1.),
        ("Learning rate", "lr", "Learning rate", 1.),
        ("Training throughput", "samples_per_second", "Samples / second", 1.),
        ("GPU allocated memory", "gpu_allocated_mib", "GiB", 1. / 1024),
        ("GPU reserved memory", "gpu_reserved_mib", "GiB", 1. / 1024),
        ("Peak GPU allocated memory", "gpu_peak_allocated_mib", "GiB", 1. / 1024),
    ):
        panels.append((title, ylabel, key, None, None, factor))
    panels = [panel for index, panel in enumerate(panels)
              if index == 0 or any(key in history.series for key in panel[2:5] if key)]
    rows = math.ceil(len(panels) / 3)
    figure = Figure(figsize=(16, 3.45 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, 3, squeeze=False).ravel()

    def sparse_line(axis, key, label, color, factor, marker):
        if key is None:
            return
        x, y = history.points(key)
        if len(x):
            axis.plot(x, y * factor, label=label, color=color, marker=marker,
                      markersize=3.8, linewidth=1.25, zorder=4)

    for index, (title, ylabel, train_key, epoch_key, val_key, factor) in enumerate(panels):
        axis = axes[index]
        x, y = history.points(train_key, max_points=None)
        if len(x):
            selected = np.linspace(0, len(x) - 1, min(len(x), 4000), dtype=int)
            smooth = len(x) >= 20 and train_key != "lr" and not train_key.startswith("gpu_")
            axis.plot(x[selected], y[selected] * factor, color="tab:blue",
                      alpha=.18 if smooth else .85, linewidth=.6 if smooth else 1.1,
                      label="Train batch" if train_key.startswith("train_") else "Measured")
            if smooth:
                window = min(100, len(y))
                cumulative = np.concatenate(([0.], np.cumsum(y)))
                means = (cumulative[window:] - cumulative[:-window]) / window
                mean_x = x[window - 1:]
                selected = np.linspace(0, len(means) - 1, min(len(means), 4000), dtype=int)
                axis.plot(mean_x[selected], means[selected] * factor, color="tab:blue",
                          linewidth=1.45, label=f"Mean ({window} batches)", zorder=3)
        sparse_line(axis, epoch_key, "Train epoch mean", "tab:red", factor, "^")
        sparse_line(axis, val_key, "Validation", "tab:orange", factor, "o")
        if index == 0:
            initial_x, initial_y = history.points("initial_train_mse")
            if len(initial_x):
                axis.plot(initial_x, initial_y, "x", color="black", label="Initial train", markersize=5)
        axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        axis.title.set_fontsize(11)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.2)
        axis.margins(x=.015, y=.08)
        if train_key == "lr":
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            if len(y) and np.ptp(y) <= np.max(np.abs(y)) * 1e-6:
                center = float(np.mean(y))
                margin = max(abs(center) * .05, 1e-9)
                axis.set_ylim(center - margin, center + margin)
        if axis.lines:
            # A shared color convention makes panels easy to compare; compact
            # legends explain only the curves actually present in this panel.
            if len(axis.lines) > 1:
                axis.legend(loc="best", fontsize=7.5, framealpha=.8)
        else:
            axis.text(.5, .5, "Waiting for measured metrics", ha="center", va="center",
                      transform=axis.transAxes, color="gray", fontsize=9)
    for axis in axes[len(panels):]:
        axis.set_visible(False)
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
                         isinstance(value, numbers.Number) or key in {
                             "objective", "metric_name", "metric_scope", "selection_metric", "training_stage"})
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
