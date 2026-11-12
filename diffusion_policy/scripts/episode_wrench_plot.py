"""Recorded wrench/joint torque and contact intervals for attention videos."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector


def backdate_confirmed_contact(causal_contact, enter_frames, exit_frames):
    """Place confirmed state changes at their first qualifying source sample.

    The detector confirms an edge on the final sample of a consecutive run.
    Offline labels can then include the entire confirming run. An unconfirmed
    exit at the end of an episode retains contact through the final sample.
    """
    causal_contact = np.asarray(causal_contact, dtype=bool)
    if causal_contact.ndim != 1 or enter_frames < 1 or exit_frames < 1:
        raise ValueError("Expected a one-dimensional contact mask and positive confirmation lengths")
    starts = np.flatnonzero(np.diff(np.r_[False, causal_contact].astype(np.int8)) == 1)
    stops = np.flatnonzero(np.diff(np.r_[causal_contact, False].astype(np.int8)) == -1) + 1
    contact = np.zeros_like(causal_contact)
    for start, stop in zip(starts, stops):
        first_contact = max(0, start - enter_frames + 1)
        first_free = stop - exit_frames + 1 if stop < len(contact) else stop
        contact[first_contact:first_free] = True
    return contact


def load_episode_wrench(dataset, policy, episode, anchors, dataset_fps):
    """Load physical data before normalization, gating, or future-window shifts.

    Contact is scanned once over the complete source episode, before any video
    stride or frame limit is applied. Confirmed transitions are backdated to
    the start of their confirmation window for offline annotation; the causal
    detector mask is retained separately. Dataset sampling phase labels include
    future windows and are intentionally not used as current contact states.
    """
    source_key = getattr(dataset, "phase_source_key", "observation.wrench")
    source = getattr(dataset, "_phase_source_data", None)
    if source is None:
        source = dataset._load_column(source_key, dtype=np.float32)
    episode_range = dataset.episode_ranges[episode]
    values = np.asarray(source[episode_range.start:episode_range.stop], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in (6, 7) or not np.isfinite(values).all():
        raise ValueError(f"Physical signal plotting requires finite [N, 6] wrench or [N, 7] joint torques in {source_key!r}; got {values.shape}")
    joint_torques = values.shape[1] == 7
    if source_key.rsplit(".", 1)[-1].startswith("tau_ext") and not joint_torques:
        raise ValueError(f"Joint torque source {source_key!r} requires seven channels; got {values.shape}")
    timestamps = np.asarray(dataset.timestamps[episode_range.start:episode_range.stop], dtype=np.float64)
    timestamp_key = getattr(dataset, "timestamp_key", "timestamp")
    if hasattr(dataset, "_timestamp_seconds"):
        timestamps = dataset._timestamp_seconds(timestamps, key=timestamp_key)
    times = timestamps - timestamps[0]
    if len(times) != len(values) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Recorded wrench timestamps must be finite and strictly increasing within an episode")
    offsets = np.asarray(anchors, dtype=np.int64) - episode_range.start
    if offsets.ndim != 1 or (offsets < 0).any() or (offsets >= len(values)).any():
        raise ValueError("Visualization anchors must belong to the requested source episode")

    detector = getattr(policy, "contact_detector", None)
    hysteresis = detector is None or hasattr(detector, "free_threshold")
    if detector is not None:
        force_dims = tuple(detector.force_dims)
        norm = detector.norm
        threshold = getattr(detector, "contact_threshold", None)
        contact_threshold = float(threshold if threshold is not None else detector.threshold)
        free_threshold = float(getattr(detector, "free_threshold", contact_threshold))
        enter_frames = int(getattr(detector, "enter_frames", 1))
        exit_frames = int(getattr(detector, "exit_frames", 1))
        rule_source = "teacher.contact_detector"
    else:
        force_dims = tuple(getattr(dataset, "contact_force_dims", (0, 1, 2)))
        norm = getattr(dataset, "contact_norm", "l2")
        threshold = getattr(dataset, "contact_threshold", None)
        if threshold is None:
            raise ValueError("Contact annotations require a teacher or dataset contact threshold")
        contact_threshold = free_threshold = float(threshold)
        enter_frames = int(getattr(dataset, "contact_enter_frames", 1))
        exit_frames = int(getattr(dataset, "contact_exit_frames", 1))
        rule_source = "dataset.contact_configuration"
    force = values[:, force_dims]
    magnitude = (np.abs(force).sum(axis=-1) if norm in {"l1", "sum"}
                 else np.linalg.norm(force, axis=-1))
    if hysteresis:
        # A separate CPU detector avoids changing the teacher's device or state.
        scan = HysteresisContactDetector(
            free_threshold=free_threshold, contact_threshold=contact_threshold,
            force_dims=force_dims, norm=norm,
            enter_frames=enter_frames, exit_frames=exit_frames,
        )
        with torch.inference_mode():
            causal_contact = scan.rollout_future(
                torch.from_numpy(values)[None, None], torch.zeros((1, 1), dtype=torch.bool)
            )[0, 0].numpy()
        contact = backdate_confirmed_contact(causal_contact, enter_frames, exit_frames)
    else:
        contact = causal_contact = magnitude > contact_threshold
    starts = np.flatnonzero(np.diff(np.r_[False, contact].astype(np.int8)) == 1)
    stops = np.flatnonzero(np.diff(np.r_[contact, False].astype(np.int8)) == -1) + 1
    sample_period = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / dataset_fps
    time_edges = np.r_[times, times[-1] + sample_period]
    intervals = np.asarray([(time_edges[start], time_edges[stop])
                            for start, stop in zip(starts, stops)], dtype=np.float64).reshape(-1, 2)
    metadata = {
        "source_key": source_key,
        "signal_type": "joint_torque" if joint_torques else "wrench",
        "channels": ([f"J{index + 1}" for index in range(7)] if joint_torques
                     else ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]),
        "units": ["Nm"] * 7 if joint_torques else ["N", "N", "N", "N m", "N m", "N m"],
        "values": ("Recorded current tau_ext joint torques, before model normalization or input gating"
                   if joint_torques else
                   "Recorded current wrench, before model normalization or input gating; not future_wrench"),
        "timestamp_key": timestamp_key,
        "time_axis": "Recorded source timestamp minus the first source timestamp of this episode",
        "source_frames": len(values), "displayed_attention_frames": len(offsets),
        "contact": {
            "source": "derived_from_recorded_tau_ext" if joint_torques else "derived_from_recorded_wrench",
            "rule_source": rule_source,
            "force_dims": list(force_dims), "norm": norm,
            "contact_threshold": contact_threshold, "free_threshold": free_threshold,
            "enter_frames": enter_frames, "exit_frames": exit_frames,
            "annotation_mode": ("offline_backdated_confirmation_windows" if hysteresis
                                else "per_frame_threshold"),
            "definition": (
                "Scan all source rows, initially free; after consecutive samples confirm entry above contact threshold or exit below free threshold, backdate that transition to the first sample of its confirmation window. Short unconfirmed runs do not change state; an unconfirmed exit at episode end retains contact. Intervals are [start, end). This is offline annotation, not the teacher's causal confirmation time."
                if hysteresis else
                "Per-source-row magnitude > teacher contact threshold, without hysteresis; intervals are [start, end)."
            ),
            "intervals_sec": intervals.tolist(),
        },
    }
    return {"values": values, "times": times, "contact": contact,
            "causal_contact": causal_contact,
            "contact_magnitude": magnitude, "anchor_offsets": offsets,
            "anchor_times": times[offsets], "contact_intervals_sec": intervals,
            "metadata": metadata}


class WrenchEpisodePlot:
    """Reuse a Matplotlib plot while moving its cursor with each image frame."""

    def __init__(self, timeline, width, height=420):
        self.timeline = timeline
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.figure.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.84, hspace=0.27)
        times, values = timeline["times"], timeline["values"]
        metadata = timeline["metadata"]
        self.joint_torques = metadata.get("signal_type") == "joint_torque"
        labels = metadata["channels"]
        if self.joint_torques:
            groups = ((range(4), "Joint torque (Nm)"), (range(4, 7), "Joint torque (Nm)"))
            colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")
        else:
            groups = ((range(3), "Force (N)"), (range(3, 6), "Torque (N m)"))
            colors = ("#1f77b4", "#ff7f0e", "#2ca02c") * 2
        self.cursors, self.markers = [], []
        for ax, (channels, ylabel) in zip(self.axes, groups):
            for start, stop in timeline["contact_intervals_sec"]:
                ax.axvspan(start, stop, color="#e9b96e", alpha=0.35, linewidth=0)
            for channel in channels:
                name, color = labels[channel], colors[channel]
                ax.plot(times, values[:, channel], label=name, color=color, linewidth=1.25)
                marker, = ax.plot([], [], "o", color=color, markersize=3)
                self.markers.append(marker)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper right", ncol=len(channels), fontsize=8, framealpha=0.85,
                      handlelength=1.2, columnspacing=0.7)
            ax.margins(x=0, y=0.15)
            self.cursors.append(ax.axvline(times[0], color="#303030", linestyle="--", linewidth=1))
        self.axes[-1].set_xlabel("Episode time (s)", fontsize=9)
        if len(times) == 1:
            self.axes[-1].set_xlim(times[0] - 0.05, times[0] + 0.05)
        self.status = self.figure.text(0.15, 0.945, "", fontsize=10, weight="bold")
        contact_cfg = timeline["metadata"]["contact"]
        if self.joint_torques:
            dims = ",".join(labels[index] for index in contact_cfg["force_dims"])
            signal = "tau_ext" if contact_cfg["force_dims"] == list(range(7)) else f"tau_ext[{dims}]"
        else:
            dims = "xyz" if contact_cfg["force_dims"] == [0, 1, 2] else str(contact_cfg["force_dims"])
            signal = f"F{dims}"
        norm = "2" if contact_cfg["norm"] == "l2" else "1"
        enter, leave = contact_cfg["contact_threshold"], contact_cfg["free_threshold"]
        thresholds = (f"> {enter:g}, < {leave:g}"
                      if contact_cfg["annotation_mode"] == "offline_backdated_confirmation_windows"
                      else f"> {enter:g}")
        self.figure.text(0.15, 0.897, f"Shaded: derived contact | ||{signal}||{norm} {thresholds}",
                         fontsize=8, color="#555555")
        confirmation_label = (
            f"{contact_cfg['enter_frames']}/{contact_cfg['exit_frames']}-frame confirmation | boundaries backdated to window start"
            if contact_cfg["annotation_mode"] == "offline_backdated_confirmation_windows"
            else f"Per-frame threshold | current recorded {'tau_ext' if self.joint_torques else 'wrench'}"
        )
        self.figure.text(0.15, 0.862, confirmation_label, fontsize=8, color="#555555")

    def render(self, current_time):
        times = self.timeline["times"]
        index = int(np.argmin(np.abs(times - current_time)))
        state = "CONTACT" if self.timeline["contact"][index] else "FREE"
        magnitude = self.timeline["contact_magnitude"][index]
        if self.joint_torques:
            self.status.set_text(f"t={current_time:.2f}s | {state} | tau_ext norm = {magnitude:.2f}")
        else:
            self.status.set_text(f"t = {current_time:.2f} s   |   {state}   |   force norm = {magnitude:.2f}")
        for cursor in self.cursors:
            cursor.set_xdata([current_time, current_time])
        for channel, marker in enumerate(self.markers):
            marker.set_data([times[index]], [self.timeline["values"][index, channel]])
        self.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[..., :3])

    def save(self, path):
        """Export the complete data plot without a selected observation cursor."""
        for artist in (*self.cursors, *self.markers):
            artist.set_visible(False)
        self.status.set_text("Recorded seven-axis tau_ext and derived contact" if self.joint_torques
                             else "Recorded six-axis wrench and derived contact")
        self.figure.savefig(path, dpi=200, facecolor="white")
        for artist in (*self.cursors, *self.markers):
            artist.set_visible(True)
