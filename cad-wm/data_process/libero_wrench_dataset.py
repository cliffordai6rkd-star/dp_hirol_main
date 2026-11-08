"""LIBERO HDF5 demo windows for q/dq/tau -> future wrench and contact phases."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .h5_direct_dataset import load_h5py

if __package__ and "." in __package__:
    from ..train.nomalizer import Normalizer
else:
    from train.nomalizer import Normalizer


STATE_KEYS = ("q", "dq", "tau")


def prediction_offsets(config):
    data = config["dataloader"]
    horizon = int(data["prediction_horizon"])
    fps = float(data["high_fps"])
    if horizon < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("prediction_horizon and high_fps must be positive")
    # Reject older configs requesting a different output rate. Both training
    # targets and inference timestamps always follow consecutive native rows.
    step = data.get("prediction_step_s")
    if step is not None and not math.isclose(float(step), 1.0 / fps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("prediction_step_s must match 1/high_fps; input and output use the native rate")
    return torch.arange(1, horizon + 1, dtype=torch.float64) / fps


def wrench_phase_labels(wrench, config, fps):
    """Offline strict-threshold debounce; precontact precedes the first onset.

    Confirmed entry/exit windows are backfilled to their first frame. Equality
    retains state and resets both streaks. Call independently per demo to reset
    contact state; these offline targets require the full confirmation window.
    """
    values = config["contact_gate"]
    if values.get("metric") != "force_xyz_l2" or values.get("label_mode") != "three_phase":
        raise ValueError("LIBERO contact labels require force_xyz_l2 and three_phase")
    free, contact = float(values["free_threshold"]), float(values["contact_threshold"])
    enter, leave = int(values["contact_enter_frames"]), int(values["contact_exit_frames"])
    pre_seconds = float(values.get("precontact_seconds", 1.0))
    if (not all(math.isfinite(v) for v in (free, contact, pre_seconds))
            or not 0 <= free <= contact or min(enter, leave) < 1 or pre_seconds < 0):
        raise ValueError("invalid DP contact thresholds, frame counts, or precontact duration")
    magnitude = torch.linalg.vector_norm(wrench[:, :3], dim=-1)
    labels = torch.zeros(len(wrench), dtype=torch.long)
    state, above, below = False, 0, 0
    for index, value in enumerate(magnitude.tolist()):
        above = above + 1 if value > contact else 0
        below = below + 1 if value < free else 0
        if not state and above >= enter:
            state, above = True, 0
            labels[index - enter + 1:index + 1] = 2
        elif state and below >= leave:
            state, below = False, 0
            labels[index - leave + 1:index + 1] = 0
        labels[index] = 2 if state else 0
    onsets = torch.nonzero(labels == 2).flatten()
    if len(onsets):
        first = int(onsets[0])
        start = max(0, math.ceil(first - pre_seconds * fps - 1e-9))
        labels[start:first] = 1
    return labels


class LiberoWrenchDataset(torch.utils.data.Dataset):
    """Read only configured low-dimensional fields; keep demos independent."""

    def __init__(self, config):
        self.config = config
        data = config["dataloader"]
        if data.get("backend") != "h5":
            raise ValueError("LiberoWrenchDataset requires dataloader.backend=h5")
        self.history_horizon = int(data["state_history_horizon"])
        self.future_horizon = int(data["prediction_horizon"])
        self.sample_rate_hz = float(data["high_fps"])
        if self.history_horizon < 1 or not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("state_history_horizon and high_fps must be positive")
        self.future_offsets = prediction_offsets(config)
        self.future_rows = torch.arange(1, self.future_horizon + 1, dtype=torch.long)
        self.future_span = self.future_horizon
        self.horizon = self.history_horizon + self.future_span
        self.pad_history = bool(data.get("pad_history", True))
        self.normalize_mode = data.get("normalize_mode", "gaussian")
        if self.normalize_mode not in (None, "gaussian"):
            raise ValueError("LIBERO wrench loading supports gaussian normalization or null")
        self.normalizer = None
        self.filter_config = {}
        self.episodes, self.valid_indices = [], []
        self.raw_idx_to_episode_start = {}
        self.importance_weight_by_sample_index = {}
        self.high_tensors = {key: [] for key in (*STATE_KEYS, "wrench")}
        phase_parts = []
        root = Path(data["root"]).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        with load_h5py().File(root, "r") as source:
            demos = source[str(data.get("h5_episode_group", "data"))]
            env = json.loads(demos.attrs.get("env_args", "{}"))
            recorded_fps = env.get("env_kwargs", {}).get("control_freq")
            if recorded_fps is not None and not math.isclose(float(recorded_fps), self.sample_rate_hz):
                raise ValueError("high_fps differs from HDF5 env_args control_freq")
            names = sorted(demos, key=lambda name: int(name.rsplit("_", 1)[-1]))
            offset = 0
            for name in names:
                demo, arrays = demos[name], {}
                for key in self.high_tensors:
                    spec = data["h5_fields"][key]
                    dataset = demo[spec["path"]]
                    array = np.asarray(dataset, dtype=np.float32)
                    if array.ndim != 2:
                        raise ValueError(f"{name}/{spec['path']} must be [T, D]")
                    if spec.get("columns") is not None:
                        array = array[:, list(spec["columns"])]
                    width = 6 if key == "wrench" else int(config["model"].get("joint_dim", 7))
                    if array.shape[1] != width or not np.isfinite(array).all():
                        raise ValueError(f"{name}/{key} must be finite [T, {width}]")
                    if key == "tau" and "dt" in dataset.attrs:
                        if not math.isclose(float(dataset.attrs["dt"]), 1.0 / self.sample_rate_hz):
                            raise ValueError(f"{name}/tau dt differs from configured high_fps")
                    arrays[key] = torch.from_numpy(array.copy())
                lengths = {len(value) for value in arrays.values()}
                if len(lengths) != 1:
                    raise ValueError(f"{name} has unaligned q/dq/tau/wrench lengths")
                length = lengths.pop()
                first = 0 if self.pad_history else self.history_horizon - 1
                if length - self.future_span <= first:
                    raise ValueError(f"{name} is too short for the configured history/future")
                self.episodes.append({"episode_index": int(name.rsplit("_", 1)[-1]),
                                      "dataset_from_index": offset,
                                      "dataset_to_index": offset + length, "name": name})
                for index in range(offset + first, offset + length - self.future_span):
                    self.valid_indices.append(index)
                    self.raw_idx_to_episode_start[index] = offset
                for key, value in arrays.items():
                    self.high_tensors[key].append(value)
                phase_parts.append(wrench_phase_labels(arrays["wrench"], config, self.sample_rate_hz))
                offset += length
        if not self.valid_indices:
            raise ValueError("HDF5 contains no usable demo windows")
        self.high_tensors = {key: torch.cat(parts) for key, parts in self.high_tensors.items()}
        self.contact = torch.cat(phase_parts)
        self.dataset = SimpleNamespace(meta=SimpleNamespace(episodes=self.episodes))

    def __len__(self):
        return len(self.valid_indices)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def fit_normalizer(self, sample_indices):
        if self.normalize_mode is None:
            return
        covered = set()
        for sample_index in sample_indices:
            anchor = self.valid_indices[int(sample_index)]
            start = self.raw_idx_to_episode_start[anchor]
            covered.update(range(max(start, anchor - self.history_horizon + 1),
                                 anchor + self.future_span + 1))
        if not covered:
            raise ValueError("cannot fit normalizer without training rows")
        rows = torch.tensor(sorted(covered), dtype=torch.long)
        stats = {}
        for key, column in self.high_tensors.items():
            values = column[rows]
            stats[key] = {"mean": values.mean(0), "std": values.std(0, unbiased=False)}
        self.set_normalizer(Normalizer(stats))

    def __getitem__(self, index):
        anchor = self.valid_indices[index]
        start = self.raw_idx_to_episode_start[anchor]
        history = torch.arange(anchor - self.history_horizon + 1, anchor + 1).clamp_min(start)
        history_valid = anchor - self.history_horizon + 1 >= start
        # Padded rows and unconfirmed high-force streaks cannot supervise
        # free-space dynamics, even when debounce still labels them free.
        free_history = (history_valid and bool((self.contact[history] == 0).all())
                        and bool((torch.linalg.vector_norm(
                            self.high_tensors["wrench"][history, :3], dim=-1
                        ) <= float(self.config["contact_gate"]["free_threshold"])).all()))
        future_indices = anchor + self.future_rows
        future = self.high_tensors["wrench"][future_indices]
        sample = {key: self.high_tensors[key][history] for key in STATE_KEYS}
        sample.update(wrench_future=future, wrench_future_raw=future,
                      contact_future=self.contact[future_indices],
                      free_dynamics_mask=torch.tensor(free_history),
                      importance_weight=torch.tensor(
                          self.importance_weight_by_sample_index.get(anchor, 1.0), dtype=torch.float32
                      ),
                      future_time=self.future_offsets.float(), sample_idx=torch.tensor(anchor))
        if self.normalizer is not None:
            for key in STATE_KEYS:
                sample[key] = self.normalizer.gaussian_normalize(key, sample[key])
            sample["wrench_future"] = self.normalizer.gaussian_normalize("wrench", future)
        return sample
