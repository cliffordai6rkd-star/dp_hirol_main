# 模型学习一个长度为horizon的window  不仅学习点到点的映射关系 也学习力的变化趋势
# 变量: q v u(action without gripper, tau)
#      wrench(lambda) Fx Fy Fz τx τy τz
# img -> phi\miu -> loss


import torch
import os
import argparse
import yaml
import logging as log
from types import SimpleNamespace

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from pathlib import Path
from train.nomalizer import Normalizer
from model.pinn_model.contact_gate import (
    ContactGateConfig,
    contact_labels_from_wrench,
    load_contact_label_cache,
)
from data_process.causal_data_filter import build_filtered_dataset_view


def _load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Dataset loading requires the lerobot package. Model/loss-only "
            "tests can run without it."
        ) from exc
    return LeRobotDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert standalone .h5/.hdf5 episode files to LeRobot v3."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/tool/config/dataset_test_cfg.yaml"),
        help="Path to the config",
    )
    return parser.parse_args()

class PINNDataset(torch.utils.data.Dataset):
    def __init__(self, config,
                    normalizer=None,
                    normalize_mode=None,
                    compute_normalizer=True,):
        # repo_id, root, 
        self.config = config
        self.data_config = config.get("dataloader") or {}
        self.backend = str(self.data_config.get("backend", "lerobot")).lower()
        root = self.data_config.get("root")
        self.root = Path(root) if root is not None else None
        self.repo_id = self.data_config.get("repo_id",None)
        self.video_backend = self.data_config.get("video_backend", "torchcodec")
        self.lowdim_keys = self.data_config.get("lowdim_keys", {})
        self.load_image = bool(self.data_config.get("load_images",True))
        if self.backend == "lerobot":
            if not self.repo_id or self.root is None:
                raise ValueError("dataloader.repo_id and dataloader.root are required")
            LeRobotDataset = _load_lerobot_dataset_class()
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root,
                video_backend=self.video_backend
            )
        elif self.backend == "h5":
            if self.root is None:
                raise ValueError("dataloader.root is required for backend=h5")
            if self.load_image:
                raise ValueError(
                    "PINNDataset backend=h5 currently supports low-dimensional "
                    "fields only; set dataloader.load_images=false"
                )
            from data_process.h5_direct_dataset import DirectH5EpisodeDataset

            h5_fields = self.data_config.get("h5_fields") or {}
            required_columns = set(self.lowdim_keys.values())
            missing_specs = sorted(required_columns - set(h5_fields))
            if missing_specs:
                raise ValueError(
                    "dataloader.h5_fields is missing direct H5 specifications "
                    f"for columns: {missing_specs}"
                )
            self.dataset = DirectH5EpisodeDataset(
                root=self.root,
                fields={name: h5_fields[name] for name in required_columns},
                timestamp_path=str(
                    self.data_config.get("h5_timestamp_path", "teleop/timestamp_us")
                ),
                timestamp_output_key=str(
                    self.data_config.get("h5_timestamp_output_key", "timestamp")
                ),
                timestamp_unit=str(
                    self.data_config.get("h5_timestamp_unit", "us")
                ),
                timestamp_output_unit=str(
                    self.data_config.get("h5_timestamp_output_unit", "s")
                ),
                patterns=tuple(
                    self.data_config.get("h5_patterns", ["*.h5", "*.hdf5"])
                ),
                max_episodes=self.data_config.get("max_episodes"),
                expected_fps=self.data_config.get("expected_fps"),
                max_cadence_error_s=float(
                    self.data_config.get("max_cadence_error_s", 1.0e-6)
                ),
            )
        else:
            raise ValueError(
                "dataloader.backend must be 'lerobot' or 'h5', "
                f"got {self.backend!r}"
            )
        self.stats_dataset, self.filter_config = build_filtered_dataset_view(
            self.dataset.hf_dataset,
            data_config=self.data_config,
            lowdim_keys=self.lowdim_keys,
            episodes=self.dataset.meta.episodes,
        )
        self.sample_rate_hz = self._resolve_sample_rate_hz()
        # self.dt = float(1/30)  # 采样frequency 30Hz

        self.horizon = int(self.data_config.get("horizon", 1))
        self.history_horizon = int(self.data_config.get("history_horizon", self.horizon))
        self.future_horizon = int(self.data_config.get("future_horizon", self.horizon))
        self.pad_history = bool(self.data_config.get("pad_history", True))

        self.valid_indices = []
        self.raw_idx_to_episode_start = {}
        self.raw_idx_to_episode_end = {}
        self._build_valid_indices()

        self.normalize_lowdim_keys = self.data_config.get("normalize_lowdim_keys",None)
        if self.normalize_lowdim_keys is None:
            self.normalize_lowdim_keys = []
        if self.load_image:
            self.image_keys = self.data_config.get("image_keys", {})
        else:
            self.image_keys = {}
        self.normalize_mode = None
        self.is_normalize = False
        self.normalizer = None
        self.normalize_fuc = None
        self.normalize_mode = normalize_mode
        if self.normalize_mode is None:
            self.normalize_mode = self.data_config.get("normalize_mode", "gaussian")

        if self.normalize_mode is not None:
            if normalizer is not None:
                self.set_normalizer(normalizer)
            elif compute_normalizer:
                self.fit_normalizer(self.valid_indices)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.is_normalize = True
        if self.normalize_mode == "gaussian":
            self.normalize_fuc = self.normalizer.gaussian_normalize
        elif self.normalize_mode == "limit":
            self.normalize_fuc = self.normalizer.limit_normalize
        elif self.normalize_mode == "quantile":
            self.normalize_fuc = self.normalizer.quantile_normalize
        else:
            raise ValueError(f"unknown normalize mode: {self.normalize_mode}")

    def fit_normalizer(self, raw_indices):
        if self.normalize_mode is None:
            return
        raw_indices = sorted(set(int(index) for index in raw_indices))
        if not raw_indices:
            raise ValueError("cannot fit normalizer with no training frames")
        normalizer = Normalizer.stats_from_dataset(
            dataset=self.stats_dataset,
            valid_indices=raw_indices,
            lowdim_keys=self.lowdim_keys,
            normalize_keys=self.normalize_lowdim_keys,
        )
        self.set_normalizer(normalizer)

    def covered_raw_indices(self, sample_indices):
        covered = set()
        for sample_idx in sample_indices:
            raw_idx = self.valid_indices[int(sample_idx)]
            episode_start = self.raw_idx_to_episode_start[raw_idx]
            covered.update(
                max(episode_start, raw_idx - self.horizon + 1 + offset)
                for offset in range(self.horizon)
            )
        return sorted(covered)

    def _build_valid_indices(self):
        episodes = self.dataset.meta.episodes

        for ep in episodes:
            start_idx = int(ep["dataset_from_index"])
            end_idx = int(ep["dataset_to_index"])
            valid_start = start_idx
            if not self.pad_history:
                valid_start = start_idx + self.horizon - 1

            for idx in range(valid_start, end_idx):
                self.valid_indices.append(idx)
                self.raw_idx_to_episode_start[idx] = start_idx
                self.raw_idx_to_episode_end[idx] = end_idx

    def __len__(self):
        return len(self.valid_indices) 
    
    def __getitem__(self, idx):

        raw_idx = self.valid_indices[idx]
        episode_start = self.raw_idx_to_episode_start[raw_idx]

        # 构造horizon窗口  用max来结局解决开头几帧的问题  相当于padding
        frame_indices = [
                max(episode_start, raw_idx - self.horizon + 1 + offset)
                for offset in range(self.horizon)
        ]


        frames = [self._read_frame(i) for i in frame_indices]

        sample = {}
        
        # ("q", "observation.joint") —> key = "q" , dataset_key = "observation.joint"
        for key, dataset_key in self.lowdim_keys.items():
            seq = [frame[dataset_key] for frame in frames]
            sample[f"{key}"] = torch.stack(seq, dim=0)

        for key in self.normalize_lowdim_keys:
            if self.is_normalize:
                sample[f"{key}"] = self.normalize_fuc(key, sample[f"{key}"])
    
        for key, dataset_key in self.image_keys.items():
            seq = [frame[dataset_key] for frame in frames]
            sample[f"image_{key}"] = torch.stack(seq, dim=0)            

        return sample
    
    def _read_frame(self, i):
        filtered_frame = self.stats_dataset[i]
        if not self.load_image:
            return filtered_frame
        frame = dict(self.dataset[i])
        for dataset_key in self.lowdim_keys.values():
            if dataset_key in filtered_frame:
                frame[dataset_key] = filtered_frame[dataset_key]
        return frame

    def _resolve_sample_rate_hz(self):
        configured = self.data_config.get(
            "filter_sample_rate_hz",
            self.data_config.get("expected_fps"),
        )
        if configured is None:
            metadata = getattr(self.dataset, "meta", None)
            configured = getattr(metadata, "fps", None)
            if configured is None:
                info = getattr(metadata, "info", None)
                if isinstance(info, dict):
                    configured = info.get("fps")
        if configured is None:
            return None
        sample_rate_hz = float(configured)
        if not torch.isfinite(torch.tensor(sample_rate_hz)) or sample_rate_hz <= 0.0:
            raise ValueError("dataloader filter sample rate must be positive and finite")
        return sample_rate_hz


class PINNHistoryFutureDataset(PINNDataset):
    def _build_valid_indices(self):
        episodes = self.dataset.meta.episodes
        self.pad_future = bool(self.data_config.get("pad_future", False))

        for ep in episodes:
            start_idx = int(ep["dataset_from_index"])
            end_idx = int(ep["dataset_to_index"])
            if self.pad_future:
                valid_end = end_idx
            else:
                valid_end = end_idx - self.future_horizon

            for idx in range(start_idx, max(start_idx, valid_end)):
                self.valid_indices.append(idx)
                self.raw_idx_to_episode_start[idx] = start_idx
                self.raw_idx_to_episode_end[idx] = end_idx

    def __getitem__(self, idx):
        raw_idx = self.valid_indices[idx]
        episode_start = self.raw_idx_to_episode_start[raw_idx]
        episode_end = self.raw_idx_to_episode_end[raw_idx]

        # raw_idx 是历史观测窗口的最后一帧，未来窗口从 raw_idx + 1 开始。
        history_indices = [
            max(episode_start, raw_idx - self.history_horizon + 1 + offset)
            for offset in range(self.history_horizon)
        ]
        future_indices = [
            min(episode_end - 1, raw_idx + 1 + offset)
            for offset in range(self.future_horizon)
        ]

        history_frames = [self._read_frame(i) for i in history_indices]
        future_frames = [self._read_frame(i) for i in future_indices]

        sample = {
            "raw_idx": torch.tensor(raw_idx, dtype=torch.long),
            "history_indices": torch.tensor(history_indices, dtype=torch.long),
            "future_indices": torch.tensor(future_indices, dtype=torch.long),
        }

        pinocchio_raw_keys = set(
            self.data_config.get("pinocchio_raw_keys", ["q", "v", "a", "tau"])
        )

        for key, dataset_key in self.lowdim_keys.items():
            history_seq = torch.stack([frame[dataset_key] for frame in history_frames], dim=0)
            future_seq = torch.stack([frame[dataset_key] for frame in future_frames], dim=0)

            sample[key] = history_seq
            sample[f"{key}_future"] = future_seq

            # Pinocchio 必须使用真实物理量；如果后续对 q/v/a/tau 做归一化，这里保留原始未来量。
            if key in pinocchio_raw_keys:
                sample[f"{key}_future_raw"] = future_seq.clone()

        for key in self.normalize_lowdim_keys:
            if self.is_normalize:
                sample[key] = self.normalize_fuc(key, sample[key])
                future_key = f"{key}_future"
                if future_key in sample:
                    sample[future_key] = self.normalize_fuc(key, sample[future_key])

        for key, dataset_key in self.image_keys.items():
            seq = [frame[dataset_key] for frame in history_frames]
            sample[f"image_{key}"] = torch.stack(seq, dim=0)

        return sample


class PackedLowdimFlowDataset(torch.utils.data.Dataset):
    """Rebuild a continuous low-dimensional stream from packed LeRobot frames.

    Each source LeRobot row is expected to contain a block shaped [S, D] for
    every configured low-dimensional feature. Source rows are ordered by the
    LeRobot frame index and block entries by their inner index, yielding the
    virtual flow index:

        flow_idx = source_frame_idx * S + sub_idx

    History and future chunks are then sliced on this virtual flow timeline.
    """

    def __init__(
        self,
        config,
        normalizer=None,
        normalize_mode=None,
        compute_normalizer=True,
    ):
        self.config = config
        self.data_config = config.get("dataloader") or {}
        self.repo_id = self.data_config.get("repo_id")
        root = self.data_config.get("root")
        self.root = Path(root) if root is not None else None
        if not self.repo_id or self.root is None:
            raise ValueError("dataloader.repo_id and dataloader.root are required.")

        LeRobotDataset = _load_lerobot_dataset_class()
        self.source_dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            video_backend=self.data_config.get("video_backend", "torchcodec"),
        )
        self.stats_dataset = self.source_dataset.hf_dataset
        self.lowdim_keys = self.data_config.get("lowdim_keys") or {}
        if not self.lowdim_keys:
            raise ValueError("dataloader.lowdim_keys must not be empty.")
        self.future_condition_keys = (
            self.data_config.get("future_condition_keys") or {}
        )
        duplicate_keys = set(self.lowdim_keys) & set(self.future_condition_keys)
        if duplicate_keys:
            raise ValueError(
                "Keys cannot appear in both lowdim_keys and "
                f"future_condition_keys: {sorted(duplicate_keys)}"
            )

        self.history_horizon = int(
            self.data_config.get("history_horizon", 8)
        )
        self.future_horizon = int(
            self.data_config.get("future_horizon", 8)
        )
        self.horizon = self.history_horizon + self.future_horizon
        self.pad_history = bool(self.data_config.get("pad_history", True))
        self.pad_future = bool(self.data_config.get("pad_future", False))
        if self.history_horizon <= 0 or self.future_horizon <= 0:
            raise ValueError(
                "history_horizon and future_horizon must be positive."
            )

        self.flow_tensors, self.packed_window_size = (
            self._load_and_flatten_lowdim_columns()
        )
        self.future_condition_offset = int(
            self.data_config.get(
                "future_condition_offset",
                self.packed_window_size - 1,
            )
        )
        if self.future_condition_offset < 0:
            raise ValueError("future_condition_offset must be non-negative.")
        self.episodes = self._build_virtual_episodes()
        self.contact_gate_config = ContactGateConfig.from_config(config)
        self._apply_contact_gate()
        # BaseTrainer expects dataset.dataset.meta.episodes. Keep the source
        # LeRobot dataset separately and expose virtual flow episode bounds here.
        self.dataset = SimpleNamespace(
            meta=SimpleNamespace(episodes=self.episodes)
        )

        self.valid_indices = []
        self.raw_idx_to_episode_start = {}
        self.raw_idx_to_episode_end = {}
        self._build_valid_indices()

        self.normalize_lowdim_keys = list(
            self.data_config.get("normalize_lowdim_keys") or []
        )
        unknown_normalize_keys = [
            key
            for key in self.normalize_lowdim_keys
            if key not in self.flow_tensors
        ]
        if unknown_normalize_keys:
            raise KeyError(
                "normalize_lowdim_keys are missing from lowdim_keys: "
                f"{unknown_normalize_keys}"
            )

        self.normalize_mode = (
            normalize_mode
            if normalize_mode is not None
            else self.data_config.get("normalize_mode", "gaussian")
        )
        self.is_normalize = False
        self.normalizer = None
        self.normalize_fuc = None
        if self.normalize_mode is not None:
            if normalizer is not None:
                self.set_normalizer(normalizer)
            elif compute_normalizer:
                self.fit_normalizer(range(self.flow_length))

    @property
    def flow_length(self):
        first_tensor = next(iter(self.flow_tensors.values()))
        return int(first_tensor.shape[0])

    def _load_and_flatten_lowdim_columns(self):
        all_keys = {**self.lowdim_keys, **self.future_condition_keys}
        dataset_keys = list(dict.fromkeys(all_keys.values()))
        formatted_dataset = self.stats_dataset.with_format(
            "torch",
            columns=dataset_keys,
            output_all_columns=False,
        )
        columns = formatted_dataset[:]

        packed_tensors = {}
        packed_window_size = None
        source_frames = None
        for key, dataset_key in all_keys.items():
            value = columns[dataset_key]
            if not torch.is_tensor(value):
                value = torch.stack(
                    [torch.as_tensor(item) for item in value],
                    dim=0,
                )
            value = value.to(dtype=torch.float32)
            if value.ndim != 3:
                raise ValueError(
                    f"Packed feature {dataset_key!r} must have shape "
                    f"[N, S, D], got {tuple(value.shape)}."
                )

            if source_frames is None:
                source_frames = int(value.shape[0])
                packed_window_size = int(value.shape[1])
            elif value.shape[:2] != (source_frames, packed_window_size):
                raise ValueError(
                    "All packed low-dimensional features must share [N, S]; "
                    f"{dataset_key!r} has {tuple(value.shape[:2])}, expected "
                    f"{(source_frames, packed_window_size)}."
                )
            packed_tensors[key] = value.contiguous()

        expected_window_size = self.data_config.get("packed_window_size")
        if (
            expected_window_size is not None
            and int(expected_window_size) != packed_window_size
        ):
            raise ValueError(
                f"Configured packed_window_size={expected_window_size}, but "
                f"LeRobot features contain S={packed_window_size}."
            )

        flattened = {
            key: value.reshape(-1, value.shape[-1]).contiguous()
            for key, value in packed_tensors.items()
        }
        return flattened, packed_window_size

    def _build_virtual_episodes(self):
        episodes = []
        for episode in self.source_dataset.meta.episodes:
            source_start = int(episode["dataset_from_index"])
            source_end = int(episode["dataset_to_index"])
            episodes.append(
                {
                    **dict(episode),
                    "source_dataset_from_index": source_start,
                    "source_dataset_to_index": source_end,
                    "dataset_from_index": (
                        source_start * self.packed_window_size
                    ),
                    "dataset_to_index": (
                        source_end * self.packed_window_size
                    ),
                }
            )
        return episodes

    def _apply_contact_gate(self):
        gate = self.contact_gate_config
        if not gate.enabled:
            return
        if gate.wrench_key not in self.flow_tensors:
            raise KeyError(
                "contact gating requires flow tensor "
                f"{gate.wrench_key!r}"
            )
        if gate.contact_key in self.flow_tensors:
            raise KeyError(
                f"contact gate key already exists: {gate.contact_key!r}"
            )
        bounds = [
            (
                int(episode["dataset_from_index"]),
                int(episode["dataset_to_index"]),
            )
            for episode in self.episodes
        ]
        if gate.label_cache_path is None:
            labels = contact_labels_from_wrench(
                self.flow_tensors[gate.wrench_key],
                bounds,
                gate,
            )
        else:
            cache_path = Path(gate.label_cache_path).expanduser()
            if not cache_path.is_absolute():
                cache_path = self.root / cache_path
            labels = load_contact_label_cache(
                cache_path,
                flow_length=self.flow_length,
                episode_bounds=bounds,
                config=gate,
            )
        self.contact_raw_wrench = self.flow_tensors[gate.wrench_key]
        self.flow_tensors[gate.contact_key] = labels
        self.flow_tensors[gate.wrench_key] = (
            self.flow_tensors[gate.wrench_key] * labels
        )

    def _build_valid_indices(self):
        for episode in self.episodes:
            episode_start = int(episode["dataset_from_index"])
            episode_end = int(episode["dataset_to_index"])
            valid_start = episode_start
            if not self.pad_history:
                valid_start = episode_start + self.history_horizon - 1
            if self.future_condition_keys:
                valid_start = max(
                    valid_start,
                    episode_start + self.future_condition_offset - 1,
                )
            valid_end = (
                episode_end
                if self.pad_future
                else episode_end - self.future_horizon
            )

            for flow_idx in range(valid_start, max(valid_start, valid_end)):
                self.valid_indices.append(flow_idx)
                self.raw_idx_to_episode_start[flow_idx] = episode_start
                self.raw_idx_to_episode_end[flow_idx] = episode_end

    def __len__(self):
        return len(self.valid_indices)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.is_normalize = True
        normalize_functions = {
            "gaussian": self.normalizer.gaussian_normalize,
            "limit": self.normalizer.limit_normalize,
            "quantile": self.normalizer.quantile_normalize,
        }
        if self.normalize_mode not in normalize_functions:
            raise ValueError(f"unknown normalize mode: {self.normalize_mode}")
        self.normalize_fuc = normalize_functions[self.normalize_mode]

    def fit_normalizer(self, flow_indices):
        if self.normalize_mode is None:
            return
        flow_indices = sorted(set(int(index) for index in flow_indices))
        if not flow_indices:
            raise ValueError("cannot fit normalizer with no training flow steps")
        cache_device = next(iter(self.flow_tensors.values())).device
        index_tensor = torch.as_tensor(
            flow_indices,
            device=cache_device,
            dtype=torch.long,
        )
        stats = {}
        for key in self.normalize_lowdim_keys:
            value = self.flow_tensors[key].index_select(0, index_tensor)
            stats[key] = {
                "mean": value.mean(dim=0),
                "std": value.std(dim=0, unbiased=False),
                "min": value.min(dim=0).values,
                "max": value.max(dim=0).values,
                "q01": torch.quantile(value, 0.01, dim=0),
                "q99": torch.quantile(value, 0.99, dim=0),
            }
        self.set_normalizer(Normalizer(stats))

    def covered_raw_indices(self, sample_indices):
        covered = set()
        for sample_idx in sample_indices:
            flow_idx = self.valid_indices[int(sample_idx)]
            episode_start = self.raw_idx_to_episode_start[flow_idx]
            episode_end = self.raw_idx_to_episode_end[flow_idx]
            history_start = max(
                episode_start,
                flow_idx - self.history_horizon + 1,
            )
            future_end = min(
                episode_end,
                flow_idx + 1 + self.future_horizon,
            )
            covered.update(range(history_start, flow_idx + 1))
            covered.update(range(flow_idx + 1, future_end))
        return sorted(covered)

    def __getitem__(self, idx):
        flow_idx = self.valid_indices[idx]
        episode_start = self.raw_idx_to_episode_start[flow_idx]
        episode_end = self.raw_idx_to_episode_end[flow_idx]
        history_indices = [
            max(
                episode_start,
                flow_idx - self.history_horizon + 1 + offset,
            )
            for offset in range(self.history_horizon)
        ]
        future_indices = [
            min(episode_end - 1, flow_idx + 1 + offset)
            for offset in range(self.future_horizon)
        ]
        history_index_tensor = torch.as_tensor(
            history_indices,
            dtype=torch.long,
        )
        future_index_tensor = torch.as_tensor(
            future_indices,
            dtype=torch.long,
        )

        sample = {
            "raw_idx": torch.tensor(flow_idx, dtype=torch.long),
            "history_indices": history_index_tensor,
            "future_indices": future_index_tensor,
            "history_source_frame_indices": (
                history_index_tensor // self.packed_window_size
            ),
            "history_sub_indices": (
                history_index_tensor % self.packed_window_size
            ),
            "future_source_frame_indices": (
                future_index_tensor // self.packed_window_size
            ),
            "future_sub_indices": (
                future_index_tensor % self.packed_window_size
            ),
        }

        raw_keys = set(
            self.data_config.get(
                "pinocchio_raw_keys",
                ["q"],
            )
        )
        for key, flow_tensor in self.flow_tensors.items():
            if key in self.future_condition_keys:
                continue
            history = flow_tensor.index_select(0, history_index_tensor)
            future = flow_tensor.index_select(0, future_index_tensor)
            sample[key] = history
            sample[f"{key}_future"] = future
            if key in raw_keys:
                sample[f"{key}_future_raw"] = future.clone()

        for key in self.normalize_lowdim_keys:
            if key in self.future_condition_keys:
                continue
            if self.is_normalize:
                sample[key] = self.normalize_fuc(key, sample[key])
                sample[f"{key}_future"] = self.normalize_fuc(
                    key,
                    sample[f"{key}_future"],
                )

        condition_indices = torch.clamp(
            future_index_tensor - self.future_condition_offset,
            min=episode_start,
            max=episode_end - 1,
        )
        sample["future_condition_indices"] = condition_indices
        for key in self.future_condition_keys:
            condition = self.flow_tensors[key].index_select(
                0,
                condition_indices,
            )
            if key in self.normalize_lowdim_keys and self.is_normalize:
                condition = self.normalize_fuc(key, condition)
            sample[f"{key}_future"] = condition
        return sample
        
if __name__ == "__main__":





    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"config :{config}")
    dataset = PINNDataset(config)
    log.info(f"dataset len : {len(dataset)}")
    assert len(dataset) > 0

    log.info(f"dataset len: {len(dataset)}")
    log.info(f"horizon: {dataset.horizon}")
    log.info(f"num episodes: {len(dataset.dataset.meta.episodes)}")
    log.info(f"first valid index: {dataset.valid_indices[0]}")


    for i in range(1,100):
        sample = dataset[i]
        log.info(f"sample keys: {sample.keys()}")
        log.info(f"sample success")
        # for k, v in sample.items():
        #     log.info(f"sample {k}: shape={v.shape}, dtype={v.dtype}")
        #     if torch.is_tensor(v) and v.is_floating_point():
        #         assert torch.isfinite(v).all(), f"{k} has nan or inf"

        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=4 ,
                                            shuffle=False,
                                            num_workers=4)

        batch = next(iter(loader))

        for k, v in batch.items():
            log.info(f"batch data shape : {k, v.shape, v.dtype}")
            if torch.is_tensor(v) and v.is_floating_point():
                assert torch.isfinite(v).all(), f"{k} has nan or inf" # 检查是否有非法数值

    
