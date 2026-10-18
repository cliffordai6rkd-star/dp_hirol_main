from typing import Dict, List, Mapping, Optional, Sequence

import copy
import logging as log
import os
import shutil
import weakref

import numpy as np
import torch
from PIL import Image

from diffusion_policy.common.lerobot_v3_io import LeRobotV3Dataset
from diffusion_policy.common.torchcodec_gpu import TorchCodecCudaFrameDecoder
from diffusion_policy.common.memory_budget import (
    compute_effective_budget_bytes,
    estimate_array_nbytes,
    format_gb,
)

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_util import absolute_pose_to_relative_pose
from diffusion_policy.common.sampler import create_indices, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.image_result_cache import (
    build_cache_metadata,
    build_in_memory_image_result_cache,
    open_or_build_image_result_cache,
    read_image_result,
    use_disk_result_cache,
    use_shared_ram_result_cache,
)
from diffusion_policy.dataset.img_randomer import Image_randomer

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer,  get_image_identity_normalizer


class _RamPreloadCacheEntry:
    def __init__(self, image_data: Dict[str, object], store=None):
        self.image_data = image_data
        # Keep an optional backing store for compatibility with older cache
        # implementations that used Zarr MemoryStore.
        self.store = store


# A single training process may construct both a train dataset and an offline
# validation view.  Keep RAM-preloaded frames process-local so an accidental
# second dataset construction does not reopen and decode every video frame.
# Weak references ensure a Hydra multirun can release a dataset's RAM after
# the corresponding job exits.
_RAM_PRELOAD_CACHE: Dict[tuple, weakref.ReferenceType] = {}


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_torch_from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)


def _stack_fixed_shape(values, dtype):
    arrays = []
    for value in values:
        arr = np.asarray(_to_numpy(value), dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def _nearest_indices(sorted_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sorted_timestamps, target_timestamps, side="left")
    right = np.clip(right, 0, len(sorted_timestamps) - 1)
    left = np.clip(right - 1, 0, len(sorted_timestamps) - 1)
    choose_right = np.abs(sorted_timestamps[right] - target_timestamps) < np.abs(
        sorted_timestamps[left] - target_timestamps
    )
    return np.where(choose_right, right, left)


def _coerce_image(image_value, expected_shape: Sequence[int]) -> np.ndarray:
    image_np = _to_numpy(image_value)
    if image_np.ndim == 4 and image_np.shape[0] == 1:
        image_np = image_np[0]
    if image_np.ndim != 3:
        raise ValueError(f"Expected 3-D image tensor, got shape {image_np.shape}")

    # Normalize to HWC before resize.
    if image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
        image_hwc = np.transpose(image_np, (1, 2, 0))
    else:
        image_hwc = image_np

    target_h, target_w = expected_shape[1], expected_shape[2]
    if image_hwc.shape[0] != target_h or image_hwc.shape[1] != target_w:
        # Pillow avoids importing OpenCV here.  OpenCV's libtiff/libjpeg
        # dependencies can conflict with the libraries bundled by PyTorch
        # when the dataset is imported after torch.
        image_hwc = np.asarray(
            Image.fromarray(np.asarray(image_hwc)).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        )

    image_chw = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)
    image_max = float(image_chw.max()) if image_chw.size else 0.0
    if image_max > 1.0:
        image_chw /= 255.0
    if image_chw.shape != tuple(expected_shape):
        raise ValueError(
            f"Image shape mismatch. Expected {tuple(expected_shape)}, got {image_chw.shape}"
        )
    return image_chw


class HirolLeRobotV3Dataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        window_sampling_strategy: str = "idx",
        image_feature_map: Optional[Mapping[str, str]] = None,
        lowdim_feature_groups: Optional[Mapping[str, Sequence[str]]] = None,
        windowed_lowdim_feature_groups: Optional[Mapping[str, Mapping]] = None,
        action_feature_fields: Optional[Sequence[str]] = None,
        action_layout: str = "per_step",
        relative_pose_actions: bool = False,
        action_reference_key: str = "action_reference",
        timestamp_key: str = "timestamp",
        timestamp_step_sec: Optional[float] = None,
        timestamp_tolerance_sec: Optional[float] = None,
        local_files_only: bool = True,
        # None lets LeRobot select its installed TorchCodec backend.
        video_backend: Optional[str] = None,
        # ``auto`` uses a CUDA TorchCodec decoder when the installed build
        # supports it, and falls back to LeRobot's normal decoder otherwise.
        video_decode_device: Optional[str] = None,
        preload_images: bool = False,
        memory_limit_gb: Optional[float] = None,
        memory_reserve_gb: float = 2.0,
        load_result_add="ram",
        image_randomer_config: Optional[Mapping] = None,
        image_cache_dtype: str = "uint8",
        image_cache_chunk_frames: int = 64,
        phase_resampling_enabled: bool = False,
        phase_sampling_weights: Optional[Mapping[str, float]] = None,
        phase_source_key: str = "observation.tau_ext",
        contact_threshold: Optional[float] = None,
        contact_force_dims: Sequence[int] = (0, 1, 2),
        contact_norm: str = "l1",
        precontact_seconds: float = 1.0,
        observation_validity_key: Optional[str] = None,
        episode_warmup_steps: int = 0,
    ):
        super().__init__()
        if window_sampling_strategy not in {"idx", "timestamp"}:
            raise ValueError(
                f"Unsupported window_sampling_strategy={window_sampling_strategy!r}. "
                "Expected 'idx' or 'timestamp'."
            )
        if action_layout not in {"per_step", "prechunked"}:
            raise ValueError(
                f"Unsupported action_layout={action_layout!r}. "
                "Expected 'per_step' or 'prechunked'."
            )

        self.shape_meta = shape_meta
        self.dataset_path = os.path.expanduser(dataset_path)
        self.horizon = int(horizon)
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = int(n_latency_steps)
        self.action_layout = action_layout
        self.relative_pose_actions = bool(relative_pose_actions)
        self.action_reference_key = str(action_reference_key)
        if not self.action_reference_key:
            raise ValueError("action_reference_key must not be empty")
        if self.relative_pose_actions and self.action_layout != "prechunked":
            raise ValueError(
                "relative_pose_actions currently requires action_layout='prechunked' "
                "so every sample has an explicit current-pose anchor"
            )
        self.window_sampling_strategy = window_sampling_strategy
        self.timestamp_key = timestamp_key
        self.timestamp_step_sec = timestamp_step_sec
        self.timestamp_tolerance_sec = timestamp_tolerance_sec
        self.episode_warmup_steps = int(episode_warmup_steps)
        if self.episode_warmup_steps < 0:
            raise ValueError("episode_warmup_steps must be non-negative")
        self.video_decode_device = (
            None if video_decode_device is None else str(video_decode_device).lower()
        )
        try:
            self.image_cache_dtype = np.dtype(image_cache_dtype)
        except TypeError as exc:
            raise ValueError(f"Unsupported image_cache_dtype={image_cache_dtype!r}") from exc
        if self.image_cache_dtype not in {np.dtype(np.uint8), np.dtype(np.float32)}:
            raise ValueError("image_cache_dtype must be 'uint8' or 'float32'")
        self.image_cache_chunk_frames = max(1, int(image_cache_chunk_frames))
        self.phase_resampling_enabled = bool(phase_resampling_enabled)
        self.phase_source_key = str(phase_source_key)
        self.contact_threshold = (
            None if contact_threshold is None else float(contact_threshold)
        )
        self.contact_force_dims = tuple(int(dim) for dim in contact_force_dims)
        self.contact_norm = "l1" if contact_norm == "sum" else str(contact_norm)
        self.precontact_seconds = float(precontact_seconds)
        default_phase_weights = {"free": 1.0, "precontact": 5.0, "contact": 2.0}
        self.phase_sampling_weights = {
            **default_phase_weights,
            **dict(phase_sampling_weights or {}),
        }
        if self.contact_threshold is not None and self.contact_threshold < 0:
            raise ValueError("contact_threshold must be non-negative")
        if self.contact_norm not in {"l1", "l2"}:
            raise ValueError("contact_norm must be 'l1', 'l2', or 'sum'")
        if self.precontact_seconds < 0:
            raise ValueError("precontact_seconds must be non-negative")
        self._video_decoder = None
        self._video_decoder_warning_emitted = False
        if self.action_layout == "prechunked":
            self.sequence_length = int(n_obs_steps or 1)
            self.sampler_pad_after = 0
        else:
            self.sequence_length = self.horizon + self.n_latency_steps
            self.sampler_pad_after = self.pad_after
        self.anchor_position = max(0, min(self.sequence_length - 1, (n_obs_steps or 1) - 1))
        self.image_data: Dict[str, object] = {}
        self.load_result_cache_path = None
        load_result_on_disk = use_disk_result_cache(load_result_add)
        distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if distributed_world_size > 1 and preload_images and not load_result_on_disk:
            load_result_add = "shared_ram"
            load_result_on_disk = True
            log.info(
                "WORLD_SIZE=%d: using one shared-RAM image cache for all ranks.",
                distributed_world_size,
            )
        self.image_randomer_config = image_randomer_config
        self.image_randomer_shared = bool(
            image_randomer_config
            and image_randomer_config.get("shared_across_sequence", False)
        )
        augmentation_enabled = bool(
            image_randomer_config
            and (
                bool(image_randomer_config.get("resize", False))
                or bool(image_randomer_config.get("random_crop", True))
                or bool(image_randomer_config.get("rotation", False))
                or bool(image_randomer_config.get("color_jitter", False))
            )
        )
        self.image_randomer = (
            Image_randomer(dict(image_randomer_config))
            if augmentation_enabled
            else None
        )


        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = [key for key, attr in obs_shape_meta.items() if attr.get("type") == "rgb"]
        self.lowdim_keys = [key for key, attr in obs_shape_meta.items() if attr.get("type") == "low_dim"]

        self.image_feature_map = dict(image_feature_map or {})
        for key in self.rgb_keys:
            self.image_feature_map.setdefault(key, f"observation.images.{key}")

        self.lowdim_feature_groups = {
            key: list(values)
            for key, values in (lowdim_feature_groups or {}).items()
        }
        self.windowed_lowdim_feature_groups = {
            str(key): dict(value)
            for key, value in (windowed_lowdim_feature_groups or {}).items()
        }
        unknown_windowed = set(self.windowed_lowdim_feature_groups) - set(self.lowdim_keys)
        if unknown_windowed:
            raise ValueError(
                "windowed_lowdim_feature_groups contains keys that are not low_dim "
                f"observations: {sorted(unknown_windowed)}"
            )
        for key in self.lowdim_keys:
            if key not in self.windowed_lowdim_feature_groups:
                self.lowdim_feature_groups.setdefault(key, [f"observation.{key}"])

        self.action_feature_fields = list(action_feature_fields or ["action"])

        self.lerobot_dataset = LeRobotV3Dataset(
            self.dataset_path,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )
        self.dataset_length = len(self.lerobot_dataset)

        self.timestamps = self._load_column(self.timestamp_key, dtype=np.float64).reshape(-1)
        self.episode_index = self._load_episode_index()
        self.episode_ends = self._build_episode_ends(self.episode_index)
        self.episode_ranges = self._build_episode_ranges(self.episode_ends)
        self._phase_source_data = (
            self._load_column(self.phase_source_key, dtype=np.float32)
            if self.phase_resampling_enabled
            else None
        )
        self.episode_step_sec = self._build_episode_step_sec(
            self.timestamps,
            self.episode_ranges,
            explicit_step_sec=timestamp_step_sec,
        )
        self._observation_valid = np.ones(self.dataset_length, dtype=bool)
        if observation_validity_key:
            flags = self._load_column(observation_validity_key, dtype=np.float64)
            if flags.size != self.dataset_length or not np.isin(flags, [0, 1]).all():
                raise ValueError(
                    f"{observation_validity_key!r} must contain one 0/1 validity flag per row"
                )
            self._observation_valid = flags.reshape(-1).astype(bool)
        self._window_anchor_timestamps = {}
        for config in self.windowed_lowdim_feature_groups.values():
            for key in (config.get("anchor_timestamp_key"), config.get("source_timestamp_key")):
                if key and key not in self._window_anchor_timestamps:
                    timestamps = self._load_column(key, dtype=np.float64).reshape(-1)
                    if timestamps.shape[0] != self.dataset_length or not np.isfinite(timestamps).all():
                        raise ValueError(f"timestamp column {key!r} must contain one finite timestamp per row")
                    self._window_anchor_timestamps[key] = self._timestamp_seconds(timestamps)

        self.lowdim_data = {}
        self.windowed_lowdim_data = {}
        self._windowed_lowdim_valid_rows = {}
        for key in self.lowdim_keys:
            if key in self.windowed_lowdim_feature_groups:
                config = self.windowed_lowdim_feature_groups[key]
                source_keys = config.get("source_keys", config.get("source_key"))
                if source_keys is None:
                    raise ValueError(
                        f"windowed lowdim feature {key!r} requires source_key(s)"
                    )
                if isinstance(source_keys, str):
                    source_keys = [source_keys]
                source = self._concat_columns(source_keys, dtype=np.float32)
                steps = int(config.get("steps", tuple(obs_shape_meta[key]["shape"])[0]))
                stride = int(config.get("stride", 1))
                start_offset = int(config.get("start_offset", -(steps - 1) * stride))
                if steps < 1 or stride < 1:
                    raise ValueError(
                        f"windowed lowdim feature {key!r} requires positive steps/stride"
                    )
                if start_offset + (steps - 1) * stride < 0 and config.get("pad_mode", "edge") not in {"edge", "clip"}:
                    raise ValueError(
                        f"windowed lowdim feature {key!r} uses negative offsets and must use pad_mode='edge'"
                    )
                self.windowed_lowdim_feature_groups[key] = {
                    **config,
                    "source_keys": list(source_keys),
                    "steps": steps,
                    "stride": stride,
                    "start_offset": start_offset,
                    "pad_mode": config.get("pad_mode", "edge"),
                }
                self._windowed_lowdim_valid_rows[key] = np.empty(self.dataset_length, dtype=bool)
                self.windowed_lowdim_data[key] = self._build_windowed_lowdim(
                    source,
                    self.windowed_lowdim_feature_groups[key],
                    valid_rows=self._windowed_lowdim_valid_rows[key],
                )
            else:
                self.lowdim_data[key] = self._concat_columns(
                    self.lowdim_feature_groups[key], dtype=np.float32
                )
        self.action_data = self._concat_columns(self.action_feature_fields, dtype=np.float32)

        for key in self.lowdim_keys:
            expected = tuple(obs_shape_meta[key]["shape"])
            actual = (
                self.windowed_lowdim_data[key].shape[1:]
                if key in self.windowed_lowdim_data
                else self.lowdim_data[key].shape[1:]
            )
            if actual != expected:
                raise ValueError(
                    f"Lowdim feature {key!r} has shape {actual}, expected {expected}. "
                    f"Source fields: {self.windowed_lowdim_feature_groups.get(key, self.lowdim_feature_groups.get(key))}"
                )

        expected_action_shape = tuple(shape_meta["action"]["shape"])
        self._validate_action_shape(expected_action_shape)
        self.action_reference_data: Optional[np.ndarray] = None
        if self.relative_pose_actions:
            if expected_action_shape != (7,):
                raise ValueError(
                    "relative_pose_actions requires action shape [7] in xyz + xyzw format"
                )
            self.action_reference_data = np.array(
                self.action_data[:, 0, :],
                dtype=np.float32,
                copy=True,
            )
            relative_action = absolute_pose_to_relative_pose(
                torch.from_numpy(self.action_data),
                torch.from_numpy(self.action_reference_data),
            )
            self.action_data = relative_action.numpy().astype(np.float32, copy=False)

        effective_budget_bytes = compute_effective_budget_bytes(
            memory_limit_gb=memory_limit_gb,
            memory_reserve_gb=memory_reserve_gb,
        )
        estimated_preload_bytes = self._estimate_image_preload_bytes()
        # Keep the decoded cache compact (uint8 by default) and do not let an
        # undersized host enter swap.  SSD caches remain reusable by later runs.
        if preload_images and not load_result_on_disk and memory_limit_gb is None:
            available_bytes = self._available_memory_bytes()
            reserve_bytes = int(float(memory_reserve_gb) * (1024 ** 3))
            if available_bytes is not None and estimated_preload_bytes + reserve_bytes > int(
                available_bytes * 0.9
            ):
                fallback_location = os.environ.get(
                    "DP_IMAGE_CACHE_FALLBACK", "ssd"
                )
                log.warning(
                    "Image preload needs %s plus reserve, but only %s host memory is available; "
                    "using %r decoded cache to avoid swap.",
                    format_gb(estimated_preload_bytes),
                    format_gb(available_bytes),
                    fallback_location,
                )
                load_result_add = fallback_location
                load_result_on_disk = use_disk_result_cache(load_result_add)
        if use_shared_ram_result_cache(load_result_add):
            shared_cache_dir = os.environ.get(
                "DP_SHARED_IMAGE_CACHE_DIR", "/dev/shm/diffusion_policy_image_cache"
            )
            os.makedirs(shared_cache_dir, exist_ok=True)
            shared_ram_free_bytes = shutil.disk_usage(shared_cache_dir).free
            reserve_bytes = int(float(memory_reserve_gb) * (1024 ** 3))
            required_bytes = estimated_preload_bytes + reserve_bytes
            if shared_ram_free_bytes < required_bytes:
                fallback_location = os.environ.get(
                    "DP_SHARED_IMAGE_CACHE_FALLBACK", "ssd"
                )
                log.warning(
                    "Shared RAM cache requires %s including reserve, but only %s is free; "
                    "falling back to %r.",
                    format_gb(required_bytes),
                    format_gb(shared_ram_free_bytes),
                    fallback_location,
                )
                load_result_add = fallback_location
                load_result_on_disk = use_disk_result_cache(load_result_add)
        self.load_result_add = load_result_add
        if effective_budget_bytes is not None:
            log.info(
                "HirolLeRobotV3Dataset RAM budget: effective=%s, estimated image-preload=%s",
                format_gb(effective_budget_bytes),
                format_gb(estimated_preload_bytes),
            )
            if preload_images and (not load_result_on_disk) and estimated_preload_bytes > effective_budget_bytes:
                log.warning(
                    "Disabling LeRobot image preload because estimated footprint %s exceeds RAM budget %s.",
                    format_gb(estimated_preload_bytes),
                    format_gb(effective_budget_bytes),
                )
                preload_images = False

        if load_result_on_disk:
            image_shapes = {
                key: tuple(self.shape_meta["obs"][key]["shape"])
                for key in self.rgb_keys
            }
            metadata = build_cache_metadata(
                source_type="hirol_lerobot_v3",
                dataset_path=dataset_path,
                dataset_length=self.dataset_length,
                rgb_keys=self.rgb_keys,
                image_shapes=image_shapes,
                extra={
                    "image_feature_map": self.image_feature_map,
                },
                dtype=self.image_cache_dtype.name,
            )

            self._video_decoder = self._make_video_decoder()

            def build_frame(frame_idx):
                return {
                    key: values[0]
                    for key, values in self._build_image_chunk([frame_idx], image_shapes).items()
                }

            def build_frame_batch(frame_indices):
                return self._build_image_chunk(frame_indices, image_shapes)

            self.image_data, self.load_result_cache_path = open_or_build_image_result_cache(
                load_result_add=load_result_add,
                dataset_path=dataset_path,
                metadata=metadata,
                build_frame_fn=build_frame,
                build_frame_batch_fn=build_frame_batch,
                desc="Build LeRobot decoded image cache",
                chunk_frames=self.image_cache_chunk_frames,
                logger=log,
            )
            self.lerobot_dataset.close()
            # Do not carry a CUDA decoder into forked DataLoader workers after
            # the full cache has been materialized.
            self._video_decoder = None
        elif preload_images:
            self._video_decoder = self._make_video_decoder()
            cache_key = (
                os.path.abspath(self.dataset_path),
                int(self.dataset_length),
                tuple(
                    (
                        key,
                        tuple(self.shape_meta["obs"][key]["shape"]),
                        self.image_feature_map[key],
                        self.image_cache_dtype.name,
                    )
                    for key in self.rgb_keys
                ),
            )
            cache_entry_ref = _RAM_PRELOAD_CACHE.get(cache_key)
            cache_entry = cache_entry_ref() if cache_entry_ref is not None else None
            if cache_entry is None:
                image_shapes = {
                    key: tuple(self.shape_meta["obs"][key]["shape"])
                    for key in self.rgb_keys
                }
                metadata = build_cache_metadata(
                    source_type="hirol_lerobot_v3",
                    dataset_path=self.dataset_path,
                    dataset_length=self.dataset_length,
                    rgb_keys=self.rgb_keys,
                    image_shapes=image_shapes,
                    extra={"image_feature_map": self.image_feature_map},
                    dtype=self.image_cache_dtype.name,
                )

                def build_frame(frame_idx):
                    return {
                        key: values[0]
                        for key, values in self._build_image_chunk([frame_idx], image_shapes).items()
                    }

                def build_frame_batch(frame_indices):
                    return self._build_image_chunk(frame_indices, image_shapes)

                self.image_data, store = build_in_memory_image_result_cache(
                    metadata=metadata,
                    build_frame_fn=build_frame,
                    build_frame_batch_fn=build_frame_batch,
                    desc="Preload LeRobot images into RAM Zarr",
                    chunk_frames=self.image_cache_chunk_frames,
                    logger=log,
                )
                cache_entry = _RamPreloadCacheEntry(self.image_data, store=store)
                _RAM_PRELOAD_CACHE[cache_key] = weakref.ref(cache_entry)
            else:
                log.info(
                    "Using process-local RAM image preload cache for %s",
                    self.dataset_path,
                )
                self.image_data = cache_entry.image_data
            self._ram_cache_entry = cache_entry
            self.lerobot_dataset.close()
            self._video_decoder = None

        val_mask = get_val_mask(
            n_episodes=len(self.episode_ends),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.val_mask = val_mask
        self.train_mask = train_mask
        self.indices = create_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.sampler_pad_after,
            episode_mask=self.train_mask,
        )
        self.indices = self._filter_indices_for_windowed_features(self.indices)
        self.phase_labels = self._build_phase_labels(self.indices)

    def _get_hf_dataset(self):
        return None

    def _load_column(self, column_name: str, dtype) -> np.ndarray:
        try:
            values = self.lerobot_dataset.get_column(column_name)
        except KeyError as exc:
            raise KeyError(f"Column {column_name!r} not found in LeRobot v3 dataset.") from exc
        return _stack_fixed_shape(values, dtype=dtype)

    def _load_episode_index(self) -> np.ndarray:
        episode_data_index = getattr(self.lerobot_dataset, "episode_data_index", None)
        if episode_data_index is not None and "from" in episode_data_index and "to" in episode_data_index:
            starts = _to_numpy(episode_data_index["from"]).astype(np.int64).reshape(-1)
            stops = _to_numpy(episode_data_index["to"]).astype(np.int64).reshape(-1)
            episode_index = np.empty((self.dataset_length,), dtype=np.int64)
            for ep_idx, (start, stop) in enumerate(zip(starts, stops)):
                episode_index[start:stop] = ep_idx
            return episode_index

        raise KeyError(
            "LeRobot v3 dataset does not expose episode_index or episode_data_index; "
            "cannot build episode-aware window sampling."
        )

    def _concat_columns(self, column_names: Sequence[str], dtype) -> np.ndarray:
        arrays = [self._load_column(column_name, dtype=dtype) for column_name in column_names]
        if len(arrays) == 1:
            return arrays[0].astype(dtype, copy=False)
        return np.concatenate(arrays, axis=-1).astype(dtype, copy=False)

    def _build_windowed_lowdim(
        self, source: np.ndarray, config: Mapping, valid_rows: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Materialize a fixed-rate low-dimensional window for every row.

        ``start_offset`` and ``stride`` are expressed in source rows.  Edge
        clipping is used at episode boundaries so history windows can be
        left-padded; future windows are filtered out of the sampler below.
        """
        steps = int(config["steps"])
        stride = int(config["stride"])
        start_offset = int(config["start_offset"])
        result = np.empty((self.dataset_length, steps, source.shape[-1]), dtype=np.float32)
        offsets = start_offset + stride * np.arange(steps, dtype=np.int64)
        offsets_sec = config.get("offsets_sec")
        anchor_key = config.get("anchor_timestamp_key")
        source_key = config.get("source_timestamp_key")
        source_timestamps = (
            self._window_anchor_timestamps[source_key]
            if source_key
            else self._timestamp_seconds(self.timestamps)
        )
        anchor_timestamps = (
            self._window_anchor_timestamps[anchor_key]
            if anchor_key
            else None
        )
        if offsets_sec is not None:
            offsets_sec = np.asarray(offsets_sec, dtype=np.float64).reshape(-1)
            if offsets_sec.shape[0] != steps:
                raise ValueError(
                    f"windowed lowdim offsets_sec has {offsets_sec.shape[0]} values, "
                    f"expected {steps}"
                )
            if anchor_timestamps is None:
                raise ValueError("offsets_sec requires anchor_timestamp_key")
        for episode_range in self.episode_ranges:
            start = int(episode_range.start)
            end = int(episode_range.stop) - 1
            if offsets_sec is not None:
                local_source_times = source_timestamps[start : end + 1]
                local_anchor_times = anchor_timestamps[start : end + 1]
                if np.any(np.diff(local_source_times) <= 0):
                    raise ValueError("windowed lowdim source timestamps must increase within each episode")
                # Preserve subsecond offsets by subtracting one common origin.
                origin = local_source_times[0]
                local_source_times = local_source_times - origin
                local_anchor_times = local_anchor_times - origin
                # Catch relative/epoch clock mismatches before nearest-neighbor
                # clipping can silently turn every token into an episode edge.
                if (
                    local_anchor_times.max() < local_source_times[0] - 1.0
                    or local_anchor_times.min() > local_source_times[-1] + 1.0
                ):
                    raise ValueError(
                        "windowed lowdim anchor/source timestamps use different time origins; "
                        "set source_timestamp_key to the state timestamp column matching "
                        f"anchor_timestamp_key={anchor_key!r}"
                    )
                targets = local_anchor_times[:, None] + offsets_sec[None, :]
                local_indices = _nearest_indices(local_source_times, targets)
                indices = local_indices + start
                # History can be edge-padded; future targets must exist in
                # the source episode rather than reuse the final force row.
                valid = (targets <= local_source_times[-1]).all(axis=1)
                if np.any(offsets_sec > 0):
                    valid &= (targets >= local_source_times[0]).all(axis=1)
            else:
                anchors = np.arange(start, end + 1, dtype=np.int64)[:, None]
                requested_indices = anchors + offsets[None, :]
                indices = np.clip(requested_indices, start, end)
                valid = (requested_indices <= end).all(axis=1)
            warmup = getattr(self, "episode_warmup_steps", 0)
            valid &= (indices >= start + warmup).all(axis=1)
            if valid_rows is not None:
                valid_rows[start : end + 1] = valid
            result[start : end + 1] = source[indices]
        return result

    @staticmethod
    def _timestamp_seconds(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmedian(np.abs(finite)) > 1.0e12:
            return values * 1.0e-9
        return values

    def _anchor_index_from_sampler_row(self, row: Sequence[int]) -> int:
        buffer_start, buffer_end, sample_start, sample_end = map(int, row)
        position = int(self.anchor_position)
        if position < sample_start:
            return buffer_start
        if position >= sample_end:
            return max(buffer_start, buffer_end - 1)
        return buffer_start + position - sample_start

    def _filter_indices_for_windowed_features(self, indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return indices
        sequences = self._sequence_indices_from_sampler_rows(indices)
        observations = sequences[:, : self.n_obs_steps]
        episodes = self.episode_index[sequences[:, self.anchor_position]]
        starts = np.r_[0, self.episode_ends[:-1]]
        keep = (sequences >= starts[episodes, None] + self.episode_warmup_steps).all(axis=1)
        keep &= self._observation_valid[observations].all(axis=1)
        for valid_rows in self._windowed_lowdim_valid_rows.values():
            keep &= valid_rows[observations].all(axis=1)
        return indices[keep]

    def _phase_magnitude(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 1:
            raise ValueError("phase values must have at least one dimension")
        if not self.contact_force_dims:
            raise ValueError("contact_force_dims cannot be empty")
        if max(self.contact_force_dims) >= values.shape[-1]:
            raise ValueError(
                f"contact_force_dims={self.contact_force_dims} exceed phase dimension "
                f"{values.shape[-1]}"
            )
        force = values[..., list(self.contact_force_dims)]
        if self.contact_norm == "l1":
            return np.abs(force).sum(axis=-1)
        return np.linalg.vector_norm(force, ord=2, axis=-1)

    def _build_phase_labels(self, indices: np.ndarray) -> np.ndarray:
        """Return 0=free, 1=pre-contact, 2=contact for each sample window.

        A window is contact-relevant when either its observed tau history or
        its future tau history contains a token above the contact threshold.
        Remaining anchors in the one-second interval before the first contact
        timestamp of their episode are marked pre-contact.
        """
        if not self.phase_resampling_enabled:
            return np.zeros((len(indices),), dtype=np.int8)
        if self.contact_threshold is None:
            raise ValueError(
                "phase resampling requires contact_threshold in the dataset config"
            )
        if len(indices) == 0:
            return np.zeros((0,), dtype=np.int8)

        window_contact = np.zeros((self.dataset_length,), dtype=bool)
        for key in ("tau_ext", "future_tau_ext"):
            values = self.windowed_lowdim_data.get(key)
            if values is None:
                continue
            window_contact |= np.asarray(
                self._phase_magnitude(values) > self.contact_threshold
            ).any(axis=1)

        source = self._phase_source_data
        if source is None:
            source = self._load_column(self.phase_source_key, dtype=np.float32)
        frame_contact = self._phase_magnitude(source) > self.contact_threshold
        timestamps = self._timestamp_seconds(self.timestamps)
        contact_onset = np.full((len(self.episode_ranges),), np.nan, dtype=np.float64)
        for episode_idx, episode_range in enumerate(self.episode_ranges):
            local = frame_contact[episode_range.start : episode_range.stop]
            contact_indices = np.flatnonzero(local)
            if contact_indices.size:
                contact_onset[episode_idx] = timestamps[
                    episode_range.start + int(contact_indices[0])
                ]

        anchors = np.asarray(
            [self._anchor_index_from_sampler_row(row) for row in indices],
            dtype=np.int64,
        )
        episodes = self.episode_index[anchors]
        anchor_timestamps = timestamps[anchors]
        labels = np.zeros((len(indices),), dtype=np.int8)
        labels[window_contact[anchors]] = 2
        precontact = np.zeros_like(labels, dtype=bool)
        for episode_idx, onset in enumerate(contact_onset):
            if not np.isfinite(onset):
                continue
            in_window = episodes == episode_idx
            precontact |= (
                in_window
                & (anchor_timestamps >= onset - self.precontact_seconds)
                & (anchor_timestamps < onset)
            )
        labels[(labels == 0) & precontact] = 1
        return labels

    def _validate_action_shape(self, expected_action_shape: Sequence[int]) -> None:
        expected_action_shape = tuple(expected_action_shape)
        source_shape = self.action_data.shape[1:]

        if self.action_layout == "per_step":
            if source_shape != expected_action_shape:
                raise ValueError(
                    f"Per-step action shape mismatch. Got {source_shape}, "
                    f"expected {expected_action_shape}. "
                    f"Source fields: {self.action_feature_fields}"
                )
            return

        required_chunk_steps = self.n_latency_steps + self.horizon
        if len(source_shape) < 1 or source_shape[1:] != expected_action_shape:
            raise ValueError(
                f"Prechunked action shape mismatch. Got {source_shape}, expected "
                f"(chunk_steps, {', '.join(map(str, expected_action_shape))}). "
                f"Source fields: {self.action_feature_fields}"
            )
        if source_shape[0] < required_chunk_steps:
            raise ValueError(
                f"Prechunked action has {source_shape[0]} steps, but horizon={self.horizon} "
                f"and n_latency_steps={self.n_latency_steps} require at least "
                f"{required_chunk_steps}."
            )

    def _sample_action(self, sequence_indices: np.ndarray) -> np.ndarray:
        if self.action_layout == "per_step":
            action = self.action_data[sequence_indices, ...]
            if self.n_latency_steps > 0:
                action = action[self.n_latency_steps :]
        else:
            anchor_idx = int(sequence_indices[self.anchor_position])
            chunk_start = self.n_latency_steps
            chunk_end = chunk_start + self.horizon
            action = self.action_data[anchor_idx, chunk_start:chunk_end, ...]

        expected_shape = (self.horizon,) + tuple(self.shape_meta["action"]["shape"])
        if action.shape != expected_shape:
            raise RuntimeError(
                f"Sampled action shape mismatch. Got {action.shape}, expected {expected_shape}. "
                f"action_layout={self.action_layout!r}"
            )
        return np.array(action, dtype=np.float32, copy=True)

    def _estimate_image_preload_bytes(self) -> int:
        total = 0
        for key in self.rgb_keys:
            expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
            total += self.dataset_length * estimate_array_nbytes(
                expected_shape, self.image_cache_dtype
            )
        return total

    @staticmethod
    def _available_memory_bytes() -> Optional[int]:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            try:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return int(pages * page_size)
            except (AttributeError, OSError, ValueError):
                return None

    def _make_video_decoder(self):
        requested = self.video_decode_device
        if requested in {None, "", "none", "cpu", "false", "0"}:
            return None
        if requested in {"auto", "cuda", "gpu", "true", "1"}:
            if not torch.cuda.is_available():
                return None
            # Use an explicit index: TorchCodec interprets bare ``cuda`` as
            # cuda:0 even when PyTorch's current device was set to another GPU.
            requested = f"cuda:{torch.cuda.current_device()}"
        if not requested.startswith("cuda"):
            raise ValueError(
                "video_decode_device must be None, 'auto', or a CUDA device such as 'cuda:0'"
            )
        try:
            decoder = TorchCodecCudaFrameDecoder(
                lerobot_dataset=self.lerobot_dataset,
                episode_ranges=self.episode_ranges,
                video_keys=[self.image_feature_map[key] for key in self.rgb_keys],
                device=requested,
            )
            log.info("Using TorchCodec CUDA video decoder on %s", requested)
            return decoder
        except Exception as exc:
            self._disable_video_decoder(exc)
            return None

    def _disable_video_decoder(self, exc: Exception) -> None:
        self._video_decoder = None
        if not self._video_decoder_warning_emitted:
            log.warning(
                "TorchCodec CUDA decoding is unavailable; falling back to LeRobot decoder: %s",
                exc,
            )
            self._video_decoder_warning_emitted = True

    @staticmethod
    def _build_episode_ends(episode_index: np.ndarray) -> np.ndarray:
        if episode_index.size == 0:
            return np.zeros((0,), dtype=np.int64)
        change_points = np.nonzero(np.diff(episode_index))[0] + 1
        return np.concatenate([change_points, [episode_index.shape[0]]]).astype(np.int64)

    @staticmethod
    def _build_episode_ranges(episode_ends: np.ndarray) -> List[range]:
        episode_ranges: List[range] = []
        start = 0
        for end in episode_ends:
            episode_ranges.append(range(start, int(end)))
            start = int(end)
        return episode_ranges

    @staticmethod
    def _build_episode_step_sec(
        timestamps: np.ndarray,
        episode_ranges: Sequence[range],
        explicit_step_sec: Optional[float],
    ) -> List[float]:
        if explicit_step_sec is not None:
            return [float(explicit_step_sec) for _ in episode_ranges]

        step_sizes = []
        for episode_range in episode_ranges:
            episode_timestamps = timestamps[episode_range.start : episode_range.stop]
            diffs = np.diff(episode_timestamps)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size == 0:
                step_sizes.append(1.0)
            else:
                step_sizes.append(float(np.median(diffs)))
        return step_sizes

    def _sample_indices_to_sequence(self, sample_idx: int) -> np.ndarray:
        # buffer数据全局   sample进行窗口采样
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[sample_idx]
        sequence_indices = np.empty((self.sequence_length,), dtype=np.int64)
        last_valid_idx = max(buffer_start_idx, buffer_end_idx - 1)
        for position in range(self.sequence_length):
            if position < sample_start_idx:
                sequence_indices[position] = buffer_start_idx
            elif position >= sample_end_idx:
                sequence_indices[position] = last_valid_idx
            else:
                sequence_indices[position] = buffer_start_idx + (position - sample_start_idx)
        return sequence_indices

    def _sequence_indices_from_sampler_rows(self, rows: np.ndarray) -> np.ndarray:
        """Resolve the same rows as __getitem__, in batches for filtering."""
        sequences = rows[:, :1] + np.arange(self.sequence_length)[None, :] - rows[:, 2:3]
        sequences = np.clip(sequences, rows[:, :1], rows[:, 1:2] - 1)
        if self.window_sampling_strategy == "timestamp":
            sequences = self._retime_anchor_indices(sequences[:, self.anchor_position])
        return sequences.astype(np.int64, copy=False)

    def _retime_sequence_indices(self, sequence_indices: np.ndarray) -> np.ndarray:
        anchors = np.asarray([sequence_indices[self.anchor_position]], dtype=np.int64)
        return self._retime_anchor_indices(anchors)[0]

    def _retime_anchor_indices(self, anchors: np.ndarray) -> np.ndarray:
        episodes = self.episode_index[anchors]
        result = np.empty((len(anchors), self.sequence_length), dtype=np.int64)
        offsets = np.arange(self.sequence_length, dtype=np.float64) - self.anchor_position
        for episode in np.unique(episodes):
            selected = episodes == episode
            episode_range = self.episode_ranges[episode]
            times = self.timestamps[episode_range.start : episode_range.stop]
            targets = self.timestamps[anchors[selected], None] + (
                offsets[None, :] * self.episode_step_sec[episode]
            )
            local_indices = _nearest_indices(times, targets)
            if self.timestamp_tolerance_sec is not None:
                deltas = np.abs(times[local_indices] - targets)
                nearest_edge = np.where(targets <= times[0], 0, len(times) - 1)
                local_indices = np.where(
                    deltas <= self.timestamp_tolerance_sec, local_indices, nearest_edge
                )
            result[selected] = episode_range.start + local_indices
        return result

    def _load_frame_feature(
        self,
        frame_idx: int,
        feature_name: str,
        expected_shape: Sequence[int],
        frame_cache: Dict[int, Dict],
    ) -> np.ndarray:
        if frame_idx not in frame_cache:
            frame_cache[frame_idx] = self.lerobot_dataset[frame_idx]
        sample = frame_cache[frame_idx]
        if feature_name not in sample:
            raise KeyError(
                f"Feature {feature_name!r} missing from LeRobot sample. "
                f"Available keys: {list(sample.keys())}"
            )
        return _coerce_image(sample[feature_name], expected_shape)

    def _build_image_chunk(
        self,
        frame_indices: Sequence[int],
        image_shapes: Mapping[str, Sequence[int]],
    ) -> Dict[str, np.ndarray]:
        """Decode one contiguous cache chunk, batching each camera request."""
        indices = [int(index) for index in frame_indices]
        decoded = {}
        for key in self.rgb_keys:
            values = None
            if self._video_decoder is not None:
                try:
                    values = self._video_decoder.decode_many(
                        indices, self.image_feature_map[key]
                    )
                except Exception as exc:
                    self._disable_video_decoder(exc)
            decoded[key] = values

        result = {key: [] for key in self.rgb_keys}
        sample_cache: Dict[int, Dict] = {}
        for local_idx, frame_idx in enumerate(indices):
            for key in self.rgb_keys:
                value = None if decoded[key] is None else decoded[key][local_idx]
                feature_name = self.image_feature_map[key]
                if value is None:
                    if frame_idx not in sample_cache:
                        sample_cache[frame_idx] = self.lerobot_dataset[frame_idx]
                    sample = sample_cache[frame_idx]
                    if feature_name not in sample:
                        raise KeyError(
                            f"Feature {feature_name!r} missing from LeRobot sample. "
                            f"Available keys: {list(sample.keys())}"
                        )
                    value = sample[feature_name]
                result[key].append(_coerce_image(value, image_shapes[key]))
        return {
            key: np.stack(values, axis=0) for key, values in result.items()
        }

    def _augment_image_sequence(self, images: np.ndarray) -> np.ndarray:
        if self.image_randomer is None:
            return images
    
        augmented_images = []
        shared_seed = None
        if self.image_randomer_shared:
            # Draw one seed for this observation window.  Replaying it inside
            # a forked RNG makes every frame and camera receive the same
            # spatial/photometric transform without perturbing DataLoader RNG.
            shared_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        for image_chw in images:
            image_hwc = np.transpose(image_chw, (1, 2, 0))
            image_hwc = np.clip(image_hwc * 255.0, 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(image_hwc)
    
            if shared_seed is None:
                augmented = self.image_randomer(image_pil)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(shared_seed)
                    augmented = self.image_randomer(image_pil)
            if torch.is_tensor(augmented):
                augmented = augmented.detach().cpu().numpy()
            augmented = np.asarray(augmented, dtype=np.float32)
    
            augmented_images.append(augmented)
    
        return np.stack(augmented_images, axis=0)

    def get_validation_dataset(self) -> "HirolLeRobotV3Dataset":
        val_set = copy.copy(self)
        val_set.indices = create_indices(
            self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.sampler_pad_after,
            episode_mask=self.val_mask,
        )
        val_set.indices = val_set._filter_indices_for_windowed_features(val_set.indices)
        val_set.phase_labels = val_set._build_phase_labels(val_set.indices)
        val_set.image_randomer = None
        val_set.image_randomer_config = None
        val_set.image_randomer_shared = False
        return val_set

    def get_phase_labels(self) -> np.ndarray:
        """Return phase labels aligned with dataset indices.

        Labels are 0=free, 1=pre-contact, and 2=contact.
        """
        return np.asarray(self.phase_labels, dtype=np.int8).copy()


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if len(self.indices) == 0:
            raise ValueError("Cannot fit a normalizer without valid training windows")
        sequences = self._sequence_indices_from_sampler_rows(self.indices)
        observation_rows = np.unique(sequences[:, : self.n_obs_steps])
        if self.action_layout == "per_step":
            actions = self.action_data[np.unique(sequences[:, self.n_latency_steps :])]
        else:
            anchors = np.unique(sequences[:, self.anchor_position])
            actions = self.action_data[
                anchors, self.n_latency_steps : self.n_latency_steps + self.horizon
            ]
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(actions, **kwargs)
        for key in self.lowdim_keys:
            values = self.windowed_lowdim_data.get(key, self.lowdim_data.get(key))
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(values[observation_rows], **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer() 
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.action_data.copy())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence_indices = self._sample_indices_to_sequence(idx)
        if self.window_sampling_strategy == "timestamp":
            sequence_indices = self._retime_sequence_indices(sequence_indices)
          
        obs_indices = sequence_indices[: self.n_obs_steps]
        frame_cache: Dict[int, Dict] = {}
        obs_dict = {}

        for key in self.rgb_keys:
            if key in self.image_data:
                images = read_image_result(self.image_data, key, obs_indices)
                if self.image_cache_dtype == np.dtype(np.uint8):
                    images = np.asarray(images, dtype=np.float32) / 255.0
            else:
                expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
                feature_name = self.image_feature_map[key]
                images = np.stack(
                    [
                        self._load_frame_feature(
                            frame_idx=int(frame_idx),
                            feature_name=feature_name,
                            expected_shape=expected_shape,
                            frame_cache=frame_cache,
                        )
                        for frame_idx in obs_indices
                    ],
                    axis=0,
                )

            obs_dict[key] = self._augment_image_sequence(images)


        for key in self.lowdim_keys:
            values = self.windowed_lowdim_data.get(key, self.lowdim_data.get(key))
            obs_dict[key] = values[obs_indices, ...].astype(np.float32, copy=False)

        action = self._sample_action(sequence_indices)
        if self.relative_pose_actions:
            anchor_idx = int(sequence_indices[self.anchor_position])
            obs_dict[self.action_reference_key] = self.action_reference_data[
                anchor_idx
            ].astype(np.float32, copy=False)

        return {
            "obs": dict_apply(obs_dict, _safe_torch_from_numpy),
            "action": _safe_torch_from_numpy(action),
        }
# 一个torch
# batch = 
#     "obs": 
#         "ee_cam_color": Tensor[B, 2, 3, 224, 224],
#         "third_person_cam_color": Tensor[B, 2, 3, 224, 224],
#         "side_cam_color": Tensor[B, 2, 3, 224, 224],
#         "state_ee": Tensor[B, 2, 15]
#     "action": Tensor[B, 16, 8],
