"""Episode-aware causal filtering for training and deployment contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
from typing import Any

import numpy as np
import torch


log = logging.getLogger(__name__)

_SUPPORTED_OPERATIONS = frozenset({"median", "moving_average", "lowpass"})
_TIMESTAMP_SCALES_TO_SECONDS = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


def normalize_dataloader_filters(
    data_config: Mapping[str, Any],
    lowdim_keys: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate and canonicalize ``dataloader.filters``.

    Operations are applied in listed order. Every supported operation is causal:
    it consumes only the current sample and samples earlier in the same episode.
    """

    raw_filters = data_config.get("filters") or {}
    if not isinstance(raw_filters, Mapping):
        raise ValueError("dataloader.filters must be a mapping")

    known_keys = None if lowdim_keys is None else set(lowdim_keys)
    normalized: dict[str, dict[str, Any]] = {}
    for key, raw_spec in raw_filters.items():
        key = str(key)
        if known_keys is not None and key not in known_keys:
            raise ValueError(
                f"dataloader.filters contains unknown low-dimensional key {key!r}"
            )
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"dataloader.filters.{key} must be a mapping")

        enabled = bool(raw_spec.get("enabled", False))
        raw_operations = raw_spec.get("operations", raw_spec.get("pipeline", ()))
        if raw_operations is None:
            raw_operations = ()
        if not isinstance(raw_operations, Sequence) or isinstance(
            raw_operations, (str, bytes)
        ):
            raise ValueError(
                f"dataloader.filters.{key}.operations must be a list"
            )

        operations = [
            _normalize_operation(operation, field=f"dataloader.filters.{key}")
            for operation in raw_operations
        ]
        raw_preprocessed = raw_spec.get("dataset_preprocessed_operations", ())
        if raw_preprocessed is None:
            raw_preprocessed = ()
        if not isinstance(raw_preprocessed, Sequence) or isinstance(
            raw_preprocessed, (str, bytes)
        ):
            raise ValueError(
                f"dataloader.filters.{key}.dataset_preprocessed_operations "
                "must be a list"
            )
        preprocessed_operations = [
            _normalize_operation(operation, field=f"dataloader.filters.{key}")
            for operation in raw_preprocessed
        ]
        if bool(raw_spec.get("source_already_filtered", False)):
            if preprocessed_operations:
                raise ValueError(
                    f"dataloader.filters.{key} must not combine "
                    "source_already_filtered with dataset_preprocessed_operations"
                )
            preprocessed_operations = list(operations)
        if operations[: len(preprocessed_operations)] != preprocessed_operations:
            raise ValueError(
                f"dataloader.filters.{key}.dataset_preprocessed_operations "
                "must be an exact prefix of operations"
            )
        if enabled and not operations:
            raise ValueError(
                f"dataloader.filters.{key} is enabled but has no operations"
            )
        normalized[key] = {
            "enabled": enabled,
            "operations": operations,
            "dataset_preprocessed_operations": preprocessed_operations,
        }
    return normalized


def _normalize_operation(operation: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(operation, Mapping):
        raise ValueError(f"{field}.operations entries must be mappings")
    unknown = set(operation) - {"type", "window", "cutoff_hz", "order"}
    if unknown:
        raise ValueError(
            f"{field} filter operation has unknown options: {sorted(unknown)}"
        )

    operation_type = str(operation.get("type", "")).strip().lower()
    aliases = {"mean": "moving_average", "moving_mean": "moving_average"}
    operation_type = aliases.get(operation_type, operation_type)
    if operation_type not in _SUPPORTED_OPERATIONS:
        raise ValueError(
            f"{field} filter type must be one of {sorted(_SUPPORTED_OPERATIONS)}, "
            f"got {operation_type!r}"
        )

    normalized: dict[str, Any] = {"type": operation_type}
    if operation_type in {"median", "moving_average"}:
        window = int(operation.get("window", 0))
        if window < 1:
            raise ValueError(f"{field} {operation_type} window must be positive")
        if operation_type == "median" and window % 2 == 0:
            raise ValueError(f"{field} median window must be odd")
        normalized["window"] = window
    else:
        cutoff_hz = float(operation.get("cutoff_hz", math.nan))
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError(f"{field} lowpass cutoff_hz must be positive and finite")
        normalized["cutoff_hz"] = cutoff_hz
        has_order = "order" in operation
        order = int(operation.get("order", 1))
        if order < 1:
            raise ValueError(f"{field} lowpass order must be positive")
        # Preserve the historical canonical form when order was omitted.  The
        # filter implementation still treats an omitted order as one stage.
        if has_order:
            normalized["order"] = order
    return normalized


def filter_episode_values(
    timestamps_s: np.ndarray | torch.Tensor,
    values: np.ndarray | torch.Tensor,
    operations: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Apply a causal operation chain to one episode."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    source = np.asarray(values)
    if source.ndim < 1 or source.shape[0] != len(timestamps) or len(source) == 0:
        raise ValueError("causal filtering requires aligned non-empty [N, ...] data")
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("causal filtering supports numeric fields only")
    if not np.isfinite(timestamps).all() or not np.isfinite(source).all():
        raise ValueError("causal filter inputs must be finite")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("causal filter timestamps must increase within each episode")

    original_shape = source.shape
    original_dtype = source.dtype
    filtered = source.astype(np.float64, copy=True).reshape(len(source), -1)
    for operation in operations:
        operation_type = str(operation["type"])
        if operation_type == "median":
            filtered = _causal_trailing_median(
                filtered, window=int(operation["window"])
            )
        elif operation_type == "moving_average":
            filtered = _causal_trailing_moving_average(
                filtered, window=int(operation["window"])
            )
        elif operation_type == "lowpass":
            # A causal order-N low-pass is represented as an N-stage cascade.
            # Each stage uses the measured interval, so timestamp jitter does
            # not introduce a hidden future sample or a fixed-rate assumption.
            for _ in range(int(operation.get("order", 1))):
                filtered = _causal_one_pole_lowpass(
                    timestamps,
                    filtered,
                    cutoff_hz=float(operation["cutoff_hz"]),
                )
        else:  # pragma: no cover - normalized configs cannot reach this branch
            raise ValueError(f"unsupported causal filter operation {operation_type!r}")
    return filtered.reshape(original_shape).astype(original_dtype, copy=False)


def _causal_trailing_median(values: np.ndarray, *, window: int) -> np.ndarray:
    output = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        samples = values[start : index + 1]
        if len(samples) < window:
            samples = np.concatenate(
                (np.repeat(samples[:1], window - len(samples), axis=0), samples),
                axis=0,
            )
        output[index] = np.median(samples, axis=0)
    return output


def _causal_trailing_moving_average(
    values: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    if window == 1:
        return values.copy()
    padded = np.concatenate(
        (np.repeat(values[:1], window - 1, axis=0), values),
        axis=0,
    )
    cumulative = np.concatenate(
        (np.zeros((1, values.shape[1]), dtype=np.float64), padded.cumsum(axis=0)),
        axis=0,
    )
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def _causal_one_pole_lowpass(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    cutoff_hz: float,
) -> np.ndarray:
    output = np.empty_like(values)
    state = values[0].copy()
    output[0] = state
    for index in range(1, len(values)):
        dt = float(timestamps_s[index] - timestamps_s[index - 1])
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
        state = alpha * values[index] + (1.0 - alpha) * state
        output[index] = state
    return output


class FilteredDatasetView(torch.utils.data.Dataset):
    """Overlay filtered tensors on a HuggingFace-like dataset."""

    def __init__(self, dataset: Any, overrides: Mapping[str, torch.Tensor]):
        self.dataset = dataset
        self.overrides = dict(overrides)

    @property
    def column_names(self):
        return list(self.dataset.column_names)

    def with_format(self, *args, **kwargs):
        return FilteredDatasetView(
            self.dataset.with_format(*args, **kwargs),
            self.overrides,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if not isinstance(item, Mapping):
            return item
        result = dict(item)
        for key, value in self.overrides.items():
            if key in result:
                result[key] = value[index]
        return result


def build_filtered_dataset_view(
    dataset: Any,
    *,
    data_config: Mapping[str, Any],
    lowdim_keys: Mapping[str, str],
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Return a dataset view containing configured filtered columns."""

    filter_config = normalize_dataloader_filters(data_config, lowdim_keys)
    active = {}
    for key, spec in filter_config.items():
        pending_operations = spec["operations"][
            len(spec["dataset_preprocessed_operations"]) :
        ]
        if spec["enabled"] and pending_operations:
            active[key] = {**spec, "operations": pending_operations}
    if not active:
        return dataset, filter_config

    timestamp_key = str(
        data_config.get(
            "filter_timestamp_key",
            data_config.get("h5_timestamp_output_key", "timestamp"),
        )
    )
    available = set(dataset.column_names)
    if timestamp_key not in available:
        raise KeyError(
            f"dataloader.filters requires timestamp column {timestamp_key!r}"
        )

    dataset_to_logical: dict[str, str] = {}
    for logical_key in active:
        dataset_key = str(lowdim_keys[logical_key])
        previous = dataset_to_logical.setdefault(dataset_key, logical_key)
        if previous != logical_key:
            raise ValueError(
                f"filtered keys {previous!r} and {logical_key!r} map to the same "
                f"dataset column {dataset_key!r}"
            )
        if dataset_key not in available:
            raise KeyError(
                f"dataloader.filters.{logical_key} maps to missing column "
                f"{dataset_key!r}"
            )

    selected = [timestamp_key, *dataset_to_logical]
    formatted = dataset.with_format(
        "torch", columns=selected, output_all_columns=False
    )
    columns = formatted[:]
    timestamp_unit = str(data_config.get("filter_timestamp_unit", "s")).lower()
    if timestamp_unit not in _TIMESTAMP_SCALES_TO_SECONDS:
        raise ValueError(
            "dataloader.filter_timestamp_unit must be one of s, ms, us, or ns"
        )
    timestamps = (
        torch.as_tensor(columns[timestamp_key], dtype=torch.float64).reshape(-1)
        * _TIMESTAMP_SCALES_TO_SECONDS[timestamp_unit]
    )
    overrides: dict[str, torch.Tensor] = {}
    for dataset_key, logical_key in dataset_to_logical.items():
        source = torch.as_tensor(columns[dataset_key])
        output = torch.empty_like(source)
        for episode in episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            filtered = filter_episode_values(
                timestamps[start:end],
                source[start:end],
                active[logical_key]["operations"],
            )
            output[start:end] = torch.as_tensor(filtered, dtype=source.dtype)
        overrides[dataset_key] = output.contiguous()
        log.info(
            "causal dataloader filter ready: key=%s column=%s operations=%s",
            logical_key,
            dataset_key,
            active[logical_key]["operations"],
        )
    return FilteredDatasetView(dataset, overrides), filter_config
