
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm


PINN_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PINN_ROOT.parent
NERO_RUNS_ROOT = WORKSPACE_ROOT / "nero_ws" / "runs"

EXAMPLE_SHAPE_META = {
    "task": "your_task_name",
    "fps": 15,
    "master_timestamp_path": "TODO/path/to/master_timestamp",
    "features": {
        "observation.images.wrist": {
            "type": "image",
            "shape": [224, 224, 3],
            "h5_path": "TODO/path/to/wrist_image_sequence",
        },
        "observation.state": {
            "type": "float32",
            "shape": ["TODO_state_dim"],
            "h5_paths": [
                "TODO/path/to/state_part_1",
                "TODO/path/to/state_part_2",
            ],
        },
        "action": {
            "type": "float32",
            "shape": ["TODO_action_dim"],
            "h5_path": "TODO/path/to/action_sequence",
        },
    },
}


_DUAL_RATE_GRIDS = frozenset({"high_past", "low_anchor", "low_future"})
_DUAL_RATE_RESAMPLERS = frozenset(
    {"linear", "pchip", "pose", "previous", "nearest", "next"}
)
_STANDARD_RESAMPLERS = frozenset(
    {"index", "linear", "pchip", "previous", "nearest"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert standalone .h5/.hdf5 episode files to LeRobot v3."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/shape_meta.yaml"),
        help="Path to the conversion config YAML/JSON.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the H5 tree and exit.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Override io.input. Relative paths are resolved from the PINN root; "
            "nero_ws/runs/... and runs/... address the sibling nero_ws run directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Override io.output with the same path rules as --input. "
            "Ignored by --inspect-only."
        ),
    )
    parser.add_argument(
        "--print-example-shape-meta",
        action="store_true",
        help="Print a minimal shape_meta template and exit.",
    )
    return parser.parse_args()

# shape_meta：读取“字段说明书”
def load_shape_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"shape_meta file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit("YAML shape_meta needs PyYAML. Use .json or install pyyaml.") from exc
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"shape_meta must be a mapping, got: {type(data).__name__}")
    return data


def load_h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: h5py. Install it before reading H5 files.") from exc
    return h5py


def load_conversion_deps() -> tuple[Any, Any, Any]:
    h5py = load_h5py()
    try:
        import numpy as np  # type: ignore
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install numpy and lerobot before conversion."
        ) from exc
    return h5py, np, LeRobotDataset


def build_conversion_spec(shape_meta: Mapping[str, Any]) -> dict[str, Any]:
    fps = normalize_fps(shape_meta.get("fps"))
    timeline = normalize_timeline_spec(shape_meta.get("timeline"), fps)
    sampling = normalize_sampling_spec(shape_meta.get("sampling"), fps)
    if timeline is not None and sampling is not None:
        raise ValueError("shape_meta must not combine timeline and sampling contracts.")
    raw_features = shape_meta.get("features")
    if not isinstance(raw_features, Mapping):
        raise ValueError("shape_meta must contain a 'features' mapping.")

    mappings = []
    lerobot_features = {}
    derived_features = {}

    for lerobot_key, raw_spec in raw_features.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Feature {lerobot_key!r} spec must be a mapping.")

        sources = normalize_h5_sources(
            str(lerobot_key),
            raw_spec,
            dual_rate=timeline is not None,
        )
        h5_paths = [source["h5_path"] for source in sources]
        timestamp_path = raw_spec.get("timestamp_path")
        align = raw_spec.get("align", "index")
        window_size = int(raw_spec.get("window_size", 1))
        transform = raw_spec.get("transform")
        combine = raw_spec.get("combine")
        grid = raw_spec.get("grid")
        resample = raw_spec.get("resample")
        max_gap_s = raw_spec.get("max_gap_s")
        allow_stale = bool(raw_spec.get("allow_stale", False))
        emit_age_key = raw_spec.get("emit_age_key")
        validate_mapping(
            str(lerobot_key),
            h5_paths,
            timestamp_path,
            align,
            window_size,
            transform,
            combine,
            max_gap_s,
        )
        validate_timeline_mapping(
            str(lerobot_key),
            raw_spec,
            timeline,
        )

        feature_spec = {
            key: value
            for key, value in raw_spec.items()
            if key not in (
                "h5_path",
                "h5_paths",
                "sources",
                "timestamp_path",
                "align",
                "window_size",
                "transform",
                "combine",
                "grid",
                "resample",
                "max_gap_s",
                "allow_stale",
                "emit_age_key",
            )}
        normalize_feature_spec(feature_spec)

        lerobot_features[str(lerobot_key)] = feature_spec
        mappings.append(
            {
                "lerobot_key": str(lerobot_key),
                "h5_paths": h5_paths,
                "sources": sources,
                "timestamp_path": timestamp_path,
                "align": align,
                "window_size": window_size,
                "transform": transform,
                "combine": combine,
                "grid": grid,
                "resample": resample,
                "max_gap_s": max_gap_s,
                "allow_stale": allow_stale,
                "emit_age_key": emit_age_key,
                "per_source_resampling": raw_spec.get("sources") is not None,
                "feature_spec": feature_spec,
            }
        )
        if emit_age_key is not None:
            if not isinstance(emit_age_key, str) or not emit_age_key:
                raise ValueError(
                    f"Feature {lerobot_key!r} emit_age_key must be a non-empty string."
                )
            if "image" in emit_age_key:
                raise ValueError(
                    f"Feature {lerobot_key!r} emit_age_key must not contain "
                    f"'image': LeRobot 0.4 misclassifies numeric stats for "
                    f"{emit_age_key!r} as image statistics."
                )
            age_source_is_previous = (
                len(sources) == 1 and sources[0]["method"] == "previous"
            )
            if not age_source_is_previous or grid != "low_anchor":
                raise ValueError(
                    f"Feature {lerobot_key!r} emit_age_key currently requires "
                    "grid=low_anchor and resample=previous."
                )
            if emit_age_key in lerobot_features or emit_age_key in derived_features:
                raise ValueError(f"Duplicate generated feature key: {emit_age_key!r}")
            derived_features[emit_age_key] = {
                "dtype": "float32",
                "shape": (1,),
            }

    lerobot_features.update(derived_features)

    if timeline is not None and timeline["store_timestamps"]:
        timing_features = {
            "timing.anchor_timestamp_ns": {
                "dtype": "int64",
                "shape": (1,),
            },
            "timing.high_timestamp_ns": {
                "dtype": "int64",
                "shape": (timeline["high_window_size"], 1),
            },
        }
        if timeline["mode"] == "camera_rows":
            timing_features["timing.action_source_timestamp_ns"] = {
                "dtype": "int64",
                "shape": (1,),
            }
        else:
            timing_features["timing.action_timestamp_ns"] = {
                "dtype": "int64",
                "shape": (timeline["action_horizon"], 1),
            }
        duplicates = sorted(set(lerobot_features) & set(timing_features))
        if duplicates:
            raise ValueError(
                "Dual-rate timing features are generated automatically and "
                f"must not be declared manually: {duplicates}"
            )
        lerobot_features.update(timing_features)

    if sampling is not None and sampling["mode"] == "fixed_rate_causal_snapshot":
        validate_causal_snapshot_mappings(
            mappings,
            master_timestamp_path=str(shape_meta["master_timestamp_path"]),
        )
    if sampling is not None and sampling["mode"] == "raw_index":
        validate_raw_index_mappings(mappings)

    return {
        "io": shape_meta.get("io", {}),
        "task": str(shape_meta.get("task", "default_task")),
        "mappings": mappings,
        "lerobot_features": lerobot_features,
        "master_timestamp_path": shape_meta["master_timestamp_path"],
        "fps": fps,
        "timeline": timeline,
        "sampling": sampling,
    }


def normalize_sampling_spec(value: Any, fps: int) -> dict[str, Any] | None:
    """Normalize raw-index or fixed-rate sampling contracts."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("shape_meta.sampling must be a mapping.")
    mode = str(value.get("mode", "")).strip().lower()
    if mode == "raw_index":
        unknown = set(value) - {"mode"}
        if unknown:
            raise ValueError(
                f"shape_meta.sampling raw_index has unknown options: "
                f"{sorted(unknown)}"
            )
        return {"mode": mode, "fps": fps}

    unknown = set(value) - {"mode", "phase", "max_staleness_s"}
    if unknown:
        raise ValueError(f"shape_meta.sampling has unknown options: {sorted(unknown)}")
    if mode != "fixed_rate_causal_snapshot":
        raise ValueError(
            "shape_meta.sampling.mode must be 'raw_index' or "
            "'fixed_rate_causal_snapshot'."
        )
    phase = str(value.get("phase", "unix_epoch")).strip().lower()
    if phase != "unix_epoch":
        raise ValueError("shape_meta.sampling.phase must be 'unix_epoch'.")
    max_staleness_s = value.get("max_staleness_s")
    if max_staleness_s is not None:
        max_staleness_s = float(max_staleness_s)
        if not math.isfinite(max_staleness_s) or max_staleness_s <= 0.0:
            raise ValueError(
                "shape_meta.sampling.max_staleness_s must be positive and finite."
            )
    return {
        "mode": mode,
        "phase": phase,
        "fps": fps,
        "max_staleness_s": max_staleness_s,
    }


def validate_causal_snapshot_mappings(
    mappings: list[dict[str, Any]],
    *,
    master_timestamp_path: str,
) -> None:
    """Require every model feature to come from one selected H5 snapshot row."""

    for mapping in mappings:
        for source in mapping["sources"]:
            method = source.get("method")
            timestamp_path = source.get("timestamp_path")
            if method == "previous" and timestamp_path != master_timestamp_path:
                raise ValueError(
                    f"Feature {mapping['lerobot_key']!r} fixed-rate causal snapshot "
                    f"source {source['h5_path']!r} must use master timestamp path "
                    f"{master_timestamp_path!r}."
                )
            if method == "index" and timestamp_path not in {
                None,
                master_timestamp_path,
            }:
                raise ValueError(
                    f"Feature {mapping['lerobot_key']!r} fixed-rate causal snapshot "
                    f"source {source['h5_path']!r} align='index' must omit "
                    f"timestamp_path or use {master_timestamp_path!r}."
                )
            if method not in {"previous", "index"}:
                raise ValueError(
                    f"Feature {mapping['lerobot_key']!r} fixed-rate causal snapshot "
                    f"source {source['h5_path']!r} must use align='previous' "
                    "or align='index'."
                )


def validate_raw_index_mappings(mappings: list[dict[str, Any]]) -> None:
    """Require every raw-index feature to read the selected master row."""

    for mapping in mappings:
        for source in mapping["sources"]:
            if source.get("method") != "index":
                raise ValueError(
                    f"Feature {mapping['lerobot_key']!r} raw_index source "
                    f"{source['h5_path']!r} must use align='index'."
                )


def normalize_timeline_spec(value: Any, fps: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("shape_meta.timeline must be a mapping.")
    mode = str(value.get("mode", "dual_rate")).lower()
    if mode not in {"dual_rate", "camera_rows"}:
        raise ValueError(
            "shape_meta.timeline.mode must be 'dual_rate' or 'camera_rows'."
        )

    if mode == "camera_rows":
        unknown = set(value) - {
            "mode",
            "history_size",
            "action_horizon",
            "max_gap_s",
            "store_timestamps",
        }
        if unknown:
            raise ValueError(
                "shape_meta.timeline camera_rows has unknown options: "
                f"{sorted(unknown)}"
            )
        history_size = int(value.get("history_size", 4))
        action_horizon = int(value.get("action_horizon", 8))
        if history_size <= 0:
            raise ValueError("timeline.history_size must be positive.")
        if action_horizon <= 0:
            raise ValueError("timeline.action_horizon must be positive.")
        max_gap_s = float(value.get("max_gap_s", 0.05))
        if not math.isfinite(max_gap_s) or max_gap_s <= 0.0:
            raise ValueError("timeline.max_gap_s must be positive and finite.")
        return {
            "mode": mode,
            "low_fps": fps,
            "high_fps": None,
            "high_window_size": history_size,
            "action_horizon": action_horizon,
            "max_gap_s": max_gap_s,
            "store_timestamps": bool(value.get("store_timestamps", True)),
        }

    low_fps = normalize_fps(value.get("low_fps", fps))
    high_fps = normalize_fps(value.get("high_fps"))
    if low_fps != fps:
        raise ValueError(
            "shape_meta.fps must equal timeline.low_fps so LeRobot row "
            "timestamps match the low-rate grid."
        )
    if high_fps % low_fps != 0:
        raise ValueError(
            "timeline.high_fps must be an integer multiple of low_fps."
        )
    high_window_size = high_fps // low_fps
    action_horizon = int(value.get("action_horizon", 0))
    if action_horizon <= 0:
        raise ValueError("timeline.action_horizon must be positive.")
    max_gap_s = float(value.get("max_gap_s", 0.05))
    if not math.isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise ValueError("timeline.max_gap_s must be positive and finite.")

    return {
        "mode": mode,
        "low_fps": low_fps,
        "high_fps": high_fps,
        "high_window_size": high_window_size,
        "action_horizon": action_horizon,
        "max_gap_s": max_gap_s,
        "store_timestamps": bool(value.get("store_timestamps", True)),
    }


def validate_timeline_mapping(
    lerobot_key: str,
    raw_spec: Mapping[str, Any],
    timeline: Mapping[str, Any] | None,
) -> None:
    grid = raw_spec.get("grid")
    resample = raw_spec.get("resample")
    per_source_resampling = raw_spec.get("sources") is not None
    if timeline is None:
        if grid is not None or resample is not None:
            raise ValueError(
                f"Feature {lerobot_key!r} uses grid/resample without a "
                "top-level timeline contract."
            )
        return

    if grid not in _DUAL_RATE_GRIDS:
        raise ValueError(
            f"Feature {lerobot_key!r} grid must be one of "
            f"{sorted(_DUAL_RATE_GRIDS)}, got {grid!r}."
        )
    if not per_source_resampling and resample not in _DUAL_RATE_RESAMPLERS:
        raise ValueError(
            f"Feature {lerobot_key!r} resample must be one of "
            f"{sorted(_DUAL_RATE_RESAMPLERS)}, got {resample!r}."
        )
    feature_max_gap_s = raw_spec.get("max_gap_s")
    if feature_max_gap_s is not None and (
        not math.isfinite(float(feature_max_gap_s))
        or float(feature_max_gap_s) <= 0.0
    ):
        raise ValueError(
            f"Feature {lerobot_key!r} max_gap_s must be positive and finite."
        )
    if (
        not per_source_resampling
        and raw_spec.get("allow_stale", False)
        and resample not in {"previous", "nearest"}
    ):
        raise ValueError(
            f"Feature {lerobot_key!r} allow_stale is only valid with "
            "point resampling (previous or nearest)."
        )
    timestamp_path = raw_spec.get("timestamp_path")
    if (
        not per_source_resampling
        and (not isinstance(timestamp_path, str) or not timestamp_path)
    ):
        raise ValueError(
            f"Feature {lerobot_key!r} needs timestamp_path in dual-rate mode."
        )

    shape = raw_spec.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValueError(
            f"Feature {lerobot_key!r} needs a non-empty shape in dual-rate mode."
        )
    if grid == "high_past" and int(shape[0]) != timeline["high_window_size"]:
        raise ValueError(
            f"Feature {lerobot_key!r} high_past shape[0] must equal "
            f"high_fps/low_fps={timeline['high_window_size']}."
        )
    if grid == "low_future" and int(shape[0]) != timeline["action_horizon"]:
        raise ValueError(
            f"Feature {lerobot_key!r} low_future shape[0] must equal "
            f"timeline.action_horizon={timeline['action_horizon']}."
        )
    source_methods = {
        source.get("resample")
        for source in (raw_spec.get("sources") or [])
        if isinstance(source, Mapping)
    }
    if resample == "pose" or "pose" in source_methods:
        transform = raw_spec.get("transform")
        if transform not in (None, "ee_pose_matrix_to_quaternion"):
            raise ValueError(
                f"Feature {lerobot_key!r} pose resampling only supports "
                "raw xyz+xyzw or ee_pose_matrix_to_quaternion."
            )


def normalize_fps(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("shape_meta must define a positive integer 'fps'.")
    try:
        fps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape_meta 'fps' must be a positive integer.") from exc
    if fps <= 0 or float(value) != fps:
        raise ValueError("shape_meta 'fps' must be a positive integer.")
    return fps


def resolve_io_path(value: str | Path) -> Path:
    """Resolve conversion paths independently of the caller's working directory.

    Existing shape-meta paths remain PINN-root relative. Two aliases make data
    collected by the sibling nero_ws repository explicit and portable:

    - ``nero_ws/runs/<name>`` -> ``<workspace>/nero_ws/runs/<name>``
    - ``runs/<name>`` -> ``<workspace>/nero_ws/runs/<name>``
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    parts = path.parts
    if len(parts) >= 2 and parts[:2] == ("nero_ws", "runs"):
        return (WORKSPACE_ROOT / path).resolve()
    if parts and parts[0] == "runs":
        return (NERO_RUNS_ROOT.joinpath(*parts[1:])).resolve()
    return (PINN_ROOT / path).resolve()


def config_path(
    config: Mapping[str, Any],
    key: str,
    required: bool = True,
    *,
    override: str | Path | None = None,
) -> Path | None:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")

    value = override if override is not None else io_config.get(key)
    if value is None:
        if required:
            raise ValueError(f"config needs io.{key}")
        return None
    return resolve_io_path(value)


def normalize_feature_spec(feature_spec: dict[str, Any]) -> None:
    """Normalize config feature metadata to what LeRobot validates against."""

    shape = feature_spec.get("shape")
    if isinstance(shape, list):
        feature_spec["shape"] = tuple(shape)


def config_bool(config: Mapping[str, Any], key: str, default: bool = False) -> bool:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")
    return bool(io_config.get(key, default))


def config_int(config: Mapping[str, Any], key: str, default: int | None = None) -> int | None:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")

    value = io_config.get(key, default)
    if value is None:
        return None
    return int(value)


def config_str(config: Mapping[str, Any], key: str, default: str) -> str:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")
    return str(io_config.get(key, default))


def normalize_h5_paths(lerobot_key: str, raw_spec: Mapping[str, Any]) -> list[str]:
    # 一对一映射写法：
    #     h5_path: teleop/q_follower

    # 聚合映射写法：
    #     h5_paths:
    #       - teleop/q_follower
    #       - teleop/gripper_state
    h5_path = raw_spec.get("h5_path")
    h5_paths = raw_spec.get("h5_paths")

    if h5_path is not None and h5_paths is not None:
        raise ValueError(f"Feature {lerobot_key!r} can use h5_path or h5_paths, not both.")

    if h5_paths is None:
        if not isinstance(h5_path, str) or not h5_path:
            raise ValueError(f"Feature {lerobot_key!r} needs h5_path or h5_paths.")
        return [h5_path]

    if not isinstance(h5_paths, list) or not h5_paths:
        raise ValueError(f"Feature {lerobot_key!r} h5_paths must be a non-empty list.")

    for path in h5_paths:
        if not isinstance(path, str) or not path:
            raise ValueError(f"Feature {lerobot_key!r} h5_paths must only contain strings.")
    return h5_paths


def normalize_h5_sources(
    lerobot_key: str,
    raw_spec: Mapping[str, Any],
    *,
    dual_rate: bool,
) -> list[dict[str, Any]]:
    """Normalize legacy mappings and per-source resampling declarations."""

    raw_sources = raw_spec.get("sources")
    method_key = "resample" if dual_rate else "align"
    if raw_sources is None:
        sources = [
            {
                "h5_path": h5_path,
                "timestamp_path": raw_spec.get("timestamp_path"),
                "method": raw_spec.get(method_key, "index"),
                "max_gap_s": raw_spec.get("max_gap_s"),
                "allow_stale": bool(raw_spec.get("allow_stale", False)),
            }
            for h5_path in normalize_h5_paths(lerobot_key, raw_spec)
        ]
        for source in sources:
            validate_source_resampling(
                lerobot_key,
                source,
                dual_rate=dual_rate,
                structured=False,
            )
        return sources

    conflicting = [
        key
        for key in (
            "h5_path",
            "h5_paths",
            "timestamp_path",
            "align",
            "resample",
            "max_gap_s",
            "allow_stale",
        )
        if key in raw_spec
    ]
    if conflicting:
        raise ValueError(
            f"Feature {lerobot_key!r} uses sources and cannot also define "
            f"top-level source options: {conflicting}."
        )
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Feature {lerobot_key!r} sources must be a non-empty list.")

    sources = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise ValueError(
                f"Feature {lerobot_key!r} sources[{index}] must be a mapping."
            )
        wrong_method_key = "align" if dual_rate else "resample"
        if wrong_method_key in raw_source:
            raise ValueError(
                f"Feature {lerobot_key!r} sources[{index}] must use "
                f"{method_key!r}, not {wrong_method_key!r}."
            )
        unknown = set(raw_source) - {
            "h5_path",
            "timestamp_path",
            method_key,
            "max_gap_s",
            "allow_stale",
        }
        if unknown:
            raise ValueError(
                f"Feature {lerobot_key!r} sources[{index}] has unknown options: "
                f"{sorted(unknown)}."
            )
        h5_path = raw_source.get("h5_path")
        if not isinstance(h5_path, str) or not h5_path:
            raise ValueError(
                f"Feature {lerobot_key!r} sources[{index}] needs h5_path."
            )
        source = {
            "h5_path": h5_path,
            "timestamp_path": raw_source.get("timestamp_path"),
            "method": raw_source.get(method_key),
            "max_gap_s": raw_source.get("max_gap_s"),
            "allow_stale": bool(raw_source.get("allow_stale", False)),
        }
        validate_source_resampling(
            lerobot_key,
            source,
            dual_rate=dual_rate,
            structured=True,
        )
        sources.append(source)
    methods = {source["method"] for source in sources}
    if "index" in methods and len(methods) > 1:
        raise ValueError(
            f"Feature {lerobot_key!r} cannot mix align='index' with "
            "timestamp-based alignment in one sources list."
        )
    return sources


def validate_source_resampling(
    lerobot_key: str,
    source: Mapping[str, Any],
    *,
    dual_rate: bool,
    structured: bool,
) -> None:
    method = source.get("method")
    supported = _DUAL_RATE_RESAMPLERS if dual_rate else _STANDARD_RESAMPLERS
    if method not in supported:
        if structured or dual_rate:
            raise ValueError(
                f"Feature {lerobot_key!r} source {source['h5_path']!r} has "
                f"unknown resampling method {method!r}; choose one of "
                f"{sorted(supported)}."
            )
        return

    timestamp_path = source.get("timestamp_path")
    max_gap_s = source.get("max_gap_s")
    allow_stale = bool(source.get("allow_stale", False))
    if method == "index":
        if max_gap_s is not None or allow_stale:
            raise ValueError(
                f"Feature {lerobot_key!r} source {source['h5_path']!r} "
                "align='index' does not use max_gap_s or allow_stale."
            )
        return

    if not isinstance(timestamp_path, str) or not timestamp_path:
        raise ValueError(
            f"Feature {lerobot_key!r} source {source['h5_path']!r} "
            f"method={method!r} needs timestamp_path."
        )
    if max_gap_s is not None and (
        not math.isfinite(float(max_gap_s)) or float(max_gap_s) <= 0.0
    ):
        raise ValueError(
            f"Feature {lerobot_key!r} source {source['h5_path']!r} "
            "max_gap_s must be positive and finite."
        )
    if allow_stale and method not in {"previous", "nearest"}:
        raise ValueError(
            f"Feature {lerobot_key!r} source {source['h5_path']!r} "
            "allow_stale is only valid with point resampling "
            "(previous/ZOH or nearest)."
        )
    if not dual_rate and method in {"linear", "pchip"} and max_gap_s is None:
        raise ValueError(
            f"Feature {lerobot_key!r} source {source['h5_path']!r} "
            f"method={method!r} needs max_gap_s."
        )


def validate_mapping(
    lerobot_key: str,
    h5_paths: list[str],
    timestamp_path: Any,
    align: Any,
    window_size: int,
    transform: Any = None,
    combine: Any = None,
    max_gap_s: Any = None,
) -> None:
    if align not in (
        "index",
        "nearest",
        "linear",
        "pchip",
        "previous",
        "nearest_past_window",
        "nearest_future_window",
    ):
        raise ValueError(f"Feature {lerobot_key!r} has unknown align mode: {align!r}")

    if align in (
        "nearest",
        "linear",
        "pchip",
        "previous",
        "nearest_past_window",
        "nearest_future_window",
    ):
        if not isinstance(timestamp_path, str) or not timestamp_path:
            raise ValueError(f"Feature {lerobot_key!r} align={align!r} needs timestamp_path.")

    if max_gap_s is not None and (
        not math.isfinite(float(max_gap_s)) or float(max_gap_s) <= 0.0
    ):
        raise ValueError(
            f"Feature {lerobot_key!r} max_gap_s must be positive and finite."
        )
    if align in ("linear", "pchip") and max_gap_s is None:
        raise ValueError(
            f"Feature {lerobot_key!r} align={align!r} needs max_gap_s so large "
            "acquisition gaps are not interpolated silently."
        )

    if align in ("nearest_past_window", "nearest_future_window"):
        if len(h5_paths) != 1:
            raise ValueError(f"Feature {lerobot_key!r} align={align!r} supports one h5_path only.")
        if window_size <= 0:
            raise ValueError(f"Feature {lerobot_key!r} window_size must be positive.")

    supported_transforms = (None, "ee_pose_matrix_to_quaternion")
    if transform not in supported_transforms:
        raise ValueError(f"Feature {lerobot_key!r} has unknown transform: {transform!r}")

    supported_combinations = (None, "subtract")
    if combine not in supported_combinations:
        raise ValueError(f"Feature {lerobot_key!r} has unknown combine mode: {combine!r}")
    if combine == "subtract" and len(h5_paths) != 2:
        raise ValueError(
            f"Feature {lerobot_key!r} combine='subtract' requires exactly two "
            "h5_paths; the first value is subtracted by the second."
        )

class H5Dataset:
    def __init__(
        self,
        input_path: Path,
        *,
        h5py: Any,
        np: Any | None = None,
        max_episodes: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.h5py = h5py
        self.np = np
        self.max_episodes = max_episodes

    def files(self) -> list[Path]:
        if self.input_path.is_file():
            h5_files = [self.input_path]
        else:
            h5_files = sorted(self.input_path.glob("*.h5")) + sorted(self.input_path.glob("*.hdf5"))

        if not h5_files:
            raise FileNotFoundError(f"No .h5/.hdf5 files found under {self.input_path}")

        if self.max_episodes is not None:
            h5_files = h5_files[: self.max_episodes]
        return h5_files

    def inspect(self):
        # 打印 H5 树结构
        for h5_path in self.files():
            # print(f"\n# {h5_path}", flush=True)
            # print("before open", flush=True)
            with self.h5py.File(h5_path, "r") as h5_file:
                # print("after open", flush=True)
                self._print_node(h5_file)
                # print("after print node", flush=True)

    def open_episode(self, h5_path: Path):
        # 用法：
        #     with h5_dataset.open_episode(path) as h5_file:

        return self.h5py.File(h5_path, "r")

    def build_episode_cache(
        self,
        h5_file,
        mappings,
        master_timestamp_path,
        fps,
        h5_path,
        timeline=None,
        sampling=None,
    ):
        """Cache dataset handles and timestamp arrays for one opened episode."""

        dataset_cache = {}
        timestamp_cache = {}

        all_dataset_paths = {master_timestamp_path}
        all_timestamp_paths = {master_timestamp_path}
        for mapping in mappings:
            for source in mapping["sources"]:
                all_dataset_paths.add(source["h5_path"])
                timestamp_path = source.get("timestamp_path")
                if timestamp_path:
                    all_timestamp_paths.add(timestamp_path)

        for field_path in all_dataset_paths | all_timestamp_paths:
            dataset_cache[field_path] = self._dataset(h5_file, field_path, h5_path)

        for timestamp_path in all_timestamp_paths:
            timestamp_cache[timestamp_path] = dataset_cache[timestamp_path][:]

        self._validate_index_sources(
            dataset_cache=dataset_cache,
            timestamp_cache=timestamp_cache,
            mappings=mappings,
            master_timestamp_path=master_timestamp_path,
            h5_path=h5_path,
        )

        if timeline is not None:
            if timeline["mode"] == "camera_rows":
                return self._build_camera_rows_episode_cache(
                    dataset_cache=dataset_cache,
                    timestamp_cache=timestamp_cache,
                    mappings=mappings,
                    master_timestamp_path=master_timestamp_path,
                    h5_path=h5_path,
                    timeline=timeline,
                )
            return self._build_dual_rate_episode_cache(
                dataset_cache=dataset_cache,
                timestamp_cache=timestamp_cache,
                mappings=mappings,
                master_timestamp_path=master_timestamp_path,
                h5_path=h5_path,
                timeline=timeline,
            )

        if sampling is not None:
            if sampling["mode"] == "raw_index":
                return self._build_raw_index_episode_cache(
                    dataset_cache=dataset_cache,
                    timestamp_cache=timestamp_cache,
                    mappings=mappings,
                    master_timestamp_path=master_timestamp_path,
                    h5_path=h5_path,
                )
            return self._build_causal_snapshot_episode_cache(
                dataset_cache=dataset_cache,
                timestamp_cache=timestamp_cache,
                mappings=mappings,
                master_timestamp_path=master_timestamp_path,
                fps=fps,
                h5_path=h5_path,
                sampling=sampling,
            )

        target_timestamps = self.uniform_timestamps(
            timestamp_cache[master_timestamp_path],
            master_timestamp_path,
            fps,
        )
        cache = {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
            "target_timestamps": target_timestamps,
            "timestamp_seconds": {},
            "aligned": {},
        }
        target_seconds = self._timestamps_seconds(
            target_timestamps,
            master_timestamp_path,
        )
        for mapping in mappings:
            method = mapping["align"]
            if all(
                source.get("method") == "index"
                for source in mapping["sources"]
            ):
                continue
            if not mapping.get("per_source_resampling") and method not in {
                "linear",
                "pchip",
                "previous",
                "nearest",
            }:
                continue
            for source in mapping["sources"]:
                timestamp_path = source["timestamp_path"]
                source_seconds = cache["timestamp_seconds"].get(timestamp_path)
                if source_seconds is None:
                    source_seconds = self._timestamps_seconds(
                        timestamp_cache[timestamp_path],
                        timestamp_path,
                    )
                    cache["timestamp_seconds"][timestamp_path] = source_seconds
                max_gap_s = source.get("max_gap_s")
                if not source.get("allow_stale") and max_gap_s is not None:
                    self._validate_resample_targets(
                        source_seconds,
                        target_seconds,
                        source["method"],
                        float(max_gap_s),
                        mapping["lerobot_key"],
                        h5_path,
                    )
            cache["aligned"][mapping["lerobot_key"]] = self._resample_mapping(
                mapping,
                target_seconds,
                cache,
            )
        return cache

    def _build_raw_index_episode_cache(
        self,
        *,
        dataset_cache,
        timestamp_cache,
        mappings,
        master_timestamp_path,
        h5_path,
    ):
        master_timestamps = self.np.asarray(
            timestamp_cache[master_timestamp_path]
        ).reshape(-1)
        if master_timestamps.size == 0:
            raise ValueError(f"Raw-index episode is empty: {h5_path}")
        if not self.np.isfinite(master_timestamps).all():
            raise ValueError(f"Raw-index timestamps must be finite in {h5_path}.")
        if master_timestamps.size > 1 and self.np.any(
            self.np.diff(master_timestamps) <= 0
        ):
            raise ValueError(
                f"Raw-index timestamps must be strictly increasing in {h5_path}."
            )
        return {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
            "target_timestamps": master_timestamps,
            "timestamp_seconds": {},
            "aligned": {},
            "raw_index": True,
        }

    def _validate_index_sources(
        self,
        *,
        dataset_cache,
        timestamp_cache,
        mappings,
        master_timestamp_path,
        h5_path,
    ) -> None:
        master_length = len(timestamp_cache[master_timestamp_path])
        for mapping in mappings:
            for source in mapping["sources"]:
                if source.get("method") != "index":
                    continue
                source_dataset = dataset_cache[source["h5_path"]]
                source_shape = tuple(source_dataset.shape)
                if not source_shape or source_shape[0] != master_length:
                    raise ValueError(
                        f"Feature {mapping['lerobot_key']!r} source "
                        f"{source['h5_path']!r} align='index' in {h5_path} "
                        f"requires exactly {master_length} rows to match "
                        f"{master_timestamp_path!r}, got shape {source_shape}."
                    )

    def _build_causal_snapshot_episode_cache(
        self,
        *,
        dataset_cache,
        timestamp_cache,
        mappings,
        master_timestamp_path,
        fps,
        h5_path,
        sampling,
    ):
        master_timestamps = timestamp_cache[master_timestamp_path]
        target_timestamps = self.fixed_phase_timestamps(
            master_timestamps,
            master_timestamp_path,
            fps,
        )
        master_seconds = self._timestamps_seconds(
            master_timestamps,
            master_timestamp_path,
        )
        target_seconds = self._timestamps_seconds(
            target_timestamps,
            master_timestamp_path,
        )
        snapshot_indices = self._point_sample_indices(
            master_seconds,
            target_seconds,
            "previous",
        )
        ages_s = target_seconds - master_seconds[snapshot_indices]
        max_staleness_s = sampling.get("max_staleness_s")
        if max_staleness_s is not None and self.np.any(
            ages_s > float(max_staleness_s) + 1.0e-12
        ):
            first = int(
                self.np.flatnonzero(
                    ages_s > float(max_staleness_s) + 1.0e-12
                )[0]
            )
            raise ValueError(
                f"Causal snapshot in {h5_path} is stale by {ages_s[first]:.6f}s "
                f"at target={target_seconds[first]:.9f}s; maximum is "
                f"{float(max_staleness_s):.6f}s."
            )

        cache = {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
            "timestamp_seconds": {master_timestamp_path: master_seconds},
            "master_timestamp_path": master_timestamp_path,
            "target_timestamps": target_timestamps,
            "snapshot_indices": snapshot_indices,
            "snapshot_age_s": ages_s,
            "aligned": {},
            "causal_snapshot": True,
        }
        for mapping in mappings:
            cache["aligned"][mapping["lerobot_key"]] = (
                self._sample_snapshot_mapping(mapping, snapshot_indices, cache)
            )
        return cache

    def _sample_snapshot_mapping(self, mapping, snapshot_indices, cache):
        values = []
        for source in mapping["sources"]:
            source_values = self.np.asarray(cache["datasets"][source["h5_path"]][:])
            sampled = source_values[snapshot_indices]
            sampled = self._apply_transform(sampled, mapping.get("transform"))
            values.append(sampled)
        if len(values) == 1:
            result = values[0]
        elif mapping.get("combine") == "subtract":
            result = self._subtract_values(values, mapping["lerobot_key"])
        else:
            result = self.np.concatenate(
                [value.reshape(*value.shape[:-1], -1) for value in values],
                axis=-1,
            )
        return self.np.asarray(result).astype(
            self._dtype_from_feature(mapping["feature_spec"]),
            copy=False,
        )

    def _build_camera_rows_episode_cache(
        self,
        *,
        dataset_cache,
        timestamp_cache,
        mappings,
        master_timestamp_path,
        h5_path,
        timeline,
    ):
        """Build one output row for each valid *recorded* master-camera frame.

        No target clock is synthesized here. Observation point samples are
        causal, actions use the first source row at/after the camera timestamp,
        and ``high_past`` contains the latest raw history rows rather than
        samples on an artificial 100 Hz grid.
        """

        if self.np is None:
            raise RuntimeError("camera_rows conversion requires numpy.")
        master_raw = self.np.asarray(
            timestamp_cache[master_timestamp_path]
        ).reshape(-1)
        master_seconds = self._timestamps_seconds(master_raw, master_timestamp_path)
        if master_seconds.size == 0:
            raise ValueError(f"Camera-row episode is empty: {h5_path}")

        timestamp_seconds = {master_timestamp_path: master_seconds}
        for mapping in mappings:
            for source in self._mapping_sources(mapping):
                timestamp_path = source["timestamp_path"]
                if timestamp_path not in timestamp_seconds:
                    timestamp_seconds[timestamp_path] = self._timestamps_seconds(
                        timestamp_cache[timestamp_path], timestamp_path
                    )

        row_count = len(master_seconds)
        valid = self.np.ones(row_count, dtype=bool)
        plans = {}
        history_size = int(timeline["high_window_size"])
        eps = 1.0e-12

        for mapping in mappings:
            grid = mapping["grid"]
            mapping_plans = []
            for source in self._mapping_sources(mapping):
                source_seconds = timestamp_seconds[source["timestamp_path"]]
                method = source["method"]
                if grid == "high_past":
                    if method not in {"previous", "nearest"}:
                        raise ValueError(
                            f"Feature {mapping['lerobot_key']!r} camera_rows "
                            "high_past must use resample='previous'."
                        )
                    last = self.np.searchsorted(
                        source_seconds, master_seconds, side="right"
                    ) - 1
                    enough_history = last >= history_size - 1
                    safe_last = last.clip(
                        history_size - 1, len(source_seconds) - 1
                    )
                    offsets = self.np.arange(
                        -(history_size - 1), 1, dtype=self.np.int64
                    )
                    indices = safe_last[:, None] + offsets[None, :]
                    source_time_window = source_seconds[indices]
                    max_gap = source.get("max_gap_s")
                    if max_gap is None:
                        max_gap = mapping.get("max_gap_s")
                    if max_gap is None:
                        max_gap = timeline["max_gap_s"]
                    recent = (
                        master_seconds - source_time_window[:, -1]
                        <= float(max_gap) + eps
                    )
                    continuous = self.np.all(
                        self.np.diff(source_time_window, axis=1)
                        <= float(max_gap) + eps,
                        axis=1,
                    )
                    source_valid = enough_history & recent & continuous
                else:
                    if method in {"previous", "nearest"}:
                        raw_indices = self.np.searchsorted(
                            source_seconds, master_seconds, side="right"
                        ) - 1
                    elif method == "next":
                        raw_indices = self.np.searchsorted(
                            source_seconds, master_seconds, side="left"
                        )
                    else:
                        raise ValueError(
                            f"Feature {mapping['lerobot_key']!r} camera_rows "
                            f"does not support resample={method!r}."
                        )
                    in_range = (raw_indices >= 0) & (
                        raw_indices < len(source_seconds)
                    )
                    indices = raw_indices.clip(0, len(source_seconds) - 1)
                    source_valid = in_range
                    max_gap = source.get("max_gap_s")
                    if max_gap is None:
                        max_gap = mapping.get("max_gap_s")
                    if max_gap is None:
                        max_gap = timeline["max_gap_s"]
                    if not source.get("allow_stale"):
                        source_valid &= (
                            self.np.abs(master_seconds - source_seconds[indices])
                            <= float(max_gap) + eps
                        )
                valid &= source_valid
                mapping_plans.append(
                    {"source": source, "indices": indices}
                )
            plans[mapping["lerobot_key"]] = mapping_plans

        selected_rows = self.np.flatnonzero(valid)
        if selected_rows.size == 0:
            raise ValueError(f"No complete camera rows remain in {h5_path}.")
        anchors = master_seconds[selected_rows]
        anchor_raw = master_raw[selected_rows]
        anchor_timestamp_ns = self._timestamps_ns(
            anchor_raw, master_timestamp_path
        )

        cache = {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
            "timestamp_seconds": timestamp_seconds,
            "target_timestamps": anchor_raw,
            "anchor_timestamp_ns": anchor_timestamp_ns,
            "selected_master_indices": selected_rows,
            "resampled": {},
            "camera_rows": True,
            "store_timestamps": bool(timeline["store_timestamps"]),
        }
        high_timestamp_ns = None
        action_source_timestamp_ns = None
        for mapping in mappings:
            values = []
            selected_plans = plans[mapping["lerobot_key"]]
            for plan in selected_plans:
                source = plan["source"]
                indices = self.np.asarray(plan["indices"])[selected_rows]
                source_times = timestamp_seconds[source["timestamp_path"]]
                if mapping["grid"] == "high_past" and high_timestamp_ns is None:
                    high_timestamp_ns = self._timestamps_ns(
                        timestamp_cache[source["timestamp_path"]],
                        source["timestamp_path"],
                    )[indices]
                if (
                    source["method"] == "next"
                    and action_source_timestamp_ns is None
                ):
                    action_source_timestamp_ns = self._timestamps_ns(
                        timestamp_cache[source["timestamp_path"]],
                        source["timestamp_path"],
                    )[indices]

                if self._is_media_feature(mapping["feature_spec"]):
                    if len(selected_plans) != 1:
                        raise ValueError(
                            f"Media feature {mapping['lerobot_key']!r} must have "
                            "one source."
                        )
                    cached = {"indices": indices, "mapping": mapping}
                    if mapping.get("emit_age_key") is not None:
                        cached["age_s"] = (
                            anchors - source_times[indices]
                        ).astype(self.np.float32)
                    cache["resampled"][mapping["lerobot_key"]] = cached
                    values = []
                    break

                source_values = self.np.asarray(
                    dataset_cache[source["h5_path"]][:]
                )[indices]
                source_values = self._apply_transform(
                    source_values, mapping.get("transform")
                )
                values.append(source_values)

            if self._is_media_feature(mapping["feature_spec"]):
                continue
            if len(values) == 1:
                result = values[0]
            elif mapping.get("combine") == "subtract":
                result = self._subtract_values(values, mapping["lerobot_key"])
            else:
                result = self.np.concatenate(
                    [value.reshape(*value.shape[:-1], -1) for value in values],
                    axis=-1,
                )
            cache["resampled"][mapping["lerobot_key"]] = self.np.asarray(
                result
            ).astype(
                self._dtype_from_feature(mapping["feature_spec"]), copy=False
            )

        if high_timestamp_ns is None:
            high_timestamp_ns = self.np.repeat(
                anchor_timestamp_ns[:, None], history_size, axis=1
            )
        if action_source_timestamp_ns is None:
            action_source_timestamp_ns = anchor_timestamp_ns.copy()
        cache["high_timestamp_ns"] = high_timestamp_ns
        cache["action_source_timestamp_ns"] = action_source_timestamp_ns
        return cache

    def _build_dual_rate_episode_cache(
        self,
        *,
        dataset_cache,
        timestamp_cache,
        mappings,
        master_timestamp_path,
        h5_path,
        timeline,
    ):
        if self.np is None:
            raise RuntimeError("Dual-rate conversion requires numpy.")
        master_seconds = self._timestamps_seconds(
            timestamp_cache[master_timestamp_path],
            master_timestamp_path,
        )
        if master_seconds.size < 2:
            raise ValueError(f"Dual-rate master timeline is too short in {h5_path}")

        high_fps = int(timeline["high_fps"])
        low_fps = int(timeline["low_fps"])
        high_steps = int(timeline["high_window_size"])
        action_steps = int(timeline["action_horizon"])
        high_dt = 1.0 / high_fps
        low_dt = 1.0 / low_fps
        high_span = (high_steps - 1) * high_dt
        action_span = (action_steps - 1) * low_dt

        anchor_start = float(master_seconds[0])
        anchor_end = float(master_seconds[-1])
        timestamp_seconds = {}
        for mapping in mappings:
            for source in self._mapping_sources(mapping):
                timestamp_path = source["timestamp_path"]
                source_seconds = timestamp_seconds.get(timestamp_path)
                if source_seconds is None:
                    source_seconds = self._timestamps_seconds(
                        timestamp_cache[timestamp_path],
                        timestamp_path,
                    )
                    timestamp_seconds[timestamp_path] = source_seconds
                grid = mapping["grid"]
                stale_hold = source.get("allow_stale") and source["method"] in {
                    "previous",
                    "nearest",
                }
                if grid == "high_past":
                    anchor_start = max(
                        anchor_start,
                        float(source_seconds[0]) + high_span,
                    )
                    if not stale_hold:
                        anchor_end = min(anchor_end, float(source_seconds[-1]))
                elif grid == "low_future":
                    anchor_start = max(anchor_start, float(source_seconds[0]))
                    if not stale_hold:
                        anchor_end = min(
                            anchor_end,
                            float(source_seconds[-1]) - action_span,
                        )
                else:
                    anchor_start = max(anchor_start, float(source_seconds[0]))
                    if not stale_hold:
                        anchor_end = min(anchor_end, float(source_seconds[-1]))

        origin = float(master_seconds[0])
        first_tick = int(math.ceil((anchor_start - origin) * low_fps - 1.0e-9))
        last_tick = int(math.floor((anchor_end - origin) * low_fps + 1.0e-9))
        if last_tick < first_tick:
            raise ValueError(
                f"No complete dual-rate rows remain in {h5_path}: "
                f"anchor interval=[{anchor_start:.9f}, {anchor_end:.9f}]"
            )
        low_ticks = self.np.arange(first_tick, last_tick + 1, dtype=self.np.int64)
        anchors = origin + low_ticks.astype(self.np.float64) / low_fps
        high_offsets = (
            self.np.arange(-(high_steps - 1), 1, dtype=self.np.float64)
            / high_fps
        )
        action_offsets = (
            self.np.arange(action_steps, dtype=self.np.float64) / low_fps
        )
        high_targets = anchors[:, None] + high_offsets[None, :]
        action_targets = anchors[:, None] + action_offsets[None, :]
        origin_ns = self.np.int64(self.np.rint(origin * 1.0e9))
        anchor_timestamp_ns = origin_ns + self.np.rint(
            low_ticks.astype(self.np.float64) * 1.0e9 / low_fps
        ).astype(self.np.int64)
        high_offset_ns = self.np.rint(high_offsets * 1.0e9).astype(self.np.int64)
        action_offset_ns = self.np.rint(action_offsets * 1.0e9).astype(self.np.int64)
        grids = {
            "high_past": high_targets,
            "low_anchor": anchors,
            "low_future": action_targets,
        }

        cache = {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
            "timestamp_seconds": timestamp_seconds,
            "target_timestamps": anchors,
            "dual_rate": True,
            "high_targets": high_targets,
            "action_targets": action_targets,
            "anchor_timestamp_ns": anchor_timestamp_ns,
            "high_timestamp_ns": anchor_timestamp_ns[:, None] + high_offset_ns[None, :],
            "action_timestamp_ns": (
                anchor_timestamp_ns[:, None] + action_offset_ns[None, :]
            ),
            "resampled": {},
            "store_timestamps": bool(timeline["store_timestamps"]),
        }
        for mapping in mappings:
            targets = grids[mapping["grid"]]
            sources = self._mapping_sources(mapping)
            for source in sources:
                source_max_gap_s = source.get("max_gap_s")
                if source_max_gap_s is None:
                    source_max_gap_s = timeline["max_gap_s"]
                if source.get("allow_stale"):
                    self._point_sample_indices(
                        timestamp_seconds[source["timestamp_path"]],
                        targets,
                        source["method"],
                    )
                else:
                    self._validate_resample_targets(
                        timestamp_seconds[source["timestamp_path"]],
                        targets,
                        source["method"],
                        float(source_max_gap_s),
                        mapping["lerobot_key"],
                        h5_path,
                    )
            if self._is_media_feature(mapping["feature_spec"]):
                if len(sources) != 1:
                    raise ValueError(
                        f"Media feature {mapping['lerobot_key']!r} must have one source."
                    )
                source = sources[0]
                indices = self._point_sample_indices(
                    timestamp_seconds[source["timestamp_path"]],
                    targets,
                    source["method"],
                )
                cache["resampled"][mapping["lerobot_key"]] = {
                    "indices": indices,
                    "mapping": mapping,
                }
                if mapping.get("emit_age_key") is not None:
                    source_times = timestamp_seconds[source["timestamp_path"]]
                    cache["resampled"][mapping["lerobot_key"]]["age_s"] = (
                        self.np.asarray(targets, dtype=self.np.float64)
                        - source_times[indices]
                    ).astype(self.np.float32)
            else:
                cache["resampled"][mapping["lerobot_key"]] = (
                    self._resample_mapping(mapping, targets, cache)
                )
        return cache

    @staticmethod
    def _is_media_feature(feature_spec):
        dtype = str(feature_spec.get("dtype", feature_spec.get("type", ""))).lower()
        return dtype in {"image", "video"}

    def _timestamps_seconds(self, timestamps, timestamp_path):
        values = self.np.asarray(timestamps, dtype=self.np.float64).reshape(-1)
        if values.size == 0 or not self.np.isfinite(values).all():
            raise ValueError(f"Timestamp array {timestamp_path!r} is empty or non-finite")
        if values.size > 1 and self.np.any(self.np.diff(values) <= 0):
            raise ValueError(
                f"Timestamp array {timestamp_path!r} must be strictly increasing"
            )
        return values * self._timestamp_seconds_scale(timestamp_path)

    def _validate_resample_targets(
        self,
        source_times,
        targets,
        method,
        max_gap_s,
        feature_name,
        h5_path,
    ):
        flat_targets = self.np.asarray(targets, dtype=self.np.float64).reshape(-1)
        if method in {"linear", "pchip", "pose"}:
            left, right, _ = self._linear_sample_plan(source_times, flat_targets)
            gaps = source_times[right] - source_times[left]
        else:
            indices = self._point_sample_indices(source_times, flat_targets, method)
            gaps = self.np.abs(flat_targets - source_times[indices])
        invalid = gaps > max_gap_s + 1.0e-12
        if self.np.any(invalid):
            first = int(self.np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Feature {feature_name!r} in {h5_path} cannot be resampled: "
                f"gap={gaps[first]:.6f}s exceeds max_gap_s={max_gap_s:.6f}s "
                f"at target={flat_targets[first]:.9f}s"
            )

    def _point_sample_indices(self, source_times, targets, method):
        flat = self.np.asarray(targets, dtype=self.np.float64).reshape(-1)
        if method in {"previous", "nearest"}:
            # Causal point sampling: both names select the nearest sample at or
            # before each target.  This keeps offline conversion consistent
            # with online inference, where future frames are unavailable.
            indices = self.np.searchsorted(source_times, flat, side="right") - 1
        elif method == "next":
            # Supervision-only point sampling: select the first recorded source
            # row at or after the target timestamp. This is intentionally
            # non-causal and must not be used for observation features.
            indices = self.np.searchsorted(source_times, flat, side="left")
        else:
            raise ValueError(f"Point sampling does not support method={method!r}")
        if self.np.any(indices < 0) or self.np.any(indices >= len(source_times)):
            raise ValueError("Resampling target lies outside the source timestamp range")
        return indices.reshape(self.np.asarray(targets).shape)

    def _linear_sample_plan(self, source_times, targets):
        right = self.np.searchsorted(source_times, targets, side="left")
        exact = self.np.zeros_like(right, dtype=bool)
        in_range = right < len(source_times)
        exact[in_range] = self.np.isclose(
            source_times[right[in_range]],
            targets[in_range],
            rtol=0.0,
            atol=1.0e-9,
        )
        left = self.np.where(exact, right, right - 1)
        if self.np.any(left < 0) or self.np.any(right >= len(source_times)):
            raise ValueError("Interpolation target lies outside the source timestamp range")
        denominator = source_times[right] - source_times[left]
        alpha = self.np.zeros_like(targets, dtype=self.np.float64)
        nonzero = denominator > 0.0
        alpha[nonzero] = (
            targets[nonzero] - source_times[left[nonzero]]
        ) / denominator[nonzero]
        return left, right, alpha

    @staticmethod
    def _mapping_sources(mapping, method=None):
        sources = mapping.get("sources")
        if sources is not None:
            return sources
        default_method = method
        if default_method is None:
            default_method = mapping.get("resample") or mapping.get(
                "align",
                "index",
            )
        return [
            {
                "h5_path": h5_path,
                "timestamp_path": mapping.get("timestamp_path"),
                "method": default_method,
                "max_gap_s": mapping.get("max_gap_s"),
                "allow_stale": bool(mapping.get("allow_stale", False)),
            }
            for h5_path in mapping["h5_paths"]
        ]

    def _resample_mapping(self, mapping, targets, cache, method=None):
        values = []
        for source in self._mapping_sources(mapping, method=method):
            dataset = cache["datasets"][source["h5_path"]]
            source_values = self.np.asarray(dataset[:])
            values.append(
                self._resample_values(
                    source_values,
                    cache["timestamp_seconds"][source["timestamp_path"]],
                    targets,
                    source["method"],
                    mapping.get("transform"),
                )
            )
        if len(values) == 1:
            result = values[0]
        elif mapping.get("combine") == "subtract":
            result = self._subtract_values(values, mapping["lerobot_key"])
        else:
            result = self.np.concatenate(
                [value.reshape(*value.shape[:-1], -1) for value in values],
                axis=-1,
            )
        return result.astype(
            self._dtype_from_feature(mapping["feature_spec"]),
            copy=False,
        )

    def _resample_values(
        self,
        source_values,
        source_times,
        targets,
        method,
        transform,
    ):
        target_shape = self.np.asarray(targets).shape
        flat_targets = self.np.asarray(targets, dtype=self.np.float64).reshape(-1)
        if method in {"previous", "nearest", "next"}:
            indices = self._point_sample_indices(source_times, flat_targets, method)
            sampled = source_values[indices]
            if transform is not None:
                sampled = self._apply_transform(sampled, transform)
            return sampled.reshape(target_shape + sampled.shape[1:])

        if method == "pchip":
            try:
                from scipy.interpolate import PchipInterpolator
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "align=pchip requires scipy. Install scipy before conversion."
                ) from exc
            if len(source_times) < 2:
                raise ValueError("PCHIP interpolation requires at least two samples.")
            origin = float(source_times[0])
            # SciPy can overflow reciprocal slopes that are effectively zero;
            # PCHIP handles those branches as zero derivatives.
            with self.np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                interpolator = PchipInterpolator(
                    self.np.asarray(source_times, dtype=self.np.float64) - origin,
                    self.np.asarray(source_values),
                    axis=0,
                    extrapolate=False,
                )
            sampled = interpolator(flat_targets - origin)
            if not self.np.isfinite(sampled).all():
                raise ValueError("PCHIP interpolation produced non-finite values.")
            if transform is not None:
                sampled = self._apply_transform(sampled, transform)
            return sampled.reshape(target_shape + sampled.shape[1:])

        left, right, alpha = self._linear_sample_plan(source_times, flat_targets)
        left_values = source_values[left]
        right_values = source_values[right]
        if method == "pose":
            if transform is not None:
                left_values = self._apply_transform(left_values, transform)
                right_values = self._apply_transform(right_values, transform)
            left_values = self.np.asarray(left_values, dtype=self.np.float64)
            right_values = self.np.asarray(right_values, dtype=self.np.float64)
            if left_values.shape[-1] != 7 or right_values.shape[-1] != 7:
                raise ValueError("Pose resampling expects xyz+xyzw values")
            position = left_values[..., :3] + alpha[:, None] * (
                right_values[..., :3] - left_values[..., :3]
            )
            quaternion = self._slerp_xyzw(
                left_values[..., 3:],
                right_values[..., 3:],
                alpha,
            )
            sampled = self.np.concatenate((position, quaternion), axis=-1)
        else:
            expand = (slice(None),) + (None,) * (left_values.ndim - 1)
            weight = alpha[expand]
            sampled = left_values + weight * (right_values - left_values)
            if transform is not None:
                sampled = self._apply_transform(sampled, transform)
        return sampled.reshape(target_shape + sampled.shape[1:])

    def _slerp_xyzw(self, left, right, alpha):
        left = left / self.np.linalg.norm(left, axis=-1, keepdims=True)
        right = right / self.np.linalg.norm(right, axis=-1, keepdims=True)
        dot = self.np.sum(left * right, axis=-1)
        flip = dot < 0.0
        right = self.np.where(flip[:, None], -right, right)
        dot = self.np.clip(self.np.abs(dot), 0.0, 1.0)
        close = dot > 0.9995
        result = self.np.empty_like(left)
        if self.np.any(close):
            result[close] = left[close] + alpha[close, None] * (
                right[close] - left[close]
            )
        far = ~close
        if self.np.any(far):
            theta = self.np.arccos(dot[far])
            sin_theta = self.np.sin(theta)
            result[far] = (
                self.np.sin((1.0 - alpha[far]) * theta)[:, None]
                / sin_theta[:, None]
                * left[far]
                + self.np.sin(alpha[far] * theta)[:, None]
                / sin_theta[:, None]
                * right[far]
            )
        result /= self.np.linalg.norm(result, axis=-1, keepdims=True)
        return result

    def clear_episode_cache(self, cache) -> None:
        """Release per-episode cached arrays and H5 dataset handles."""

        if not cache:
            return
        cache.get("datasets", {}).clear()
        cache.get("timestamps", {}).clear()
        cache.get("timestamp_seconds", {}).clear()
        cache.get("resampled", {}).clear()
        cache.get("aligned", {}).clear()
        cache.pop("target_timestamps", None)
        cache.pop("high_targets", None)
        cache.pop("action_targets", None)
        cache.pop("anchor_timestamp_ns", None)
        cache.pop("high_timestamp_ns", None)
        cache.pop("action_timestamp_ns", None)
        cache.pop("action_source_timestamp_ns", None)
        cache.pop("selected_master_indices", None)
        cache.clear()

    def episode_length(self, h5_file, master_timestamp_path, h5_path, cache=None):
        if cache is not None and "target_timestamps" in cache:
            return int(cache["target_timestamps"].shape[0])
        master_ts = self._timestamp_array(h5_file, master_timestamp_path, h5_path, cache)
        return int(master_ts.shape[0])

    def uniform_timestamps(self, master_timestamps, master_timestamp_path: str, fps: int):
        if self.np is None:
            raise RuntimeError("numpy is required to generate a uniform timeline.")

        timestamps = self.np.asarray(master_timestamps).reshape(-1)
        if timestamps.size == 0:
            raise ValueError("Cannot generate a uniform timeline from empty master timestamps.")
        if not self.np.isfinite(timestamps).all():
            raise ValueError("Master timestamps must be finite.")
        if timestamps.size > 1 and self.np.any(self.np.diff(timestamps) <= 0):
            raise ValueError("Master timestamps must be strictly increasing.")

        scale = self._timestamp_seconds_scale(master_timestamp_path)
        duration_seconds = float(timestamps[-1] - timestamps[0]) * scale
        frame_count = int(self.np.floor(duration_seconds * fps + 1e-9)) + 1
        offsets = self.np.arange(frame_count, dtype=self.np.float64) / (fps * scale)
        target = float(timestamps[0]) + offsets

        if self.np.issubdtype(timestamps.dtype, self.np.integer):
            target = self.np.rint(target).astype(timestamps.dtype)
        else:
            target = target.astype(timestamps.dtype, copy=False)
        if target.size > 1 and self.np.any(self.np.diff(target) <= 0):
            raise ValueError("Configured fps is too high for the master timestamp resolution.")
        return target

    def fixed_phase_timestamps(
        self,
        master_timestamps,
        master_timestamp_path: str,
        fps: int,
        *,
        phase_offset_s: float = 0.0,
    ):
        """Build an absolute 50 Hz-style grid shared with online inference."""

        timestamps = self.np.asarray(master_timestamps).reshape(-1)
        if timestamps.size == 0 or not self.np.isfinite(timestamps).all():
            raise ValueError("Cannot generate a fixed grid from invalid timestamps.")
        if timestamps.size > 1 and self.np.any(self.np.diff(timestamps) <= 0):
            raise ValueError("Master timestamps must be strictly increasing.")

        scale = self._timestamp_seconds_scale(master_timestamp_path)
        units_per_second = int(round(1.0 / scale))
        if not math.isclose(units_per_second * scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Fixed-phase sampling requires an integral timestamp unit.")
        if units_per_second % fps != 0:
            raise ValueError(
                f"fps={fps} does not divide {units_per_second} timestamp units per second."
            )
        period = units_per_second // fps
        if not self.np.issubdtype(timestamps.dtype, self.np.integer):
            raise ValueError("Fixed-phase sampling requires integer source timestamps.")

        phase_offset_units_float = float(phase_offset_s) * units_per_second
        phase_offset_units = int(round(phase_offset_units_float))
        if not math.isclose(
            phase_offset_units_float,
            phase_offset_units,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "Fixed-phase offset must be exactly representable in the master "
                "timestamp unit."
            )
        if phase_offset_units < 0 or phase_offset_units >= period:
            raise ValueError(
                f"Fixed-phase offset must be in [0, {1.0 / fps:.9f}) seconds."
            )

        first = int(timestamps[0])
        last = int(timestamps[-1])
        first_tick = (
            -(-(first - phase_offset_units) // period) * period
            + phase_offset_units
        )
        last_tick = (
            (last - phase_offset_units) // period * period
            + phase_offset_units
        )
        if last_tick < first_tick:
            raise ValueError("Episode contains no complete fixed-rate sampling tick.")
        return self.np.arange(
            first_tick,
            last_tick + period,
            period,
            dtype=timestamps.dtype,
        )

    def estimate_fps_from_master_timestamps(self, master_timestamp_path: str) -> int:
        if self.np is None:
            raise RuntimeError("numpy is required to estimate fps from timestamps.")

        dt_chunks = []
        for h5_path in self.files():
            with self.open_episode(h5_path) as h5_file:
                timestamps = self.np.asarray(
                    self._timestamp_array(h5_file, master_timestamp_path, h5_path),
                    dtype=self.np.float64,
                ).reshape(-1)

            if timestamps.shape[0] < 2:
                continue

            dt = self.np.diff(timestamps)
            dt = dt[self.np.isfinite(dt) & (dt > 0)]
            if dt.size:
                dt_chunks.append(dt)

        if not dt_chunks:
            raise ValueError(
                f"Cannot estimate fps: no positive timestamp deltas found at "
                f"{master_timestamp_path!r}."
            )

        all_dt = self.np.concatenate(dt_chunks)
        median_dt_seconds = float(self.np.median(all_dt)) * self._timestamp_seconds_scale(
            master_timestamp_path
        )
        if median_dt_seconds <= 0:
            raise ValueError(f"Invalid median timestamp delta: {median_dt_seconds}")

        fps = int(round(1.0 / median_dt_seconds))
        if fps <= 0:
            raise ValueError(f"Invalid estimated fps: {fps}")

        print(
            f"estimated fps={fps} from {master_timestamp_path} "
            f"(median_dt={median_dt_seconds:.9f}s, num_deltas={all_dt.size})"
        )
        return fps

    @staticmethod
    def _timestamp_seconds_scale(timestamp_path: str) -> float:
        name = str(timestamp_path).lower()
        if name.endswith("_us") or "timestamp_us" in name:
            return 1e-6
        if name.endswith("_ms") or "timestamp_ms" in name:
            return 1e-3
        if name.endswith("_ns") or "timestamp_ns" in name:
            return 1e-9
        return 1.0

    def _timestamps_ns(self, timestamps, timestamp_path):
        """Convert timestamps without losing integer microsecond precision."""

        raw = self.np.asarray(timestamps).reshape(-1)
        scale_to_ns = self._timestamp_seconds_scale(timestamp_path) / 1.0e-9
        integral_scale = int(round(scale_to_ns))
        if self.np.issubdtype(raw.dtype, self.np.integer) and math.isclose(
            scale_to_ns, integral_scale, rel_tol=0.0, abs_tol=1.0e-12
        ):
            return raw.astype(self.np.int64) * integral_scale
        return self.np.rint(
            raw.astype(self.np.float64) * scale_to_ns
        ).astype(self.np.int64)

    def read_frame(self, h5_file, frame_idx, mappings, h5_path, master_timestamp_path, cache=None):
        # 从 H5 里读出一帧，返回 LeRobotDataset.add_frame 需要的字典
        if cache is not None and cache.get("camera_rows"):
            return self._read_camera_rows_frame(frame_idx, mappings, cache)
        if cache is not None and cache.get("dual_rate"):
            return self._read_dual_rate_frame(frame_idx, mappings, cache)
        if cache is not None and "target_timestamps" in cache:
            target_t = cache["target_timestamps"][frame_idx]
        else:
            master_ts = self._timestamp_array(h5_file, master_timestamp_path, h5_path, cache)
            target_t = master_ts[frame_idx]

        frame = {}

        for mapping in mappings:
            key = mapping["lerobot_key"]
            frame[key] = self._read_mapped_value(
                h5_file=h5_file,
                mapping=mapping,
                frame_idx=frame_idx,
                target_t=target_t,
                h5_path=h5_path,
                cache=cache,
            )

        return frame

    def _read_camera_rows_frame(self, frame_idx, mappings, cache):
        frame = {}
        for mapping in mappings:
            cached = cache["resampled"][mapping["lerobot_key"]]
            if isinstance(cached, Mapping) and "indices" in cached:
                indices = self.np.asarray(cached["indices"][frame_idx])
                source = self._mapping_sources(mapping)[0]
                dataset = cache["datasets"][source["h5_path"]]
                if indices.shape == ():
                    value = dataset[int(indices)]
                else:
                    value = self.np.stack(
                        [dataset[int(index)] for index in indices.reshape(-1)],
                        axis=0,
                    ).reshape(indices.shape + dataset.shape[1:])
                value = self._apply_transform(value, mapping.get("transform"))
                frame[mapping["lerobot_key"]] = self._to_lerobot_value(value)
                age_key = mapping.get("emit_age_key")
                if age_key is not None:
                    frame[age_key] = self.np.asarray(
                        [cached["age_s"][frame_idx]], dtype=self.np.float32
                    )
            else:
                frame[mapping["lerobot_key"]] = self._to_lerobot_value(
                    cached[frame_idx]
                )

        if cache.get("store_timestamps"):
            frame["timing.anchor_timestamp_ns"] = self.np.asarray(
                [cache["anchor_timestamp_ns"][frame_idx]], dtype=self.np.int64
            )
            frame["timing.high_timestamp_ns"] = cache[
                "high_timestamp_ns"
            ][frame_idx, :, None].copy()
            frame["timing.action_source_timestamp_ns"] = self.np.asarray(
                [cache["action_source_timestamp_ns"][frame_idx]],
                dtype=self.np.int64,
            )
        return frame

    def _read_dual_rate_frame(self, frame_idx, mappings, cache):
        frame = {}
        for mapping in mappings:
            cached = cache["resampled"][mapping["lerobot_key"]]
            if isinstance(cached, Mapping) and "indices" in cached:
                indices = self.np.asarray(cached["indices"][frame_idx])
                source = self._mapping_sources(mapping)[0]
                dataset = cache["datasets"][source["h5_path"]]
                if indices.shape == ():
                    value = dataset[int(indices)]
                else:
                    value = self.np.stack(
                        [dataset[int(index)] for index in indices.reshape(-1)],
                        axis=0,
                    ).reshape(indices.shape + dataset.shape[1:])
                value = self._apply_transform(value, mapping.get("transform"))
                frame[mapping["lerobot_key"]] = self._to_lerobot_value(value)
                age_key = mapping.get("emit_age_key")
                if age_key is not None:
                    frame[age_key] = self.np.asarray(
                        [cached["age_s"][frame_idx]],
                        dtype=self.np.float32,
                    )
            else:
                frame[mapping["lerobot_key"]] = self._to_lerobot_value(
                    cached[frame_idx]
                )

        if cache.get("store_timestamps"):
            frame["timing.anchor_timestamp_ns"] = self.np.asarray(
                [cache["anchor_timestamp_ns"][frame_idx]],
                dtype=self.np.int64,
            )
            frame["timing.high_timestamp_ns"] = cache[
                "high_timestamp_ns"
            ][frame_idx, :, None].copy()
            frame["timing.action_timestamp_ns"] = cache[
                "action_timestamp_ns"
            ][frame_idx, :, None].copy()
        return frame


    def _nearest_past_window_indices(self, timestamps, target_t, window_size):
        history_idx = self._history_index_from_timestamps(timestamps, target_t)
        indices = self.np.arange(history_idx - window_size + 1, history_idx + 1)
        return self.np.clip(indices, 0, len(timestamps) - 1)

    def _nearest_future_window_indices(self, timestamps, target_t, window_size):
        anchor_idx = self._history_index_from_timestamps(timestamps, target_t)
        indices = self.np.arange(anchor_idx, anchor_idx + window_size)
        return self.np.clip(indices, 0, len(timestamps) - 1)

    def _read_nearest_past_window(self, h5_file, h5_field_path, timestamp_path, target_t, h5_path, window_size, cache=None):
        timestamps = self._timestamp_array(h5_file, timestamp_path, h5_path, cache)
        values = self._dataset_cached(h5_file, h5_field_path, h5_path, cache)

        indices = self._nearest_past_window_indices(timestamps, target_t, window_size)
        # h5py fancy indexing requires strictly increasing indices, but left-padding
        # intentionally creates duplicates such as [0, 0, 0, 0]. Read one by one.
        window = [values[int(index)] for index in indices]
        return self.np.asarray(window).astype("float32")

    def _read_nearest_future_window(self, h5_file, h5_field_path, timestamp_path, target_t, h5_path, window_size, cache=None):
        timestamps = self._timestamp_array(h5_file, timestamp_path, h5_path, cache)
        values = self._dataset_cached(h5_file, h5_field_path, h5_path, cache)

        indices = self._nearest_future_window_indices(timestamps, target_t, window_size)
        # Right-padding intentionally creates duplicate indices near the episode end.
        window = [values[int(index)] for index in indices]
        return self.np.asarray(window).astype("float32")




    def _read_mapped_value(self, h5_file, mapping, frame_idx, target_t, h5_path, cache=None):
        align = mapping.get("align", "index")

        lerobot_key = mapping.get("lerobot_key")
        if (
            cache is not None
            and lerobot_key is not None
            and lerobot_key in cache.get("aligned", {})
        ):
            return cache["aligned"][lerobot_key][frame_idx]

        if align == "index":
            value = self._read_paths_at_index(
                h5_file,
                mapping["h5_paths"],
                frame_idx,
                h5_path,
                mapping["feature_spec"],
                cache,
                combine=mapping.get("combine"),
                feature_name=mapping.get("lerobot_key"),
            )
        elif align == "nearest":
            source_idx = self._nearest_index(
                h5_file, mapping["timestamp_path"], target_t, h5_path, cache
            )
            value = self._read_paths_at_index(
                h5_file,
                mapping["h5_paths"],
                source_idx,
                h5_path,
                mapping["feature_spec"],
                cache,
                combine=mapping.get("combine"),
                feature_name=mapping.get("lerobot_key"),
            )
        elif align in ("linear", "pchip", "previous"):
            if cache is None or mapping["lerobot_key"] not in cache.get("aligned", {}):
                raise RuntimeError(
                    f"align={align} requires a precomputed episode cache for "
                    f"{mapping['lerobot_key']!r}."
                )
            return cache["aligned"][mapping["lerobot_key"]][frame_idx]
        elif align == "nearest_past_window":
            value = self._read_nearest_past_window(
                h5_file=h5_file,
                h5_field_path=mapping["h5_paths"][0],
                timestamp_path=mapping["timestamp_path"],
                target_t=target_t,
                window_size=mapping["window_size"],
                h5_path=h5_path,
                cache=cache,
            )
        elif align == "nearest_future_window":
            value = self._read_nearest_future_window(
                h5_file=h5_file,
                h5_field_path=mapping["h5_paths"][0],
                timestamp_path=mapping["timestamp_path"],
                target_t=target_t,
                window_size=mapping["window_size"],
                h5_path=h5_path,
                cache=cache,
            )
        else:
            raise ValueError(f"Unknown align mode: {align}")

        return self._apply_transform(value, mapping.get("transform"))

    def _apply_transform(self, value, transform):
        if transform is None:
            return value
        if transform == "ee_pose_matrix_to_quaternion":
            return self._ee_pose_matrix_to_quaternion(value)
        raise ValueError(f"Unknown transform: {transform!r}")

    def _ee_pose_matrix_to_quaternion(self, value):
        """Convert (..., 4, 4) poses to (..., 7) xyz + quaternion in xyzw order."""
        if self.np is None:
            raise RuntimeError("numpy is required to convert ee_pose matrices.")

        poses = self.np.asarray(value, dtype=self.np.float64)
        if poses.shape[-2:] != (4, 4):
            raise ValueError(
                "ee_pose_matrix_to_quaternion expects shape (..., 4, 4), "
                f"got {poses.shape}"
            )

        flat_poses = poses.reshape(-1, 4, 4)
        converted = self.np.empty((flat_poses.shape[0], 7), dtype=self.np.float32)
        for index, pose in enumerate(flat_poses):
            converted[index, :3] = pose[:3, 3]
            converted[index, 3:] = self._rotation_matrix_to_quaternion_xyzw(pose[:3, :3])
        return converted.reshape(poses.shape[:-2] + (7,))

    def _rotation_matrix_to_quaternion_xyzw(self, matrix):
        matrix = self.np.asarray(matrix, dtype=self.np.float64)
        trace = self.np.trace(matrix)
        if trace > 0:
            scale = self.np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = self.np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = self.np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = self.np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

        quaternion = self.np.asarray([qx, qy, qz, qw], dtype=self.np.float64)
        norm = self.np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("rotation matrix produced a zero-norm quaternion")
        quaternion /= norm
        if quaternion[3] < 0:
            quaternion = -quaternion
        return quaternion.astype(self.np.float32)
    

    def _nearest_index(self, h5_file, timestamp_path, target_t, h5_path, cache=None):
        timestamps = self._timestamp_array(h5_file, timestamp_path, h5_path, cache)
        return self._history_index_from_timestamps(timestamps, target_t)

    def _history_index_from_timestamps(self, timestamps, target_t):
        timestamps = self.np.asarray(timestamps).reshape(-1)
        if timestamps.size == 0:
            raise ValueError("Cannot align against an empty timestamp array.")
        if timestamps.size > 1 and self.np.any(self.np.diff(timestamps) < 0):
            raise ValueError("Alignment timestamps must be sorted in ascending order.")
        history_idx = int(self.np.searchsorted(timestamps, target_t, side="right") - 1)
        if history_idx < 0:
            raise ValueError(
                f"No historical sample exists at or before target timestamp {target_t!r}; "
                f"first source timestamp is {timestamps[0]!r}."
            )
        return history_idx

    def _read_paths_at_index(
        self,
        h5_file,
        h5_paths,
        source_idx,
        h5_path,
        feature_spec,
        cache=None,
        *,
        combine=None,
        feature_name=None,
    ):
        values = [
            self._read_value(
                h5_file=h5_file,
                h5_field_path=h5_field_path,
                frame_idx=source_idx,
                h5_path=h5_path,
                cache=cache,
            )
            for h5_field_path in h5_paths]
    
        if len(values) == 1:
            return self._coerce_single_value(values[0], feature_spec)

        if combine == "subtract":
            value = self._subtract_values(values, feature_name)
            return self._coerce_single_value(value, feature_spec)
    
        return self._concat_values(values)

    def _subtract_values(self, values, feature_name=None):
        if self.np is None:
            raise RuntimeError("combine='subtract' requires numpy.")
        if len(values) != 2:
            raise ValueError("combine='subtract' requires exactly two values.")

        minuend = self.np.asarray(values[0])
        subtrahend = self.np.asarray(values[1])
        if minuend.shape != subtrahend.shape:
            label = f" for {feature_name!r}" if feature_name else ""
            raise ValueError(
                f"combine='subtract'{label} requires matching shapes, got "
                f"{minuend.shape} and {subtrahend.shape}."
            )
        return minuend - subtrahend


    def _read_value(self, h5_file: Any, h5_field_path: str, frame_idx: int, h5_path: Path, cache=None) -> Any:
        # 从一个 H5 dataset 里取一个值。
        # 标量 dataset：直接取 dataset[()]；
        # 时序 dataset：取 dataset[frame_idx]。
        # 后面如果需要对图片转 RGB、换通道、拼 state、裁剪 action，
        # 可以从这里拆出 `_read_image_value()` / `_read_state_value()

        dataset = self._dataset_cached(h5_file, h5_field_path, h5_path, cache)
        if len(dataset.shape) == 0:
            value = dataset[()]
        else:
            value = dataset[frame_idx]
        return self._to_lerobot_value(value)

    def _coerce_single_value(self, value, feature_spec):
        if self.np is None:
            return value

        if not isinstance(feature_spec, Mapping):
            return value

        dtype_name = feature_spec.get("dtype")
        if dtype_name in ("image", "video"):
            return value

        array = self.np.asarray(value)
        shape = feature_spec.get("shape")
        if (shape == [1] or shape == (1,)) and array.shape == ():
            array = array.reshape(1)

        return array.astype(self._dtype_from_feature(feature_spec), copy=False)

    def _concat_values(self, values: list[Any]) -> Any:
        """把多个低维 H5 字段聚合成一个一维向量。

        例子：
            teleop/q_follower      shape=(7,)
            teleop/gripper_state   shape=()
            -> observation.state   shape=(8,)

        注意：这个函数默认把多维数组 flatten 后再拼接，所以只建议用于
        state/action 这类低维字段，不建议用于图片。
        """

        if self.np is None:
            raise RuntimeError("Aggregating h5_paths requires numpy.")

        arrays = []
        for value in values:
            array = self.np.asarray(value)
            if array.shape == ():
                array = array.reshape(1)
            elif array.ndim > 1:
                array = array.reshape(-1)
            arrays.append(array)

        return self.np.concatenate(arrays, axis=0).astype("float32")

    def _dtype_from_feature(self, feature_spec):
        dtype_name = feature_spec.get("dtype", feature_spec.get("type", "float32"))
        try:
            return self.np.dtype(dtype_name)
        except TypeError:
            return self.np.float32

    def _dataset(self, h5_file, h5_field_path, h5_path) -> Any:
        if h5_field_path not in h5_file:
            raise KeyError(f"{h5_field_path!r} not found in {h5_path}")

        dataset = h5_file[h5_field_path]
        if not isinstance(dataset, self.h5py.Dataset):
            raise TypeError(f"{h5_field_path!r} is not a dataset in {h5_path}")
        return dataset

    def _dataset_cached(self, h5_file, h5_field_path, h5_path, cache=None):
        if cache is not None and h5_field_path in cache["datasets"]:
            return cache["datasets"][h5_field_path]
        return self._dataset(h5_file, h5_field_path, h5_path)

    def _timestamp_array(self, h5_file, timestamp_path, h5_path, cache=None):
        if cache is not None and timestamp_path in cache["timestamps"]:
            return cache["timestamps"][timestamp_path]
        return self._dataset(h5_file, timestamp_path, h5_path)[:]

    def _to_lerobot_value(self, value: Any) -> Any:
        if self.np is None:
            return value

        value = self.np.asarray(value)
        if value.shape == ():
            return value.item()
        return value

    def _print_node(self, node: Any, prefix: str = "") -> None:
        for key in sorted(node.keys()):
            item = node[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, self.h5py.Dataset):
                print(f"{path}: dataset shape={item.shape} dtype={item.dtype}{self._format_attrs(item.attrs)}")
            elif isinstance(item, self.h5py.Group):
                print(f"{path}/: group{self._format_attrs(item.attrs)}")
                self._print_node(item, path)

    @staticmethod
    def _format_attrs(attrs: Any) -> str:
        if len(attrs) == 0:
            return ""
        parts = [f"{key}={attrs[key]!r}" for key in sorted(attrs.keys())]
        return " attrs={" + ", ".join(parts) + "}"



class LeRobotV3Dataset:
    def __init__(
        self,
        LeRobotDataset,
        *,
        repo_id: str,
        root: Path,
        fps: int,
        features: Mapping[str, Any],
        no_videos: bool,
    ) -> None:
        create_kwargs = {
            "repo_id": repo_id,
            "root": root,
            "fps": fps,
            "features": dict(features),
            "use_videos": not no_videos,
            "video": not no_videos,
        }
        supported_kwargs = filter_supported_kwargs(LeRobotDataset.create, create_kwargs)
        self.dataset = LeRobotDataset.create(**supported_kwargs)

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        frame = dict(frame)
        frame.setdefault("task", task)
        kwargs = {"frame": frame, "task": task}
        try:
            self.dataset.add_frame(**filter_supported_kwargs(self.dataset.add_frame, kwargs))
        except TypeError:
            self.dataset.add_frame(frame)

    def save_episode(self, task: str) -> None:
        kwargs = {"task": task}
        try:
            self.dataset.save_episode(**filter_supported_kwargs(self.dataset.save_episode, kwargs))
        except TypeError:
            self.dataset.save_episode()

    def finalize(self) -> None:
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            finalize()

    def push_to_hub(self) -> None:
        if not hasattr(self.dataset, "push_to_hub"):
            raise AttributeError("Installed LeRobotDataset has no push_to_hub().")
        self.dataset.push_to_hub()


def filter_supported_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """兼容不同 LeRobot 版本：只传当前函数支持的参数。"""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}



# 主流程：H5Dataset 读，LeRobotV3Dataset 写

def run_inspect(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    input_path = config_path(
        config,
        "input",
        override=getattr(args, "input", None),
    )
    h5py = load_h5py()
    h5_dataset = H5Dataset(
        input_path,
        h5py=h5py,
        max_episodes=config_int(config, "max_episodes"),
    )
    h5_dataset.inspect()


def resolve_raw_index_fps(config: Mapping[str, Any], h5_dataset: H5Dataset) -> dict[str, Any]:
    """Infer only the nominal LeRobot fps; raw-index rows remain untouched."""

    if config.get("fps") is not None:
        return dict(config)
    sampling = config.get("sampling")
    mode = (
        str(sampling.get("mode", "")).strip().lower()
        if isinstance(sampling, Mapping)
        else ""
    )
    if mode != "raw_index":
        return dict(config)
    master_timestamp_path = config.get("master_timestamp_path")
    if not isinstance(master_timestamp_path, str) or not master_timestamp_path:
        raise ValueError("shape_meta must define master_timestamp_path.")
    resolved = dict(config)
    resolved["fps"] = h5_dataset.estimate_fps_from_master_timestamps(
        master_timestamp_path
    )
    return resolved

def run_conversion(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    h5py, np, LeRobotDataset = load_conversion_deps()

    h5_dataset = H5Dataset(
        config_path(
            config,
            "input",
            override=getattr(args, "input", None),
        ),
        h5py=h5py,
        np=np,
        max_episodes=config_int(config, "max_episodes"),
    )
    config = resolve_raw_index_fps(config, h5_dataset)
    spec = build_conversion_spec(config)
    fps = spec["fps"]
    lerobot_dataset = LeRobotV3Dataset(
        LeRobotDataset,
        repo_id=config_str(config, "repo_id", "local/h5_to_lerobot_v3"),
        root=config_path(
            config,
            "output",
            override=getattr(args, "output", None),
        ),
        fps=fps,
        features=spec["lerobot_features"],
        no_videos=config_bool(config, "no_videos"),
    )

    h5_files = h5_dataset.files()
    episode_iter = tqdm(h5_files, desc="episodes", unit="episode")
    try:
        for h5_path in episode_iter:
            episode_iter.set_postfix_str(h5_path.name)
            with h5_dataset.open_episode(h5_path) as h5_file:
                cache = None
                try:
                    cache = h5_dataset.build_episode_cache(
                        h5_file,
                        spec["mappings"],
                        spec["master_timestamp_path"],
                        spec["fps"],
                        h5_path,
                        timeline=spec["timeline"],
                        sampling=spec["sampling"],
                    )
                    episode_length = h5_dataset.episode_length(
                        h5_file,
                        spec["master_timestamp_path"],
                        h5_path,
                        cache,
                    )
                    frame_iter = tqdm(
                        range(episode_length),
                        desc=f"frames {h5_path.name}",
                        unit="frame",
                        leave=False,
                    )
                    for frame_idx in frame_iter:
                        frame = h5_dataset.read_frame(
                            h5_file,
                            frame_idx,
                            spec["mappings"],
                            h5_path,
                            spec["master_timestamp_path"],
                            cache,
                        )
                        lerobot_dataset.add_frame(frame, task=spec["task"])
                finally:
                    h5_dataset.clear_episode_cache(cache)
            lerobot_dataset.save_episode(task=spec["task"])
    finally:
        lerobot_dataset.finalize()

    if config_bool(config, "push_to_hub"):
        lerobot_dataset.push_to_hub()


def main() -> None:
    args = parse_args()

    if args.print_example_shape_meta:
        print(json.dumps(EXAMPLE_SHAPE_META, indent=2, ensure_ascii=False))
        return

    if args.inspect_only:
        run_inspect(args)
        return

    run_conversion(args)


if __name__ == "__main__":
    main()
