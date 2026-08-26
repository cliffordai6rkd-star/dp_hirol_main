"""Optional TorchCodec CUDA frame access for LeRobot video datasets.

TorchCodec is intentionally imported lazily.  The project still runs with the
normal LeRobot decoder when the installed TorchCodec wheel has no CUDA support
or when the LeRobot metadata does not expose video paths.
"""

from __future__ import annotations

import inspect
import logging
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch


log = logging.getLogger(__name__)


def _as_path(value, root: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


class TorchCodecCudaFrameDecoder:
    """Decode LeRobot frames with a CUDA TorchCodec decoder.

    LeRobot metadata changed slightly between 0.3 and 0.4.  Path resolution is
    therefore deliberately introspective and fails closed: if paths cannot be
    resolved, callers can fall back to ``LeRobotDataset.__getitem__``.
    """

    def __init__(
        self,
        *,
        lerobot_dataset,
        episode_ranges: Sequence[range],
        video_keys: Sequence[str],
        device: str = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        try:
            from torchcodec.decoders import VideoDecoder
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("torchcodec.decoders.VideoDecoder is unavailable") from exc

        self._decoder_cls = VideoDecoder
        self._dataset = lerobot_dataset
        self._meta = getattr(lerobot_dataset.dataset, "meta", None)
        self._root = Path(lerobot_dataset.root)
        self._episode_ranges = list(episode_ranges)
        self._episode_stops = [item.stop for item in self._episode_ranges]
        self.device = str(device)
        self._paths: Dict[tuple, Path] = {}
        self._decoders: Dict[tuple, object] = {}
        self._path_first_episode_start: Dict[Path, int] = {}

        for episode_idx in range(len(self._episode_ranges)):
            for video_key in video_keys:
                path = self._resolve_video_path(episode_idx, video_key)
                if path is not None:
                    self._paths[(episode_idx, video_key)] = path
                    self._path_first_episode_start[path] = min(
                        self._path_first_episode_start.get(path, self._episode_ranges[episode_idx].start),
                        self._episode_ranges[episode_idx].start,
                    )

        missing = [
            key
            for key in video_keys
            if not any((ep, key) in self._paths for ep in range(len(self._episode_ranges)))
        ]
        if missing:
            raise RuntimeError(
                "LeRobot metadata does not expose video paths for: "
                + ", ".join(map(str, missing))
            )

        # Probe one decoder now.  This gives a clear, early fallback for the
        # common torchcodec==0.5 CPU-only wheel (Unsupported device: cuda).
        first_key = next(iter(self._paths))
        first_path = self._paths[first_key]
        self._decoders[(first_path, first_key[1])] = self._open_decoder(first_path)

    @property
    def available(self) -> bool:
        return bool(self._paths)

    def _open_decoder(self, path: Path):
        try:
            return self._decoder_cls(str(path), device=self.device)
        except TypeError:
            # A few development snapshots used a keyword-only ``source``.
            return self._decoder_cls(source=str(path), device=self.device)

    def _resolve_video_path(self, episode_idx: int, video_key: str) -> Optional[Path]:
        meta = self._meta
        if meta is None:
            return None

        for method_name in ("get_video_file_path", "get_video_path"):
            method = getattr(meta, method_name, None)
            if not callable(method):
                continue
            # Prefer keyword arguments based on the actual signature.
            try:
                params = inspect.signature(method).parameters
                kwargs = {}
                for name in params:
                    lowered = name.lower()
                    if "episode" in lowered:
                        kwargs[name] = episode_idx
                    elif "video" in lowered or lowered in {"key", "name"}:
                        kwargs[name] = video_key
                if kwargs:
                    path = _as_path(method(**kwargs), self._root)
                    if path is not None:
                        return path
            except Exception:
                pass
            for args in ((episode_idx, video_key), (video_key, episode_idx)):
                try:
                    path = _as_path(method(*args), self._root)
                    if path is not None:
                        return path
                except Exception:
                    continue

        # Older metadata objects exposed a ``video_paths`` mapping directly.
        paths = getattr(meta, "video_paths", None)
        if isinstance(paths, Mapping) and video_key in paths:
            values = paths[video_key]
            if isinstance(values, (str, Path)):
                return _as_path(values, self._root)
            try:
                return _as_path(values[episode_idx], self._root)
            except (IndexError, KeyError, TypeError):
                pass

        # A simple one-file-per-episode layout is common in converted datasets.
        candidates = sorted(
            (self._root / "videos" / video_key).glob("**/*.mp4")
        )
        if len(candidates) == len(self._episode_ranges):
            return candidates[episode_idx]
        return None

    @staticmethod
    def _frame_data(frame) -> np.ndarray:
        """Convert one TorchCodec frame to a CPU NumPy array."""
        data = getattr(frame, "data", frame)
        if isinstance(data, Mapping):
            data = data.get("data", data.get("frame"))
        if torch.is_tensor(data):
            # Keep the host transfer in one place.  ``decode_many`` transfers
            # the whole batch at once, avoiding a CUDA sync for every frame.
            data = data.detach().cpu().numpy()
        data = np.asarray(data)
        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        return data

    @classmethod
    def _frame_data_batch(cls, frames, expected_count: int) -> Sequence[np.ndarray]:
        data = getattr(frames, "data", frames)
        if isinstance(data, Mapping):
            data = data.get("data", data.get("frames", data.get("frame")))
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        data = np.asarray(data)
        if data.ndim == 3 and expected_count == 1:
            data = data[None, ...]
        if data.ndim < 4 or data.shape[0] != expected_count:
            # Some TorchCodec versions return a sequence of frame objects
            # rather than a stacked tensor.
            try:
                source = data if data.ndim == 1 and len(data) == expected_count else frames
                values = [cls._frame_data(item) for item in source]
            except TypeError:
                values = [cls._frame_data(data)]
            if len(values) == expected_count:
                return values
            raise RuntimeError(
                f"TorchCodec returned {data.shape} for {expected_count} requested frames"
            )
        return [np.asarray(data[index]) for index in range(expected_count)]

    def _decode_many(self, decoder, local_indices: Sequence[int]) -> Sequence[np.ndarray]:
        local_indices = [int(index) for index in local_indices]
        if hasattr(decoder, "get_frames_at"):
            try:
                frames = decoder.get_frames_at(local_indices)
            except TypeError:
                frames = decoder.get_frames_at(indices=local_indices)
            return self._frame_data_batch(frames, len(local_indices))
        if len(local_indices) == 1 and hasattr(decoder, "get_frame_at"):
            return [self._frame_data(decoder.get_frame_at(local_indices[0]))]
        return [self._frame_data(decoder[index]) for index in local_indices]

    def _decode_one(self, decoder, local_index: int) -> np.ndarray:
        """Backward-compatible single-frame helper."""
        return self._decode_many(decoder, [local_index])[0]

    def _decoder_for(self, path: Path, video_key: str):
        cache_key = (path, video_key)
        decoder = self._decoders.get(cache_key)
        if decoder is None:
            decoder = self._open_decoder(path)
            self._decoders[cache_key] = decoder
        return decoder

    def decode_many(self, frame_indices: Sequence[int], video_key: str) -> Sequence[np.ndarray]:
        """Decode several global frame indices with one call per video file.

        Requests can straddle episode/video boundaries; indices are grouped by
        decoder and returned in the same order as the input sequence.
        """
        indices = [int(index) for index in frame_indices]
        if not indices:
            return []
        grouped = {}
        for position, frame_idx in enumerate(indices):
            episode_idx = bisect_right(self._episode_stops, frame_idx)
            if episode_idx >= len(self._episode_ranges):
                episode_idx = len(self._episode_ranges) - 1
            path = self._paths.get((episode_idx, video_key))
            if path is None:
                raise KeyError(f"No video path for episode={episode_idx}, key={video_key}")
            local_index = frame_idx - self._path_first_episode_start[path]
            grouped.setdefault((path, video_key), []).append((position, local_index))

        result = [None] * len(indices)
        for (path, key), requests in grouped.items():
            decoder = self._decoder_for(path, key)
            values = self._decode_many(decoder, [item[1] for item in requests])
            for (position, _), value in zip(requests, values):
                result[position] = value
        return result

    def decode(self, frame_idx: int, video_key: str) -> np.ndarray:
        return self.decode_many([frame_idx], video_key)[0]
