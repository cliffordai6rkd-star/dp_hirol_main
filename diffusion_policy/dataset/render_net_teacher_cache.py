"""Resumable teacher targets, indexed exactly like the source dataset."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_VERSION = 1


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def dataset_source_signature(dataset_path):
    """Fingerprint source file names, sizes and modification times, without decoding."""
    root = Path(dataset_path).resolve()
    digest = hashlib.sha256()
    for current, directories, files in os.walk(root):
        directories.sort()
        for name in sorted(files):
            path = Path(current, name)
            stat = path.stat()
            digest.update(json.dumps([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns]).encode())
    return {"path": str(root), "files_signature": digest.hexdigest()}


class TeacherTargetCache:
    """Append an ordered prefix; commit its progress only after arrays are flushed.

    A completed cache is read through memory maps. Rows are copied on access so
    DataLoader workers never mutate the shared on-disk supervision.
    """

    def __init__(self, directory, identity, length, *, token_dtype="float32"):
        self.directory = Path(directory)
        self.identity = identity
        self.length = int(length)
        if token_dtype not in {"float16", "float32"}:
            raise ValueError("Teacher cache token_dtype must be float16 or float32")
        self.token_dtype = np.dtype(token_dtype)
        self.metadata = None
        self._arrays = None
        self._pid = None
        self.in_ram = False
        metadata_path = self.directory / "metadata.json"
        if metadata_path.is_file():
            self.metadata = json.loads(metadata_path.read_text())
            if self.metadata["identity"] != identity or self.metadata["length"] != self.length:
                raise ValueError(f"Teacher cache identity/sample order mismatch: {self.directory}")
            if self._array_dtype("tokens") != self.token_dtype:
                raise ValueError(f"Teacher cache token dtype mismatch: {self.directory}")
            self._open_arrays("r")

    @property
    def completed(self):
        return 0 if self.metadata is None else int(self.metadata["completed_samples"])

    @property
    def complete(self):
        return self.metadata is not None and self.completed == self.length

    @property
    def nbytes(self):
        return sum(self.length * int(np.prod(spec["shape"])) * self._array_dtype(key).itemsize
                   for key, spec in self.metadata["arrays"].items())

    def _array_dtype(self, key):
        # Version 1 float32 caches have only a top-level dtype field.
        return np.dtype(self.metadata["arrays"][key].get("dtype", self.metadata["dtype"]))

    def load_into_ram(self):
        if not self.complete:
            raise RuntimeError("Cannot load an incomplete teacher cache into RAM")
        if self.in_ram:
            return
        self._open_arrays("r")
        self._arrays = {key: np.array(values, copy=True) for key, values in self._arrays.items()}
        for values in self._arrays.values():
            values.setflags(write=False)
        self.in_ram = True

    def _open_arrays(self, mode):
        self.close()
        self._arrays = {}
        for key, spec in self.metadata["arrays"].items():
            values = np.load(self.directory / spec["file"], mmap_mode=mode, allow_pickle=False)
            if tuple(values.shape) != (self.length, *spec["shape"]) or values.dtype != self._array_dtype(key):
                raise ValueError(f"Invalid teacher cache array: {self.directory / spec['file']}")
            self._arrays[key] = values
        self._pid = os.getpid()

    def close(self):
        """Release memory maps before deleting the backing teacher targets."""
        for values in (self._arrays or {}).values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._arrays = None
        self._pid = None
        self.in_ram = False

    def append(self, tokens, patch_attention):
        dtype = torch.float16 if self.token_dtype == np.float16 else torch.float32
        values = {"tokens": tokens.detach().to(device="cpu", dtype=dtype).numpy()}
        values.update({f"attention:{key}": value.detach().float().cpu().numpy()
                       for key, value in patch_attention.items()})
        size = len(values["tokens"])
        start, end = self.completed, self.completed + size
        if size < 1 or end > self.length or any(len(value) != size for value in values.values()):
            raise ValueError("Teacher cache append does not match its remaining sample range")
        # Check after quantization too: a finite float32 target can overflow FP16.
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Non-finite teacher targets after cache conversion: {key}")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.metadata is None:
            arrays = {key: {"file": f"array_{index}.npy", "shape": list(value.shape[1:]),
                            "dtype": str(value.dtype)}
                      for index, (key, value) in enumerate(values.items())}
            self.metadata = {"identity": self.identity, "length": self.length, "dtype": str(self.token_dtype),
                             "arrays": arrays, "completed_samples": 0}
            self._arrays = {key: np.lib.format.open_memmap(
                self.directory / spec["file"], mode="w+", dtype=self._array_dtype(key),
                shape=(self.length, *spec["shape"]),
            ) for key, spec in arrays.items()}
        else:
            self._open_arrays("r+")
        if set(values) != set(self._arrays):
            raise ValueError("Teacher cache attention camera keys changed")
        for key, value in values.items():
            if list(value.shape[1:]) != self.metadata["arrays"][key]["shape"]:
                raise ValueError(f"Teacher cache target shape changed: {key}")
            self._arrays[key][start:end] = value
            self._arrays[key].flush()
        self.metadata["completed_samples"] = end
        temporary = self.directory / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temporary.replace(self.directory / "metadata.json")

    def read(self, index):
        if not self.complete:
            raise RuntimeError("All teacher targets must be cached before training starts")
        if self._arrays is None or (not self.in_ram and self._pid != os.getpid()):
            self._open_arrays("r")
        tokens = torch.from_numpy(np.array(self._arrays["tokens"][index], copy=True))
        attention = {key.removeprefix("attention:"): torch.from_numpy(np.array(value[index], copy=True))
                     for key, value in self._arrays.items() if key.startswith("attention:")}
        return {"tokens": tokens, "patch_attention": attention}

    def __getstate__(self):
        state = dict(self.__dict__)
        if not self.in_ram:
            state["_arrays"] = None
            state["_pid"] = None
        return state


class CachedTeacherDataset(Dataset):
    def __init__(self, dataset, cache):
        if len(dataset) != cache.length or not cache.complete:
            raise ValueError("Teacher cache must cover the complete source split")
        self.dataset = dataset
        self.cache = cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "teacher_targets": self.cache.read(index)}
