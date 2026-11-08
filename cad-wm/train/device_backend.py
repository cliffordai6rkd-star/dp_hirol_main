"""Small device backend helpers for CUDA, CPU, and Ascend NPU."""

from contextlib import nullcontext
import importlib
import torch


def is_npu_device(device) -> bool:
    return str(device).split(":", 1)[0].lower() == "npu"


def ensure_npu():
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "train.device=npu:* requires the Ascend torch_npu package. "
            "Install the torch_npu build matching your PyTorch and CANN versions."
        ) from exc
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch_npu was imported but torch.npu is unavailable")


def device_count(device_type: str) -> int:
    if device_type == "cuda":
        return torch.cuda.device_count()
    if device_type == "npu":
        ensure_npu()
        return int(torch.npu.device_count())
    return 0


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        ensure_npu()
        torch.npu.synchronize(device)


def manual_seed_all(seed: int, device_type: str):
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif device_type == "npu":
        ensure_npu()
        seed_fn = getattr(torch.npu, "manual_seed_all", None)
        if seed_fn is not None:
            seed_fn(seed)
        else:
            torch.npu.manual_seed(seed)


def get_rng_state_all(device_type: str):
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_rng_state_all()
    if device_type == "npu":
        ensure_npu()
        getter = getattr(torch.npu, "get_rng_state_all", None)
        return getter() if getter is not None else torch.npu.get_rng_state()
    return None


def set_rng_state_all(state, device_type: str):
    if state is None:
        return
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state)
    elif device_type == "npu":
        ensure_npu()
        setter = getattr(torch.npu, "set_rng_state_all", None)
        if setter is not None:
            setter(state)
        else:
            torch.npu.set_rng_state(state)


def autocast(device, enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def grad_scaler(device, enabled):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device.type)
    except (AttributeError, TypeError):
        if device.type == "npu":
            ensure_npu()
            return torch.npu.amp.GradScaler()
        return torch.cuda.amp.GradScaler()
