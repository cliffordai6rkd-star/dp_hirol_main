import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from diffusion_policy.common.teacher_cache_lifecycle import TeacherCacheLifecycle
from diffusion_policy.dataset.render_net_teacher_cache import TeacherTargetCache
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_ddp import TinyDataset, TinyPreprocessor, TinyRenderer, TinyTeacher


@pytest.mark.parametrize("exit_mode", ["complete", "cache_only", "build_error", "train_error"])
def test_workspace_cleans_only_its_teacher_cache_on_completion_and_errors(monkeypatch, tmp_path, exit_mode):
    source = tmp_path / "data"
    source.mkdir()
    (source / "keep.json").write_text("source data")
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"teacher")
    cache_parent = tmp_path / "cache"
    unrelated = cache_parent / "another-teacher"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.npy").write_bytes(b"another run's targets")
    teacher_cfg = OmegaConf.create({"policy": {"test": True}, "task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *a, **kw: TinyTeacher())
    monkeypatch.setattr(distill, "build_render_net", lambda *a: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda policy: TinyPreprocessor())
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: TinyDataset(source))
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         "diffusion_policy/config/train_distill_render_net.yaml")
    del cfg["hydra"]
    cfg.distillation.objective = "token_mse" # Cache lifecycle uses the legacy obs-only test policy.
    assert cfg.teacher_cache.cleanup_on_exit is True
    # Old saved configs must get the new default as well.
    del cfg.teacher_cache.cleanup_on_exit
    cfg.teacher_checkpoint = str(checkpoint)
    cfg.teacher_cache.directory = str(cache_parent)
    cfg.teacher_cache.batch_size = 2
    cfg.teacher_cache.num_workers = 1
    cfg.teacher_cache.only = exit_mode == "cache_only"
    cfg.training.device = "cpu"
    cfg.training.amp.enabled = False
    cfg.training.compile.enabled = False
    cfg.training.num_epochs = 1
    cfg.render_net.architecture_version = 1
    cfg.render_net.high_attention_render.enabled = False
    cfg.logging.progress_bar = False
    cfg.logging.plot_every_steps = 0
    for loader in (cfg.dataloader, cfg.val_dataloader):
        loader.batch_size = 4
        loader.num_workers = 1
        loader.persistent_workers = True
        loader.pin_memory = False
    ws = distill.TrainDistillRenderNetWorkspace(cfg, str(tmp_path / "run"))
    if exit_mode == "build_error":
        original_append = TeacherTargetCache.append

        def fail_after_write(self, *args):
            original_append(self, *args)
            raise RuntimeError("interrupted cache build")

        monkeypatch.setattr(TeacherTargetCache, "append", fail_after_write)
    if exit_mode == "train_error":
        def fail_training(batch):
            raise RuntimeError("training failed")

        monkeypatch.setattr(ws, "_batch_loss", fail_training)
    if exit_mode.endswith("error"):
        with pytest.raises(RuntimeError, match="interrupted cache build|training failed"):
            ws.run()
    else:
        ws.run()
    assert ws.teacher_cache_paths
    for path in ws.teacher_cache_paths.values():
        assert not Path(path).parent.exists()
    assert (unrelated / "keep.npy").read_bytes() == b"another run's targets"
    assert (source / "keep.json").read_text() == "source data"
    assert checkpoint.read_bytes() == b"teacher"
    assert (tmp_path / "run/config.yaml").is_file()
    if exit_mode == "complete":
        assert (tmp_path / "run/checkpoints/latest.ckpt").is_file()


def test_cleanup_waits_for_other_readers_and_closes_memory_maps(tmp_path):
    root = tmp_path / "cache"
    first, second = TeacherCacheLifecycle(), TeacherCacheLifecycle()
    first.acquire(root)
    first.acquire(root)  # Re-registering a cache must not leak a shared lease.
    cache = TeacherTargetCache(root / "train", {}, 2)
    cache.append(torch.ones(2, 3, 4), {})
    first.track_cache(cache)
    mappings = [array._mmap for array in cache._arrays.values()]
    second.acquire(root)
    other_cache = TeacherTargetCache(root / "train", {}, 2)
    second.track_cache(other_cache)
    first.close()
    assert root.exists() and all(mapping.closed for mapping in mappings)
    torch.testing.assert_close(other_cache.read(0)["tokens"], torch.ones(3, 4))
    second.close()
    second.close()  # Exception unwinding/exit cleanup may run more than once.
    assert not root.exists()


def test_opt_out_keeps_cache_but_stops_persistent_workers_and_restores_handlers(tmp_path):
    root = tmp_path / "cache"
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    with TeacherCacheLifecycle(cleanup_on_exit=False) as lifecycle:
        lifecycle.acquire(root)
        cache = TeacherTargetCache(root / "train", {}, 2)
        cache.append(torch.ones(2, 3, 4), {})
        lifecycle.track_cache(cache)
        loader = DataLoader(TensorDataset(torch.arange(4)), num_workers=1, persistent_workers=True)
        iterator = lifecycle.track_iterator(iter(loader))
        workers = list(iterator._workers)
        next(iterator)
        assert all(worker.is_alive() for worker in workers)
    assert root.exists() and all(not worker.is_alive() for worker in workers)
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_stop_signals_remove_partial_cache_in_a_real_process(tmp_path, stop_signal):
    root = tmp_path / "cache"
    script = """
import signal
import sys
from pathlib import Path
from diffusion_policy.common.teacher_cache_lifecycle import TeacherCacheLifecycle

signal.signal(signal.SIGHUP, signal.SIG_DFL)
with TeacherCacheLifecycle() as lifecycle:
    root = Path(sys.argv[1])
    lifecycle.acquire(root)
    root.mkdir()
    (root / 'partial.npy').write_bytes(b'partial teacher targets')
    print('ready', flush=True)
    while True:
        signal.pause()
"""
    proc = subprocess.Popen([sys.executable, "-c", script, str(root)],
                            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(proc.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=15), "Cache subprocess did not become ready"
        assert proc.stdout.readline().strip() == "ready"
        os.kill(proc.pid, stop_signal)
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 128 + stop_signal, (stdout, stderr)
        assert not root.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=15)
