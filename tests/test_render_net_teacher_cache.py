import json
import io
import shutil
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache
from diffusion_policy.workspace import train_distill_render_net_workspace as distill


@pytest.mark.parametrize("enabled", [True, False])
def test_tqdm_reports_resumed_samples_even_without_a_terminal(monkeypatch, enabled):
    output = io.StringIO()
    monkeypatch.setattr(distill, "tqdm", partial(tqdm, file=output))
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({"logging": {"progress_bar": enabled, "tqdm_interval_sec": 0}})
    with workspace._progress(None, "Decode + teacher cache [train]", total=10,
                             initial=6, unit="sample") as progress:
        progress.update(2)
    text = output.getvalue()
    if enabled:
        assert "6/10" in text and "8/10" in text and "80%" in text and "sample/s" in text
    else:
        assert text == ""


class SourceDataset:
    image_randomer = None

    def __init__(self, path, ids):
        self.dataset_path = str(path)
        self.indices = np.asarray(ids, dtype=np.int64).reshape(-1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {"obs": {"value": torch.tensor([float(self.indices[index, 0])])}}


def test_cache_resumes_exact_prefix_and_keeps_shuffled_samples_aligned(tmp_path):
    identity = {"checkpoint": "fixed", "indices": "fixed"}
    tokens = torch.arange(5).float()[:, None, None].expand(-1, 6, 8).clone()
    attention = torch.rand(5, 2, 4)
    cache = TeacherTargetCache(tmp_path / "cache", identity, 5)
    cache.append(tokens[:2], {"wrist": attention[:2]})
    with pytest.raises(RuntimeError, match="before training"):
        cache.read(0)
    cache = TeacherTargetCache(tmp_path / "cache", identity, 5)
    assert cache.completed == 2 and not cache.complete
    cache.append(tokens[2:], {"wrist": attention[2:]})
    cache = TeacherTargetCache(tmp_path / "cache", identity, 5)
    dataset = CachedTeacherDataset(SourceDataset(tmp_path, range(5)), cache)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    for batch in loader:
        ids = batch["obs"]["value"][:, 0].long()
        torch.testing.assert_close(batch["teacher_targets"]["tokens"], tokens[ids], rtol=0, atol=0)
        torch.testing.assert_close(batch["teacher_targets"]["patch_attention"]["wrist"], attention[ids], rtol=0, atol=0)
    cache.read(1)["tokens"].zero_()
    torch.testing.assert_close(cache.read(1)["tokens"], tokens[1], rtol=0, atol=0)
    assert cache.metadata["dtype"] == "float32"
    with pytest.raises(ValueError, match="identity/sample order"):
        TeacherTargetCache(tmp_path / "cache", {**identity, "indices": "changed"}, 5)


def test_ram_cache_is_read_only_and_shared_with_fork_workers(tmp_path, monkeypatch):
    cache = TeacherTargetCache(tmp_path / "cache", {"teacher": "fixed"}, 5)
    tokens = torch.arange(5).float()[:, None, None].expand(-1, 6, 8).clone()
    cache.append(tokens, {"wrist": tokens[:, :2, :4]})
    expected_bytes = tokens.numel() * 4 + tokens[:, :2, :4].numel() * 4
    assert cache.nbytes == expected_bytes
    cache.load_into_ram()
    assert cache.in_ram
    assert all(not isinstance(array, np.memmap) and not array.flags.writeable
               for array in cache._arrays.values())

    def forbid_reopen(*args):
        raise AssertionError("Fork workers must use the inherited RAM arrays")

    monkeypatch.setattr(cache, "_open_arrays", forbid_reopen)
    dataset = CachedTeacherDataset(SourceDataset(tmp_path, range(5)), cache)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context="fork")
    for batch in loader:
        ids = batch["obs"]["value"][:, 0].long()
        torch.testing.assert_close(batch["teacher_targets"]["tokens"], tokens[ids], rtol=0, atol=0)
    cache.read(1)["tokens"].zero_()
    torch.testing.assert_close(cache.read(1)["tokens"], tokens[1], rtol=0, atol=0)


def test_fp16_cache_resumes_without_quantizing_attention_and_reduces_bytes(tmp_path):
    tokens = torch.linspace(-3.0, 3.0, 120).reshape(5, 3, 8)
    attention = torch.rand(5, 2, 4)
    directory, identity = tmp_path / "half", {"teacher": "fixed", "storage": "float16"}
    cache = TeacherTargetCache(directory, identity, 5, token_dtype="float16")
    cache.append(tokens[:2], {"wrist": attention[:2]})
    cache = TeacherTargetCache(directory, identity, 5, token_dtype="float16")
    assert cache.completed == 2
    cache.append(tokens[2:], {"wrist": attention[2:]})
    cache = TeacherTargetCache(directory, identity, 5, token_dtype="float16")
    assert cache.nbytes == tokens.numel() * 2 + attention.numel() * 4
    for in_ram in (False, True):
        if in_ram:
            cache.load_into_ram()
        for index in range(5):
            row = cache.read(index)
            assert row["tokens"].dtype == torch.float16
            torch.testing.assert_close(row["tokens"], tokens[index].half(), rtol=0, atol=0)
            torch.testing.assert_close(row["patch_attention"]["wrist"], attention[index], rtol=0, atol=0)
    with pytest.raises(ValueError, match="token dtype mismatch"):
        TeacherTargetCache(directory, identity, 5, token_dtype="float32")


def test_original_float32_cache_metadata_is_still_readable(tmp_path):
    cache = TeacherTargetCache(tmp_path, {"teacher": "fixed"}, 2)
    tokens = torch.randn(2, 3, 8)
    cache.append(tokens, {"wrist": torch.rand(2, 2, 4)})
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for spec in metadata["arrays"].values():
        del spec["dtype"]
    metadata_path.write_text(json.dumps(metadata))
    restored = TeacherTargetCache(tmp_path, {"teacher": "fixed"}, 2)
    assert restored.nbytes == (tokens.numel() + 2 * 2 * 4) * 4
    torch.testing.assert_close(restored.read(0)["tokens"], tokens[0], rtol=0, atol=0)


def test_fp16_cache_rejects_overflow_before_committing_progress(tmp_path):
    cache = TeacherTargetCache(tmp_path, {}, 1, token_dtype="float16")
    with pytest.raises(FloatingPointError, match="after cache conversion"):
        cache.append(torch.full((1, 2, 3), 1e6), {})
    assert cache.completed == 0 and not (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("remove_output_during_cache", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("validation_count", [0, 3])
def test_complete_train_validation_pass_is_reused_when_percentiles_change(
    monkeypatch, tmp_path, remove_output_during_cache, dtype, validation_count,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.json").write_text('{}')
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"teacher weights")
    output = tmp_path / "runs" / "run"
    output.mkdir(parents=True)
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace._output_dir = str(output)
    workspace.cfg = OmegaConf.create({
        "teacher_checkpoint": str(checkpoint),
        "teacher_cache": {"enabled": True, "directory": str(tmp_path / "cache"), "batch_size": 2,
                          "num_workers": 0, "load_into_ram": True, "dtype": dtype},
        "training": {"amp": {"allow_tf32": False}},
    })
    workspace.teacher_cfg = OmegaConf.create({"policy": {"fixed": True}})
    workspace.teacher_state_key = "ema_model"
    workspace.device = torch.device("cpu")
    workspace.amp_enabled = False
    workspace.amp_dtype = torch.bfloat16
    workspace.attention_enabled = True
    workspace.attention_config = {"percentiles": [10, 90], "teacher_seed": 42}
    workspace.teacher_policy = SimpleNamespace()
    workspace.global_step = 0
    calls = []

    def targets(teacher, obs, **kwargs):
        assert kwargs["return_patch_attention"]
        if remove_output_during_cache and not calls:
            shutil.rmtree(output.parent)
        calls.append(obs["value"].clone())
        return obs["value"][:, None, :].expand(-1, 6, 8), {"wrist": obs["value"][:, None, :].expand(-1, 2, 4)}

    monkeypatch.setattr(distill, "teacher_attention_targets", targets)
    train = SourceDataset(source, range(5))
    val = SourceDataset(source, range(20, 20 + validation_count))
    config = OmegaConf.create({"dataset_path": str(source)})
    cached_train, cached_val = workspace._prepare_cached_datasets(train, val, config)
    expected_calls = 3 + (validation_count + 1) // 2
    assert len(calls) == expected_calls and workspace.global_step == 0
    for dataset in (cached_train, cached_val):
        if len(dataset) == 0:
            assert dataset is val and "validation" not in workspace.teacher_cache_paths
            continue
        assert dataset.cache.in_ram
        for sample in dataset:
            assert (sample["teacher_targets"]["tokens"] == sample["obs"]["value"].item()).all()
    previous_paths = dict(workspace.teacher_cache_paths)
    if remove_output_during_cache:
        shutil.rmtree(output.parent)
    workspace.attention_config["percentiles"] = [60, 99]
    workspace._prepare_cached_datasets(train, val, config)
    assert len(calls) == expected_calls and workspace.teacher_cache_paths == previous_paths
    assert json.loads((output / "teacher_cache.json").read_text())["status"] == "complete"
    workspace.cfg.teacher_cache.dtype = "float16" if dtype == "float32" else "float32"
    workspace._prepare_cached_datasets(train, val, config)
    assert len(calls) == 2 * expected_calls and workspace.teacher_cache_paths != previous_paths
