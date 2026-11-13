"""Persist real/physical-zero teacher token pairs in the same sample order."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from diffusion_policy.dataset import render_net_teacher_cache as cache_module
from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache


class _Samples:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"sample_id": index}


def _targets(length=7):
    real = torch.arange(length * 3 * 4, dtype=torch.float32).reshape(length, 3, 4) / 7
    # Vary the difference by sample and channel; an unweighted average of
    # chunk means must not accidentally equal the complete-dataset mean.
    zero = real + torch.arange(length, dtype=torch.float32)[:, None, None] + .125
    return real, zero, {"side": real[:, :2, :2].square(), "wrist": real[:, :2, :3]}


def _slice_attention(attention, start, stop=None):
    return {key: value[start:stop] for key, value in attention.items()}


def test_pair_resumes_committed_prefix_and_collates_aligned_samples(tmp_path):
    identity = {"teacher": "fixed", "target_contract": "physical_zero_pair_v1"}
    real, zero, attention = _targets()
    cache = TeacherTargetCache(tmp_path, identity, len(real))
    cache.append(real[:3], _slice_attention(attention, 0, 3), zero_tokens=zero[:3])
    assert cache.completed == 3 and not cache.complete
    with pytest.raises(RuntimeError, match="before training"):
        cache.read(0)
    with pytest.raises(RuntimeError, match="complete"):
        cache.force_difference_statistics()
    # Simulate a process dying after writing an uncommitted row. Only the
    # metadata prefix is authoritative, so resume must overwrite this row.
    for array in cache._arrays.values():
        array[3] = -999
        array.flush()
    cache.close()
    resumed = TeacherTargetCache(tmp_path, identity, len(real))
    assert resumed.completed == 3
    resumed.append(real[3:], _slice_attention(attention, 3), zero_tokens=zero[3:])
    resumed.close()
    complete = TeacherTargetCache(tmp_path, identity, len(real))
    dataset = CachedTeacherDataset(_Samples(len(real)), complete)
    loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2,
                        generator=torch.Generator().manual_seed(130))
    seen = []
    for batch in loader:
        ids = batch["sample_id"]
        seen.extend(ids.tolist())
        targets = batch["teacher_targets"]
        torch.testing.assert_close(targets["tokens"], real[ids], rtol=0, atol=0)
        torch.testing.assert_close(targets["zero_tokens"], zero[ids], rtol=0, atol=0)
        for key in attention:
            torch.testing.assert_close(targets["patch_attention"][key], attention[key][ids], rtol=0, atol=0)
    assert sorted(seen) == list(range(len(real)))
    expected_elements = real.numel() + zero.numel() + sum(value.numel() for value in attention.values())
    assert complete.nbytes == expected_elements * 4
    complete.read(2)["zero_tokens"].zero_()
    torch.testing.assert_close(complete.read(2)["zero_tokens"], zero[2], rtol=0, atol=0)
    complete.load_into_ram()
    assert all(not value.flags.writeable for value in complete._arrays.values())
    torch.testing.assert_close(complete.read(2)["zero_tokens"], zero[2], rtol=0, atol=0)


def test_statistics_stream_all_elements_including_last_partial_chunk(tmp_path, monkeypatch):
    real, zero, attention = _targets()
    cache = TeacherTargetCache(tmp_path, {"target_contract": "paired"}, len(real))
    cache.append(real[:2], _slice_attention(attention, 0, 2), zero_tokens=zero[:2])
    cache.close()
    cache = TeacherTargetCache(tmp_path, {"target_contract": "paired"}, len(real))
    cache.append(real[2:], _slice_attention(attention, 2), zero_tokens=zero[2:])
    cache.close()
    calls = []
    subtract = cache_module.np.subtract

    def record_chunks(left, right, **kwargs):
        assert isinstance(left, np.memmap) and isinstance(right, np.memmap)
        calls.append(left.shape[0])
        return subtract(left, right, **kwargs)

    monkeypatch.setattr(cache_module.np, "subtract", record_chunks)
    statistics = cache.force_difference_statistics(chunk_size=3)
    assert calls == [3, 3, 1]
    expected = (real.double() - zero.double()).square()
    assert statistics["samples"] == len(real)
    assert statistics["elements"] == expected.numel()
    assert statistics["sum_square"] == pytest.approx(expected.sum().item(), rel=1e-14)
    assert statistics["mean_square"] == pytest.approx(expected.mean().item(), rel=1e-14)
    assert not cache.in_ram


@pytest.mark.parametrize("invalid", ["shape", "nan", "infinity", "fp16_overflow"])
def test_invalid_pairs_do_not_commit_or_create_metadata(tmp_path, invalid):
    real, zero, _ = _targets(2)
    dtype = "float16" if invalid == "fp16_overflow" else "float32"
    if invalid == "shape":
        zero = zero[:, :-1]
    elif invalid == "nan":
        zero[0, 0, 0] = float("nan")
    elif invalid == "infinity":
        zero[0, 0, 0] = float("inf")
    else:
        zero[0, 0, 0] = 1e6
    cache = TeacherTargetCache(tmp_path, {}, 2, token_dtype=dtype)
    error = ValueError if invalid == "shape" else FloatingPointError
    with pytest.raises(error, match="zero_tokens"):
        cache.append(real, {}, zero_tokens=zero)
    assert cache.completed == 0
    assert not (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("paired_first", [True, False])
def test_resume_cannot_switch_between_paired_and_legacy_targets(tmp_path, paired_first):
    real, zero, _ = _targets(3)
    cache = TeacherTargetCache(tmp_path, {}, 3)
    cache.append(real[:1], {}, zero_tokens=zero[:1] if paired_first else None)
    cache.close()
    resumed = TeacherTargetCache(tmp_path, {}, 3)
    with pytest.raises(ValueError, match="target keys changed"):
        resumed.append(real[1:], {}, zero_tokens=None if paired_first else zero[1:])
    assert resumed.completed == 1
    assert json.loads((tmp_path / "metadata.json").read_text())["completed_samples"] == 1
    resumed.append(real[1:], {}, zero_tokens=zero[1:] if paired_first else None)
    assert resumed.complete


def test_legacy_cache_remains_readable_without_inventing_zero_targets(tmp_path):
    real, _, attention = _targets(2)
    cache = TeacherTargetCache(tmp_path, {"legacy": True}, len(real))
    cache.append(real, attention)
    cache.close()
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for spec in metadata["arrays"].values():
        del spec["dtype"]
    metadata_path.write_text(json.dumps(metadata))
    restored = TeacherTargetCache(tmp_path, {"legacy": True}, len(real))
    row = restored.read(1)
    assert set(row) == {"tokens", "patch_attention"}
    torch.testing.assert_close(row["tokens"], real[1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="paired zero_tokens"):
        restored.force_difference_statistics()
    with pytest.raises(ValueError, match="identity/sample order"):
        TeacherTargetCache(tmp_path, {"legacy": True, "target_contract": "paired"}, len(real))
