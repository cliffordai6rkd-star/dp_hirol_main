"""Paired torque-context supervision through the complete student workspace."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_civi_action_heads import _batch
from test_render_net_visual_distillation import _workspace as _visual_workspace


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _workspace(monkeypatch, tmp_path, upsample=True):
    workspace = _visual_workspace(monkeypatch, tmp_path, upsample)
    with open_dict(workspace.cfg.distillation):
        workspace.cfg.distillation.force_context = {
            "enabled": True, "difference_weight": 1.0, "region_weight": 1.0,
            "difference_scale": 1.0, "region_scale": 1.0,
        }
    workspace.force_difference_scale = 1.0
    workspace.force_region_scale = 1.0
    Path(workspace.output_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        workspace.model.output.weight.normal_(std=.02)
    workspace.model.set_stage_mode(True)
    return workspace


def _teacher_pair(teacher, obs):
    zero_obs = {**obs, teacher.wrench_key: torch.zeros_like(obs[teacher.wrench_key])}
    with torch.no_grad():
        return {
            "tokens": teacher._encode_context(obs, apply_curriculum_mask=False),
            "zero_tokens": teacher._encode_context(zero_obs, apply_curriculum_mask=False),
        }


def _forbidden(*args, **kwargs):
    pytest.fail("force-context supervision must not execute ACT or legacy region heads")


@pytest.mark.parametrize("upsample", [True, False])
def test_pair_uses_one_renderer_and_visual_encoder_with_gradients_to_both_branches(
    monkeypatch, tmp_path, upsample,
):
    workspace = _workspace(monkeypatch, tmp_path, upsample)
    workspace.cfg.distillation.force_context.difference_weight = 1.7
    workspace.cfg.distillation.force_context.region_weight = .3
    workspace.force_difference_scale = .25
    workspace.force_region_scale = .7
    teacher, renderer = workspace.teacher_policy, workspace.model
    obs = _batch()["obs"]
    pair = _teacher_pair(teacher, obs)
    monkeypatch.setattr(teacher.model, "forward", _forbidden)
    monkeypatch.setattr(workspace, "_region_targets", _forbidden)
    monkeypatch.setattr(workspace, "_denoiser_losses", _forbidden)
    teacher_calls, rendered_calls, encoded_calls = [], [], []
    original_teacher = teacher._encode_context
    original_encode = teacher.obs_encoder.encode_image_context

    def teacher_spy(observations, **kwargs):
        teacher_calls.append(observations[teacher.wrench_key].detach().clone())
        return original_teacher(observations, **kwargs)

    def encode_spy(images, **kwargs):
        tokens = original_encode(images, **kwargs)
        encoded_calls.append(tokens)
        return tokens

    def render_hook(module, args, output):
        rgb = output[0] if isinstance(output, tuple) else output
        for image in rgb.values():
            image.retain_grad()
        rendered_calls.append(rgb)

    monkeypatch.setattr(teacher, "_encode_context", teacher_spy)
    monkeypatch.setattr(teacher.obs_encoder, "encode_image_context", encode_spy)
    handle = renderer.register_forward_hook(render_hook)
    try:
        loss = workspace._batch_loss({"obs": obs})
        assert len(teacher_calls) == 2
        torch.testing.assert_close(teacher_calls[0], obs[teacher.wrench_key], rtol=0, atol=0)
        assert teacher_calls[1].count_nonzero() == 0
        assert len(rendered_calls) == len(encoded_calls) == 1
        student_real, student_zero = encoded_calls[0].chunk(2, dim=0)
        expected_mse = F.mse_loss(student_real.float(), pair["tokens"].float())
        expected_delta = F.mse_loss(
            student_real.float() - student_zero.float(),
            pair["tokens"].float() - pair["zero_tokens"].float(),
        )
        assert workspace.loss_metrics["token_mse"] == pytest.approx(expected_mse.item())
        assert workspace.loss_metrics["force_delta_mse"] == pytest.approx(expected_delta.item())
        assert workspace.loss_metrics["force_delta_loss"] == pytest.approx(expected_delta.item() / .25)
        images = teacher.prepare_context_inputs(obs)[0]
        residual_tokens = []
        for key in teacher.obs_encoder.image_keys:
            residual = (rendered_calls[0][key][:len(student_real)] - images[key]).abs().mean(dim=2)
            grid = teacher.obs_encoder.context_patch_grid_shapes[key]
            pooled = F.adaptive_avg_pool2d(residual.flatten(0, 1).unsqueeze(1), grid)
            residual_tokens.append(pooled.reshape(len(student_real), teacher.n_obs_steps, -1))
        residual_tokens = torch.cat(residual_tokens, dim=-1).flatten(1)
        sensitivity = (pair["tokens"].float() - pair["zero_tokens"].float()).square().mean(dim=-1).sqrt()
        weights = sensitivity / (sensitivity + .7)
        expected_region = ((1 - weights) * residual_tokens).mean()
        assert workspace.loss_metrics["context_region_loss"] == pytest.approx(expected_region.item())
        expected_total = (workspace.loss_metrics["token_mse"]
                          + 1.7 * workspace.loss_metrics["force_delta_loss"]
                          + .3 * workspace.loss_metrics["context_region_loss"])
        assert loss.item() == pytest.approx(expected_total)
        assert workspace.loss_metrics["loss"] == pytest.approx(loss.item())
        assert workspace.metric_name == "force context loss"
        assert workspace._selection_mse(loss) == pytest.approx(loss.item())
        loss.backward()
        for rgb in rendered_calls[0].values():
            assert rgb.shape[0] == 2 * obs["tau_ext"].shape[0]
            real_gradient, zero_gradient = rgb.grad.chunk(2, dim=0)
            assert real_gradient.abs().sum() > 0
            assert zero_gradient.abs().sum() > 0
        for name, parameter in renderer.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            else:
                assert parameter.grad is None, name
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in teacher.parameters())
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in renderer.wrench_encoder.parameters())
    finally:
        handle.remove()
    # A small gradient step should improve this deterministic paired objective.
    baseline = loss.item()
    with torch.no_grad():
        for parameter in renderer.parameters():
            if parameter.requires_grad:
                parameter.add_(parameter.grad, alpha=-1e-5)
    assert workspace._batch_loss({"obs": obs, "teacher_targets": pair}).item() < baseline


@pytest.mark.parametrize("upsample", [True, False])
def test_cached_pair_bypasses_teacher_and_preserves_total_loss_and_gradients(
    monkeypatch, tmp_path, upsample,
):
    workspace = _workspace(monkeypatch, tmp_path, upsample)
    obs = _batch()["obs"]
    pair = _teacher_pair(workspace.teacher_policy, obs)
    online_loss = workspace._batch_loss({"obs": obs})
    online_metrics = dict(workspace.loss_metrics)
    online_loss.backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in workspace.model.named_parameters() if parameter.requires_grad}
    workspace.optimizer.zero_grad(set_to_none=True)
    monkeypatch.setattr(workspace.teacher_policy, "_encode_context", _forbidden)
    cached_loss = workspace._batch_loss({"obs": obs, "teacher_targets": pair})
    torch.testing.assert_close(cached_loss, online_loss, rtol=0, atol=0)
    for key in ("token_mse", "force_delta_mse", "force_delta_loss", "context_region_loss", "loss"):
        assert workspace.loss_metrics[key] == online_metrics[key]
    cached_loss.backward()
    for name, parameter in workspace.model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        else:
            assert parameter.grad is None


def test_validation_uses_ema_and_selects_combined_objective(monkeypatch, tmp_path):
    workspace = _workspace(monkeypatch, tmp_path)
    obs = _batch()["obs"]
    pair = _teacher_pair(workspace.teacher_policy, obs)
    workspace.ema_model = copy.deepcopy(workspace.model).eval().requires_grad_(False)
    workspace.model.eval()
    batch = {"obs": obs, "teacher_targets": pair}
    with torch.no_grad():
        expected = workspace._batch_loss(batch).item()
    token_only = workspace.loss_metrics["token_mse"]
    assert expected > token_only
    first = workspace._validate([batch])
    assert first == pytest.approx(expected)
    assert first == pytest.approx(
        workspace.validation_metrics["token_mse"]
        + workspace.validation_metrics["force_delta_loss"]
        + workspace.validation_metrics["context_region_loss"]
    )
    with torch.no_grad():
        workspace.model.output.bias.add_(3.0)
    assert workspace._validate([batch]) == first
    with torch.no_grad():
        workspace.ema_model.output.bias.add_(1.0)
    assert workspace._validate([batch]) != pytest.approx(first)


class _ObservationDataset:
    image_randomer = None

    def __init__(self, path, obs, count, *, offset=0):
        self.dataset_path = str(path)
        self.obs = obs
        self.indices = np.arange(offset, offset + count, dtype=np.int64).reshape(-1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = index % next(iter(self.obs.values())).shape[0]
        return {"obs": {key: value[row].clone() for key, value in self.obs.items()}}


def test_cache_preparation_saves_both_splits_and_reuses_pair_without_teacher_calls(
    monkeypatch, tmp_path,
):
    workspace = _workspace(monkeypatch, tmp_path)
    source = tmp_path / "data"
    source.mkdir()
    Path(workspace.cfg.teacher_checkpoint).write_bytes(b"fixed tiny teacher")
    with open_dict(workspace.cfg.teacher_cache):
        workspace.cfg.teacher_cache.update({
            "enabled": True, "cleanup_on_exit": False, "dtype": "float32",
            "precision": "float32", "directory": str(tmp_path / "cache"),
            "batch_size": 2, "num_workers": 0, "load_into_ram": False,
        })
    obs = _batch()["obs"]
    train = _ObservationDataset(source, obs, 3)
    validation = _ObservationDataset(source, obs, 1, offset=10)
    original = workspace.teacher_policy._encode_context
    calls = []

    def spy(observations, **kwargs):
        calls.append(observations["tau_ext"].detach().clone())
        return original(observations, **kwargs)

    monkeypatch.setattr(workspace.teacher_policy, "_encode_context", spy)
    prepared = workspace._prepare_cached_datasets(
        train, validation, OmegaConf.create({"dataset_path": str(source)}),
    )
    assert len(calls) == 6  # Two train batches and one validation batch, two targets each.
    for cached in prepared:
        assert isinstance(cached, CachedTeacherDataset)
        assert cached.cache.complete
        assert cached.cache.metadata["arrays"]["zero_tokens"]["dtype"] == "float32"
        assert "target_contract" in cached.cache.identity
        for batch in DataLoader(cached, batch_size=2):
            expected = _teacher_pair(workspace.teacher_policy, batch["obs"])
            for key in expected:
                torch.testing.assert_close(batch["teacher_targets"][key], expected[key], rtol=1e-6, atol=1e-7)
    monkeypatch.setattr(workspace.teacher_policy, "_encode_context", _forbidden)
    reused = workspace._prepare_cached_datasets(
        train, validation, OmegaConf.create({"dataset_path": str(source)}),
    )
    assert [str(item.cache.directory) for item in reused] == [str(item.cache.directory) for item in prepared]
    assert all(item.cache.directory.exists() for item in reused)


@pytest.mark.parametrize("zero_effect", [False, True])
def test_automatic_scales_use_complete_training_cache_only_and_persist(tmp_path, monkeypatch, zero_effect):
    workspace = _workspace(monkeypatch, tmp_path)
    workspace.cfg.distillation.force_context.difference_scale = None
    workspace.cfg.distillation.force_context.region_scale = None
    workspace.force_difference_scale = None
    workspace.force_region_scale = None
    obs = _batch()["obs"]
    real = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4) / 10
    zero = real.clone() if zero_effect else real + torch.arange(1, 6, dtype=torch.float32)[:, None, None]
    train_cache = TeacherTargetCache(tmp_path / "train_cache", {"split": "train"}, 5)
    train_cache.append(real, {}, zero_tokens=zero)
    training = CachedTeacherDataset(_ObservationDataset(tmp_path, obs, 5), train_cache)
    validation_cache = TeacherTargetCache(tmp_path / "validation_cache", {"split": "validation"}, 1)
    validation_cache.append(real[:1] * 100, {}, zero_tokens=zero[:1] * 100)
    monkeypatch.setattr(validation_cache, "force_difference_statistics", _forbidden)
    original = train_cache.force_difference_statistics
    calls = []

    def train_statistics(**kwargs):
        calls.append("train")
        return original(**kwargs)

    monkeypatch.setattr(train_cache, "force_difference_statistics", train_statistics)
    workspace._prepare_force_context_scales(training)
    assert calls == ["train"]
    expected = (real.double() - zero.double()).square().mean().item()
    assert workspace.force_difference_scale == pytest.approx(max(expected, 1e-8))
    assert workspace.force_region_scale == pytest.approx(max(expected ** .5, 1e-4))
    assert workspace.cfg.distillation.force_context.difference_scale == workspace.force_difference_scale
    assert workspace.cfg.distillation.force_context.region_scale == workspace.force_region_scale
    statistics = json.loads((Path(workspace.output_dir) / "force_context_statistics.json").read_text())
    assert statistics["source"] == "train" and statistics["baseline"] == "raw_zero"
    assert statistics["samples"] == 5 and statistics["elements"] == real.numel()
    assert statistics["mean_square"] == pytest.approx(expected)
    assert statistics["difference_scale"] == workspace.force_difference_scale
    assert statistics["region_scale"] == workspace.force_region_scale
    # Once fixed in the resolved config, neither validation nor resuming the
    # same experiment is allowed to recalibrate on a different sample set.
    monkeypatch.setattr(train_cache, "force_difference_statistics", _forbidden)
    workspace._prepare_force_context_scales(training)
    assert workspace.force_difference_scale == pytest.approx(max(expected, 1e-8))
