"""Student-only calibration, stage ownership, caching and deployment parity."""

import copy
from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import dill
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn import functional as F

from diffusion_policy.common.render_net_training_logger import MetricHistory
from diffusion_policy.dataset.render_net_teacher_cache import TeacherTargetCache
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.render_net_attention import (
    teacher_context_effect_targets, teacher_zero_context_target,
)
from diffusion_policy.model.vision.student_context_adapter import StudentContextAdapter
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts.visualize_render_net_episode import encode_student_context
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext
from test_response_weighted_distillation import make_workspace, renderer_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def adapted_workspace(kind="linear"):
    workspace, obs, cache = make_workspace()
    config = renderer_config()
    config["separate_gate_path"] = True
    config["context_adapter"] = {"enabled": True, "kind": kind, "hidden_dim": 8}
    workspace.model = build_render_net(6, config)
    cache["zero_tokens"] = teacher_zero_context_target(workspace.teacher_policy, obs)
    workspace.training_stage = "adapter"
    workspace.model.set_training_stage("adapter")
    return workspace, obs, cache, config


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_adapter_starts_at_identity_and_receives_gradients(kind):
    adapter = StudentContextAdapter(16, kind=kind, hidden_dim=8)
    tokens = torch.randn(2, 9, 16, requires_grad=True)
    torch.testing.assert_close(adapter(tokens), tokens, rtol=0, atol=0)
    adapter(tokens).square().mean().backward()
    assert tokens.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())


def test_adapter_runs_after_visual_embeddings_before_norm_without_owning_teacher_parameters():
    workspace, obs, _, _ = adapted_workspace()
    teacher, adapter = workspace.teacher_policy, workspace.model.context_adapter
    before = {key: value.clone() for key, value in teacher.state_dict().items()}
    with torch.no_grad():
        adapter.mapping.bias.copy_(torch.linspace(-1., 1., 16))
    images = {k: obs[k] for k in teacher.obs_encoder.image_keys}
    seen = []
    hook = teacher.obs_encoder.context_norm.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    raw = teacher.obs_encoder.encode_image_context(images)
    adapted = teacher.obs_encoder.encode_image_context(images, context_adapter=adapter)
    hook.remove()
    torch.testing.assert_close(seen[1], adapter(seen[0]))
    torch.testing.assert_close(adapted, teacher.obs_encoder.context_norm(adapter(seen[0])))
    assert not torch.allclose(raw, adapted)
    assert teacher.state_dict().keys() == before.keys()
    for key in before:
        torch.testing.assert_close(teacher.state_dict()[key], before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="exclusive"):
        teacher.obs_encoder.encode_image_context(images, context_adapter=adapter, apply_fusion_ffn=True)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_calibration_uses_only_original_images_and_updates_only_adapter(kind):
    workspace, obs, cache, _ = adapted_workspace(kind)
    model = workspace.model
    model.set_stage_mode(True)
    before = {key: value.clone() for key, value in model.state_dict().items()
              if not key.startswith("context_adapter.")}
    forbidden = lambda *args: (_ for _ in ()).throw(AssertionError("RGB/condition path ran"))
    hooks = [module.register_forward_pre_hook(forbidden)
             for module in (model.backbone, model.condition_network)]
    optimizer = torch.optim.Adam(model.context_adapter.parameters(), lr=.02)
    try:
        initial = workspace._loss(obs, cache).item()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = workspace._loss(obs, cache)
            loss.backward()
            optimizer.step()
        assert workspace._loss(obs, cache).item() < initial * .8
        changed = dict(obs, wrench=obs["wrench"] * 100 + 20,
                       contact_alpha=1 - obs["contact_alpha"])
        torch.testing.assert_close(workspace._loss(obs, cache), workspace._loss(changed, cache))
    finally:
        for hook in hooks:
            hook.remove()
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.rgb_parameters())
    assert all(p.grad is None for p in workspace.teacher_policy.parameters())
    with pytest.raises(ValueError, match="zero_tokens"):
        workspace._loss(obs, {k: v for k, v in cache.items() if k != "zero_tokens"})


def test_paired_targets_recompute_zero_alpha_and_cache_both_contexts(tmp_path):
    workspace, obs, _, _ = adapted_workspace()
    normal, response, zero = teacher_context_effect_targets(workspace.teacher_policy, obs, return_zero=True)
    torch.testing.assert_close(zero, teacher_zero_context_target(workspace.teacher_policy, obs))
    altered = dict(obs, contact_alpha=1 - obs["contact_alpha"])
    torch.testing.assert_close(zero, teacher_zero_context_target(workspace.teacher_policy, altered))
    cache = TeacherTargetCache(tmp_path / "paired", {"paired_zero_context": "raw_zero_v1"}, 2)
    cache.append(normal, response, zero_tokens=zero)
    reopened = TeacherTargetCache(tmp_path / "paired", cache.identity, 2)
    torch.testing.assert_close(reopened.read(1)["zero_tokens"], zero[1])
    assert not torch.equal(zero, normal)


def test_workspace_builds_separate_paired_cache_for_adapter_supervision(tmp_path):
    workspace, obs, _, _ = adapted_workspace()
    source = tmp_path / 'source'
    source.mkdir()
    checkpoint = tmp_path / 'teacher.ckpt'
    checkpoint.write_bytes(b'fixed-teacher-identity')
    workspace._output_dir = str(tmp_path)
    workspace.cfg.teacher_checkpoint = str(checkpoint)
    workspace.cfg.teacher_cache = {
        'enabled': True, 'directory': str(tmp_path / 'cache'), 'batch_size': 2,
        'num_workers': 0, 'dtype': 'float16', 'precision': 'float32',
    }
    workspace.cfg.training.seed = 42
    workspace.cfg.training.amp = {'allow_tf32': False}
    workspace.cfg.logging = {'progress_bar': False}
    workspace.amp_dtype = torch.bfloat16
    workspace.teacher_cfg = OmegaConf.create({'policy': {'fixed': True}})
    workspace.teacher_state_key = 'ema_model'

    class Source:
        image_randomer = None
        dataset_path = str(source)
        indices = torch.arange(2).numpy().reshape(-1, 1)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {'obs': {key: value[index] for key, value in obs.items()}}

    dataset = Source()
    _, identity = workspace._teacher_cache_spec(dataset, dataset, OmegaConf.create({}))
    assert identity['paired_zero_context'] == 'raw_zero_v1'
    expected = teacher_zero_context_target(workspace.teacher_policy, obs)
    cached, _ = workspace._prepare_cached_datasets_local(dataset, dataset, OmegaConf.create({}))
    torch.testing.assert_close(cached[1]['teacher_targets']['zero_tokens'].float(), expected[1],
                               rtol=.001, atol=.001)
    adapter = workspace.model.context_adapter
    workspace.model.context_adapter = None
    _, legacy_identity = workspace._teacher_cache_spec(dataset, dataset, OmegaConf.create({}))
    assert 'paired_zero_context' not in legacy_identity
    workspace.model.context_adapter = adapter


def test_rgb_and_condition_stages_freeze_adapter_but_rgb_gradients_pass_through_it():
    workspace, obs, cache, _ = adapted_workspace()
    model = workspace.model
    with torch.no_grad():
        model.context_adapter.mapping.bias.copy_(torch.linspace(-.2, .2, 16))
    frozen = copy.deepcopy(model.context_adapter.state_dict())
    for stage in ("intensity", "rgb"):
        workspace.training_stage = stage
        model.set_training_stage(stage)
        model.set_stage_mode(True)
        model.zero_grad(set_to_none=True)
        workspace._loss(obs, cache).backward()
        assert all(not p.requires_grad and p.grad is None for p in model.context_adapter.parameters())
        if stage == "rgb":
            assert model.output.weight.grad.abs().sum() > 0
            assert all(p.grad is None for p in model.condition_network.parameters())
    for key, value in frozen.items():
        torch.testing.assert_close(model.context_adapter.state_dict()[key], value, rtol=0, atol=0)


def test_best_adapter_and_condition_boundaries_preserve_frozen_snapshots(tmp_path):
    workspace, _, _, _ = adapted_workspace()
    workspace._output_dir = str(tmp_path)
    workspace.epoch = workspace.stage_epoch = workspace.adapter_epochs = 1
    workspace.stage_step = 4
    workspace.intensity_epochs = 2
    workspace.ema_model = copy.deepcopy(workspace.model).eval().requires_grad_(False)
    workspace.ema = EMAModel(workspace.ema_model)
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
    workspace.lr_scheduler = None
    with torch.no_grad():
        workspace.ema_model.context_adapter.mapping.bias.fill_(.5)
    snapshot = copy.deepcopy(workspace.ema_model.state_dict())
    path = workspace.get_checkpoint_path("best_adapter")
    path.parent.mkdir()
    torch.save({"state_dicts": {"ema_model": snapshot}}, path, pickle_module=dill)
    assert workspace._maybe_transition_stage()
    assert workspace.training_stage == "intensity" and workspace.stage_epoch == 0
    assert workspace.best_val_loss == float("inf") and workspace.ema.optimization_step == 0
    assert all(not p.requires_grad for p in workspace.model.context_adapter.parameters())
    torch.testing.assert_close(workspace.model.context_adapter.mapping.bias, torch.full((16,), .5))
    workspace.epoch, workspace.stage_epoch = 21, 1
    assert not workspace._maybe_transition_stage()  # Uses stage age, not global epoch.
    workspace.stage_epoch = 2
    assert workspace._maybe_transition_stage()
    assert workspace.training_stage == "rgb"
    torch.testing.assert_close(workspace.model.context_adapter.mapping.bias, torch.full((16,), .5))


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_saved_student_retains_adapter_and_inference_matches_training_context(kind):
    workspace, obs, _, config = adapted_workspace(kind)
    model = workspace.model
    with torch.no_grad():
        last = model.context_adapter.mapping if kind == "linear" else model.context_adapter.mapping[-1]
        last.bias.copy_(torch.linspace(-.4, .4, 16))
    model.set_training_stage("rgb")
    model.eval()
    restored = build_render_net(6, config).eval().requires_grad_(False)
    restored.load_state_dict(model.state_dict(), strict=True)
    teacher = workspace.teacher_policy
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    with torch.no_grad():
        expected = student_context_tokens(teacher, model, obs)
        actual = encode_student_context(restored, preprocessor, obs, teacher.obs_encoder, 2)
        changed = dict(obs, contact_alpha=1 - obs["contact_alpha"])
        repeated = encode_student_context(restored, preprocessor, changed, teacher.obs_encoder, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    assert any(key.startswith("context_adapter.") for key in restored.state_dict())


def test_validation_selects_calibration_mse_instead_of_normal_response_metric():
    workspace, obs, cache, _ = adapted_workspace()
    workspace.epoch, workspace.max_val_batches = 0, None
    workspace.cfg.logging = {"progress_bar": False}
    batch = {"obs": obs, "teacher_targets": cache}
    workspace.model.eval()
    expected = workspace._adapter_loss(obs, cache).item()
    assert workspace._validate_batches([batch]) == pytest.approx(expected)
    assert workspace.validation_metrics["adapter_mse"] == pytest.approx(expected)


def test_adapter_history_is_separate_from_rgb_token_loss():
    history = MetricHistory()
    history.record({"objective": "adapter_warmup", "global_step": 1,
                    "train_mse": .9, "train_adapter_mse": .9, "val_mse": .8})
    history.record({"objective": "token_mse", "global_step": 2, "train_mse": .2})
    assert history.points("train_adapter_mse")[1].tolist() == [.9]
    assert history.points("val_adapter_mse")[1].tolist() == [.8]
    assert history.points("train_mse")[1].tolist() == [.2]


def _adapter_ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        workspace, obs, _, _ = adapted_workspace()
        workspace.distributed = DistributedContext(rank=rank, local_rank=rank, world_size=2)
        workspace.cfg.training.distributed = {"find_unused_parameters": True}
        workspace._output_dir = directory
        workspace.ema = workspace.ema_model = None
        workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
        workspace.lr_scheduler = None
        workspace.epoch = workspace.stage_epoch = workspace.stage_step = 0
        workspace.adapter_epochs = workspace.intensity_epochs = 1
        obs['side'] = obs['side'] + .02 * rank
        normal, responses, zero = teacher_context_effect_targets(workspace.teacher_policy, obs, return_zero=True)
        cache = {'tokens': normal, 'patch_attention': responses, 'zero_tokens': zero}
        workspace._setup_distributed_model()
        for stage in ('adapter', 'intensity', 'rgb'):
            assert workspace.training_stage == stage
            workspace.model.set_stage_mode(True)
            for _ in range(2):
                workspace.optimizer.zero_grad(set_to_none=True)
                workspace._loss(obs, cache).backward()
                workspace.optimizer.step()
            packed = torch.cat([p.detach().flatten() for p in workspace.model.parameters()])
            replicas = [torch.empty_like(packed) for _ in range(2)]
            dist.all_gather(replicas, packed)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
            workspace.stage_epoch = 1
            workspace.epoch += 1
            if stage != 'rgb':
                assert workspace._maybe_transition_stage()
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_rebuilds_when_trainable_stage_changes(tmp_path):
    mp.spawn(_adapter_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
