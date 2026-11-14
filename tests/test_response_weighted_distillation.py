"""Response normalization and the shared condition -> frozen RGB training flow."""

import copy

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import RenderNet, build_render_net
from diffusion_policy.model.vision.render_net_attention import teacher_context_effect_targets
from diffusion_policy.model.vision.response_distillation import response_weighted_token_mse
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts.visualize_render_net_episode import render_observations
from diffusion_policy.workspace.train_distill_render_net_workspace import (
    TrainDistillRenderNetWorkspace, student_context_tokens,
)
from test_render_net_distillation import make_attention_teacher


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def renderer_config():
    return dict(
        base_channels=4, conditioning_dim=16, wrench_history_steps=7,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4, residual_scale=.5,
        spatial_modulation={"enabled": True},
        condition_prediction={"enabled": True, "hidden_dim": 8},
        condition_backbone={"base_channels": 4, "patch_size": 8, "embed_dim": 16,
                            "num_heads": 4},
        high_attention_render={"enabled": True, "threshold": 0., "initial_strength": .4},
    )


def make_workspace():
    teacher, obs, _ = make_attention_teacher()
    obs["contact_alpha"] = torch.tensor([[0., .25], [.75, 1.]])
    tokens, responses = teacher_context_effect_targets(teacher, obs)
    # Exercise independent camera/time layout and an empty response map.
    responses["side"][0, 0].zero_()
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({
        "distillation": {"objective": "token_mse", "use_teacher_ffn": False,
                         "response_loss_weight": 1., "response_noise_floor": 1e-6,
                         "alpha_loss_weight": 1.},
        "optimizer": {"lr": .001}, "training": {"lr_scheduler": {"enabled": False}},
    })
    workspace.model = RenderNet(6, **renderer_config())
    workspace.teacher_policy = teacher
    workspace.device = torch.device("cpu")
    workspace.amp_enabled = False
    workspace.attention_enabled = True
    workspace.attention_config = {"source": "context", "selection": "soft",
                                  "normalization": "image_minmax", "pixel_loss_weight": 0.}
    workspace.region_source = "context"
    workspace.context_patch_loss_weight = workspace.intensity_loss_weight = 1.
    workspace.smoothness_loss_weight = .02
    workspace.shape_info = {}
    workspace.render_stages_enabled = True
    workspace.training_stage = "intensity"
    return workspace, obs, {"tokens": tokens, "patch_attention": responses}


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_raw_response_weighting_uses_effective_weight_sum_and_stops_teacher_gradients():
    student = torch.tensor([[[1., 1.], [2., 2.]], [[3., 3.], [4., 4.]]], requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    response = torch.tensor([[9., 1.], [0., 0.]], requires_grad=True)
    global_loss, weighted, numerator, denominator = response_weighted_token_mse(
        student, teacher, response, noise_floor=0.,
    )
    assert global_loss.item() == pytest.approx(7.5)
    assert numerator.item() == pytest.approx(13.)
    assert denominator.item() == pytest.approx(10.)
    assert weighted.item() == pytest.approx(1.3)
    weighted.backward()
    assert student.grad[0, 0, 0] / student.grad[0, 1, 0] == pytest.approx(4.5)
    assert student.grad[1].count_nonzero() == 0
    assert response.grad is None and teacher.grad is None


@pytest.mark.parametrize("magnitude", [0., 1e-8])
def test_empty_or_numerical_noise_responses_leave_global_alignment(magnitude):
    student = torch.randn(2, 4, 3, requires_grad=True)
    global_loss, response_loss, _, denominator = response_weighted_token_mse(
        student, torch.zeros_like(student), torch.full((2, 4), magnitude),
    )
    assert response_loss.item() == denominator.item() == 0.
    (global_loss + response_loss).backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def test_condition_pretraining_encodes_once_per_camera_and_skips_rgb_and_student_dino():
    workspace, obs, cache = make_workspace()
    model = workspace.model
    model.set_training_stage("intensity")
    model.set_stage_mode(True)
    calls = []
    handle = model.condition_network.backbone.stem.register_forward_hook(lambda *args: calls.append(1))

    def forbidden(*args):
        raise AssertionError("Condition pretraining ran RGB or student DINO")

    hooks = [module.register_forward_pre_hook(forbidden) for module in
             (model.backbone, workspace.teacher_policy.obs_encoder.backbone)]
    try:
        loss = workspace._loss(obs, cache)
        loss.backward()
    finally:
        handle.remove()
        for hook in hooks:
            hook.remove()
    assert len(calls) == 2
    assert has_grad(model.condition_network.region_head)
    assert has_grad(model.condition_network.alpha_head)
    assert has_grad(model.condition_network.backbone.stem)
    assert all(p.grad is None and not p.requires_grad for p in model.rgb_parameters())


def test_rgb_loss_uses_teacher_response_and_freezes_both_condition_heads():
    workspace, obs, cache = make_workspace()
    model = workspace.model
    workspace.training_stage = "rgb"
    model.set_training_stage("rgb")
    model.set_stage_mode(True)
    assert not model.condition_network.training
    model.zero_grad(set_to_none=True)
    condition_before = copy.deepcopy(model.condition_network.state_dict())
    optimizer = torch.optim.AdamW(model.rgb_parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = workspace._loss(obs, cache)
        assert loss.item() == pytest.approx(workspace.loss_metrics["token_mse"]
                                          + workspace.loss_metrics["response_mse"])
        loss.backward()
        optimizer.step()
    assert has_grad(model.backbone.stem) and has_grad(model.output)
    assert all(p.grad is None for p in model.condition_network.parameters())
    for name, value in condition_before.items():
        torch.testing.assert_close(model.condition_network.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.grad is None for p in workspace.teacher_policy.parameters())

    # Labels can change the teacher targets; they cannot change student rendering.
    model.eval()
    with torch.no_grad():
        first = student_context_tokens(workspace.teacher_policy, model, obs)
        changed = dict(obs, contact_alpha=1 - obs["contact_alpha"])
        second = student_context_tokens(workspace.teacher_policy, model, changed)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_best_condition_snapshot_is_restored_and_selection_resets_at_rgb_boundary(tmp_path):
    workspace, obs, cache = make_workspace()
    workspace._output_dir = str(tmp_path)
    workspace.epoch = workspace.intensity_epochs = 1
    workspace.ema_model = copy.deepcopy(workspace.model).eval().requires_grad_(False)
    workspace.ema = EMAModel(workspace.ema_model)
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
    workspace.lr_scheduler = None
    workspace.model.set_training_stage("intensity")
    with torch.no_grad():
        workspace.ema_model.condition_network.alpha_head[-1].bias.fill_(2.)
    snapshot = {key: value.clone() for key, value in workspace.ema_model.state_dict().items()}
    checkpoint = workspace.get_checkpoint_path("best_condition")
    checkpoint.parent.mkdir()
    torch.save({"state_dicts": {"ema_model": snapshot}}, checkpoint, pickle_module=dill)
    workspace.best_val_loss = workspace.early_stopping_best_loss = .1
    assert workspace._maybe_transition_stage()
    assert workspace.training_stage == "rgb"
    assert workspace.best_val_loss == float("inf")
    assert workspace.ema.optimization_step == 0
    torch.testing.assert_close(workspace.model.condition_network.alpha_head[-1].bias,
                               torch.tensor([2.]))
    assert all(not p.requires_grad for p in workspace.model.condition_network.parameters())


def test_saved_model_reloads_and_deployment_ignores_offline_alpha():
    workspace, obs, _ = make_workspace()
    model = workspace.model.eval()
    with torch.no_grad():
        model.output.bias.fill_(.4)
    restored = build_render_net(6, renderer_config()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    preprocessor = WrenchPreprocessor.from_policy(workspace.teacher_policy)
    images, rendered = render_observations(restored, preprocessor, obs, model.image_keys, 2)
    changed = dict(obs, contact_alpha=torch.ones_like(obs["contact_alpha"]))
    _, rendered_changed = render_observations(restored, preprocessor, changed, model.image_keys, 2)
    for key in rendered:
        torch.testing.assert_close(rendered[key], rendered_changed[key], rtol=0, atol=0)
        assert (rendered[key] - images[key]).abs().max() > 0


def test_alpha_is_causal_across_observation_times():
    model = RenderNet(6, **renderer_config()).eval()
    images = {key: torch.rand(2, 2, 3, 16, 24) for key in model.image_keys}
    wrench = torch.randn(2, 2, 7, 6)
    _, _, first = model(images, wrench, condition_only=True)
    changed = {key: image.clone() for key, image in images.items()}
    for image in changed.values():
        image[:, 1].zero_()
    wrench[:, 1] += 20
    _, _, second = model(changed, wrench, condition_only=True)
    torch.testing.assert_close(first["contact_alpha"][:, 0], second["contact_alpha"][:, 0])


def test_validation_aggregates_response_numerators_before_normalizing():
    workspace, _, _ = make_workspace()
    workspace.training_stage = "rgb"
    workspace.epoch = 0
    workspace.max_val_batches = None
    records = iter([
        {"token_mse": 1., "student_loss": 2., "response_numerator": 9., "response_weight_sum": 9.},
        {"token_mse": 9., "student_loss": 18., "response_numerator": 9., "response_weight_sum": 1.},
    ])

    def batch_loss(batch):
        workspace.loss_metrics = next(records)
        return torch.tensor(workspace.loss_metrics["student_loss"])

    workspace._batch_loss = batch_loss
    result = workspace._validate_batches([{"obs": {"side": torch.zeros(1)}}] * 2)
    assert result == pytest.approx(5. + 18. / 10.)
    assert workspace.validation_metrics["response_mse"] == pytest.approx(1.8)


def test_precontact_teacher_preserves_continuous_wrench_and_preprocessor_contract():
    teacher, obs, _ = make_attention_teacher()
    teacher.precontact_seconds = .5
    obs["wrench"].fill_(.01)
    _, normalized, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
    torch.testing.assert_close(normalized, teacher.normalizer["wrench"].normalize(obs["wrench"]))
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    torch.testing.assert_close(preprocessor(obs)[0], normalized)
    assert preprocessor.get_config()["precontact_seconds"] == .5
