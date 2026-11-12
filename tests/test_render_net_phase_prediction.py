"""Causal contact estimates control rendering without offline phase labels."""

import torch
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net import RenderNet


def _renderer(**kwargs):
    phase_prediction = kwargs.pop("phase_prediction", {"enabled": True, "hidden_dim": 8})
    return RenderNet(
        7, base_channels=4, conditioning_dim=8, wrench_history_steps=8,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4,
        phase_prediction=phase_prediction, **kwargs,
    )


def _observations():
    return ({key: torch.rand(2, 2, 3, 16, 24) for key in ("side", "wrist")},
            torch.randn(2, 2, 8, 7))


def test_phase_and_alpha_predictions_do_not_read_later_observations():
    torch.manual_seed(72)
    renderer = _renderer().eval()
    images, wrench = _observations()
    predictions = renderer.predict_contact_phase(images, wrench)
    altered_images = {key: value.clone() for key, value in images.items()}
    for image in altered_images.values():
        image[:, 1].normal_(mean=30, std=5)
    altered_wrench = wrench.clone()
    altered_wrench[:, 1].normal_(mean=40, std=8)
    altered = renderer.predict_contact_phase(altered_images, altered_wrench)
    assert predictions["phase_logits"].shape == (2, 2, 3)
    assert predictions["contact_alpha"].shape == (2, 2)
    for key in predictions:
        torch.testing.assert_close(predictions[key][:, 0], altered[key][:, 0], rtol=0, atol=0)
        assert not torch.equal(predictions[key][:, 1], altered[key][:, 1])


def test_phase_ce_and_alpha_regression_train_the_shared_spatial_temporal_features():
    torch.manual_seed(73)
    renderer = _renderer(high_attention_render={"enabled": True, "threshold": 0.})
    images, wrench = _observations()
    _, masks, predictions = renderer(
        images, torch.zeros_like(wrench), return_phase=True,
    )
    phase = torch.tensor([[0, 1], [1, 2]])
    alpha = torch.tensor([[0., .25], [.75, 1.]])
    loss = (F.cross_entropy(predictions["phase_logits"].flatten(0, 1), phase.flatten())
            + F.mse_loss(predictions["contact_alpha"], alpha)
            + sum(mask.square().mean() for mask in masks.values()))
    loss.backward()
    assert renderer.mask_head.weight.grad is not None
    assert renderer.mask_head.weight.grad[0].abs().sum() > 0
    assert renderer.mask_head.weight.grad[1:5].abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in renderer.phase_temporal_encoder.parameters())


def test_renderer_uses_causal_alpha_instead_of_future_mask_and_serializes_predictor():
    torch.manual_seed(74)
    renderer = _renderer().eval()
    images, wrench = _observations()
    with torch.no_grad():
        renderer.output.bias.fill_(.4)
        free, _, predictions = renderer(
            images, wrench, future_contact=torch.zeros(2, 2, 8, dtype=torch.bool),
            return_phase=True,
        )
        contact = renderer(images, wrench, future_contact=torch.ones(2, 2, 8, dtype=torch.bool))
        closed = renderer(images, wrench, contact_alpha=torch.zeros(2, 2))
    assert predictions is not None
    for key in images:
        torch.testing.assert_close(free[key], contact[key], rtol=0, atol=0)
        assert (free[key] - images[key]).abs().max() > 0
        torch.testing.assert_close(closed[key], images[key], rtol=0, atol=0)
    restored = _renderer().eval()
    restored.load_state_dict(renderer.state_dict())
    with torch.no_grad():
        restored_predictions = restored.predict_contact_phase(images, wrench)
    for key in predictions:
        torch.testing.assert_close(predictions[key], restored_predictions[key], rtol=0, atol=0)


def test_region_phase_and_alpha_are_channels_of_one_shared_spatial_head():
    torch.manual_seed(75)
    renderer = _renderer(high_attention_render={"enabled": True, "threshold": 0.})
    images, wrench = _observations()
    rendered, masks, predictions = renderer(images, wrench, return_aux=True, return_phase=True)
    assert set(masks) == set(images)
    assert set(rendered) == set(images)
    assert renderer.mask_head.out_channels == 5
    assert predictions["phase_logits"].shape == (2, 2, 3)
    assert predictions["contact_alpha"].shape == (2, 2)


def test_condition_only_runs_shared_head_without_rgb_rendering():
    renderer = _renderer(high_attention_render={"enabled": True, "threshold": 0.})
    images, wrench = _observations()
    rendered, masks, predictions = renderer(
        images, wrench, condition_only=True, return_aux=True, return_phase=True,
    )
    assert rendered is None
    assert set(masks) == set(images)
    for key, mask in masks.items():
        assert mask.shape == (*images[key].shape[:2], 1, *images[key].shape[-2:])
    assert predictions["phase_logits"].shape == (2, 2, 3)
    assert predictions["contact_alpha"].shape == (2, 2)


def test_shared_head_accepts_a_subset_of_configured_camera_observations():
    renderer = _renderer(high_attention_render={"enabled": True, "threshold": 0.}).eval()
    images, wrench = _observations()
    images = {"wrist": images["wrist"]}
    with torch.no_grad():
        rendered, masks, predictions = renderer(images, wrench, return_phase=True)
        _, warmup_masks, warmup_predictions = renderer(
            images, wrench, condition_only=True, return_aux=True, return_phase=True,
        )
    assert set(rendered) == set(masks) == set(warmup_masks) == {"wrist"}
    torch.testing.assert_close(masks["wrist"], warmup_masks["wrist"], rtol=0, atol=0)
    for key in predictions:
        torch.testing.assert_close(predictions[key], warmup_predictions[key], rtol=0, atol=0)


def test_unified_phase_region_head_follows_gate_training_stage():
    renderer = _renderer(high_attention_render={"enabled": True}, separate_gate_path=True)
    renderer.set_training_stage("intensity")
    assert all(parameter.requires_grad for parameter in renderer.phase_temporal_encoder.parameters())
    assert all(parameter.requires_grad for parameter in renderer.mask_head.parameters())
    renderer.set_training_stage("rgb")
    renderer.set_stage_mode(training=True)
    assert not renderer.phase_temporal_encoder.training
    assert all(not parameter.requires_grad for parameter in renderer.phase_temporal_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in renderer.mask_head.parameters())
    renderer.set_training_stage("joint")
    assert all(parameter.requires_grad for parameter in renderer.phase_temporal_encoder.parameters())
    assert all(parameter.requires_grad for parameter in renderer.mask_head.parameters())


def test_disabled_phase_prediction_keeps_old_one_channel_mask_head():
    renderer = RenderNet(7, base_channels=4, conditioning_dim=8, embed_dim=16,
                         num_heads=4, patch_size=8, decoder_layers=3)
    assert not renderer.phase_prediction_enabled
    assert renderer.phase_temporal_encoder is None
    assert renderer.mask_head is None


def test_teacher_initialized_wrench_encoders_remain_frozen_across_stages():
    renderer = _renderer(
        phase_prediction={"enabled": True, "hidden_dim": 8, "freeze_wrench_encoder": True},
        high_attention_render={"enabled": True}, separate_gate_path=True,
    )
    assert renderer.freeze_wrench_encoder_on_init
    renderer.freeze_wrench_encoder()
    frozen_modules, frozen_parameters = renderer._wrench_conditioning_parameters()
    frozen_ids = {id(parameter) for parameter in frozen_parameters}
    for module in frozen_modules:
        frozen_ids.update(id(parameter) for parameter in module.parameters())
    for stage in ("condition_warmup", "rgb", "joint"):
        renderer.set_training_stage(stage)
        renderer.set_stage_mode(training=True)
        assert all(not parameter.requires_grad for module in frozen_modules
                   for parameter in module.parameters())
        assert all(not parameter.requires_grad for parameter in frozen_parameters)
        assert not (frozen_ids & {id(parameter) for parameter in renderer.gate_parameters()})
        assert not (frozen_ids & {id(parameter) for parameter in renderer.rgb_parameters()})
    renderer.set_training_stage("condition_warmup")
    assert any(parameter.requires_grad for parameter in renderer.gate_parameters())
    assert all(not parameter.requires_grad for parameter in renderer.rgb_parameters())
