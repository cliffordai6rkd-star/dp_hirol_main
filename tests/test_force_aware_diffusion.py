from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.common.pose_util import (
    absolute_pose_to_relative_pose,
    relative_pose_to_absolute_pose,
)
from diffusion_policy.model.diffusion.context_transformer_for_diffusion import (
    ContextTransformerForDiffusion,
)
from diffusion_policy.model.vision.contact_curriculum import (
    FactrBlurScheduler,
    ContactDetector,
    HysteresisContactDetector,
    VectorThresholdGate,
    gaussian_2d_smoothing,
    select_contact_blur_mask,
)
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
    fill_future_wrench_prefix,
    normalize_pose_quaternions,
)


class FakeDinoBackbone(nn.Module):
    def __init__(self, hidden_size=16, num_register_tokens=2, patch_count=6):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
        )
        self.patch_count = patch_count
        self.projection = nn.Linear(3, hidden_size)
        self.forward_calls = 0

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        self.forward_calls += 1
        pooled = pixel_values.mean(dim=(-2, -1))
        cls = self.projection(pooled).unsqueeze(1)
        registers = torch.zeros(
            pixel_values.shape[0],
            self.config.num_register_tokens,
            self.config.hidden_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        patches = self.projection(pooled).unsqueeze(1).expand(
            -1, self.patch_count, -1
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat([cls, registers, patches], dim=1)
        )


def make_obs_encoder(
    n_emb=16,
    temporal_encoder="gru",
    image_keys=None,
    camera_attention_mode="joint",
    pre_cross_self_attn_layers=0,
):
    return ForceAwareObsEncoder(
        pretrained_model_name_or_path="unused",
        n_emb=n_emb,
        n_head=4,
        max_obs_steps=2,
        wrench_dim=6,
        wrench_history_steps=8,
        force_temporal_encoder=temporal_encoder,
        force_encoder_layers=1,
        dropout=0.0,
        freeze_backbone=True,
        image_keys=image_keys,
        camera_attention_mode=camera_attention_mode,
        pre_cross_self_attn_layers=pre_cross_self_attn_layers,
        backbone=FakeDinoBackbone(hidden_size=12),
    )


def make_normalizer(wrench_keys=("wrench_ext",), action_dim=7):
    normalizer = LinearNormalizer()
    for wrench_key in wrench_keys:
        normalizer[wrench_key] = SingleFieldLinearNormalizer.create_fit(
            torch.linspace(-20.0, 20.0, 600).reshape(100, 6),
            last_n_dims=1,
        )
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-2.0, 2.0, 100 * action_dim).reshape(100, action_dim),
        last_n_dims=1,
    )
    return normalizer


def make_policy(
    blur_scope="full_context",
    relative_pose_actions=False,
    image_keys=("wrist",),
    future_phase=False,
    pre_cross_self_attn_layers=0,
    image_blur_schedule=None,
    action_dim=7,
    action_representation="ee_pose",
):
    encoder = make_obs_encoder(
        image_keys=image_keys, pre_cross_self_attn_layers=pre_cross_self_attn_layers
    )
    expert = ContextTransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=9,
        n_emb=16,
        n_head=4,
        n_layer=1,
        n_cond_layers=1,
        max_context_tokens=16,
        p_drop_attn=0.0,
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    obs_meta = {
        **{
            key: {"shape": [3, 16, 16], "type": "rgb"}
            for key in image_keys
        },
        "wrench_ext": {"shape": [8, 6], "type": "low_dim"},
    }
    policy_kwargs = {"wrench_key": "wrench_ext"}
    if future_phase:
        obs_meta = {
            **{
                key: {"shape": [3, 16, 16], "type": "rgb"}
                for key in image_keys
            },
            "tau_ext": {"shape": [8, 7], "type": "low_dim"},
            "future_wrench": {"shape": [8, 6], "type": "low_dim"},
            "future_tau_ext": {"shape": [8, 7], "type": "low_dim"},
        }
        policy_kwargs = {
            "wrench_key": "future_wrench",
            "phase_wrench_key": "tau_ext",
            "future_phase_wrench_key": "future_tau_ext",
            "free_threshold": 1.0,
        }
    policy_contact_threshold = 2.0 if future_phase else 5.0
    policy = ForceAwareDiffusionTransformerPolicy(
        shape_meta={
            "obs": obs_meta,
            "action": {"shape": [action_dim]},
        },
        noise_scheduler=scheduler,
        horizon=9,
        n_action_steps=8,
        n_obs_steps=2,
        relative_pose_actions=relative_pose_actions,
        action_representation=action_representation,
        num_inference_steps=2,
        n_emb=16,
        contact_threshold=policy_contact_threshold,
        image_blur_scope=blur_scope,
        image_blur_schedule=image_blur_schedule,
        obs_encoder=encoder,
        action_expert=expert,
        **policy_kwargs,
    )
    policy.set_normalizer(
        make_normalizer(("wrench_ext", "future_wrench") if future_phase else ("wrench_ext",), action_dim)
    )
    return policy


def test_contact_detector_uses_physical_wrench_units():
    wrench = torch.zeros(2, 2, 8, 6)
    wrench[0, -1, -2, 0] = 5.1
    wrench[1, -1, -2, 0] = 4.9
    contact = ContactDetector(5.0, history_reducer="max")(wrench)
    assert contact.tolist() == [[False, True], [False, False]]


def test_hysteresis_contact_detector_holds_contact_until_free_threshold():
    detector = HysteresisContactDetector(
        free_threshold=1.0,
        contact_threshold=2.0,
        force_dims=(0,),
        history_reducer="last",
    )
    signal = torch.tensor(
        [[[[2.2], [0.0]], [[1.5], [2.2]], [[1.5], [1.5]], [[0.8], [0.8]]]]
    )
    assert detector(signal).tolist() == [[False, True, True, False]]
    boundary_signal = torch.tensor(
        [[[[2.1]], [[1.0]], [[0.99]]]]
    )
    assert detector(boundary_signal).tolist() == [[True, True, False]]


def test_hysteresis_contact_detector_supports_l1_norm_and_future_rollout():
    detector = HysteresisContactDetector(
        free_threshold=1.0,
        contact_threshold=1.5,
        force_dims=(0, 1),
        history_reducer="last",
        norm="l1",
    )
    # L1=1.6 enters contact while L2 would still be below 1.5.
    current = torch.tensor([[[[0.8, 0.8]]]])
    assert detector(current).tolist() == [[True]]

    future = torch.tensor(
        [[[[0.2, 0.2], [0.4, 0.4], [0.9, 0.8], [1.2, 0.8]]]]
    )
    rollout = detector.rollout_future(future, initial_state=torch.tensor([[False]]))
    assert rollout.tolist() == [[[False, False, True, True]]]


def test_hysteresis_contact_detector_requires_five_consecutive_frames_for_causal_scan():
    detector = HysteresisContactDetector(
        free_threshold=8.0,
        contact_threshold=8.0,
        force_dims=(0,),
        norm="l2",
        enter_frames=5,
        exit_frames=5,
    )
    wrench = torch.tensor(
        [[[[9.0], [9.0], [9.0], [9.0], [9.0],
           [7.0], [7.0], [7.0], [7.0], [7.0]]]]
    )
    # The causal scan keeps the configured debounce.  Future windows use a
    # horizon-aware suffix rule tested below instead.
    states = detector._scan_tokens(wrench.reshape(1, -1), initial_state=torch.tensor([False]))
    assert states.tolist() == [[False, False, False, False, True,
                                True, True, True, True, False]]


def test_future_rollout_accepts_late_three_frame_contact_without_chatter():
    detector = HysteresisContactDetector(
        free_threshold=8.0,
        contact_threshold=8.0,
        force_dims=(0,),
        norm="l2",
        enter_frames=5,
        exit_frames=5,
    )
    # Contact begins at the sixth of eight future samples.  Only the final
    # three samples remain, so a fixed five-frame debounce would erase it.
    future = torch.zeros(1, 1, 8, 1)
    future[..., 5:, 0] = 9.0
    states = detector.rollout_future(future, initial_state=torch.tensor([[False]]))
    assert states.tolist() == [[[False, False, False, False, False, True, True, True]]]

    # A short early spike still does not enter contact, preserving debounce
    # against noise and repeated state switching.
    noisy = torch.zeros(1, 1, 8, 1)
    noisy[..., 1, 0] = 9.0
    noisy_states = detector.rollout_future(noisy, initial_state=torch.tensor([[False]]))
    assert noisy_states.count_nonzero() == 0


def test_future_wrench_prefix_uses_first_contact_vector():
    raw_wrench = torch.tensor(
        [[[[1.0, -0.5], [2.0, -1.0], [3.0, -1.5], [4.0, -2.0]]]]
    )
    future_contact = torch.tensor([[[False, False, True, True]]])
    filled, has_contact = fill_future_wrench_prefix(raw_wrench, future_contact)
    expected = torch.tensor(
        [[[[3.0, -1.5], [3.0, -1.5], [3.0, -1.5], [4.0, -2.0]]]]
    )
    assert torch.equal(filled, expected)
    assert has_contact.tolist() == [[True]]

    free_only = torch.zeros_like(future_contact)
    filled_free, has_free_contact = fill_future_wrench_prefix(
        raw_wrench, free_only
    )
    assert torch.equal(filled_free, torch.zeros_like(raw_wrench))
    assert has_free_contact.tolist() == [[False]]


def test_render_net_receives_exact_teacher_wrench_and_preprocessor_roundtrips():
    teacher = make_policy(future_phase=True).eval().requires_grad_(False)
    teacher.normalizer["future_wrench"] = SingleFieldLinearNormalizer.create_fit(
        torch.arange(600).reshape(100, 6).float() + 5,
    )
    obs = {"wrist": torch.rand(4, 2, 3, 16, 16), "tau_ext": torch.zeros(4, 2, 8, 7),
           "future_tau_ext": torch.zeros(4, 2, 8, 7),
           "future_wrench": torch.arange(4 * 2 * 8 * 6).reshape(4, 2, 8, 6).float()}
    # Free window, future contact with prefix, latched contact in the hysteresis
    # band, and future contact followed by release.
    obs["future_tau_ext"][1, :, 3:, 0] = 3.0
    obs["tau_ext"][2, :, :, 0] = 3.0
    obs["future_tau_ext"][2, :, :, 0] = 1.5
    obs["future_tau_ext"][3, :, :3, 0] = 3.0
    captured = {}

    def capture_teacher(module, args, kwargs):
        captured["teacher"] = kwargs["wrench_history"].clone()

    handle = teacher.obs_encoder.register_forward_pre_hook(capture_teacher, with_kwargs=True)
    try:
        with torch.no_grad():
            teacher._encode_context(obs, apply_curriculum_mask=False)
    finally:
        handle.remove()
    renderer = RenderNet(6, base_channels=4, conditioning_dim=8)
    handle = renderer.register_forward_pre_hook(
        lambda module, args: captured.update(student=args[1].detach().clone()))
    try:
        student_context_tokens(teacher, renderer, obs)
    finally:
        handle.remove()
    torch.testing.assert_close(captured["student"], captured["teacher"], rtol=0, atol=0)
    assert torch.count_nonzero(captured["student"][0]) == 0
    assert torch.count_nonzero(teacher.normalizer["future_wrench"].normalize(
        torch.zeros_like(obs["future_wrench"][0]))) > 0
    expected_prefix = obs["future_wrench"][1, :, 3:4].expand(-1, 4, -1)
    torch.testing.assert_close(captured["student"][1, :, :4],
                               teacher.normalizer["future_wrench"].normalize(expected_prefix), rtol=0, atol=0)
    torch.testing.assert_close(captured["student"][2], teacher.normalizer["future_wrench"].normalize(
        obs["future_wrench"][2]), rtol=0, atol=0)
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    restored = WrenchPreprocessor.from_config(preprocessor.get_config())
    restored.load_state_dict(preprocessor.state_dict(), strict=True)
    torch.testing.assert_close(restored(obs)[0], captured["teacher"], rtol=0, atol=0)
    obs["future_tau_ext"].zero_()
    obs["tau_ext"].zero_()
    assert torch.count_nonzero(restored(obs)[0]) == 0


def test_vector_threshold_gate_zeroes_complete_tau_vectors_and_is_serialized():
    gate = VectorThresholdGate(threshold=1.0, norm="l1")
    tau = torch.tensor(
        [[[[0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.19, 0.19, 0.19, 0.19, 0.19, 0.0, 0.0],
           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]
    )
    gated = gate(tau)
    assert torch.equal(gated[0, 0, 0], torch.zeros(7))
    assert torch.equal(gated[0, 0, 1], torch.zeros(7))
    assert torch.equal(gated[0, 0, 2], tau[0, 0, 2])
    assert "threshold" in gate.state_dict()
    assert "enabled" in gate.state_dict()
    assert "norm_code" in gate.state_dict()


def test_policy_freeze_selectors_exclude_cross_attention_from_optimizer():
    policy = make_policy()
    policy.set_freeze_modules(["cross_attn"])
    cross_ids = {
        id(parameter) for parameter in policy.obs_encoder.cross_attention.parameters()
    }
    assert cross_ids
    assert all(
        not parameter.requires_grad
        for parameter in policy.obs_encoder.cross_attention.parameters()
    )
    assert any(
        parameter.requires_grad for parameter in policy.obs_encoder.cross_ffn.parameters()
    )
    optimizer = policy.get_optimizer(learning_rate=1.0e-4)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert cross_ids.isdisjoint(optimizer_ids)
    assert policy.get_freeze_config()["freeze_modules"] == ("cross_attn",)


def test_factr_blur_scheduler_has_fixed_quarter_boundaries():
    scheduler = FactrBlurScheduler(total_steps=200_000)
    assert scheduler.sigma(0) == pytest.approx(5.0)
    assert scheduler.sigma(49_999) == pytest.approx(5.0)
    assert scheduler.sigma(50_000) == pytest.approx(5.0)
    assert scheduler.sigma(100_000) == pytest.approx(2.5)
    assert scheduler.sigma(149_999) > 0
    assert scheduler.sigma(150_000) == pytest.approx(0.0)
    assert scheduler.sigma(200_000) == pytest.approx(0.0)
    assert scheduler.sigma(300_000) == pytest.approx(0.0)


def test_factr_blur_scheduler_scales_with_total_steps():
    scheduler = FactrBlurScheduler(total_steps=100_000)
    assert scheduler.sigma(25_000) == pytest.approx(5.0)
    assert scheduler.sigma(50_000) == pytest.approx(2.5)
    assert scheduler.sigma(75_000) == pytest.approx(0.0)
    assert scheduler.sigma(100_000) == pytest.approx(0.0)


def test_contact_blur_scope_is_deterministic():
    contact = torch.tensor([[False, True], [True, False]])
    current = select_contact_blur_mask(contact, "current_observation")
    full = select_contact_blur_mask(contact, "full_context")
    assert current.tolist() == [[False, True], [False, False]]
    assert full.tolist() == [[True, True], [False, False]]


def test_factr_gaussian_smoothing_uses_zero_padded_2d_kernel():
    images = torch.rand(2, 2, 3, 16, 24)
    blurred = gaussian_2d_smoothing(images, 1.0)
    assert blurred.shape == images.shape
    assert blurred.dtype == images.dtype
    assert gaussian_2d_smoothing(images, 0) is images
    # Zero padding is part of FACTR's implementation, so a constant image
    # loses mass at the border instead of remaining exactly constant.
    constant = torch.ones(1, 3, 16, 16)
    assert gaussian_2d_smoothing(constant, 1.0).mean() < 1.0
    impulse = torch.zeros(1, 3, 64, 64)
    impulse[:, :, 32, 32] = 1
    assert gaussian_2d_smoothing(impulse, 10).max() < gaussian_2d_smoothing(impulse, 5).max()


@pytest.mark.parametrize("temporal_encoder", ["gru", "lstm", "transformer", "none"])
def test_force_encoder_returns_eight_tokens_per_observation(temporal_encoder):
    encoder = make_obs_encoder(temporal_encoder=temporal_encoder)
    encoder.train()
    context = encoder(
        image=torch.rand(2, 2, 3, 16, 16),
        wrench_history=torch.randn(2, 2, 8, 6),
    )
    assert context.shape == (2, 16, 16)
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    context.sum().backward()
    assert all(parameter.grad is None for parameter in encoder.backbone.parameters())
    assert encoder.wrench_projection.weight.grad is not None


def test_force_encoder_encodes_every_configured_camera():
    encoder = make_obs_encoder(image_keys=("wrist", "side"))
    context = encoder(
        images={
            "wrist": torch.rand(2, 2, 3, 16, 16),
            "side": torch.rand(2, 2, 3, 16, 16),
        },
        wrench_history=torch.randn(2, 2, 8, 6),
    )
    assert encoder.image_keys == ("side", "wrist")
    assert encoder.backbone.forward_calls == 1
    assert context.shape == (2, 16, 16)


def test_force_encoder_groups_cameras_with_different_resolutions():
    encoder = make_obs_encoder(image_keys=("wrist", "side"))
    context = encoder(
        images={
            "wrist": torch.rand(1, 2, 3, 16, 16),
            "side": torch.rand(1, 2, 3, 12, 20),
        },
        wrench_history=torch.randn(1, 2, 8, 6),
    )
    assert encoder.backbone.forward_calls == 2
    assert context.shape == (1, 16, 16)


def test_force_encoder_per_camera_attention_has_independent_softmaxes():
    encoder = make_obs_encoder(
        image_keys=("wrist", "side"),
        camera_attention_mode="per_camera",
    )
    encoder.set_attention_capture(True)
    context = encoder(
        images={
            "wrist": torch.rand(1, 2, 3, 16, 16),
            "side": torch.rand(1, 2, 3, 16, 16),
        },
        wrench_history=torch.randn(1, 2, 8, 6),
    )

    assert context.shape == (1, 16, 16)
    assert set(encoder.last_cross_attention_weights) == {"side", "wrist"}
    for weights in encoder.last_cross_attention_weights.values():
        assert weights.shape == (2, 4, 8, 6)
        assert torch.allclose(
            weights.sum(dim=-1),
            torch.ones(2, 4, 8),
            atol=1e-5,
        )
    context.sum().backward()
    assert encoder.camera_fusion.weight.grad is not None


@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
def test_pre_cross_self_attention_splits_updated_image_and_wrench_tokens(camera_mode):
    encoder = make_obs_encoder(
        image_keys=("wrist", "side"), camera_attention_mode=camera_mode,
        pre_cross_self_attn_layers=1,
    ).eval()
    encoder.set_attention_capture(True)
    seen = {}
    cross_inputs = []

    def remember_force(module, args, output):
        seen["gru"] = output[0].detach().clone()

    def remember_joint_input(module, args):
        seen["input"] = args[0].detach().clone()

    def remember_joint_output(module, args, output):
        seen["output"] = output.detach().clone()

    def remember_cross(module, args, kwargs):
        cross_inputs.append({key:kwargs[key].detach().clone() for key in ("query", "key", "value")})

    handles = [
        encoder.force_encoder.register_forward_hook(remember_force),
        encoder.pre_cross_self_attention.register_forward_pre_hook(remember_joint_input),
        encoder.pre_cross_self_attention.register_forward_hook(remember_joint_output),
        encoder.cross_attention.register_forward_pre_hook(remember_cross, with_kwargs=True),
    ]
    try:
        context = encoder(
            images={key: torch.rand(2, 2, 3, 16, 16) for key in ("side", "wrist")},
            wrench_history=torch.randn(2, 2, 8, 6),
        )
    finally:
        for handle in handles:
            handle.remove()
    # Independently for each batch/observation: side 0:6, wrist 6:12,
    # wrench 12:20. The GRU runs before joint self-attention.
    assert seen["input"].shape == (4, 20, 16)
    torch.testing.assert_close(seen["input"][:, 12:], seen["gru"])
    assert not torch.allclose(seen["output"], seen["input"])
    expected_query = encoder.force_norm(seen["output"][:, 12:])
    assert len(cross_inputs) == (1 if camera_mode == "joint" else 2)
    for index, values in enumerate(cross_inputs):
        image_slice = slice(0, 12) if camera_mode == "joint" else slice(index * 6, (index + 1) * 6)
        expected_images = seen["output"][:, image_slice]
        torch.testing.assert_close(values["query"], expected_query)
        torch.testing.assert_close(values["key"], encoder.image_norm(expected_images))
        torch.testing.assert_close(values["value"], expected_images)
    assert context.shape == (2, 16, 16)
    weights = encoder.last_cross_attention_weights
    maps = [weights] if camera_mode == "joint" else list(weights.values())
    for values in maps:
        torch.testing.assert_close(values.sum(-1), torch.ones(4, 4, 8))
    (context * torch.randn_like(context)).mean().backward()
    assert all(p.grad is not None for p in encoder.pre_cross_self_attention.parameters())
    assert encoder.force_encoder.weight_ih_l0.grad is not None


@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
def test_pre_cross_self_attention_keeps_image_inputs(camera_mode):
    encoder = make_obs_encoder(
        image_keys=("wrist", "side"), camera_attention_mode=camera_mode,
        pre_cross_self_attn_layers=1,
    ).eval()
    wrench = torch.randn(1, 2, 8, 6)
    images = {key: torch.rand(1, 2, 3, 16, 16) for key in ("side", "wrist")}
    changed = {key: value + 1.0 for key, value in images.items()}
    with torch.inference_mode():
        first = encoder(images=images, wrench_history=wrench)
        same = encoder(images=images, wrench_history=wrench)
        second = encoder(images=changed, wrench_history=wrench)
    torch.testing.assert_close(first, same, rtol=0, atol=0)
    assert not torch.allclose(first, second)


def test_pre_cross_self_attention_trains_and_checkpoint_restores():
    policy = make_policy(image_keys=("side", "wrist"), pre_cross_self_attn_layers=2)
    batch = {
        "obs": {
            "side":torch.rand(2, 2, 3, 16, 16), "wrist":torch.rand(2, 2, 3, 16, 16),
            "wrench_ext":torch.randn(2, 2, 8, 6),
        },
        "action":torch.randn(2, 9, 7),
    }
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)
    restored = make_policy(image_keys=("side", "wrist"), pre_cross_self_attn_layers=2)
    restored.load_state_dict(policy.state_dict(), strict=True)
    policy.eval()
    restored.eval()
    with torch.inference_mode():
        first = policy.predict_action(batch["obs"], generator=torch.Generator().manual_seed(42))
        second = restored.predict_action(batch["obs"], generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first["action"], second["action"])
    restored.apply_freeze_config({"freeze_modules": ["cross_fusion"]})
    assert all(not p.requires_grad for p in restored.obs_encoder.pre_cross_self_attention.parameters())


def test_zero_pre_cross_layers_has_no_new_checkpoint_parameters():
    encoder = make_obs_encoder(image_keys=("side", "wrist"))
    assert isinstance(encoder.pre_cross_self_attention, nn.Identity)
    assert not any(key.startswith("pre_cross_self_attention.") for key in encoder.state_dict())


def test_pre_cross_layer_count_cannot_be_negative():
    with pytest.raises(ValueError, match="pre_cross_self_attn_layers must be non-negative"):
        make_obs_encoder(pre_cross_self_attn_layers=-1)


def test_multi_camera_policy_discovers_rgb_keys_from_shape_meta():
    policy = make_policy(image_keys=("wrist", "side"))
    assert "input_gate.threshold" in policy.state_dict()
    assert "input_gate.enabled" in policy.state_dict()
    assert "input_gate.norm_code" in policy.state_dict()
    assert all(
        parameter.numel() > 0
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    policy.train()
    batch = {
        "obs": {
            "wrist": torch.rand(2, 2, 3, 16, 16),
            "side": torch.rand(2, 2, 3, 16, 16),
            "wrench_ext": torch.zeros(2, 2, 8, 6),
        },
        "action": torch.randn(2, 9, 7),
    }
    # Exercise nn.Module.forward, which is the path used by DDP.
    loss = policy(batch, optimizer_step=0)
    assert policy.image_keys == ("side", "wrist")
    assert loss.ndim == 0 and torch.isfinite(loss)

    del batch["obs"]["side"]
    with pytest.raises(KeyError, match="side"):
        policy.compute_loss(batch, optimizer_step=0)


def test_context_transformer_output_shape():
    model = ContextTransformerForDiffusion(
        input_dim=7,
        output_dim=7,
        horizon=8,
        n_emb=16,
        n_head=4,
        n_layer=1,
        max_context_tokens=16,
    )
    output = model(
        sample=torch.randn(3, 8, 7),
        timestep=torch.tensor([1, 2, 3]),
        context=torch.randn(3, 16, 16),
    )
    assert output.shape == (3, 8, 7)


def test_policy_loss_trains_contact_and_backpropagates():
    policy = make_policy()
    policy.train()
    batch = {
        "obs": {
            "wrist": torch.rand(2, 2, 3, 16, 16),
            "wrench_ext": torch.zeros(2, 2, 8, 6),
        },
        "action": torch.randn(2, 9, 7),
    }
    batch["obs"]["wrench_ext"][0, -1, -1, 0] = 10.0
    loss = policy.compute_loss(batch, optimizer_step=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert policy.last_curriculum_metrics["curriculum/contact_fraction"] == pytest.approx(0.25)
    assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == pytest.approx(0.0)
    loss.backward()
    assert policy.model.output_head.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in policy.obs_encoder.backbone.parameters()
    )


def test_policy_blur_uses_future_contact_for_future_phase_windows():
    policy = make_policy(
        future_phase=True,
        image_blur_schedule={
            "total_steps": 100,
            "start_sigma": 3,
            "end_sigma": 0,
            "start_fraction": 0.25,
            "end_fraction": 0.75,
        },
    )
    policy.train()
    batch = {
        "obs": {
            "wrist": torch.rand(2, 2, 3, 16, 16),
            "tau_ext": torch.zeros(2, 2, 8, 7),
            "future_wrench": torch.zeros(2, 2, 8, 6),
            "future_tau_ext": torch.zeros(2, 2, 8, 7),
        },
        "action": torch.randn(2, 9, 7),
    }
    # Both observations in the first sample are currently free, but their
    # future window reaches contact.  The second sample remains free.
    batch["obs"]["future_tau_ext"][0, :, :, 0] = 3.0
    loss = policy.compute_loss(batch, optimizer_step=0)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert policy.last_curriculum_metrics["curriculum/contact_fraction"] == pytest.approx(0.0)
    assert policy.last_curriculum_metrics["curriculum/future_contact_fraction"] == pytest.approx(0.5)
    # full_context blurs both observation images for only the first sample.
    assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == pytest.approx(0.5)


def test_policy_blur_uses_future_contact_and_preserves_free_rgb():
    policy = make_policy(future_phase=True, image_blur_schedule={
        "total_steps": 100, "start_sigma": 3, "end_sigma": 0,
        "start_fraction": 0.25, "end_fraction": 0.75,
    }).train()
    obs = {
        "wrist": torch.rand(2, 2, 3, 16, 16),
        "tau_ext": torch.zeros(2, 2, 8, 7),
        "future_wrench": torch.zeros(2, 2, 8, 6),
        "future_tau_ext": torch.zeros(2, 2, 8, 7),
    }
    obs["future_tau_ext"][0, :, :, 0] = 3
    captured = []
    handle = policy.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs["images"]["wrist"].detach().clone()),
        with_kwargs=True,
    )
    try:
        policy._encode_context(obs, apply_curriculum_mask=True)
    finally:
        handle.remove()
    assert not torch.equal(captured[0][0], obs["wrist"][0])
    torch.testing.assert_close(captured[0][1], obs["wrist"][1], rtol=0, atol=0)
    assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == 0.5


@pytest.mark.parametrize("action_dim", [7, 8])
def test_policy_prediction_is_clear_and_returns_eight_action_steps(action_dim):
    policy = make_policy(action_dim=action_dim)
    policy.eval()
    obs = {
        "wrist": torch.rand(2, 2, 3, 16, 16),
        "wrench_ext": torch.zeros(2, 2, 8, 6),
    }
    obs["wrench_ext"][:, -1, -1, 0] = 10.0
    result = policy.predict_action(obs, generator=torch.Generator().manual_seed(1))
    assert result["action_pred"].shape == (2, 9, action_dim)
    assert result["action"].shape == (2, 8, action_dim)
    assert "action_target" not in result
    assert torch.allclose(
        torch.linalg.vector_norm(result["action"][..., 3:7], dim=-1),
        torch.ones(2, 8),
        atol=1e-5,
    )
    assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == 0.0


def test_joint_position_prediction_preserves_joint_angles_and_gripper(monkeypatch):
    policy = make_policy(action_dim=8, action_representation="joint_position")
    policy.eval()
    # Joint angles 4--7 must not be projected onto a unit quaternion.
    target = torch.tensor([0.1, -0.4, 0.2, -2.4, 0.3, 1.8, 0.7, -1.0]).repeat(2, 9, 1)
    normalized = policy.normalizer["action"].normalize(target)
    monkeypatch.setattr(policy, "conditional_sample", lambda *args, **kwargs: normalized)
    obs = {
        "wrist": torch.rand(2, 2, 3, 16, 16),
        "wrench_ext": torch.zeros(2, 2, 8, 6),
    }
    result = policy.predict_action(obs)
    torch.testing.assert_close(result["model_action_pred"], target)
    torch.testing.assert_close(result["action"], target[:, 1:9])


def test_joint_position_actions_cannot_use_relative_pose_transform():
    with pytest.raises(ValueError, match="relative_pose_actions requires action_representation"):
        make_policy(relative_pose_actions=True, action_representation="joint_position")


@pytest.mark.parametrize("action_dim", [7, 8])
def test_pose_quaternion_normalization_has_identity_fallback(action_dim):
    action = torch.zeros(1, 2, action_dim)
    if action_dim == 8:
        action[..., 7] = torch.tensor([-1.0, 1.0])
    action[0, 1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    normalized = normalize_pose_quaternions(action)
    assert torch.allclose(
        normalized[..., 3:7],
        torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]),
    )
    torch.testing.assert_close(normalized[..., 7:], action[..., 7:])


def test_relative_pose_round_trip_and_quaternion_sign_invariance():
    reference = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.7071068, 0.7071068]]
    )
    target = torch.tensor(
        [[
            [1.1, 2.2, 2.9, 0.0, 0.0, 1.0, 0.0],
            [0.9, 2.0, 3.3, 0.0, 0.0, -1.0, 0.0],
        ]]
    )
    relative = absolute_pose_to_relative_pose(target, reference)
    restored = relative_pose_to_absolute_pose(relative, reference)

    assert torch.allclose(relative[0, 0, :3], torch.tensor([0.1, 0.2, -0.1]))
    assert torch.allclose(relative[0, 0, 3:7], relative[0, 1, 3:7], atol=1e-6)
    assert torch.allclose(restored[..., :3], target[..., :3], atol=1e-6)
    quaternion_similarity = torch.abs(
        (restored[..., 3:7] * target[..., 3:7]).sum(dim=-1)
    )
    assert torch.allclose(quaternion_similarity, torch.ones_like(quaternion_similarity))


def test_relative_policy_restores_absolute_prediction_for_controller():
    policy = make_policy(relative_pose_actions=True)
    policy.eval()
    relative_prediction = torch.zeros(2, 9, 7)
    relative_prediction[..., 0] = 0.05
    relative_prediction[..., 6] = 1.0
    normalized_prediction = policy.normalizer["action"].normalize(relative_prediction)
    policy.conditional_sample = lambda shape, context, generator=None: (
        normalized_prediction.to(device=context.device, dtype=context.dtype)
    )
    obs = {
        "wrist": torch.rand(2, 2, 3, 16, 16),
        "wrench_ext": torch.zeros(2, 2, 8, 6),
        "action_reference": torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    result = policy.predict_action(obs)

    assert torch.allclose(result["model_action_pred"], relative_prediction, atol=1e-5)
    assert torch.allclose(
        result["action_pred"][..., :3],
        obs["action_reference"][:, None, :3]
        + torch.tensor([0.05, 0.0, 0.0]),
        atol=1e-5,
    )
    assert result["action"].shape == (2, 8, 7)
    assert torch.allclose(
        result["action"][..., :3],
        obs["action_reference"][:, None, :3] + torch.tensor([0.05, 0.0, 0.0]),
        atol=1e-5,
    )


def test_relative_policy_requires_current_pose_reference():
    policy = make_policy(relative_pose_actions=True)
    policy.eval()
    obs = {
        "wrist": torch.rand(1, 2, 3, 16, 16),
        "wrench_ext": torch.zeros(1, 2, 8, 6),
    }
    with pytest.raises(KeyError, match="current absolute EE pose"):
        policy.predict_action(obs)
