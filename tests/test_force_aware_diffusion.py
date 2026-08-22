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
    ContactAwareImageMasker,
    ContactDetector,
    MaskProbabilityScheduler,
    VectorThresholdGate,
)
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
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


def make_obs_encoder(n_emb=16, temporal_encoder="gru", image_keys=None):
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
        backbone=FakeDinoBackbone(hidden_size=12),
    )


def make_normalizer():
    normalizer = LinearNormalizer()
    normalizer["wrench_ext"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-20.0, 20.0, 600).reshape(100, 6),
        last_n_dims=1,
    )
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-2.0, 2.0, 700).reshape(100, 7),
        last_n_dims=1,
    )
    return normalizer


def make_policy(
    mask_scope="full_context",
    relative_pose_actions=False,
    image_keys=("wrist",),
):
    encoder = make_obs_encoder(image_keys=image_keys)
    expert = ContextTransformerForDiffusion(
        input_dim=7,
        output_dim=7,
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
    policy = ForceAwareDiffusionTransformerPolicy(
        shape_meta={
            "obs": {
                **{
                    key: {"shape": [3, 16, 16], "type": "rgb"}
                    for key in image_keys
                },
                "wrench_ext": {"shape": [8, 6], "type": "low_dim"},
            },
            "action": {"shape": [7]},
        },
        noise_scheduler=scheduler,
        horizon=9,
        n_action_steps=8,
        n_obs_steps=2,
        wrench_key="wrench_ext",
        relative_pose_actions=relative_pose_actions,
        num_inference_steps=2,
        n_emb=16,
        contact_threshold=5.0,
        image_mask_scope=mask_scope,
        mask_schedule={
            "schedule_type": "cosine",
            "start_step": 0,
            "end_step": 100,
            "start_probability": 1.0,
            "end_probability": 0.0,
        },
        obs_encoder=encoder,
        action_expert=expert,
    )
    policy.set_normalizer(make_normalizer())
    return policy


def test_contact_detector_uses_physical_wrench_units():
    wrench = torch.zeros(2, 2, 8, 6)
    wrench[0, -1, -2, 0] = 5.1
    wrench[1, -1, -2, 0] = 4.9
    contact = ContactDetector(5.0, history_reducer="max")(wrench)
    assert contact.tolist() == [[False, True], [False, False]]


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


@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "exponential"])
def test_mask_scheduler_has_exact_endpoints(schedule_type):
    scheduler = MaskProbabilityScheduler(
        schedule_type=schedule_type,
        start_step=10,
        end_step=110,
        start_probability=0.9,
        end_probability=0.1,
    )
    assert scheduler.probability(0) == pytest.approx(0.9)
    assert scheduler.probability(10) == pytest.approx(0.9)
    assert 0.1 < scheduler.probability(60) < 0.9
    assert scheduler.probability(110) == pytest.approx(0.1)
    assert scheduler.probability(1000) == pytest.approx(0.1)


def test_piecewise_scheduler_and_mask_scopes():
    scheduler = MaskProbabilityScheduler(
        schedule_type="piecewise",
        piecewise_steps=[0, 10, 20],
        piecewise_probabilities=[1.0, 0.5, 0.0],
    )
    assert scheduler.probability(5) == pytest.approx(0.75)
    contact = torch.tensor([[False, True], [True, False]])
    current = ContactAwareImageMasker("current_observation")(
        contact, probability=1.0
    )
    full = ContactAwareImageMasker("full_context")(contact, probability=1.0)
    assert current.tolist() == [[False, True], [False, False]]
    assert full.tolist() == [[True, True], [False, False]]


@pytest.mark.parametrize("temporal_encoder", ["gru", "lstm", "transformer", "none"])
def test_force_encoder_returns_eight_tokens_per_observation(temporal_encoder):
    encoder = make_obs_encoder(temporal_encoder=temporal_encoder)
    encoder.train()
    context = encoder(
        image=torch.rand(2, 2, 3, 16, 16),
        wrench_history=torch.randn(2, 2, 8, 6),
        image_token_mask=torch.tensor([[False, True], [False, False]]),
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
        image_token_mask=torch.tensor([[False, True], [False, False]]),
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


def test_policy_loss_masks_training_contact_and_backpropagates():
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
    assert policy.last_curriculum_metrics["curriculum/masked_image_fraction"] == pytest.approx(0.5)
    loss.backward()
    assert policy.model.output_head.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in policy.obs_encoder.backbone.parameters()
    )


def test_policy_prediction_is_unmasked_and_returns_eight_action_steps():
    policy = make_policy()
    policy.eval()
    obs = {
        "wrist": torch.rand(2, 2, 3, 16, 16),
        "wrench_ext": torch.zeros(2, 2, 8, 6),
    }
    obs["wrench_ext"][:, -1, -1, 0] = 10.0
    result = policy.predict_action(obs, generator=torch.Generator().manual_seed(1))
    assert result["action_pred"].shape == (2, 9, 7)
    assert result["action"].shape == (2, 8, 7)
    assert "action_target" not in result
    assert torch.allclose(
        torch.linalg.vector_norm(result["action"][..., 3:7], dim=-1),
        torch.ones(2, 8),
        atol=1e-5,
    )
    assert policy.last_curriculum_metrics["curriculum/masked_image_fraction"] == 0.0


def test_pose_quaternion_normalization_has_identity_fallback():
    action = torch.zeros(1, 2, 7)
    action[0, 1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    normalized = normalize_pose_quaternions(action)
    assert torch.allclose(
        normalized[..., 3:7],
        torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]),
    )


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
