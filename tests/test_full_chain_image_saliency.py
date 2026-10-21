from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.scripts.full_chain_image_saliency import full_chain_image_saliency


class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze_backbone = True
        self.backbone = nn.Linear(24, 2, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.arange(1, 49).reshape(2, 24) / 48)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def forward(self, image):
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(image.flatten(1))
        return self.backbone(image.flatten(1))


class RecursiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor([0.2, -0.4]))

    def forward(self, sample, timestep, context):
        return sample * self.coefficient + context


class RecursivePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_keys = ("side", "wrist")
        self.obs_encoder = FrozenEncoder()
        self.model = RecursiveModel()
        self.forward_count = 0

    def predict_action(self, obs, generator=None):
        self.forward_count += 1
        assert not obs["wrench"].requires_grad
        context = self.obs_encoder(obs["side"]) + 2 * self.obs_encoder(obs["wrist"]) + obs["wrench"]
        sample = torch.randn(context.shape, generator=generator)
        for timestep in (9, 5, 0):
            prediction = self.model(sample, timestep, context)
            sample = 0.7 * sample + 0.3 * prediction
        return {"action": sample, "extra": [sample + 1]}


def toy_obs():
    return {"side": torch.rand(1, 2, 3, 2, 2), "wrist": torch.rand(1, 2, 3, 2, 2),
            "wrench": torch.zeros(1, 2, requires_grad=True)}


def selector(policy, prediction, obs):
    return prediction["action"] * prediction["action"].new_tensor([2, 0.25])


def test_exact_full_chain_matches_analytic_recursive_jacobian_and_scales_output_coordinates():
    policy = RecursivePolicy()
    obs = toy_obs()
    result = full_chain_image_saliency(policy, obs, selector, estimator="exact",
                                       generator=torch.Generator().manual_seed(11))
    # Every earlier denoiser contributes: da/dcontext=.3*(1+A+A^2).
    # Omitting the first two calls would incorrectly give only .3.
    recurrence = 0.7 + 0.3 * policy.model.coefficient.detach()
    gain = 0.3 * (1 + recurrence + recurrence.square()) * torch.tensor([2, 0.25])
    jacobian = gain[:, None] * policy.obs_encoder.backbone.weight.detach()
    expected = (jacobian.square().sum(0).reshape(1, 2, 3, 2, 2).sum(2) / 2).sqrt()
    torch.testing.assert_close(result.maps["side"], expected)
    torch.testing.assert_close(result.maps["wrist"], expected * 2)
    assert policy.forward_count == 1
    assert result.diagnostics["num_model_calls"] == 3
    assert result.diagnostics["timesteps"] == [9, 5, 0]
    assert result.diagnostics["target_count"] == result.diagnostics["num_vjps"] == 2
    assert all(result.diagnostics["connected_cameras"].values())
    assert not result.prediction["action"].requires_grad
    assert not result.prediction["extra"][0].requires_grad
    assert all(not value.requires_grad and value.device.type == "cpu" for value in result.maps.values())


def test_scalar_output_hutchinson_is_exact_for_any_probe_sign():
    policy = RecursivePolicy()
    obs = toy_obs()
    scalar = lambda p, prediction, o: prediction["action"][..., :1]
    exact = full_chain_image_saliency(policy, obs, scalar, estimator="exact")
    for seed in (0, 2, 111):
        estimated = full_chain_image_saliency(policy, obs, scalar, num_probes=1, probe_seed=seed)
        for key in policy.image_keys:
            torch.testing.assert_close(estimated.maps[key], exact.maps[key])


def test_exact_jacobian_does_not_cancel_opposite_output_gradients():
    policy = RecursivePolicy()
    obs = toy_obs()
    scalar = lambda p, prediction, o: prediction["action"][..., :1]

    def opposite_outputs(model, prediction, inputs):
        value = scalar(model, prediction, inputs)
        return torch.cat([value, -value], dim=-1)

    one = full_chain_image_saliency(policy, obs, scalar, estimator="exact")
    opposite = full_chain_image_saliency(policy, obs, opposite_outputs, estimator="exact")
    for key in policy.image_keys:
        torch.testing.assert_close(opposite.maps[key], one.maps[key])
        assert opposite.maps[key].max() > 0


def test_policy_output_clipping_is_preserved_and_saturated_local_gradients_are_zero():
    policy = RecursivePolicy()

    def clipped_target(model, prediction, inputs):
        # This mirrors an output saturation inside a real scheduler. The
        # visualizer must measure the actual graph rather than bypass clipping.
        return (prediction["action"] + 100).clamp(-1, 1)

    result = full_chain_image_saliency(policy, toy_obs(), clipped_target, estimator="exact")
    assert all(torch.count_nonzero(value) == 0 for value in result.maps.values())
    assert all(result.diagnostics["connected_cameras"].values())


def test_hutchinson_reuses_fixed_independent_probes_without_touching_global_or_sampler_rng():
    policy = RecursivePolicy()
    obs = toy_obs()
    initial_rng = torch.random.get_rng_state().clone()
    first = full_chain_image_saliency(policy, obs, selector, num_probes=5, probe_seed=100,
                                      generator=torch.Generator().manual_seed(42))
    after_first_rng = torch.random.get_rng_state().clone()
    second = full_chain_image_saliency(policy, obs, selector, num_probes=5, probe_seed=100,
                                       generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(initial_rng, after_first_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), initial_rng, rtol=0, atol=0)
    for key in policy.image_keys:
        torch.testing.assert_close(first.maps[key], second.maps[key], rtol=0, atol=0)
    torch.testing.assert_close(first.prediction["action"], second.prediction["action"], rtol=0, atol=0)
    third = full_chain_image_saliency(policy, obs, selector, num_probes=5, probe_seed=101,
                                      generator=torch.Generator().manual_seed(42))
    # Probe RNG must not change the diffusion trajectory.
    torch.testing.assert_close(first.prediction["action"], third.prediction["action"], rtol=0, atol=0)
    assert not torch.equal(first.maps["side"], third.maps["side"])


def test_outer_no_grad_supported_original_inputs_gradients_flags_and_prediction_preserved():
    policy = RecursivePolicy().train()
    policy.obs_encoder.backbone.eval()
    policy.model.coefficient.grad = torch.tensor([3.0, 4.0])
    original_grad = policy.model.coefficient.grad
    obs = toy_obs()
    obs["side"].requires_grad_(True)
    obs["side"].grad = torch.ones_like(obs["side"])
    image_grad = obs["side"].grad
    state = {key: value.clone() for key, value in policy.state_dict().items()}
    flags = [module.training for module in policy.modules()]
    requires = [parameter.requires_grad for parameter in policy.parameters()]
    with torch.no_grad():
        ordinary = policy.predict_action({key: value.detach() for key, value in obs.items()},
                                         generator=torch.Generator().manual_seed(42))
        result = full_chain_image_saliency(policy, obs, selector, estimator="exact",
                                           generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(result.prediction["action"], ordinary["action"], rtol=0, atol=0)
    assert policy.model.coefficient.grad is original_grad
    assert obs["side"].grad is image_grad
    assert obs["wrench"].grad is None
    assert policy.obs_encoder.freeze_backbone is True
    assert [module.training for module in policy.modules()] == flags
    assert [parameter.requires_grad for parameter in policy.parameters()] == requires
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert not policy.model._forward_pre_hooks


def test_selecting_one_camera_keeps_other_inputs_constant_and_matches_exact_full_selection():
    policy = RecursivePolicy()
    obs = toy_obs()
    both = full_chain_image_saliency(policy, obs, selector, estimator="exact")
    side = full_chain_image_saliency(policy, obs, selector, image_keys=["side"], estimator="exact")
    assert set(side.maps) == {"side"}
    torch.testing.assert_close(side.maps["side"], both.maps["side"])


@pytest.mark.parametrize("failure", ["selector", "forward", "detached"])
def test_failure_restores_flags_parameters_hooks_and_existing_gradients(failure):
    policy = RecursivePolicy().train()
    policy.obs_encoder.backbone.eval()
    flags = [module.training for module in policy.modules()]
    requires = [parameter.requires_grad for parameter in policy.parameters()]
    policy.model.coefficient.grad = torch.tensor([6.0, 7.0])
    original_grad = policy.model.coefficient.grad
    if failure == "forward":
        def broken_forward(*args, **kwargs):
            raise RuntimeError("forward failed")
        policy.model.forward = broken_forward
        selected = selector
    elif failure == "selector":
        def selected(*args):
            raise RuntimeError("selector failed")
    else:
        selected = lambda p, prediction, o: prediction["action"].detach()
    with pytest.raises(RuntimeError):
        full_chain_image_saliency(policy, toy_obs(), selected)
    assert policy.obs_encoder.freeze_backbone is True
    assert [module.training for module in policy.modules()] == flags
    assert [parameter.requires_grad for parameter in policy.parameters()] == requires
    assert policy.model.coefficient.grad is original_grad
    assert not policy.model._forward_pre_hooks


@pytest.mark.parametrize("kwargs,match", [
    ({"estimator": "unknown"}, "estimator"), ({"num_probes": 0}, "num_probes"),
    ({"num_probes": 1.5}, "integers"), ({"image_keys": []}, "distinct"),
    ({"image_keys": ["side", "side"]}, "distinct"), ({"image_keys": ["absent"]}, "Missing"),
])
def test_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        full_chain_image_saliency(RecursivePolicy(), toy_obs(), selector, **kwargs)


def test_inference_mode_rejected_explicitly():
    with torch.inference_mode(), pytest.raises(RuntimeError, match="leave torch.inference_mode"):
        full_chain_image_saliency(RecursivePolicy(), toy_obs(), selector)


class PatchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, patch_size=2, num_register_tokens=0)
        self.patches = nn.Conv2d(3, 4, kernel_size=2, stride=2)

    def forward(self, pixel_values, **kwargs):
        patches = self.patches(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat([patches.mean(1, keepdim=True), patches], 1))


def test_real_force_aware_frozen_dino_lstm_dit_and_ddim_chain_are_differentiable():
    torch.manual_seed(31)
    shapes = {"side": (3, 4, 4), "wrist": (3, 4, 4)}
    encoder = ForceAwareObsEncoder(
        "unused", image_shapes=shapes, backbone=PatchBackbone(), n_emb=4, n_head=2,
        max_obs_steps=2, wrench_history_steps=2, force_temporal_encoder="lstm",
        force_encoder_layers=1, cross_attn_query="image", cross_attn_layers=1,
        dropout=0.0, freeze_backbone=True,
    )
    policy = ForceAwareDiffusionTransformerPolicy(
        shape_meta={"obs": {**{key: {"shape": shape, "type": "rgb"} for key, shape in shapes.items()},
                             "wrench_ext": {"shape": (2, 6), "type": "low_dim"}},
                    "action": {"shape": (7,)}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=12, prediction_type="epsilon", clip_sample=False),
        horizon=3, n_obs_steps=2, n_action_steps=2, num_inference_steps=3, wrench_history_steps=2,
        obs_encoder=encoder, n_emb=4, n_head=2, n_layer=1, n_cond_layers=1,
        p_drop_attn=0.0,
    ).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(32, 7), "wrench_ext": torch.randn(32, 6)})
    policy.set_normalizer(normalizer)
    obs = {key: torch.rand(1, 2, *shape) for key, shape in shapes.items()}
    obs["wrench_ext"] = torch.randn(1, 2, 2, 6)

    def normalized_executed_target(model, prediction, inputs):
        start = model.action_start_index
        target = prediction["model_action_pred"][:, start:start + model.n_action_steps]
        return model.normalizer["action"].normalize(target)

    with torch.no_grad():
        expected = policy.predict_action(obs, generator=torch.Generator().manual_seed(7))
    result = full_chain_image_saliency(policy, obs, normalized_executed_target, num_probes=2,
                                       generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(result.prediction["action"], expected["action"], rtol=2e-4, atol=2e-5)
    assert result.diagnostics["num_model_calls"] == 3
    assert result.diagnostics["target_count"] == 14
    for value in result.maps.values():
        assert value.shape == (1, 2, 4, 4)
        assert torch.isfinite(value).all() and value.max() > 0
    assert policy.obs_encoder.freeze_backbone is True
    assert not policy.obs_encoder.backbone.training
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(not parameter.requires_grad for parameter in policy.obs_encoder.backbone.parameters())
