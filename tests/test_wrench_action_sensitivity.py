"""Compare cached local replay with actual full-encoder interventions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler
from torch import nn

from test_image_query_attention_capture import inputs, make_policy as make_encoder_policy
from diffusion_policy.model.diffusion.context_transformer_for_diffusion import ContextTransformerForDiffusion
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.scripts import visualize_wrench_action_sensitivity_episode as sensitivity


class SmallPolicy(nn.Module):
    n_obs_steps = 2
    horizon = 3
    action_dim = 2
    num_inference_steps = 2
    wrench_key = phase_wrench_key = "wrench"
    future_phase_wrench_key = None
    conditional_sample = ForceAwareDiffusionTransformerPolicy.conditional_sample

    def __init__(self, mode="joint", residual=True):
        super().__init__()
        self.obs_encoder = make_encoder_policy(camera_mode=mode, layers=1).obs_encoder
        self.obs_encoder.cross_attn_query_residual = residual
        self.model = ContextTransformerForDiffusion(
            input_dim=2, output_dim=2, horizon=3, n_emb=4, n_head=2,
            n_layer=2, n_cond_layers=1, max_context_tokens=12, p_drop_attn=0.,
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=6, prediction_type="epsilon")
        self.scheduler_step_kwargs = {}
        self.eval()

    def prepare_context_inputs(self, obs):
        images = {key: obs[key] for key in self.obs_encoder.image_keys}
        return images, obs["wrench"], torch.zeros((1, 2), dtype=torch.bool), None

    def _encode_context(self, obs, apply_curriculum_mask=False):
        images, wrench, _, _ = self.prepare_context_inputs(obs)
        return self.obs_encoder(images=images, wrench_history=wrench)


def observations():
    values = inputs()
    return {**values["images"], "wrench": values["wrench_history"]}


@torch.inference_mode()
def full_intervention(policy, obs, reference_update, token):
    """Rerun the entire encoder, changing only its actual MHA output."""
    encoder = policy.obs_encoder
    patches = sum(encoder.patch_counts.values())
    observation, patch = divmod(token, patches)
    offset = 0

    def replace(module, args, output):
        nonlocal offset
        update = output[0].clone()
        if offset <= patch < offset + update.shape[1]:
            update[observation, patch - offset] = reference_update[observation, patch]
        offset += update.shape[1]
        return update, output[1]

    handle = encoder.cross_attention.register_forward_hook(replace)
    try:
        return policy._encode_context(obs)
    finally:
        handle.remove()


@pytest.mark.parametrize("mode", ["joint", "per_camera"])
@pytest.mark.parametrize("residual", [False, True])
def test_local_replay_matches_full_encoder_for_every_observation_camera_patch(mode, residual):
    policy = SmallPolicy(mode, residual)
    obs = observations()
    query, normal, reference = sensitivity.capture_fusion_pair(policy, obs, "raw_zero")
    indices = torch.arange(12)
    local = sensitivity.replacement_contexts(policy.obs_encoder, query, normal, reference, indices)
    for token in indices.tolist():
        full = full_intervention(policy, obs, reference["ca_update"], token)
        torch.testing.assert_close(local[token:token + 1], full, rtol=1e-5, atol=1e-6)
        other = indices[indices != token]
        torch.testing.assert_close(local[token, other], normal["context"].reshape(12, 4)[other], rtol=0, atol=0)
    assert not policy.obs_encoder.image_norm._forward_pre_hooks
    assert not policy.obs_encoder.cross_attention._forward_hooks


def test_sensitivity_matches_bruteforce_full_fusion_and_dit_at_fixed_noise():
    policy = SmallPolicy()
    obs = observations()
    result = sensitivity.patch_action_sensitivity(policy, obs, seed=11, batch_size=5)
    _, normal, reference = sensitivity.capture_fusion_pair(policy, obs, "raw_zero")
    expected = np.zeros((2, 2, 6), dtype=np.float32)
    with torch.inference_mode():
        for token in range(12):
            full = full_intervention(policy, obs, reference["ca_update"], token)
            for step, timestep in enumerate(result["timesteps"]):
                noisy_action = torch.from_numpy(result["normal_noisy_action"][step:step + 1])
                original = policy.model(noisy_action, int(timestep), normal["context"].reshape(1, 12, 4))
                replaced = policy.model(noisy_action, int(timestep), full)
                expected[step].flat[token] = float(torch.linalg.vector_norm(original - replaced))
    np.testing.assert_allclose(result["per_step"], expected, rtol=3e-3, atol=3e-7)
    assert expected.max() > 1e-6
    assert result["noop_epsilon_l2_max"] < 1e-6
    again = sensitivity.patch_action_sensitivity(policy, obs, seed=11, batch_size=7)
    np.testing.assert_allclose(again["per_step"], result["per_step"], rtol=3e-3, atol=3e-7)


def test_identical_wrench_reference_has_exact_zero_sensitivity():
    policy = SmallPolicy()
    obs = observations()
    obs["wrench"].zero_()
    result = sensitivity.patch_action_sensitivity(policy, obs)
    np.testing.assert_array_equal(result["per_step"], 0.)
    assert result["intervened_tokens"] == 0
    assert not policy.model._forward_hooks


def test_nonzero_injection_can_have_zero_action_sensitivity():
    policy = SmallPolicy()
    obs = observations()
    _, normal, reference = sensitivity.capture_fusion_pair(policy, obs, "raw_zero")
    assert torch.linalg.vector_norm(normal["ca_update"] - reference["ca_update"]) > 0

    class ContextIndependentModel(nn.Module):
        def forward(self, sample, timestep, context):
            return sample * .1

    policy.model = ContextIndependentModel()
    result = sensitivity.patch_action_sensitivity(policy, obs)
    np.testing.assert_array_equal(result["per_step"], 0.)
    assert result["intervened_tokens"] == 12


def test_trace_cleans_up_hook_on_failure():
    policy = SmallPolicy()
    policy.model.forward = lambda *args: (_ for _ in ()).throw(RuntimeError("intentional"))
    with pytest.raises(RuntimeError, match="intentional"):
        sensitivity.normal_denoising_trace(policy, torch.zeros(1, 12, 4), 42)
    assert not policy.model._forward_hooks
