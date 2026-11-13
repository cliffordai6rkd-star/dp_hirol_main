"""ACT and DiT must consume the same DINO/LSTM fused observation context."""

import io
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn
from torch.nn import functional as F

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.vision.contact_curriculum import gaussian_2d_smoothing
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)


class _PatchBackbone(nn.Module):
    """Small spatial backbone with DINO's CLS/register/patch token layout."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16, patch_size=8, num_register_tokens=2
        )
        self.projection = nn.Conv2d(3, 16, kernel_size=8, stride=8)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        registers = patches.new_zeros(patches.shape[0], 2, patches.shape[-1])
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls, registers, patches), dim=1)
        )


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _make_policy(
    head="act", *, loss_type="l1", blur=False, blur_scope="full_context",
    fusion_downsample_factor=1,
):
    torch.manual_seed(23)
    image_shapes = {key: (3, 16, 16) for key in ("side", "wrist")}
    encoder = ForceAwareObsEncoder(
        pretrained_model_name_or_path="unused",
        n_emb=16,
        n_head=4,
        max_obs_steps=2,
        wrench_dim=7,
        wrench_history_steps=8,
        force_temporal_encoder="lstm",
        force_encoder_layers=2,
        dropout=0.0,
        freeze_backbone=True,
        image_shapes=image_shapes,
        pre_cross_self_attn_layers=1,
        cross_attn_layers=0,
        cross_attn_query="image",
        fusion_downsample_factor=fusion_downsample_factor,
        backbone=_PatchBackbone(),
    )
    scheduler = None if head == "act" else DDIMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    policy = ForceAwareDiffusionTransformerPolicy(
        shape_meta={
            "obs": {
                **{key: {"shape": list(shape), "type": "rgb"}
                   for key, shape in image_shapes.items()},
                "tau_ext": {"shape": [8, 7], "type": "low_dim"},
            },
            "action": {"shape": [7]},
        },
        noise_scheduler=scheduler,
        horizon=9,
        n_action_steps=8,
        n_obs_steps=2,
        wrench_key="tau_ext",
        n_emb=16,
        n_head=4,
        n_layer=1,
        n_cond_layers=0,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        contact_threshold=5.0,
        contact_force_dims=tuple(range(7)),
        precontact_seconds=0.5,
        input_gate_enabled=False,
        action_representation="joint_position",
        action_head_backbone=head,
        act_loss_type=loss_type,
        num_inference_steps=2,
        obs_encoder=encoder,
        fusion_downsample_factor=fusion_downsample_factor,
        image_blur_scope=blur_scope,
        image_blur_schedule=(
            dict(total_steps=100, start_sigma=2.0, end_sigma=0.0)
            if blur else None
        ),
    )
    normalizer = LinearNormalizer()
    normalizer["tau_ext"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-9, 15, 700).reshape(100, 7)
    )
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        torch.linspace(-2, 8, 700).reshape(100, 7)
    )
    policy.set_normalizer(normalizer)
    return policy


def _batch():
    generator = torch.Generator().manual_seed(41)
    return {
        "obs": {
            "side": torch.rand(2, 2, 3, 16, 16, generator=generator),
            "wrist": torch.rand(2, 2, 3, 16, 16, generator=generator),
            "tau_ext": torch.randn(2, 2, 8, 7, generator=generator),
        },
        "action": torch.rand(2, 9, 7, generator=generator) * 3 + 1,
    }


@pytest.mark.parametrize("loss_type", ["l1", "mse"])
def test_act_regresses_normalized_actions_without_diffusion(loss_type):
    policy = _make_policy(loss_type=loss_type).eval()
    batch = _batch()
    assert policy.noise_scheduler is None
    context = policy._encode_context(batch["obs"], apply_curriculum_mask=False)
    prediction = policy.model(context)
    target = policy.normalizer["action"].normalize(batch["action"])
    expected = (F.l1_loss if loss_type == "l1" else F.mse_loss)(prediction, target)
    actual = policy.compute_loss(batch, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)


@pytest.mark.parametrize("head", ["act", "dit"])
def test_both_heads_receive_complete_context_and_train_lstm_fusion(head):
    policy = _make_policy(head).train()
    batch = _batch()
    batch["obs"]["tau_ext"].requires_grad_()
    received = []
    handle = policy.model.register_forward_pre_hook(
        lambda module, args: received.append(args[-1])
    )
    try:
        loss = policy.compute_loss(batch, generator=torch.Generator().manual_seed(8))
        loss.backward()
    finally:
        handle.remove()
    # Each observation retains 4 patches/camera * 2 cameras + 8 LSTM outputs.
    assert received[0].shape == (2, 32, 16)
    assert policy.obs_encoder.context_tokens_per_observation == 16
    for module in (
        policy.obs_encoder.image_projection,
        policy.obs_encoder.wrench_projection,
        policy.obs_encoder.force_encoder,
        policy.obs_encoder.pre_cross_self_attention,
        policy.model.decoder,
    ):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    # Keeping the LSTM path means both older and most recent torques influence loss.
    tau_grad = batch["obs"]["tau_ext"].grad
    assert tau_grad[:, :, 0].abs().sum() > 0
    assert tau_grad[:, :, -1].abs().sum() > 0
    assert all(p.grad is None for p in policy.obs_encoder.backbone.parameters())


def test_joint_self_attention_does_not_use_contact_alpha_as_a_force_gate():
    policy = _make_policy().eval()
    obs = _batch()["obs"]
    zero_alpha = dict(obs, contact_alpha=torch.zeros(2, 2))
    one_alpha = dict(obs, contact_alpha=torch.ones(2, 2))
    zeros = policy._encode_context(zero_alpha, apply_curriculum_mask=False)
    ones = policy._encode_context(one_alpha, apply_curriculum_mask=False)
    torch.testing.assert_close(zeros, ones, rtol=0, atol=0)
    changed_history = {**obs, "tau_ext": obs["tau_ext"].clone()}
    changed_history["tau_ext"][:, :, 0] += 4
    changed = policy._encode_context(changed_history, apply_curriculum_mask=False)
    assert not torch.allclose(changed, zeros)


def test_curriculum_changes_only_selected_images_and_eval_is_clear():
    policy = _make_policy(blur=True)
    batch = _batch()
    batch["obs"]["tau_ext"].zero_()
    batch["obs"]["tau_ext"][0, -1] = 3
    received = []

    def remember_inputs(module, args, kwargs):
        received.append({
            "images": {key: value.detach().clone() for key, value in kwargs["images"].items()},
            "wrench": kwargs["wrench_history"].detach().clone(),
        })

    handle = policy.obs_encoder.register_forward_pre_hook(remember_inputs, with_kwargs=True)
    try:
        policy.train()
        policy.compute_loss(batch, optimizer_step=0)
        assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == 0.5
        policy.eval()
        policy.compute_loss(batch)
        assert policy.last_curriculum_metrics["curriculum/blur_sigma"] == 0.0
    finally:
        handle.remove()
    torch.testing.assert_close(received[0]["wrench"], received[1]["wrench"], rtol=0, atol=0)
    for camera in ("side", "wrist"):
        raw = batch["obs"][camera]
        torch.testing.assert_close(received[0]["images"][camera][0], gaussian_2d_smoothing(raw[0], 2.0))
        torch.testing.assert_close(received[0]["images"][camera][1], raw[1], rtol=0, atol=0)
        torch.testing.assert_close(received[1]["images"][camera], raw, rtol=0, atol=0)


@pytest.mark.parametrize("head", ["act", "dit"])
@pytest.mark.parametrize("step,sigma", [(0, 2.0), (10, 2.0), (45, 1.0), (80, 0.0)])
def test_global_curriculum_blurs_all_phases_and_cameras_but_not_eval(head, step, sigma):
    policy = _make_policy(head, blur=True, blur_scope="all")
    batch = _batch()
    # One contacting window and one entirely free window; alpha must not select RGB.
    batch["obs"]["tau_ext"].zero_()
    batch["obs"]["tau_ext"][0, -1] = 3
    batch["obs"]["contact_alpha"] = torch.tensor([[0.5, 1.0], [0.0, 0.0]])
    received = []

    def remember_inputs(module, args, kwargs):
        received.append({
            "images": {key: value.detach().clone() for key, value in kwargs["images"].items()},
            "wrench": kwargs["wrench_history"].detach().clone(),
        })

    handle = policy.obs_encoder.register_forward_pre_hook(remember_inputs, with_kwargs=True)
    try:
        policy.train()
        policy.compute_loss(batch, optimizer_step=step)
        assert policy.last_curriculum_metrics["curriculum/blur_sigma"] == pytest.approx(sigma)
        assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == (
            1.0 if sigma > 0 else 0.0
        )
        policy.eval()
        policy.compute_loss(batch)
        assert policy.last_curriculum_metrics["curriculum/blur_sigma"] == 0.0
        policy.predict_action(batch["obs"])
    finally:
        handle.remove()

    for camera in ("side", "wrist"):
        raw = batch["obs"][camera]
        torch.testing.assert_close(
            received[0]["images"][camera], gaussian_2d_smoothing(raw, sigma)
        )
        for evaluation in received[1:]:
            torch.testing.assert_close(evaluation["images"][camera], raw, rtol=0, atol=0)
    for evaluation in received[1:]:
        torch.testing.assert_close(received[0]["wrench"], evaluation["wrench"], rtol=0, atol=0)


def test_act_predictions_are_deterministic_and_checkpoint_preserves_action_units():
    policy = _make_policy().eval()
    obs = _batch()["obs"]
    first = policy.predict_action(obs, generator=torch.Generator().manual_seed(9))
    other = policy.predict_action(
        obs, generator=[torch.Generator().manual_seed(2), torch.Generator().manual_seed(3)]
    )
    assert first["action"].shape == (2, 8, 7)
    assert first["model_action_pred"].shape == (2, 9, 7)
    for key in first:
        torch.testing.assert_close(first[key], other[key], rtol=0, atol=0)
    context = policy._encode_context(obs, apply_curriculum_mask=False)
    expected = policy.normalizer["action"].unnormalize(policy.model(context))
    torch.testing.assert_close(first["model_action_pred"], expected)
    torch.testing.assert_close(first["action"], expected[:, 1:9])

    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    buffer.seek(0)
    restored = _make_policy().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    actual = restored.predict_action(obs)
    for key in first:
        torch.testing.assert_close(first[key], actual[key], rtol=0, atol=0)


def test_dit_keeps_noise_training_and_iterative_sampling(monkeypatch):
    policy = _make_policy("dit").train()
    batch = _batch()
    counts = {"add_noise": 0, "step": 0}
    scheduler = policy.noise_scheduler
    original_add_noise, original_step = scheduler.add_noise, scheduler.step

    def add_noise(*args, **kwargs):
        counts["add_noise"] += 1
        return original_add_noise(*args, **kwargs)

    def step(*args, **kwargs):
        counts["step"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(scheduler, "add_noise", add_noise)
    monkeypatch.setattr(scheduler, "step", step)
    loss = policy.compute_loss(batch, generator=torch.Generator().manual_seed(9))
    assert torch.isfinite(loss)
    assert counts == {"add_noise": 1, "step": 0}
    policy.eval()
    prediction = policy.predict_action(batch["obs"], generator=torch.Generator().manual_seed(5))
    assert prediction["action"].shape == (2, 8, 7)
    assert counts["step"] == policy.num_inference_steps == 2
