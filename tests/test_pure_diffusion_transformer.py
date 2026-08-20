from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.dinov3_multi_image_obs_encoder import (
    DINOv3MultiImageObsEncoder,
)
from diffusion_policy.policy.diffusion_transformer_image_policy import (
    DiffusionTransformerImagePolicy,
)


SHAPE_META = {
    "obs": {
        "wrist": {"shape": [3, 16, 24], "type": "rgb"},
        "side": {"shape": [3, 16, 24], "type": "rgb"},
    },
    "action": {"shape": [7]},
}


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_register_tokens=2)
        self.projection = nn.Linear(3, 8)
        self.last_input_shape = None
        self.interpolate_pos_encoding = None

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        self.last_input_shape = tuple(pixel_values.shape)
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)
        pooled = pixel_values.mean(dim=(-2, -1))
        token = self.projection(pooled)
        return SimpleNamespace(last_hidden_state=token[:, None].repeat(1, 7, 1))


def _encoder() -> DINOv3MultiImageObsEncoder:
    return DINOv3MultiImageObsEncoder(
        shape_meta=SHAPE_META,
        pretrained_model_name_or_path="unused",
        output_dim=16,
        freeze_backbone=True,
        backbone=_Backbone(),
    )


def _policy() -> DiffusionTransformerImagePolicy:
    policy = DiffusionTransformerImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=10,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        ),
        obs_encoder=_encoder(),
        horizon=8,
        n_action_steps=7,
        n_obs_steps=2,
        num_inference_steps=2,
        n_layer=2,
        n_cond_layers=1,
        n_head=4,
        n_emb=32,
        causal_attn=False,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(32, 8, 7)}, last_n_dims=1)
    policy.set_normalizer(normalizer)
    return policy


def test_dual_camera_encoder_shares_and_freezes_dinov3() -> None:
    encoder = _encoder()
    output = encoder(
        {
            "side": torch.rand(3, 3, 16, 24),
            "wrist": torch.rand(3, 3, 16, 24),
        }
    )

    assert output.shape == (3, 16)
    assert encoder.backbone.last_input_shape == (6, 3, 16, 24)
    assert encoder.backbone.interpolate_pos_encoding
    assert not any(parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert any(parameter.requires_grad for parameter in encoder.fusion.parameters())


def test_transformer_policy_predicts_noise_from_images_only() -> None:
    policy = _policy()
    obs = {
        "side": torch.rand(2, 2, 3, 16, 24),
        "wrist": torch.rand(2, 2, 3, 16, 24),
    }
    action = torch.randn(2, 8, 7)
    action[..., 3:] = torch.nn.functional.normalize(action[..., 3:], dim=-1)

    loss = policy.compute_loss({"obs": obs, "action": action})
    output = policy.predict_action(obs)

    assert torch.isfinite(loss)
    assert output["action_pred"].shape == (2, 8, 7)
    assert output["action"].shape == (2, 7, 7)
    assert output["action_target"].shape == (2, 7)
    torch.testing.assert_close(
        torch.linalg.vector_norm(output["action"][..., 3:], dim=-1),
        torch.ones(2, 7),
    )
