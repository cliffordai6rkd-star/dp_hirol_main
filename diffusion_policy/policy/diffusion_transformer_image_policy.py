from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _normalize_pose_quaternions(action: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if action.shape[-1] != 7:
        return action
    return torch.cat(
        (action[..., :3], F.normalize(action[..., 3:], dim=-1, eps=eps)),
        dim=-1,
    )


class DiffusionTransformerImagePolicy(BaseImagePolicy):
    """Pure-image diffusion policy with a Transformer noise predictor."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: SchedulerMixin,
        obs_encoder: nn.Module,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int | None = None,
        n_layer: int = 8,
        n_cond_layers: int = 2,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        normalize_pose_quaternions: bool = True,
        **scheduler_step_kwargs,
    ) -> None:
        super().__init__()
        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError("action shape must be one-dimensional per timestep")
        action_dim = int(action_shape[0])
        if horizon < 1 or n_obs_steps < 1 or n_action_steps < 1:
            raise ValueError("horizon and observation/action step counts must be positive")
        action_start_index = int(n_obs_steps) - 1
        if action_start_index + int(n_action_steps) > int(horizon):
            raise ValueError("n_action_steps does not fit in the diffusion horizon")
        obs_feature_dim = int(obs_encoder.output_shape()[0])
        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=int(horizon),
            n_obs_steps=int(n_obs_steps),
            cond_dim=obs_feature_dim,
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_emb=int(n_emb),
            p_drop_emb=float(p_drop_emb),
            p_drop_attn=float(p_drop_attn),
            causal_attn=bool(causal_attn),
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=int(n_cond_layers),
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.action_start_index = action_start_index
        self.action_dim = action_dim
        # A seven-dimensional action is not necessarily xyz+xyzw.  Joint
        # position actions must bypass quaternion projection at inference.
        self.normalize_pose_quaternions = bool(normalize_pose_quaternions)
        self.obs_feature_dim = obs_feature_dim
        self.image_keys = tuple(getattr(obs_encoder, "rgb_keys", ()))
        if not self.image_keys:
            raise ValueError("obs_encoder must expose at least one rgb_key")
        self.image_key = self.image_keys[0] if len(self.image_keys) == 1 else None
        # These attributes keep the pure visual policy compatible with the
        # FDP-style workspace protocol without introducing force inputs.
        self.relative_pose_actions = False
        self.action_reference_key = "action_reference"
        self.optimizer_step = 0
        self.last_curriculum_metrics: Dict[str, float] = {}
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else noise_scheduler.config.num_train_timesteps
        )
        self.scheduler_step_kwargs = scheduler_step_kwargs

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_optimizer_step(self, optimizer_step: int) -> None:
        self.optimizer_step = int(optimizer_step)

    def _encode_observation(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.image_keys) - set(obs_dict)
        if missing:
            raise KeyError(f"missing image observations: {sorted(missing)}")
        first = obs_dict[self.image_keys[0]]
        if first.ndim != 5 or first.shape[1] < self.n_obs_steps:
            raise ValueError("image observations must have shape [B,To,C,H,W]")
        batch_size = int(first.shape[0])
        flattened = {
            key: obs_dict[key][:, : self.n_obs_steps].reshape(
                -1, *obs_dict[key].shape[2:]
            )
            for key in self.image_keys
        }
        features = self.obs_encoder(flattened)
        return features.reshape(batch_size, self.n_obs_steps, self.obs_feature_dim)

    def conditional_sample(
        self,
        shape: tuple[int, int, int],
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            shape,
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps,
            device=condition.device,
        )
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, condition)
            trajectory = self.noise_scheduler.step(
                prediction,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not implemented")
        condition = self._encode_observation(obs_dict)
        action_prediction = self.normalizer["action"].unnormalize(
            self.conditional_sample(
                (condition.shape[0], self.horizon, self.action_dim),
                condition,
                generator=generator,
            )
        )
        if self.normalize_pose_quaternions:
            action_prediction = _normalize_pose_quaternions(action_prediction)
        start = self.action_start_index
        action = action_prediction[:, start : start + self.n_action_steps]
        return {
            "action": action,
            "action_pred": action_prediction,
            "model_action_pred": action_prediction,
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer_step: int | None = None,
    ) -> torch.Tensor:
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask loss weighting is not implemented")
        if optimizer_step is not None:
            self.set_optimizer_step(optimizer_step)
        action = batch["action"]
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"action must have shape [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        normalized_action = self.normalizer["action"].normalize(action)
        condition = self._encode_observation(batch["obs"])
        noise = torch.randn_like(normalized_action)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (normalized_action.shape[0],),
            device=normalized_action.device,
        ).long()
        noisy_action = self.noise_scheduler.add_noise(
            normalized_action,
            noise,
            timesteps,
        )
        prediction = self.model(noisy_action, timesteps, condition)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = normalized_action
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                normalized_action,
                noise,
                timesteps,
            )
        else:
            raise ValueError(f"unsupported prediction type {prediction_type!r}")
        return F.mse_loss(prediction, target)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer_step: int | None = None,
    ) -> torch.Tensor:
        return self.compute_loss(batch, optimizer_step=optimizer_step)

    def get_optimizer(
        self,
        learning_rate: float,
        action_expert_weight_decay: float = 1.0e-3,
        fusion_weight_decay: float = 1.0e-6,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        action_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        fusion_parameters = [
            parameter
            for parameter in self.obs_encoder.parameters()
            if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            [
                {
                    "params": action_parameters,
                    "weight_decay": float(action_expert_weight_decay),
                },
                {
                    "params": fusion_parameters,
                    "weight_decay": float(fusion_weight_decay),
                },
            ],
            lr=float(learning_rate),
            betas=betas,
        )
