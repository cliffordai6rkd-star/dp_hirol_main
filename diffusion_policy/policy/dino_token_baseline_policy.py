"""Token baselines using the same action DiT, losses and training workspace."""

from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import (
    DinoTokenBaselineObsEncoder,
)
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)


class DinoTokenBaselinePolicy(ForceAwareDiffusionTransformerPolicy):
    def __init__(
        self, baseline_mode="dino", baseline_self_attn_layers=1, obs_encoder=None, **kwargs
    ):
        if baseline_mode not in {"dino", "self_attn"}:
            raise ValueError("baseline_mode must be 'dino' or 'self_attn'")
        if obs_encoder is None:
            obs_meta = kwargs["shape_meta"]["obs"]
            wrench_key = kwargs.get("wrench_key", "wrench_ext")
            obs_encoder = DinoTokenBaselineObsEncoder(
                pretrained_model_name_or_path=kwargs["dino_model_name_or_path"],
                image_shapes={key: meta["shape"] for key, meta in obs_meta.items()
                              if meta.get("type") == "rgb"},
                mode=baseline_mode,
                n_emb=kwargs.get("n_emb", 256), n_head=kwargs.get("n_head", 4),
                max_obs_steps=kwargs["n_obs_steps"],
                wrench_dim=obs_meta[wrench_key]["shape"][-1],
                wrench_history_steps=kwargs.get("wrench_history_steps", 8),
                self_attn_layers=baseline_self_attn_layers,
                dropout=kwargs.get("p_drop_attn", 0.1),
                freeze_backbone=kwargs.get("freeze_dino_backbone", True),
                local_files_only=kwargs.get("dino_local_files_only", True),
                force_temporal_encoder=kwargs.get("force_temporal_encoder", "none"),
                force_encoder_layers=kwargs.get("force_encoder_layers", 0),
            )
        if obs_encoder.mode != baseline_mode:
            raise ValueError("obs_encoder.mode must match baseline_mode")
        super().__init__(obs_encoder=obs_encoder, **kwargs)
        self.baseline_mode = baseline_mode

    def _encode_context(self, obs_dict, apply_curriculum_mask, generator=None):
        # Force labels remain in the dataset for identical splits/resampling,
        # but the visual baseline never reads them in its forward computation.
        images = {key: obs_dict[key][:, :self.n_obs_steps] for key in self.image_keys}
        if any(value.shape[1] != self.n_obs_steps for value in images.values()):
            raise ValueError(f"baseline requires {self.n_obs_steps} observation steps")
        wrench = None
        if self.baseline_mode == "self_attn":
            wrench = self.normalizer[self.wrench_key].normalize(
                obs_dict[self.wrench_key][:, :self.n_obs_steps]
            )
        self.last_curriculum_metrics = {
            "curriculum/mask_probability": 0.0,
            "curriculum/masked_image_fraction": 0.0,
        }
        return self.obs_encoder(images=images, wrench_history=wrench)
