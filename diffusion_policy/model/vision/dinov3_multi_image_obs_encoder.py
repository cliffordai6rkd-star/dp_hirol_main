from __future__ import annotations

import inspect
from typing import Mapping, Sequence

import torch
import torch.nn as nn


class DINOv3MultiImageObsEncoder(nn.Module):
    """Encode multiple RGB observations with one shared DINOv3 backbone."""

    def __init__(
        self,
        shape_meta: Mapping,
        pretrained_model_name_or_path: str,
        output_dim: int = 256,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
        pooling: str = "cls",
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        obs_meta = shape_meta.get("obs", {})
        self.rgb_keys = tuple(
            sorted(key for key, value in obs_meta.items() if value.get("type") == "rgb")
        )
        non_rgb_keys = sorted(set(obs_meta) - set(self.rgb_keys))
        if not self.rgb_keys:
            raise ValueError("DINOv3 encoder requires at least one RGB observation")
        if non_rgb_keys:
            raise ValueError(
                "pure visual DINOv3 encoder does not accept low-dimensional keys: "
                f"{non_rgb_keys}"
            )
        self.key_shape_map = {
            key: tuple(int(value) for value in obs_meta[key]["shape"])
            for key in self.rgb_keys
        }
        for key, shape in self.key_shape_map.items():
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"RGB observation {key!r} must have shape [3,H,W]")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        pooling = str(pooling).lower()
        if pooling not in {"cls", "mean_patch"}:
            raise ValueError("pooling must be 'cls' or 'mean_patch'")

        if backbone is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "DINOv3 requires transformers>=4.56 in the training environment"
                ) from exc
            backbone = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DINOv3 backbone config must expose hidden_size")

        self.backbone = backbone
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.freeze_backbone = bool(freeze_backbone)
        self.pooling = pooling
        self.output_dim = int(output_dim)
        self.num_register_tokens = int(
            getattr(getattr(backbone, "config", None), "num_register_tokens", 0)
        )
        forward_parameters = inspect.signature(self.backbone.forward).parameters
        self._interpolate_positions = "interpolate_pos_encoding" in forward_parameters
        fused_dim = len(self.rgb_keys) * int(hidden_size)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self._set_backbone_frozen()

    def _set_backbone_frozen(self) -> None:
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        kwargs = {"pixel_values": images}
        if self._interpolate_positions:
            kwargs["interpolate_pos_encoding"] = True
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**kwargs)
        else:
            outputs = self.backbone(**kwargs)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError("DINOv3 must return last_hidden_state [B,N,D]")
        if self.pooling == "cls":
            return tokens[:, 0]
        patch_start = 1 + self.num_register_tokens
        if tokens.shape[1] <= patch_start:
            raise RuntimeError("DINOv3 output does not contain patch tokens")
        return tokens[:, patch_start:].mean(dim=1)

    def forward(self, obs_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        provided_keys = tuple(sorted(str(key) for key in obs_dict))
        if provided_keys != self.rgb_keys:
            raise ValueError(
                "DINOv3 observation keys do not match shape_meta: "
                f"expected={list(self.rgb_keys)}, got={list(provided_keys)}"
            )
        batch_size = None
        images = []
        for key in self.rgb_keys:
            value = obs_dict[key]
            expected_shape = self.key_shape_map[key]
            if value.ndim != 4 or tuple(value.shape[1:]) != expected_shape:
                raise ValueError(
                    f"obs[{key!r}] must have shape [B,{','.join(map(str, expected_shape))}], "
                    f"got {tuple(value.shape)}"
                )
            if batch_size is None:
                batch_size = int(value.shape[0])
            elif value.shape[0] != batch_size:
                raise ValueError("all camera observations must share batch size")
            images.append(value)
        assert batch_size is not None
        features = self._encode(torch.cat(images, dim=0))
        features = features.reshape(len(self.rgb_keys), batch_size, -1)
        features = torch.moveaxis(features, 0, 1).reshape(batch_size, -1)
        return self.fusion(features)

    def output_shape(self) -> tuple[int]:
        return (self.output_dim,)
