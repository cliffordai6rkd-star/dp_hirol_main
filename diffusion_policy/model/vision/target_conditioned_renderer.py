"""Experimental RGB decoder conditioned on spatial teacher target features.

The teacher is supplied by the caller, never registered or updated here.
This interface requires image patch tokens; non-spatial action tokens need a
different layout adapter before they can condition this decoder.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class TargetConditionedRenderer(nn.Module):
    def __init__(self, reference, context_dim, patch_counts):
        super().__init__()
        if reference.rendering_mode != "residual" or reference.dual_path_enabled:
            raise ValueError("Target decoder expects the current residual CNN renderer")
        if reference.wrench_projection.out_features != context_dim:
            raise ValueError("Teacher context width must match the reference attention input")
        self.image_keys = tuple(reference.image_keys)
        self.patch_counts = {key: int(patch_counts[key]) for key in self.image_keys}
        self.context_dim = int(context_dim)
        self.residual_scale = float(reference.residual_scale)
        self.backbone = copy.deepcopy(reference.backbone)
        self.camera_embedding = copy.deepcopy(reference.camera_embedding)
        self.spatial_modulations = copy.deepcopy(reference.spatial_modulations)
        self.spatial_modulation_enabled = reference.spatial_modulation_enabled
        self.context_projection = nn.Sequential(
            nn.Linear(2 * context_dim, self.backbone.embed_dim), nn.GELU(),
            nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim),
        )
        nn.init.zeros_(self.context_projection[-1].weight)
        nn.init.zeros_(self.context_projection[-1].bias)
        self.requires_grad_(True)

    def forward(self, images, visual_tokens, target_tokens, spatial_gates, phase_gate):
        """Decode Z_target - Z_image using shared, spatially aligned weights.

        Tokens use [B, observation * all_camera_patches, D]. Gates are supplied
        explicitly so baseline and candidate use exactly the same support.
        """
        batch, steps = images[self.image_keys[0]].shape[:2]
        expected = (batch, steps * sum(self.patch_counts.values()), self.context_dim)
        if visual_tokens.shape != expected or target_tokens.shape != expected:
            raise ValueError(f"Expected image-layout context tokens {expected}")
        visual = visual_tokens.reshape(batch, steps, -1, self.context_dim)
        response = (target_tokens - visual_tokens).reshape_as(visual)
        rendered = {}
        offset = 0
        for camera, key in enumerate(self.image_keys):
            image = images[key]
            count = self.patch_counts[key]
            grid_h = math.ceil(image.shape[-2] / self.backbone.patch_size)
            grid_w = math.ceil(image.shape[-1] / self.backbone.patch_size)
            if grid_h * grid_w != count:
                raise ValueError("Teacher and decoder patch grids must align")
            z0 = visual[:, :, offset:offset + count].reshape(batch * steps, count, -1)
            delta = response[:, :, offset:offset + count].reshape_as(z0)
            offset += count
            embedding = self.camera_embedding.weight[camera].reshape(1, 1, -1)
            features, skips, height, width = self.backbone.encode(
                image.flatten(0, 1), delta, embedding,
            )
            update = self.context_projection(torch.cat((z0, delta), dim=-1))
            update = update.transpose(1, 2).reshape(batch * steps, -1, grid_h, grid_w)
            features = features + update
            fused = features
            gate = spatial_gates[key].flatten(0, 1)
            for index, (decoder, skip) in enumerate(zip(
                    self.backbone.decoders,
                    reversed(skips[-self.backbone.decoder_layers:]))):
                if self.backbone.conv_block == "bottleneck":
                    features = decoder(features, skip)
                else:
                    features = decoder(torch.cat((F.interpolate(
                        features, size=skip.shape[-2:], mode="bilinear", align_corners=False,
                    ), skip), dim=1))
                if self.spatial_modulation_enabled and index < len(self.spatial_modulations):
                    features = self.spatial_modulations[index](features, fused, gate)
            if features.shape[-2:] != (height, width):
                features = F.interpolate(features, size=(height, width),
                                         mode="bilinear", align_corners=False)
            raw = self.backbone.output(features[:, :, :height, :width]).tanh().reshape_as(image)
            support = spatial_gates[key].float() * phase_gate[..., None, None, None]
            out = (image.float() + self.residual_scale * support * raw.float()).clamp(0, 1)
            rendered[key] = torch.where(support > 0, out, image.float())
        return rendered
