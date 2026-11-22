"""Lightweight reusable RGB + torque-history phase gate.

Inputs use the same image and wrench preprocessing as the teacher checkpoint.
The output classes are free, precontact and contact; classes 1/2 open rendering.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ImageTauPhaseGate(nn.Module):
    """Two-camera CNN + tau_ext GRU + MLP three-class phase classifier."""

    def __init__(self, image_keys=("side", "wrist"), wrench_dim=7,
                 hidden_dim=16, gru_layers=1, image_size=64,
                 cnn_channels=(4, 8, 12), pool_size=1, dropout=0.0,
                 image_offsets=(0,), center_images=False, cnn_norm=False):
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.hidden_dim = int(hidden_dim)
        self.image_size = int(image_size)
        self.image_offsets = tuple(int(offset) for offset in image_offsets)
        self.center_images = bool(center_images)
        if (not self.image_offsets or self.image_offsets[-1] != 0
                or tuple(sorted(set(self.image_offsets))) != self.image_offsets):
            raise ValueError("Image offsets must be distinct causal offsets ending at zero")
        first, second, third = (int(value) for value in cnn_channels)
        convolution_layers = []
        for incoming, outgoing, kernel in ((3, first, 5), (first, second, 3), (second, third, 3)):
            convolution_layers.append(nn.Conv2d(incoming, outgoing, kernel, stride=2, padding=kernel // 2))
            if cnn_norm:
                convolution_layers.append(nn.GroupNorm(1, outgoing))
            convolution_layers.append(nn.SiLU())
        self.image_cnn = nn.Sequential(*convolution_layers,
            nn.AdaptiveAvgPool2d((int(pool_size), int(pool_size))), nn.Flatten(),
            nn.Linear(third * int(pool_size) ** 2, hidden_dim), nn.SiLU(),
        )
        self.image_fusion = nn.Sequential(
            nn.Linear(len(self.image_keys) * hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.image_temporal = (nn.GRU(hidden_dim, hidden_dim, batch_first=True)
                               if len(self.image_offsets) > 1 else None)
        self.fusion_dropout = nn.Dropout(float(dropout))
        self.tau_projection = nn.Sequential(nn.Linear(int(wrench_dim), hidden_dim), nn.SiLU())
        self.tau_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=int(gru_layers),
                              batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, images, tau_ext):
        if not images or tuple(sorted(images)) != tuple(sorted(self.image_keys)):
            raise ValueError("Image keys do not match the phase gate")
        if tau_ext.ndim != 4:
            raise ValueError("tau_ext must be [B, observation_steps, history_steps, wrench_dim]")
        batch, steps, history, dim = tau_ext.shape
        image_steps = len(self.image_offsets)
        image_features = []
        for key in self.image_keys:
            image = images[key]
            if image.ndim == 5 and image_steps == 1:
                image = image.unsqueeze(2)
            if (image.ndim != 6 or image.shape[:3] != (batch, steps, image_steps)
                    or image.shape[3] != 3):
                raise ValueError("Each image must match [B, observation_steps, image_history, 3, H, W]")
            image = image.reshape(batch * steps * image_steps, *image.shape[3:]).float()
            # The gate only needs coarse phase cues.  Shrinking before the CNN
            # avoids spending most of the training time on full-resolution frames.
            if image.shape[-2:] != (self.image_size, self.image_size):
                image = F.interpolate(image, size=(self.image_size, self.image_size),
                                      mode="area")
            feature = self.image_cnn(image * 2 - 1 if self.center_images else image)
            image_features.append(feature)
        visual = self.image_fusion(torch.cat(image_features, dim=-1))
        if self.image_temporal is not None:
            _, visual_hidden = self.image_temporal(visual.reshape(batch * steps, image_steps, -1))
            visual = visual_hidden[-1]
        tau = self.tau_projection(tau_ext.reshape(batch * steps, history, dim).float())
        _, hidden = self.tau_gru(tau)
        force = hidden[-1]
        logits = self.head(self.fusion_dropout(torch.cat((visual, force), dim=-1))).reshape(batch, steps, 3)
        return logits


def load_phase_gate(checkpoint, device="cpu"):
    """Load a reusable phase gate checkpoint saved by this experiment."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]
    model = ImageTauPhaseGate(**architecture)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_phase_gate_preprocessor(checkpoint, device="cpu"):
    """Load the saved torque filter and normalizer without loading the teacher."""
    from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    preprocessor = WrenchPreprocessor.from_config(payload["wrench_preprocessing"])
    preprocessor.load_state_dict(payload["wrench_preprocessor"], strict=True)
    return preprocessor.to(device).eval().requires_grad_(False)

