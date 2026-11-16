"""Predict the teacher's binary phase switch from measured causal history."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalPhaseGate(nn.Module):
    def __init__(self, image_keys, history_steps=16, image_offsets=(-12, -8, -4, 0),
                 hidden_dim=64, image_size=112, threshold=0.5, loss_weight=1.0,
                 state_source_keys=("observation.joint", "observation.velocity",
                                    "observation.tau_ext_unthresholded")):
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.history_steps = int(history_steps)
        self.image_offsets = tuple(int(value) for value in image_offsets)
        self.image_size = int(image_size)
        self.threshold = float(threshold)
        self.loss_weight = float(loss_weight)
        if (self.history_steps < 1 or not self.image_offsets
                or tuple(sorted(set(self.image_offsets))) != self.image_offsets
                or self.image_offsets[-1] != 0 or self.image_offsets[0] < 1 - self.history_steps):
            raise ValueError("Phase image offsets must be distinct causal offsets ending at zero, within history")
        if len(self.image_keys) != 2 or len(state_source_keys) != 3:
            raise ValueError("Phase prediction requires two cameras and joint/velocity/torque histories")
        if not 0 < self.threshold < 1 or not self.loss_weight > 0:
            raise ValueError("Phase threshold must be in (0,1) and loss weight positive")
        self.register_buffer("state_mean", torch.zeros(21))
        self.register_buffer("state_std", torch.ones(21))
        self.register_buffer("normalizer_fitted", torch.tensor(False))
        # One shared image encoder; camera order is preserved in the GRU input.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(256, hidden_dim), nn.SiLU(),
        )
        self.image_temporal = nn.GRU(2 * hidden_dim, hidden_dim, batch_first=True)
        self.state_projection = nn.Sequential(nn.Linear(21, hidden_dim), nn.SiLU())
        self.state_temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, 1))

    @torch.no_grad()
    def fit_normalizer(self, training_states):
        values = torch.as_tensor(training_states, dtype=torch.float32, device=self.state_mean.device)
        if values.ndim != 2 or values.shape[1] != 21 or not len(values) or not torch.isfinite(values).all():
            raise ValueError("Fit phase normalization on finite training-only [N,21] measured states")
        self.state_mean.copy_(values.mean(dim=0))
        self.state_std.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.normalizer_fitted.fill_(True)

    def forward(self, inputs):
        if not bool(self.normalizer_fitted):
            raise RuntimeError("Phase normalizer must be fitted on the training split before inference")
        state = inputs["state"]
        if state.ndim != 4 or state.shape[-2:] != (self.history_steps, 21):
            raise ValueError("Phase state must have shape [B,To,history_steps,21]")
        batch, steps = state.shape[:2]
        summaries = []
        for key in self.image_keys:
            image = inputs["images"][key]
            if (image.ndim != 6 or image.shape[:3] != (batch, steps, len(self.image_offsets))
                    or image.shape[3] != 3):
                raise ValueError("Phase RGB must have shape [B,To,4,3,H,W] for each camera")
            pixels = image.flatten(0, 2).float()
            if image.dtype == torch.uint8:
                pixels = pixels / 255.0
            pixels = F.interpolate(pixels, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
            features = self.image_encoder((pixels - 0.5) / 0.5)
            summaries.append(features.reshape(batch * steps, len(self.image_offsets), -1))
        _, image_hidden = self.image_temporal(torch.cat(summaries, dim=-1))
        normalized = (state.float() - self.state_mean) / self.state_std
        projected = self.state_projection(normalized.reshape(batch * steps, self.history_steps, 21))
        _, state_hidden = self.state_temporal(projected)
        logits = self.head(torch.cat((image_hidden[-1], state_hidden[-1]), dim=-1)).reshape(batch, steps).float()
        probability = logits.sigmoid()
        # Token gradients cannot change the timing classifier through this switch.
        gate = (probability.detach() >= self.threshold).to(logits.dtype)
        return {"phase_logits": logits, "p_open": probability, "contact_alpha": gate}
