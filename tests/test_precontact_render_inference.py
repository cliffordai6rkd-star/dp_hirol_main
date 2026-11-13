"""Deployment uses causal predictions rather than offline onset labels."""

import pytest
import torch

from diffusion_policy.scripts import visualize_render_net_episode as visual


class CausalPreprocessor:
    n_obs_steps = 2

    def __init__(self):
        self.received_keys = []

    def __call__(self, obs, *, return_future_contact):
        self.received_keys.append(set(obs))
        wrench = obs["tau_ext"][:, :self.n_obs_steps]
        contact = torch.zeros(wrench.shape[:2], dtype=torch.bool)
        return wrench, contact, contact.unsqueeze(-1)


class PredictedAlphaRenderer:
    phase_prediction_enabled = True
    high_attention_enabled = False

    def __init__(self):
        self.alphas = []
        self.wrenches = []
        self.prediction_calls = 0

    def predict_contact_phase(self, images, wrench):
        self.prediction_calls += 1
        # A nonzero activation is possible below the physical contact threshold.
        alpha = (images["side"].mean(dim=(-3, -2, -1))
                 + wrench.mean(dim=(-2, -1))).sigmoid()
        logits = torch.stack((1 - alpha, alpha, torch.zeros_like(alpha)), dim=-1)
        return {"contact_alpha": alpha, "phase_logits": logits}

    def __call__(self, images, wrench, *, future_contact=None, contact_alpha=None,
                 return_aux=False, use_attention_mask=None):
        self.wrenches.append(wrench.clone())
        self.alphas.append(None if contact_alpha is None else contact_alpha.clone())
        alpha = (self.predict_contact_phase(images, wrench)["contact_alpha"]
                 if contact_alpha is None else contact_alpha)
        rendered = {key: image + 0.1 * alpha[..., None, None, None]
                    for key, image in images.items()}
        masks = {key: torch.ones_like(image[:, :, :1]) for key, image in images.items()}
        return (rendered, masks) if return_aux else rendered


def observations():
    return {"side": torch.full((1, 2, 3, 4, 4), 0.2),
            "tau_ext": torch.full((1, 2, 8, 7), 0.01),
            "contact_alpha": torch.zeros(1, 2),
            "contact_phase": torch.zeros(1, 2, dtype=torch.long)}


@pytest.mark.parametrize("with_labels", [False, True])
def test_visualization_uses_predicted_alpha_without_offline_labels(with_labels):
    model, preprocessor = PredictedAlphaRenderer(), CausalPreprocessor()
    obs = observations()
    if not with_labels:
        obs.pop("contact_alpha")
        obs.pop("contact_phase")
    images, rendered = visual.render_observations(model, preprocessor, obs, ["side"], 2)
    assert model.alphas == [None]
    assert model.prediction_calls == 1
    assert "contact_alpha" not in preprocessor.received_keys[0]
    assert "contact_phase" not in preprocessor.received_keys[0]
    torch.testing.assert_close(model.wrenches[0], obs["tau_ext"])
    assert (rendered["side"] > images["side"]).all()


def test_legacy_visualization_preserves_explicit_alpha_override():
    model, preprocessor = PredictedAlphaRenderer(), CausalPreprocessor()
    model.phase_prediction_enabled = False
    obs = observations()
    images, rendered = visual.render_observations(model, preprocessor, obs, ["side"], 2)
    torch.testing.assert_close(model.alphas[0], obs["contact_alpha"])
    assert model.prediction_calls == 0
    torch.testing.assert_close(rendered["side"], images["side"])
