"""Pre-contact labels open fusion without suppressing weak causal torque."""

import torch

from test_contact_alpha import make_current_teacher, current_observations
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor


def test_precontact_preprocessing_retains_subthreshold_history_and_roundtrips():
    teacher = make_current_teacher().eval().requires_grad_(False)
    teacher.precontact_seconds = .5
    obs = current_observations()
    obs["wrench"].fill_(.2)
    obs["contact_alpha"] = torch.tensor([[0., .5], [.25, 1.]])
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    normalized, contact, _ = preprocessor(obs)
    assert not contact.any()
    torch.testing.assert_close(normalized, teacher.normalizer["wrench"].normalize(obs["wrench"]))
    restored = WrenchPreprocessor.from_config(preprocessor.get_config())
    restored.load_state_dict(preprocessor.state_dict())
    assert restored.precontact_seconds == .5
    torch.testing.assert_close(restored(obs)[0], normalized)


def test_teacher_uses_explicit_precontact_alpha_through_final_context_norm():
    torch.manual_seed(23)
    teacher = make_current_teacher().eval().requires_grad_(False)
    teacher.precontact_seconds = .5
    obs = current_observations()
    obs["wrench"].fill_(.2)
    obs["contact_alpha"] = torch.full((2, 2), .5)
    images, wrench, contact, future_contact = teacher.prepare_context_inputs_with_future_contact(obs)
    assert not contact.any()
    with torch.no_grad():
        actual = teacher._encode_context(obs, apply_curriculum_mask=False)
        expected = teacher.obs_encoder(images=images, wrench_history=wrench,
                                        future_contact=future_contact,
                                        contact_alpha=obs["contact_alpha"])
        closed = teacher._encode_context({**obs, "contact_alpha": torch.zeros(2, 2)},
                                         apply_curriculum_mask=False)
        image_only = teacher.obs_encoder.encode_image_context(images, apply_fusion_ffn=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(closed, image_only)
    assert not torch.allclose(actual, closed)
