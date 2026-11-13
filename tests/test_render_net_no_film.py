"""Region conditioning reaches RGB through the final gate, not decoder FiLM."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.model.vision.render_net import build_render_net
from test_render_net_shared_condition import _observations, _renderer


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shared", [False, True])
def test_render_net_has_no_film_parameters_or_runtime_switch(shared):
    model = _renderer(shared=shared)
    assert not hasattr(model, "spatial_modulations")
    assert not hasattr(model, "spatial_modulation_enabled")
    assert not any("modulation" in name.lower() or "film" in name.lower()
                   for name in model.state_dict())


def test_builder_accepts_explicitly_disabled_legacy_film_without_changing_weights():
    config = {"base_channels": 4, "conditioning_dim": 8, "embed_dim": 16,
              "patch_size": 8, "num_heads": 4, "wrench_history_steps": 8}
    torch.manual_seed(2202)
    current = build_render_net(7, config)
    legacy = {**config, "spatial_modulation": {"enabled": False}}
    torch.manual_seed(2202)
    restored = build_render_net(7, legacy)
    assert legacy["spatial_modulation"] == {"enabled": False}
    assert current.state_dict().keys() == restored.state_dict().keys()
    for name, expected in current.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_builder_rejects_enabled_legacy_film_instead_of_silently_changing_architecture():
    with pytest.raises(ValueError, match="FiLM"):
        build_render_net(7, {"spatial_modulation": {"enabled": True}})


def test_current_student_presets_do_not_expose_removed_film_option():
    config_dir = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    presets = sorted(config_dir.glob("train_distill_render_net*.yaml"))
    assert presets
    for path in presets:
        config = OmegaConf.load(path)
        assert "spatial_modulation" not in config.get("render_net", {}), path.name


@pytest.mark.parametrize("shared", [False, True])
def test_region_changes_only_final_rgb_gate_while_decoder_features_stay_fixed(shared):
    torch.manual_seed(2201)
    model = _renderer(shared=shared).eval()
    model.residual_scale = .1
    images, wrench = _observations()
    images = {key: torch.full_like(image, .5) for key, image in images.items()}
    decoder_outputs = []
    handle = model.backbone.output.register_forward_hook(
        lambda module, args, output: decoder_outputs.append(output.detach().clone())
    )
    try:
        rendered_by_gate = []
        with torch.no_grad():
            model.output.weight.normal_(std=.03)
            model.mask_head.weight[0].zero_()
            for probability in (.25, .75):
                model.mask_head.bias[0].fill_(torch.logit(torch.tensor(probability)))
                rendered, masks = model(
                    images, wrench, contact_alpha=torch.ones(2, 2), return_aux=True,
                )
                rendered_by_gate.append(rendered)
                for mask in masks.values():
                    torch.testing.assert_close(mask, torch.full_like(mask, probability))
        cameras = len(images)
        assert len(decoder_outputs) == 2 * cameras
        for first, second in zip(decoder_outputs[:cameras], decoder_outputs[cameras:]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for key, image in images.items():
            first = rendered_by_gate[0][key] - image
            second = rendered_by_gate[1][key] - image
            assert first.abs().max() > 0
            # Subtracting the unchanged float32 RGB base adds rounding error.
            torch.testing.assert_close(second, 3 * first, rtol=1e-5, atol=2e-7)
    finally:
        handle.remove()
