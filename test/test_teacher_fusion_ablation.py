"""Architecture equivalence, closed-gate invariance and FiLM gradient checks."""

from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "test")]

import pytest
import torch

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.model.vision.teacher_fusion_ablation import TeacherFusionAblationEncoder
from test_configurable_force_fusion import PatchBackbone, IMAGE_SHAPES
from run_teacher_fusion_ablation import common_initialization_hash, evaluate


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def encoder(mode):
    torch.manual_seed(42)
    kwargs = dict(
        pretrained_model_name_or_path="unused", image_shapes=IMAGE_SHAPES, backbone=PatchBackbone(),
        n_emb=16, n_head=4, max_obs_steps=2, wrench_dim=6, wrench_history_steps=8,
        force_temporal_encoder="lstm", force_encoder_layers=2, dropout=0.,
        cross_attn_query="image", cross_attn_layers=2, cross_attn_query_residual=True,
    )
    return (ForceAwareObsEncoder(**kwargs) if mode == "original"
            else TeacherFusionAblationEncoder(**kwargs, fusion_mode=mode)).eval()


def inputs():
    return dict(images={key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()},
                wrench_history=torch.randn(2, 2, 8, 6), contact_alpha=torch.tensor([[0., .3], [.8, 1.]]))


def test_baseline_matches_current_teacher_exactly_and_hybrid_starts_as_baseline():
    original, baseline, hybrid = [encoder(mode) for mode in ("original", "baseline", "hybrid")]
    obs = inputs()
    expected = original(**obs)
    torch.testing.assert_close(baseline(**obs), expected, rtol=0, atol=0)
    torch.testing.assert_close(hybrid(**obs), expected, rtol=0, atol=0)
    assert set(original.state_dict()) == set(baseline.state_dict())
    assert common_initialization_hash(original) == common_initialization_hash(baseline)
    assert common_initialization_hash(original) == common_initialization_hash(hybrid)


@pytest.mark.parametrize("mode", ["baseline", "film_only", "hybrid"])
def test_closed_alpha_matches_original_visual_ffn_path_even_after_film_training(mode):
    model = encoder(mode)
    obs = inputs()
    for film in model.film_layers:
        torch.nn.init.normal_(film.mlp[-1].weight, std=.1)
        torch.nn.init.normal_(film.mlp[-1].bias, std=.1)
    obs["contact_alpha"].zero_()
    with torch.no_grad():
        output = model(**obs)
        visual = model.encode_image_context(obs["images"], apply_fusion_ffn=True)
        obs["wrench_history"] = torch.randn_like(obs["wrench_history"]) * 20
        changed = model(**obs)
    torch.testing.assert_close(output, visual, rtol=0, atol=0)
    torch.testing.assert_close(output, changed, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["film_only", "hybrid"])
def test_zero_initial_film_receives_gradient_then_reaches_pool_and_lstm(mode):
    model = encoder(mode).train()
    obs = inputs()
    obs["contact_alpha"].fill_(1.)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for film in model.film_layers:
        assert torch.count_nonzero(film.mlp[-1].weight) == 0
        assert torch.count_nonzero(film.mlp[-1].bias) == 0
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**obs)
        (output * torch.randn_like(output)).mean().backward()
        assert all(film.mlp[-1].weight.grad.abs().sum() > 0 for film in model.film_layers)
        if step:
            assert model.torque_pool.query.grad.abs().sum() > 0
            assert model.force_encoder.weight_ih_l0.grad.abs().sum() > 0
        optimizer.step()
    assert all(p.grad is None for p in model.backbone.parameters())
    state = copy.deepcopy(model.state_dict())
    restored = encoder(mode)
    restored.load_state_dict(state, strict=True)
    model.eval()
    torch.testing.assert_close(restored(**obs), model(**obs), rtol=0, atol=0)


def test_film_only_has_no_attention_parameters_and_common_initialization_is_preserved():
    baseline, film = encoder("baseline"), encoder("film_only")
    assert not any("cross_attention." in name for name, _ in film.named_parameters())
    assert common_initialization_hash(baseline) == common_initialization_hash(film)


def test_intervention_evaluation_reuses_clear_dino_and_fixed_noise_without_changing_alpha():
    from omegaconf import OmegaConf
    from test_force_aware_diffusion import make_policy

    policy = make_policy().eval()
    policy.precontact_seconds = .5
    policy.obs_encoder.cross_attn_query = "image"

    class Samples(torch.utils.data.Dataset):
        phase_labels = torch.tensor([0, 1, 2]).numpy()
        episode_index = torch.tensor([3, 4, 5]).numpy()
        indices = torch.arange(3).numpy()[:, None]

        def __init__(self):
            self.rows = [{"obs": {"wrist": torch.rand(2, 3, 16, 16),
                                  "wrench_ext": torch.randn(2, 8, 6),
                                  "contact_alpha": torch.full((2,), alpha),
                                  "contact_phase": torch.full((2,), phase)},
                          "action": torch.randn(9, 7)}
                         for phase, alpha in enumerate((0., .5, 1.))]

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return self.rows[index]

        def _anchor_index_from_sampler_row(self, row):
            return int(row[0])

    samples = Samples()
    shuffled = torch.stack([row["obs"]["wrench_ext"] for row in samples.rows]).roll(1, 0)
    config = OmegaConf.create({"val_dataloader": {"batch_size": 2}, "ablation": {"seed": 42}})
    cache = {}
    result, rows, _ = evaluate(policy, samples, shuffled, config, torch.device("cpu"), cache,
                              sample_actions=True)
    calls = policy.obs_encoder.backbone.forward_calls
    torch.randn(100)
    again, repeated, _ = evaluate(policy, samples, shuffled, config, torch.device("cpu"), cache,
                                  sample_actions=True)
    assert policy.obs_encoder.backbone.forward_calls == calls == 2
    assert result == again and rows == repeated
    assert result["free"]["context_response_rms"] == 0
    assert result["free"]["action_prediction_difference_rms"] == 0
    assert result["contact"]["context_response_rms"] > 0
    assert samples.rows[1]["obs"]["contact_alpha"].tolist() == [.5, .5]
