"""Dataset phases select spatial/token supervision without hiding free-phase labels."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from diffusion_policy.workspace import train_distill_render_net_workspace as distill


def make_workspace(monkeypatch, *, phases=(1, 2), stage="joint"):
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({
        "distillation": {"objective": "token_mse", "token_region_phases": phases,
                         "phase_ce_weight": .7, "alpha_loss_weight": .3},
    })
    workspace.device = torch.device("cpu")
    workspace.amp_enabled = False
    workspace.attention_enabled = True
    workspace.attention_config = {"pixel_loss_weight": 1.}
    workspace.context_patch_loss_weight = workspace.intensity_loss_weight = 1.
    workspace.smoothness_loss_weight = .02
    workspace.shape_info = {}
    workspace.training_stage = stage
    workspace.teacher_policy = SimpleNamespace(n_obs_steps=2)
    workspace.model = torch.nn.Module()
    workspace.model.phase_prediction_enabled = True
    # Errors distinguish all four observation slots, including free/contact
    # transitions within the same sample. Each slot contains four patch tokens.
    workspace.model.tokens = torch.nn.Parameter(
        torch.tensor([[10., 2.], [3., 20.]])[..., None, None].expand(2, 2, 4, 1)
        .clone().reshape(2, 8, 1))
    workspace.model.field = torch.nn.Parameter(torch.tensor([
        [[[10., 11.], [12., 13.]], [[.1, .2], [.3, .4]]],
        [[[.2, .4], [.6, .8]], [[20., 22.], [24., 26.]]],
    ]).unsqueeze(2))
    workspace.model.phase_logits = torch.nn.Parameter(torch.zeros(2, 2, 3))
    workspace.model.alpha = torch.nn.Parameter(torch.full((2, 2), .25))

    def student_context(teacher, model, obs, **kwargs):
        tokens = None if kwargs.get("condition_only") else model.tokens
        return (tokens, {"side": model.field}, obs["contact_alpha"] > 0,
                {"phase_logits": model.phase_logits, "contact_alpha": model.alpha})

    monkeypatch.setattr(distill, "student_context_tokens", student_context)
    workspace._region_masks = lambda patches, obs: {
        "side": patches["side"].reshape(2, 2, 1, 2, 2)}
    workspace._region_patch_targets = lambda patches: patches
    workspace._patch_pool = lambda masks: {"side": masks["side"].flatten(2)}
    obs = {"contact_phase": torch.tensor([[0, 1], [2, 0]]),
           "contact_alpha": torch.tensor([[0., .5], [1., 0.]])}
    targets = {"tokens": torch.zeros(2, 8, 1),
               "patch_attention": {"side": torch.zeros(2, 2, 4)}}
    return workspace, obs, targets


def test_phase_selected_token_and_all_region_terms_have_no_free_slot_gradient(monkeypatch):
    workspace, obs, targets = make_workspace(monkeypatch)
    loss = workspace._loss(obs, cached_targets=targets)
    metrics = workspace.loss_metrics
    assert metrics["token_region_observation_count"] == 2
    assert metrics["token_mse_measured"]
    assert metrics["token_mse"] == pytest.approx(6.5)
    assert metrics["mask_mse"] == pytest.approx(.1875)
    assert metrics["context_patch_mse"] == pytest.approx(.09375)
    assert metrics["intensity_smoothness"] == pytest.approx(.225)
    assert metrics["intensity_loss"] == pytest.approx(.1875 + .09375 + .02 * .225)
    expected_condition = (.7 * F.cross_entropy(workspace.model.phase_logits.flatten(0, 1),
                                               obs["contact_phase"].flatten())
                          + .3 * F.mse_loss(workspace.model.alpha, obs["contact_alpha"]))
    assert loss.item() == pytest.approx(6.5 + metrics["intensity_loss"] + expected_condition.item())
    loss.backward()
    selected = obs["contact_phase"] > 0
    token_gradient = workspace.model.tokens.grad.reshape(2, 2, 4, 1)
    assert token_gradient[~selected].count_nonzero() == 0
    assert token_gradient[selected].abs().sum() > 0
    assert workspace.model.field.grad[~selected].count_nonzero() == 0
    assert workspace.model.field.grad[selected].abs().sum() > 0
    # Free labels still train both phase classification and alpha closure.
    assert (workspace.model.phase_logits.grad[~selected].abs().sum(-1) > 0).all()
    assert (workspace.model.alpha.grad[~selected].abs() > 0).all()

    with torch.no_grad():
        workspace.model.tokens.reshape(2, 2, 4, 1)[~selected] += 1000
        workspace.model.field[~selected] += 1000
    unchanged = workspace._loss(obs, cached_targets=targets)
    assert unchanged.item() == pytest.approx(loss.item())


def test_free_only_batch_returns_graph_connected_zero_spatial_losses_but_trains_conditions(monkeypatch):
    workspace, obs, targets = make_workspace(monkeypatch)
    obs["contact_phase"].zero_()
    obs["contact_alpha"].zero_()
    loss = workspace._loss(obs, cached_targets=targets)
    metrics = workspace.loss_metrics
    assert metrics["token_region_observation_count"] == 0
    assert not metrics["token_mse_measured"]
    for key in ("token_mse", "mask_mse", "context_patch_mse", "intensity_smoothness", "intensity_loss"):
        assert metrics[key] == 0
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert workspace.model.tokens.grad.count_nonzero() == 0
    assert workspace.model.field.grad.count_nonzero() == 0
    assert workspace.model.phase_logits.grad.abs().sum() > 0
    assert workspace.model.alpha.grad.abs().sum() > 0


def test_missing_phase_filter_keeps_legacy_full_observation_supervision(monkeypatch):
    workspace, obs, targets = make_workspace(monkeypatch, phases=None)
    workspace.cfg.distillation.pop("token_region_phases")
    workspace._loss(obs, cached_targets=targets)
    assert workspace.token_region_phases is None
    assert workspace.loss_metrics["token_region_observation_count"] == 4
    assert workspace.loss_metrics["token_mse"] == pytest.approx(128.25)


def test_warmup_only_masks_region_and_continues_phase_alpha_supervision(monkeypatch):
    workspace, obs, targets = make_workspace(monkeypatch, stage="condition_warmup")
    loss = workspace._loss(obs, cached_targets=targets)
    assert not workspace.loss_metrics["token_mse_measured"]
    assert workspace.loss_metrics["token_region_observation_count"] == 2
    loss.backward()
    assert workspace.model.tokens.grad is None
    assert workspace.model.field.grad[obs["contact_phase"] == 0].count_nonzero() == 0
    assert workspace.model.phase_logits.grad[obs["contact_phase"] == 0].abs().sum() > 0


@pytest.mark.parametrize("phases", [[], [3], [-1], [1.5], [True], "contact", 1])
def test_invalid_phase_selection_is_rejected(monkeypatch, phases):
    workspace, _, _ = make_workspace(monkeypatch, phases=phases)
    with pytest.raises(ValueError, match="token_region_phases"):
        workspace._validate_distillation_config()


def test_phase_selection_requires_labels_and_cannot_silently_mask_denoiser_objective(monkeypatch):
    workspace, obs, targets = make_workspace(monkeypatch)
    workspace.cfg.distillation.objective = "denoiser_mse"
    with pytest.raises(ValueError, match="requires objective=token_mse"):
        workspace._validate_distillation_config()
    workspace.cfg.distillation.objective = "token_mse"
    del obs["contact_phase"]
    with pytest.raises(KeyError, match="contact_phase"):
        workspace._loss(obs, cached_targets=targets)


def test_training_ddp_weights_selected_losses_by_global_observation_count(monkeypatch):
    workspace, obs, targets = make_workspace(monkeypatch)
    workspace.distributed = SimpleNamespace(enabled=True, world_size=2)
    workspace._sum = lambda values: [3.]
    loss = workspace._loss(obs, cached_targets=targets)
    metrics = workspace.loss_metrics
    expected = ((metrics["token_mse"] + metrics["intensity_loss"]) * 4 / 3
                + .7 * metrics["phase_ce"] + .3 * metrics["alpha_mse"])
    assert loss.item() == pytest.approx(expected)
    # Local metrics retain their own means; the parent loop can aggregate them
    # with the supplied count. Uneven validation shards must never all-reduce.
    assert metrics["token_mse"] == pytest.approx(6.5)
    workspace.model.eval()
    workspace._sum = lambda _: (_ for _ in ()).throw(AssertionError("validation collective"))
    with torch.no_grad():
        assert workspace._loss(obs, cached_targets=targets).item() == pytest.approx(6.5)
