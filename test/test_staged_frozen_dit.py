"""Regression for DiT supervision through the staged renderer and frozen gates."""

from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import dill
import pytest
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_denoiser_distillation import make_workspace
from test_response_weighted_distillation import make_workspace as make_staged_workspace, renderer_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("token_weight,demo_weight", [(0., 0.), (.2, .7)])
def test_staged_rgb_optimizes_dit_and_validates_with_fixed_noise(token_weight, demo_weight):
    workspace, batch = make_workspace(token_loss_weight=token_weight, demo_loss_weight=demo_weight)
    workspace.render_stages_enabled = True
    workspace.training_stage = "rgb"
    loss = workspace._batch_loss(batch)
    target, prediction = workspace.teacher_policy.model.calls
    expected = F.mse_loss(prediction["output"], target["output"])
    expected = (expected + token_weight * workspace.loss_metrics["token_mse"]
                + demo_weight * workspace.loss_metrics["demo_mse"])
    torch.testing.assert_close(loss.detach(), expected)
    assert workspace._selection_mse(loss) == workspace.loss_metrics["denoiser_mse"]
    loss.backward()
    assert workspace.model.offset.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in workspace.teacher_policy.parameters())
    loader = torch.utils.data.DataLoader([batch, batch], batch_size=None)
    first = workspace._validate(loader)
    torch.randn(10)
    second = workspace._validate(loader)
    assert first == second == workspace.validation_metrics["denoiser_mse"]


def test_frozen_transformer_propagates_to_rgb_but_not_adapter_or_conditions():
    workspace, obs, cache = make_staged_workspace()
    workspace.cfg.distillation.update(objective="denoiser_mse", response_loss_weight=0.)
    config = renderer_config()
    config["context_adapter"] = {"enabled": True, "kind": "linear"}
    workspace.model = RenderNet(6, **config)
    workspace.training_stage = "rgb"
    workspace.model.set_training_stage("rgb")
    workspace.model.set_stage_mode(True)
    workspace.model.output.reset_parameters()
    workspace.teacher_policy.normalizer.fit({"action": torch.randn(32, 7)})
    frozen = copy.deepcopy(workspace.model.state_dict())
    teacher_before = copy.deepcopy(workspace.teacher_policy.state_dict())
    batch = {"obs": obs, "action": torch.randn(2, 4, 7), "teacher_targets": cache}
    optimizer = torch.optim.AdamW(workspace.model.rgb_parameters(), lr=.001)
    loss = workspace._batch_loss(batch)
    assert loss.item() == pytest.approx(workspace.loss_metrics["denoiser_mse"])
    loss.backward()
    for module in (workspace.model.output, workspace.model.backbone.stem, workspace.model.wrench_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for module in (workspace.model.context_adapter, workspace.model.condition_network):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    optimizer.step()
    for name, value in workspace.model.state_dict().items():
        if name.startswith(("context_adapter.", "condition_network.")):
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    for name, value in workspace.teacher_policy.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in workspace.teacher_policy.parameters())


def test_warm_start_uses_ema_weights_and_fresh_training_state(monkeypatch, tmp_path):
    from test_render_net_ddp import _config, TinyTeacher, TinyRenderer, TinyPreprocessor

    teacher_cfg = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: TinyTeacher())
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    model = TinyRenderer()
    with torch.no_grad():
        model.offset.weight.fill_(.314)
    path = tmp_path / "student.ckpt"
    torch.save({"state_dicts": {"ema_model": model.state_dict()},
                "pickles": {"epoch": dill.dumps(75), "global_step": dill.dumps(83175)}},
               path, pickle_module=dill)
    cfg = _config(tmp_path)
    cfg.training.use_ema = True
    cfg.training.warm_start_from = str(path)
    cfg.distillation.objective = "denoiser_mse"
    workspace = distill.TrainDistillRenderNetWorkspace(cfg, str(tmp_path / "new_run"))
    assert workspace.training_stage == "rgb"
    assert workspace.epoch == workspace.global_step == workspace.stage_epoch == 0
    assert workspace.best_val_loss == float("inf") and workspace.optimizer.state == {}
    assert workspace.ema.optimization_step == 0 and workspace.resume_from is None
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(workspace.model.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(workspace.ema_model.state_dict()[name], expected, rtol=0, atol=0)
