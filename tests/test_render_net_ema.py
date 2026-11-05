"""Student EMA is trained, evaluated, and resumed without changing the teacher."""

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_ddp import _config, TinyDataset, TinyPreprocessor, TinyRenderer, TinyTeacher


@pytest.fixture
def make_workspace(monkeypatch, tmp_path):
    teacher_config = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_config))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: TinyTeacher())
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: TinyDataset(tmp_path / "data"))
    monkeypatch.setattr(distill.RenderNetTrainingLogger, "save_plot", lambda self: None)

    def make(name, *, epochs=3, resume=None, enabled=True):
        config = _config(tmp_path)
        config.teacher_cache.enabled = False
        config.optimizer.lr = 0.01
        config.training.num_epochs = epochs
        config.training.early_stopping.enabled = False
        config.training.lr_scheduler = {"enabled": False}
        config.training.resume_from = str(resume) if resume is not None else None
        if enabled == "missing":
            config.training.pop("use_ema", None)
            config.pop("ema", None)
        else:
            config.training.use_ema = enabled
            config.ema = {"update_after_step": 0, "inv_gamma": 1.0, "power": 0.75,
                          "min_value": 0.0, "max_value": 0.999}
        return distill.TrainDistillRenderNetWorkspace(config, str(tmp_path / name))

    return make


def assert_same_parameters(first, second):
    for name, expected in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], expected, rtol=0, atol=0)


def test_training_updates_student_and_ema_but_keeps_teacher_frozen(make_workspace):
    workspace = make_workspace("training")
    initial = {name: value.detach().clone() for name, value in workspace.model.state_dict().items()}
    teacher_initial = {name: value.detach().clone() for name, value in workspace.teacher_policy.state_dict().items()}
    online_training_calls = []
    ema_calls = []
    workspace.model.register_forward_pre_hook(
        lambda model, args: online_training_calls.append((model.training, torch.is_grad_enabled())))
    workspace.ema_model.register_forward_pre_hook(
        lambda model, args: ema_calls.append((model.training, torch.is_grad_enabled())))
    workspace.run()

    assert workspace.global_step == workspace.ema.optimization_step == 6
    assert sum(training and grad for training, grad in online_training_calls) == 6
    assert ema_calls and all(not training and not grad for training, grad in ema_calls)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in workspace.ema_model.parameters())
    assert any(not torch.equal(value, initial[name]) for name, value in workspace.model.state_dict().items())
    assert any(not torch.equal(value, initial[name]) for name, value in workspace.ema_model.state_dict().items())
    assert any(not torch.equal(value, workspace.model.state_dict()[name])
               for name, value in workspace.ema_model.state_dict().items())
    for name, value in workspace.teacher_policy.state_dict().items():
        torch.testing.assert_close(value, teacher_initial[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in workspace.teacher_policy.parameters())

    payload = torch.load(workspace.get_checkpoint_path("latest"), pickle_module=dill)
    assert {"model", "ema_model", "ema"} <= payload["state_dicts"].keys()
    assert "averaged_model" not in payload["state_dicts"]["ema"]
    assert payload["state_dicts"]["ema"]["optimization_step"] == 6


def test_validation_depends_on_ema_weights_not_online_weights(make_workspace):
    workspace = make_workspace("validation")
    loader = workspace._loader(TinyDataset("unused", range(8, 13)),
                               workspace.cfg.val_dataloader, validation=True)
    with torch.no_grad():
        for parameter in workspace.model.parameters():
            parameter.fill_(0.0)
        for parameter in workspace.ema_model.parameters():
            parameter.fill_(0.2)
    first = workspace._validate(loader)
    with torch.no_grad():
        for parameter in workspace.model.parameters():
            parameter.fill_(10.0)
    assert workspace._validate(loader) == first
    with torch.no_grad():
        for parameter in workspace.ema_model.parameters():
            parameter.fill_(-0.2)
    assert workspace._validate(loader) != pytest.approx(first)


def test_resumed_ema_matches_uninterrupted_training(make_workspace):
    reference = make_workspace("reference", epochs=3)
    reference.run()
    first = make_workspace("first", epochs=2)
    first.run()
    resumed = make_workspace("resumed", epochs=3, resume=first.get_checkpoint_path("latest"))
    assert resumed.ema.optimization_step == first.ema.optimization_step == 4
    assert_same_parameters(first.ema_model, resumed.ema_model)
    resumed.run()
    assert resumed.ema.optimization_step == reference.ema.optimization_step == 6
    assert resumed.ema.decay == reference.ema.decay
    assert_same_parameters(reference.model, resumed.model)
    assert_same_parameters(reference.ema_model, resumed.ema_model)
    assert resumed.latest_val_loss == reference.latest_val_loss


def test_legacy_checkpoint_initializes_ema_from_restored_student(make_workspace):
    old = make_workspace("old", epochs=2, enabled="missing")
    old.run()
    payload = torch.load(old.get_checkpoint_path("latest"), pickle_module=dill)
    assert "ema_model" not in payload["state_dicts"] and "ema" not in payload["state_dicts"]
    resumed = make_workspace("upgraded", epochs=3, resume=old.get_checkpoint_path("latest"))
    assert resumed.global_step == 4 and resumed.ema.optimization_step == 0
    assert_same_parameters(old.model, resumed.model)
    assert_same_parameters(old.model, resumed.ema_model)
    resumed.run()
    assert resumed.global_step == 6 and resumed.ema.optimization_step == 2


def test_ema_can_be_disabled_when_resuming_checkpoint_with_ema(make_workspace):
    first = make_workspace("with_ema", epochs=2)
    first.run()
    resumed = make_workspace("without_ema", epochs=3,
                             resume=first.get_checkpoint_path("latest"), enabled=False)
    assert resumed.ema is None and resumed.ema_model is None
    assert_same_parameters(first.model, resumed.model)
    resumed.run()
    payload = torch.load(resumed.get_checkpoint_path("latest"), pickle_module=dill)
    assert "ema_model" not in payload["state_dicts"] and "ema" not in payload["state_dicts"]
