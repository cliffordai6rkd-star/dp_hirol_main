import math
import time
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


class StagedRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.ones(1))
        self.rgb = torch.nn.Parameter(torch.zeros(1))

    def set_training_stage(self, stage):
        self.gate.requires_grad_(stage == "intensity")
        self.rgb.requires_grad_(stage == "rgb")

    @torch.no_grad()
    def initialize_rgb_from_gate(self):
        self.rgb.copy_(self.gate)

    def gate_parameters(self):
        return (self.gate,)

    def rgb_parameters(self):
        return (self.rgb,)


@pytest.fixture
def workspace(tmp_path):
    cfg = OmegaConf.create({
        "distillation": {"objective": "token_mse"},
        "training": {"early_stopping": {
            "enabled": True, "patience": 5, "min_delta": 1e-4, "start_at_rgb": True,
        }},
    })
    result = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    BaseWorkspace.__init__(result, cfg, output_dir=str(tmp_path))
    result.model = StagedRenderer()
    result.optimizer = torch.optim.SGD(result.model.parameters(), lr=1e-3)
    result.lr_scheduler = None
    result.training_stage = "intensity"
    result.render_stages_enabled = True
    result.intensity_epochs = 5
    result.epoch = result.global_step = result.stage_epoch = result.stage_step = 0
    result.best_val_loss = result.early_stopping_best_loss = 0.1
    result.epochs_without_improvement = 0
    result.early_stopping_enabled = True
    result.patience = 5
    result.min_delta = 1e-4
    result.started_at = time.monotonic()
    result.validation_loss = 0.1
    result._validate = lambda loader: result.validation_loss
    result._gpu_metrics = lambda: {}
    result._summary = lambda status: None
    result.save_checkpoint = lambda **kwargs: None
    return result


def validate(workspace, loss):
    workspace.epoch += 1
    workspace.global_step += 1
    workspace.validation_loss = loss
    logger = SimpleNamespace(log=lambda row: None, save_plot=lambda: None)
    return workspace._validate_and_save(object(), logger)


def test_gate_warmup_reaches_rgb_then_waits_for_five_bad_rgb_epochs(workspace):
    for _ in range(5):
        assert not workspace._maybe_transition_stage()
        assert not validate(workspace, 0.1)
        assert workspace.epochs_without_improvement == 0

    # Even a worse RGB baseline must start a fresh early-stopping window.
    workspace.epochs_without_improvement = 4
    assert workspace._maybe_transition_stage()
    assert workspace.training_stage == "rgb"
    assert not workspace.model.gate.requires_grad
    assert workspace.model.rgb.requires_grad
    assert workspace.model.rgb.item() == workspace.model.gate.item()
    assert workspace.epochs_without_improvement == 0
    assert math.isinf(workspace.early_stopping_best_loss)
    assert not validate(workspace, 0.8)
    assert workspace.early_stopping_best_loss == 0.8

    for bad_epochs in range(1, 6):
        # Re-entering/resuming the existing RGB stage must retain its count.
        assert not workspace._maybe_transition_stage()
        assert validate(workspace, 0.8) == (bad_epochs == 5)
        assert workspace.epochs_without_improvement == bad_epochs


def test_rgb_improvement_respects_min_delta_and_restarts_patience(workspace):
    workspace.epoch = 5
    workspace._maybe_transition_stage()
    assert not validate(workspace, 0.8)
    assert not validate(workspace, 0.79995)
    assert workspace.epochs_without_improvement == 1
    assert workspace.early_stopping_best_loss == 0.8
    assert not validate(workspace, 0.7998)
    assert workspace.epochs_without_improvement == 0
    assert workspace.early_stopping_best_loss == 0.7998


def test_existing_configs_retain_early_stopping_during_intensity(workspace):
    del workspace.cfg.training.early_stopping.start_at_rgb
    for epoch in range(1, 6):
        assert validate(workspace, 0.1) == (epoch == 5)
        assert workspace.epochs_without_improvement == epoch


def test_disabling_early_stopping_keeps_rgb_training_running(workspace):
    workspace.early_stopping_enabled = False
    workspace.training_stage = "rgb"
    for _ in range(7):
        assert not validate(workspace, 0.1)
