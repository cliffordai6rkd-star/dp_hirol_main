"""Exercise condition warmup and joint distillation through the real renderer."""

from datetime import timedelta
import json
import copy

import dill
import pytest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from omegaconf import OmegaConf

from test_render_net_distillation import make_attention_teacher
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext


def _workspace():
    teacher, obs, _ = make_attention_teacher()
    teacher.precontact_seconds = .5
    obs["contact_phase"] = torch.tensor([[0, 1], [1, 2]])
    obs["contact_alpha"] = torch.tensor([[0., .25], [.75, 1.]])
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({
        "distillation": {"objective": "token_mse", "use_teacher_ffn": True,
                         "phase_ce_weight": .7, "alpha_loss_weight": .3},
        "training": {"distributed": {"find_unused_parameters": True}},
    })
    workspace.teacher_policy = teacher
    workspace.device = torch.device("cpu")
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        patch_size=8, decoder_layers=3, num_heads=4, separate_gate_path=True,
        high_attention_render={"enabled": True, "threshold": 0., "detach_for_rgb": False},
        spatial_modulation={"enabled": True},
        phase_prediction={"enabled": True, "hidden_dim": 8},
    )
    workspace.model.freeze_wrench_encoder()
    workspace.attention_enabled = True
    workspace.attention_config = {"enabled": True, "source": "context",
                                  "baseline": "raw_zero", "normalization": "image_minmax",
                                  "pixel_loss_weight": 0.}
    workspace.region_source = "context"
    workspace.context_patch_loss_weight = 1.
    workspace.intensity_loss_weight = 1.
    workspace.smoothness_loss_weight = .02
    workspace.amp_enabled = False
    workspace.shape_info = None
    workspace.render_stages_enabled = True
    workspace.condition_warmup_epochs = 1
    workspace.training_stage = "condition_warmup"
    workspace.epoch = 0
    workspace.stage_step = workspace.stage_epoch = 0
    workspace.model.set_training_stage("condition_warmup")
    workspace.model.set_stage_mode(True)
    return workspace, obs


def _targets(workspace, obs):
    with torch.no_grad():
        tokens = workspace.teacher_policy._encode_context(obs, apply_curriculum_mask=False)
    patches = {key: torch.linspace(0, 1, count).expand(2, 2, -1).clone()
               for key, count in workspace.teacher_policy.obs_encoder.patch_counts.items()}
    return {"tokens": tokens, "patch_attention": patches}


def test_condition_warmup_bypasses_rgb_then_joint_updates_both_paths_and_uses_predicted_alpha():
    torch.manual_seed(913)
    workspace, obs = _workspace()
    targets = _targets(workspace, obs)
    seen_alpha = []
    hook = workspace.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen_alpha.append(kwargs.get("contact_alpha")), with_kwargs=True,
    )
    rgb_calls = []
    rgb_hook = workspace.model.backbone.register_forward_hook(lambda *args: rgb_calls.append(1))
    optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
    condition_before = workspace.model.mask_head.weight.detach().clone()
    rgb_before = workspace.model.output.weight.detach().clone()
    try:
        loss = workspace._loss(obs, cached_targets=targets)
        expected = (.7 * workspace.loss_metrics["phase_ce"]
                    + .3 * workspace.loss_metrics["alpha_mse"]
                    + workspace.loss_metrics["intensity_loss"])
        assert abs(loss.item() - expected) < 1e-6
        assert not workspace.loss_metrics["token_mse_measured"]
        assert workspace._selection_mse(loss) == loss.item()
        assert workspace.shape_info["student_token_shape"] == list(targets["tokens"].shape)
        assert not rgb_calls
        loss.backward()
        optimizer.step()
        assert not torch.equal(condition_before, workspace.model.mask_head.weight)
        torch.testing.assert_close(rgb_before, workspace.model.output.weight, rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in workspace.model.wrench_encoder.parameters())
        workspace.epoch = 1
        assert workspace._maybe_transition_stage()
        assert workspace.training_stage == "joint"
        assert workspace.model.mask_head.weight.requires_grad
        assert workspace.model.output.weight.requires_grad
        assert not workspace._maybe_transition_stage()
        workspace.model.set_stage_mode(True)
        optimizer.zero_grad(set_to_none=True)
        loss = workspace._loss(obs, cached_targets=targets)
        metrics = workspace.loss_metrics
        expected = (metrics["token_mse"] + .7 * metrics["phase_ce"]
                    + .3 * metrics["alpha_mse"] + metrics["intensity_loss"])
        assert abs(loss.item() - expected) < 1e-6
        loss.backward()
        assert workspace.model.output.weight.grad.abs().sum() > 0
        assert workspace.model.mask_head.weight.grad.abs().sum() > 0
        assert rgb_calls and all(value is None for value in seen_alpha)
        assert metrics["token_mse_measured"]
        assert workspace._selection_mse(loss) == metrics["token_mse"]
        assert not any(parameter.grad is not None for parameter in workspace.teacher_policy.parameters())
    finally:
        hook.remove()
        rgb_hook.remove()


def _ddp_worker(rank, init_file):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                                   world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(914)
        workspace, obs = _workspace()
        workspace.distributed = DistributedContext(rank=rank, local_rank=rank, world_size=2)
        workspace._setup_distributed_model()
        # Different data on each rank makes missing joint-stage reducer hooks
        # observable as diverging RGB parameters after warmup.
        obs = {key: value.clone() for key, value in obs.items()}
        obs["side"].add_(rank * .1)
        obs["wrench"].add_(rank * .2)
        targets = _targets(workspace, obs)
        optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
        for step in range(3):
            if step == 1:
                workspace.epoch = 1
                workspace._maybe_transition_stage()
            workspace.model.set_stage_mode(True)
            optimizer.zero_grad(set_to_none=True)
            workspace._loss(obs, cached_targets=targets).backward()
            optimizer.step()
        state = torch.cat([parameter.detach().flatten() for parameter in workspace.model.parameters()])
        replicas = [torch.empty_like(state) for _ in range(2)]
        distributed.all_gather(replicas, state)
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    finally:
        distributed.destroy_process_group()


def test_ddp_condition_warmup_to_joint_keeps_all_trainable_replicas_synchronized(tmp_path):
    multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / "ddp_init"),), nprocs=2, join=True)


@pytest.mark.parametrize("strict", [False, True, "plateau"])
def test_real_workspace_runs_condition_then_render_with_stage_selection(monkeypatch, tmp_path, strict):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         "diffusion_policy/config/train_distill_render_net_student.yaml")
    if "hydra" in cfg:
        del cfg.hydra
    cfg.distillation.phase_ce_weight = .7
    cfg.distillation.alpha_loss_weight = .3
    cfg.teacher_checkpoint = str(tmp_path / "teacher.ckpt")
    cfg.teacher_cache.enabled = False
    cfg.training.device = "cpu"
    cfg.training.amp.enabled = False
    cfg.training.num_epochs = 3 if strict == "plateau" else 2
    cfg.training.render_stages = {
        "enabled": True, "condition_warmup_epochs": 2 if strict == "plateau" else 1,
        "rgb_warmup_epochs": 1 if strict else 0, "joint_after_rgb": not strict,
        "condition_early_stopping": {"enabled": True, "patience": 1, "min_delta": .01},
    }
    if strict:
        cfg.render_net.condition_backbone.update({"base_channels": 4, "embed_dim": 8,
                                                 "num_heads": 2, "patch_size": 8,
                                                 "image_self_attn_layers": 0, "cross_attn_layers": 1,
                                                 "decoder_layers": 0})
    else:
        cfg.render_net.pop("condition_backbone", None)
    cfg.render_net.shared_condition_backbone = False
    cfg.render_net.separate_gate_path = True
    cfg.logging.plot_every_steps = 0
    cfg.render_net.phase_prediction.hidden_dim = 8
    cfg.render_net.backbone.update({"image_encoder": "cnn", "patch_size": 8,
                                    "base_channels": 4, "embed_dim": 16,
                                    "image_self_attn_layers": 1, "cross_attn_layers": 1})
    for loader in (cfg.dataloader, cfg.val_dataloader):
        loader.batch_size = 2
        loader.num_workers = 0
        loader.persistent_workers = False
        loader.pin_memory = False
    teacher, obs, _ = make_attention_teacher()
    teacher.obs_encoder.force_encoder = torch.nn.LSTM(16, 16, batch_first=True)
    teacher.obs_encoder.force_temporal_encoder_type = "lstm"
    obs["contact_phase"] = torch.tensor([[0, 1], [1, 2]])
    obs["contact_alpha"] = torch.tensor([[0., .25], [.75, 1.]])
    teacher_cfg = OmegaConf.create({"policy": {"precontact_seconds": .5},
                                   "task": {"dataset": {}}})
    monkeypatch.setattr(distill, "resolve_student_teacher", lambda _: (tmp_path / "teacher.ckpt", teacher_cfg))
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *a, **kw: teacher)

    class Dataset:
        def __init__(self):
            self.closed = False
            self.lerobot_dataset = SimpleNamespace(close=self.close)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"obs": {key: value[index] for key, value in obs.items()}}

        def get_validation_dataset(self):
            return self

        def close(self):
            self.closed = True

    dataset = Dataset()
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: dataset)
    workspace = distill.TrainDistillRenderNetWorkspace(cfg, output_dir=str(tmp_path))
    if strict == "plateau":
        validate = workspace._validate
        stage_losses = iter([1., 2., .5])
        def plateau_validation(loader):
            validate(loader)
            return next(stage_losses)
        monkeypatch.setattr(workspace, "_validate", plateau_validation)
    workspace.run()
    records = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    train = [record for record in records if "train_mse" in record]
    assert len(train) == 2
    assert train[0]["objective"] == "condition_warmup"
    assert not train[0]["train_token_mse_measured"]
    assert train[0]["train_mse"] == pytest.approx(train[0]["train_loss"])
    assert train[0]["train_loss"] > 0
    assert train[1]["objective"] == "token_mse"
    assert train[1]["train_token_mse_measured"]
    assert train[1]["train_mse"] == train[1]["train_token_mse"]
    assert workspace.training_stage == ("rgb" if strict else "joint")
    assert workspace.global_step == 2
    assert workspace.model.mask_head.weight.requires_grad == (not strict)
    if strict:
        assert workspace.stage_epoch == 1
        initial = [record for record in records if "initial_val_mse" in record][0]
        assert initial["objective"] == "condition_warmup"
        assert not initial["val_token_mse_measured"]
        assert not initial["train_token_mse_measured"]
        condition_val = [record for record in records if "val_mse" in record and record["epoch"] == 1][0]
        assert condition_val["objective"] == "condition_warmup"
        assert not condition_val["val_token_mse_measured"]
        assert (tmp_path / "checkpoints/condition_best.ckpt").is_file()
        assert workspace.best_epoch == 2
        assert workspace.ema.optimization_step == 1
        gate_ids = {id(parameter) for parameter in workspace.model.gate_parameters()}
        for name, parameter in workspace.model.named_parameters():
            if id(parameter) in gate_ids:
                torch.testing.assert_close(parameter, workspace.condition_best_state[name], rtol=0, atol=0)
                torch.testing.assert_close(workspace.ema_model.state_dict()[name], parameter, rtol=0, atol=0)
        if strict == "plateau":
            assert workspace.condition_best_epoch == 0
            assert workspace.epoch == 2 < workspace.num_epochs
            assert workspace.stop_reason == "render_max_epochs"
            assert workspace.stage_transition_reason.startswith("condition_plateau:")
    assert workspace.model.output.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in workspace.model.wrench_encoder.parameters())
    assert not any(parameter.grad is not None for parameter in teacher.parameters())
    assert workspace.teacher_policy.precontact_seconds == .5
    assert dataset.closed
    assert (tmp_path / "checkpoints/latest.ckpt").is_file()


def _strict_workspace(tmp_path):
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    workspace, obs = _workspace()
    workspace.joint_after_rgb = False
    workspace.rgb_warmup_epochs = 2
    workspace.condition_warmup_epochs = 3
    workspace.condition_early_stopping = {"enabled": True, "patience": 1, "min_delta": 0.01}
    workspace.condition_transition_pending = False
    workspace.condition_best_state = None
    workspace.condition_best_metric = float("inf")
    workspace.condition_best_epoch = None
    workspace.cfg.optimizer = {"lr": .001}
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
    workspace.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(workspace.optimizer, patience=0)
    workspace.ema_model = copy.deepcopy(workspace.model)
    workspace.ema = EMAModel(workspace.ema_model)
    workspace.global_step = 0
    workspace.best_val_loss = workspace.early_stopping_best_loss = float("inf")
    workspace.epochs_without_improvement = 0
    workspace.best_epoch = workspace.best_step = 0
    workspace.min_delta = 0.01
    workspace.patience = 1
    workspace.early_stopping_enabled = True
    workspace._output_dir = str(tmp_path)
    return workspace, obs


def test_strict_condition_validation_bypasses_rgb_and_uses_condition_loss(tmp_path):
    workspace, obs = _strict_workspace(tmp_path)
    workspace.model.eval()
    calls = []
    handle = workspace.ema_model.backbone.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        loss = workspace._loss(obs, cached_targets=_targets(workspace, obs))
    handle.remove()
    assert not calls
    assert not workspace.loss_metrics["token_mse_measured"]
    expected = (.7 * workspace.loss_metrics["phase_ce"]
                + .3 * workspace.loss_metrics["alpha_mse"]
                + workspace.loss_metrics["intensity_loss"])
    assert workspace._selection_mse(loss) == pytest.approx(expected)
    assert workspace._selection_weight(2) == 2


def test_strict_transition_restores_best_ema_and_freezes_entire_condition_path(tmp_path):
    workspace, obs = _strict_workspace(tmp_path)
    workspace.best_val_loss = .25
    workspace.epoch = 1
    with torch.no_grad():
        workspace.ema_model.mask_head.weight.fill_(.125)
        workspace.model.mask_head.weight.fill_(.25)
    workspace._remember_best_condition()
    selected_state = {key: value.clone() for key, value in workspace.condition_best_state.items()}
    workspace.optimizer.param_groups[0]["lr"] = 1e-5
    workspace.lr_scheduler.step(.1)
    workspace.lr_scheduler.step(.2)
    workspace.epochs_without_improvement = 7
    workspace.condition_transition_pending = True
    assert workspace._maybe_transition_stage()
    assert workspace.training_stage == "rgb"
    assert workspace.stage_transition_reason == "condition_plateau:condition_warmup->rgb"
    assert workspace.stage_epoch == 0 and workspace.stage_step == 0
    assert workspace.optimizer.param_groups[0]["lr"] == .001
    assert workspace.lr_scheduler.best == float("inf")
    assert workspace.best_val_loss == float("inf") and workspace.epochs_without_improvement == 0
    assert workspace.ema.optimization_step == 0
    gate_ids = {id(parameter) for parameter in workspace.model.gate_parameters()}
    frozen = {name: parameter.detach().clone() for name, parameter in workspace.model.named_parameters()
              if id(parameter) in gate_ids}
    assert frozen
    for name, value in frozen.items():
        torch.testing.assert_close(value, selected_state[name], rtol=0, atol=0)
        assert not dict(workspace.model.named_parameters())[name].requires_grad
    for _ in range(2):
        workspace.model.set_stage_mode(True)
        workspace.optimizer.zero_grad(set_to_none=True)
        loss = workspace._loss(obs, cached_targets=_targets(workspace, obs))
        loss.backward()
        workspace.optimizer.step()
        workspace.ema.step(workspace.model)
    for name, value in frozen.items():
        torch.testing.assert_close(workspace.model.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(workspace.ema_model.state_dict()[name], value, rtol=0, atol=0)
    workspace.epoch = 100
    assert not workspace._maybe_transition_stage()
    assert workspace.training_stage == "rgb"


def test_condition_checkpoint_retains_best_snapshot_and_pending_transition(tmp_path):
    workspace, _ = _strict_workspace(tmp_path)
    workspace.best_val_loss = .2
    workspace.epoch = 1
    workspace._remember_best_condition()
    workspace.condition_transition_pending = True
    workspace.save_checkpoint(tag="latest", use_thread=False)
    payload = torch.load(workspace.get_checkpoint_path("latest"), pickle_module=dill)
    resumed, _ = _strict_workspace(tmp_path)
    resumed.load_payload(payload, include_keys=resumed.include_keys)
    assert resumed.condition_transition_pending
    assert resumed.condition_best_metric == .2
    assert resumed.condition_best_epoch == 1
    assert resumed._maybe_transition_stage()
    assert resumed.training_stage == "rgb"
    for name, parameter in resumed.model.named_parameters():
        if id(parameter) in {id(value) for value in resumed.model.gate_parameters()}:
            torch.testing.assert_close(parameter, workspace.condition_best_state[name], rtol=0, atol=0)


def test_condition_validation_reduces_region_and_global_terms_separately(tmp_path, monkeypatch):
    workspace, _ = _strict_workspace(tmp_path)
    workspace.max_val_batches = None
    batches = [{"obs": {"image": torch.zeros(2, 1)}}, {"obs": {"image": torch.zeros(1, 1)}}]
    measurements = iter([
        {"phase_ce": 2., "alpha_mse": 1., "intensity_loss": 10.,
         "token_region_observation_count": 1, "token_mse_measured": False},
        {"phase_ce": 4., "alpha_mse": 2., "intensity_loss": 20.,
         "token_region_observation_count": 2, "token_mse_measured": False},
    ])
    def fake_batch_loss(batch):
        workspace.loss_metrics = next(measurements)
        return torch.tensor(workspace._condition_validation_loss(workspace.loss_metrics))
    monkeypatch.setattr(workspace, "_batch_loss", fake_batch_loss)
    actual = workspace._validate_batches(batches)
    expected = .7 * (2. * 2 + 4.) / 3 + .3 * (1. * 2 + 2.) / 3 + (10. + 20. * 2) / 3
    assert actual == pytest.approx(expected)
