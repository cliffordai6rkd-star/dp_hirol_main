import copy
import json

import dill
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.checkpoint_action_mse import (
    ACTION_MSE_KEY, record_checkpoint_action_mse, resolve_student_teacher, select_action_mse_teacher,
)
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import TrainForceAwareDiffusionWorkspace


class _Normalizer:
    def normalize(self, value):
        return value * value.new_tensor([2.0, 0.5])


class _Policy(torch.nn.Module):
    action_start_index = 1
    n_action_steps = 2

    def __init__(self, weight=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))
        self.normalizer = {"action": _Normalizer()}
        self.last_curriculum_metrics = {}
        self.prediction_calls = []

    def predict_action(self, obs, generator):
        assert not self.training and not torch.is_grad_enabled()
        self.prediction_calls.append(torch.rand((), generator=generator).item())
        return {"model_action_pred": obs["prediction"] + self.weight}

    def compute_loss(self, batch, optimizer_step):
        return self.weight.square()

    def set_optimizer_step(self, step):
        pass


class _Episodes(torch.utils.data.Dataset):
    def __init__(self):
        self.episode_index = np.repeat(np.arange(6), np.arange(1, 7))
        self.indices = np.column_stack([np.arange(len(self.episode_index))] * 4)
        self.seen_rows = []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        self.seen_rows.append(row)
        # Context and past-horizon values must be excluded from the action MSE.
        prediction = torch.tensor([[999.0, 999.0], [row + 1.0, 2.0], [3.0, 4.0], [999.0, 999.0]])
        return {"obs": {"prediction": prediction}, "action": torch.zeros_like(prediction)}


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, record, **kwargs):
        self.records.append(copy.deepcopy(record))


def _workspace(tmp_path):
    workspace = TrainForceAwareDiffusionWorkspace.__new__(TrainForceAwareDiffusionWorkspace)
    workspace.cfg = OmegaConf.create({
        "training": {
            "seed": 42, "max_val_steps": 1, "max_train_steps": None,
            "gradient_accumulate_every": 1, "max_grad_norm": None,
            "tqdm_interval_sec": 100, "checkpoint_every_optimizer_steps": 2,
        },
        "checkpoint": {
            "action_mse": {"enabled": True, "num_episodes": 2, "batch_size": 3},
            "topk": {"monitor_key": ACTION_MSE_KEY, "mode": "min", "k": 1,
                     "format_str": "step={optimizer_step:08d}-mse={checkpoint_action_mse:.8f}.ckpt"},
            "save_last_ckpt": True, "save_last_snapshot": False,
        },
    })
    workspace._output_dir = str(tmp_path)
    workspace._saving_thread = None
    workspace.model = _Policy(100.0)
    workspace.ema_model = _Policy()
    workspace.optimizer = torch.optim.SGD(workspace.model.parameters(), lr=0.01)
    workspace.optimizer_step = workspace.global_step = workspace.epoch = 0
    workspace.topk_state = {}
    workspace._checkpoint_eval_dataset = _Episodes()
    return workspace


def test_checkpoint_inference_uses_complete_fresh_episodes_and_ema(tmp_path):
    workspace = _workspace(tmp_path)
    local_logger = _Logger()
    rng_before = torch.get_rng_state().clone()
    draws = []
    for step in (2, 4, 6):
        workspace.optimizer_step = step
        dataset = workspace._checkpoint_eval_dataset
        dataset.seen_rows.clear()
        metrics = workspace._checkpoint_action_mse_and_log(
            workspace.cfg, torch.device("cpu"), _Logger(), local_logger,
        )
        assert len(set(metrics["episode_ids"])) == 2
        rows = np.flatnonzero(np.isin(dataset.episode_index, metrics["episode_ids"]))
        assert dataset.seen_rows == rows.tolist()  # No max_val_steps truncation.
        values = torch.stack([dataset[int(row)]["obs"]["prediction"][1:3] for row in rows])
        assert metrics[ACTION_MSE_KEY] == pytest.approx(values.square().mean().item())
        assert metrics["checkpoint_action_mse_normalized"] == pytest.approx(
            workspace.ema_model.normalizer["action"].normalize(values).square().mean().item()
        )
        assert metrics["num_action_values"] == values.numel()
        assert metrics["policy_state_key"] == "ema_model"
        draws.append(metrics["episode_ids"])
    assert draws[0] != draws[1]
    assert not workspace.model.prediction_calls
    assert workspace.ema_model.prediction_calls
    assert workspace.model.training and workspace.ema_model.training
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert len(local_logger.records) == 3


@pytest.mark.parametrize("explicit_null_interval", [False, True])
def test_step_checkpoints_run_inference_before_save_and_retain_lowest_score(
    tmp_path, explicit_null_interval,
):
    workspace = _workspace(tmp_path)
    if explicit_null_interval:
        workspace.cfg.checkpoint.action_mse.every_optimizer_steps = None
    manager = workspace._make_topk_manager(workspace.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(workspace.optimizer, lambda _: 1.0)
    batch = {"obs": {"prediction": torch.zeros(1, 4, 2)}, "action": torch.zeros(1, 4, 2)}
    workspace._train_epoch(
        cfg=workspace.cfg, dataloader=[batch] * 4, device=torch.device("cpu"),
        lr_scheduler=scheduler, ema=None, wandb_run=_Logger(), json_logger=_Logger(),
        topk_manager=manager, target_optimizer_steps=4,
    )
    scores = json.loads((tmp_path / "checkpoints/action_mse.json").read_text())
    assert [record["optimizer_step"] for record in scores["records"]] == [2, 4]
    assert all(len(record["episode_ids"]) == 2 for record in scores["records"])
    checkpoint, best = select_action_mse_teacher(tmp_path)
    assert best[ACTION_MSE_KEY] == min(record[ACTION_MSE_KEY] for record in scores["records"])
    assert checkpoint == (tmp_path / "checkpoints/best.ckpt").resolve()
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    saved_metrics = dill.loads(payload["pickles"]["action_mse_metrics"])
    assert saved_metrics[ACTION_MSE_KEY] == best[ACTION_MSE_KEY]
    assert saved_metrics["episode_ids"] == best["episode_ids"]


def test_independent_inference_cadence_preserves_best_and_periodic_resume(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    workspace.cfg.checkpoint.action_mse.every_optimizer_steps = 2
    workspace.cfg.training.checkpoint_every_optimizer_steps = 8
    workspace.cfg.training.gradient_accumulate_every = 2
    workspace.cfg.checkpoint.save_last_snapshot = True
    snapshots = []
    monkeypatch.setattr(
        TrainForceAwareDiffusionWorkspace, "save_snapshot",
        lambda self, tag: snapshots.append(tag),
    )
    manager = workspace._make_topk_manager(workspace.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(workspace.optimizer, lambda _: 1.0)
    logger = _Logger()
    batch = {"obs": {"prediction": torch.zeros(1, 4, 2)}, "action": torch.zeros(1, 4, 2)}
    directory = tmp_path / "checkpoints"

    def train_to(step, inference_weight):
        with torch.no_grad():
            workspace.ema_model.weight.fill_(inference_weight)
        workspace._train_epoch(
            cfg=workspace.cfg, dataloader=[batch] * 4, device=torch.device("cpu"),
            lr_scheduler=scheduler, ema=None, wandb_run=_Logger(), json_logger=logger,
            topk_manager=manager, target_optimizer_steps=step,
        )

    def load_checkpoint(path):
        return torch.load(path, map_location="cpu", pickle_module=dill)

    train_to(2, 100.0)
    first = (directory / "best.ckpt").resolve()
    assert first.is_file()
    assert (directory / "latest.ckpt").resolve() == first

    # A better evaluation between periodic saves evicts the old latest target.
    train_to(4, 0.0)
    best = (directory / "best.ckpt").resolve()
    assert best != first and not first.exists()
    assert best.is_file() and (directory / "latest.ckpt").resolve() == best
    best_payload = load_checkpoint(best)
    assert dill.loads(best_payload["pickles"]["optimizer_step"]) == 4
    assert best_payload["state_dicts"]["ema_model"]["weight"].item() == 0.0
    latest_before_rejection = (directory / "latest.ckpt").read_bytes()

    # A worse evaluation is recorded, but neither checkpoint nor snapshot is written.
    train_to(6, 200.0)
    assert (directory / "latest.ckpt").is_symlink()
    assert (directory / "latest.ckpt").read_bytes() == latest_before_rejection
    assert (directory / "best.ckpt").resolve() == best
    assert not list(directory.glob("step=00000006-*.ckpt"))
    assert snapshots == []
    scores = json.loads((directory / "action_mse.json").read_text())
    assert [record["optimizer_step"] for record in scores["records"]] == [2, 4, 6]
    assert scores["records"][-1]["checkpoint"] is None

    # The regular checkpoint still saves current training state if Top-K rejects it.
    train_to(8, 300.0)
    assert not (directory / "latest.ckpt").is_symlink()
    latest_payload = load_checkpoint(directory / "latest.ckpt")
    assert dill.loads(latest_payload["pickles"]["optimizer_step"]) == 8
    assert dill.loads(latest_payload["pickles"]["global_step"]) == 16
    assert latest_payload["state_dicts"]["ema_model"]["weight"].item() == 300.0
    assert snapshots == ["optimizer_step=00000008"]
    assert (directory / "best.ckpt").resolve() == best
    scores = json.loads((directory / "action_mse.json").read_text())
    assert [record["optimizer_step"] for record in scores["records"]] == [2, 4, 6, 8]
    assert [record["global_step"] for record in scores["records"]] == [4, 8, 12, 16]
    assert scores["best"]["optimizer_step"] == 4
    assert scores["records"][-1]["checkpoint"] is None
    assert sum(ACTION_MSE_KEY in record for record in logger.records) == 4

    resumed = _workspace(tmp_path)
    resumed.cfg = copy.deepcopy(workspace.cfg)
    resumed.load_payload(latest_payload)
    restored_manager = resumed._make_topk_manager(resumed.cfg)
    assert resumed.optimizer_step == 8
    assert restored_manager.path_value_map == manager.path_value_map
    assert (directory / "best.ckpt").resolve() == best
    selected, selected_metrics = select_action_mse_teacher(tmp_path)
    assert selected == best and selected_metrics["optimizer_step"] == 4


def test_student_reads_teacher_yaml_paths_and_selects_mse_instead_of_latest(tmp_path):
    teacher_dir = tmp_path / "teacher output"
    checkpoints = teacher_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for step, mse in ((1, 0.8), (2, 0.1), (3, 0.4)):
        checkpoint = checkpoints / f"step={step}.ckpt"
        checkpoint.touch()
        record_checkpoint_action_mse(teacher_dir, {
            "optimizer_step": step, ACTION_MSE_KEY: mse, "episode_ids": [step, step + 1],
        }, checkpoint)
    # Deliberately misleading aliases and deleted old score must not win.
    (checkpoints / "latest.ckpt").symlink_to("step=3.ckpt")
    (checkpoints / "best.ckpt").symlink_to("step=1.ckpt")
    missing = checkpoints / "deleted.ckpt"
    missing.touch()
    record_checkpoint_action_mse(teacher_dir, {
        "optimizer_step": 0, ACTION_MSE_KEY: 0.001, "episode_ids": [0, 1],
    }, missing)
    missing.unlink()
    teacher_config = tmp_path / "teacher.yaml"
    OmegaConf.save(OmegaConf.create({
        "output_dir": str(teacher_dir), "dataset_path": "/new/data/location",
        "task": {"dataset": {"dataset_path": "${dataset_path}", "horizon": 9, "val_ratio": 0.0}},
    }), teacher_config)
    cfg = OmegaConf.create({
        "teacher_config": str(teacher_config),
        "teacher_config_overrides": ["dataset_path=/override/data/location"],
        "dataset": {"val_ratio": 0.1, "video_backend": "pyav"},
    })
    checkpoint, source = resolve_student_teacher(cfg)
    assert checkpoint == checkpoints / "step=2.ckpt"
    assert cfg.teacher_checkpoint == str(checkpoint)
    assert cfg.teacher_selection[ACTION_MSE_KEY] == 0.1
    workspace = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = cfg
    workspace.teacher_source_cfg = source
    workspace.teacher_cfg = OmegaConf.create({"task": {"dataset": {"dataset_path": "/old/checkpoint/path"}}})
    dataset = workspace._dataset_config()
    assert dataset.dataset_path == "/override/data/location"
    assert dataset.horizon == 9 and dataset.val_ratio == 0.1 and dataset.video_backend == "pyav"


def test_student_requires_scored_teacher_instead_of_falling_back(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/latest.ckpt").touch()
    with pytest.raises(FileNotFoundError, match="checkpoint.action_mse.enabled=true"):
        select_action_mse_teacher(tmp_path)


@pytest.mark.parametrize("use_teacher_yaml, scored_run", [(False, False), (True, False), (True, True)])
def test_explicit_teacher_weights_override_mse_selection_and_keep_optional_data_yaml(
    tmp_path, use_teacher_yaml, scored_run,
):
    checkpoint = tmp_path / "specified teacher.ckpt"
    checkpoint.touch()
    teacher_dir = tmp_path / "teacher output"
    if scored_run:
        best = teacher_dir / "checkpoints/best.ckpt"
        best.parent.mkdir(parents=True)
        best.touch()
        record_checkpoint_action_mse(teacher_dir, {
            "optimizer_step": 1, ACTION_MSE_KEY: 0.001, "episode_ids": [0, 1],
        }, best)
    teacher_yaml = tmp_path / "teacher.yaml"
    OmegaConf.save(OmegaConf.create({
        "output_dir": str(teacher_dir), "dataset_path": "/teacher/data",
        "task": {"dataset": {"dataset_path": "${dataset_path}"}},
    }), teacher_yaml)
    cfg = OmegaConf.create({
        "teacher_config": str(teacher_yaml) if use_teacher_yaml else None,
        "teacher_config_overrides": ["dataset_path=/override/data"],
        "teacher_checkpoint": str(checkpoint),
        "teacher_selection": {ACTION_MSE_KEY: 0.001},
    })

    selected, source = resolve_student_teacher(cfg)

    assert selected == checkpoint and cfg.teacher_checkpoint == str(checkpoint)
    assert cfg.teacher_selection is None
    if use_teacher_yaml:
        assert source.task.dataset.dataset_path == "/override/data"
        assert "hydra" not in source
    else:
        assert source is None
