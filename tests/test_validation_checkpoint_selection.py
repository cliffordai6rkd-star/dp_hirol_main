import copy
from pathlib import Path
from unittest.mock import Mock

import dill
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
)


class _ScaleNormalizer:
    def normalize(self, value):
        return value * value.new_tensor([2.0, 0.5])


class _Policy(torch.nn.Module):
    action_start_index = 1
    n_action_steps = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.normalizer = {"action": _ScaleNormalizer()}
        self.last_curriculum_metrics = {}
        self.prediction_calls = []

    def compute_loss(self, batch, optimizer_step=None, generator=None):
        # Deliberately consume the global RNG to check validation isolation.
        torch.rand(1)
        return batch["loss"].mean() + self.weight.square()

    def predict_action(self, obs, generator=None):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.prediction_calls.append(torch.rand((), generator=generator).item())
        prediction = obs["prediction"] + self.weight
        return {"model_action_pred": prediction, "action": prediction[:, 1:3]}

    def set_optimizer_step(self, step):
        pass


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, record, **kwargs):
        self.records.append(copy.deepcopy(record))


def make_cfg():
    return OmegaConf.create({
        "training": {
            "seed": 42,
            "max_val_steps": None,
            "max_train_steps": None,
            "gradient_accumulate_every": 1,
            "max_grad_norm": None,
            "tqdm_interval_sec": 100,
            "checkpoint_every_optimizer_steps": 2,
        },
        "checkpoint": {
            "topk": {
                "monitor_key": "val_action_mse_raw", "mode": "min", "k": 2,
                "format_str": "optimizer_step={optimizer_step:08d}-val_action_mse={val_action_mse_raw:.6f}.ckpt",
            },
            "save_last_ckpt": True, "save_last_snapshot": False,
        },
    })


def make_workspace(tmp_path):
    workspace = TrainForceAwareDiffusionWorkspace.__new__(TrainForceAwareDiffusionWorkspace)
    workspace.cfg = make_cfg()
    workspace._output_dir = str(tmp_path)
    workspace._saving_thread = None
    workspace.model = _Policy()
    workspace.ema_model = None
    workspace.optimizer = torch.optim.SGD(workspace.model.parameters(), lr=0.01)
    workspace.global_step = 0
    workspace.optimizer_step = 0
    workspace.epoch = 0
    workspace.topk_state = {}
    return workspace


def make_batches():
    # Large errors outside action[1:3] must not affect the selected metric.
    first = torch.tensor([[[100.0, 100.0], [1.0, 2.0], [3.0, 4.0], [200.0, 200.0]]])
    second = torch.tensor([
        [[100.0, 100.0], [2.0, 2.0], [2.0, 2.0], [200.0, 200.0]],
        [[100.0, 100.0], [3.0, 3.0], [3.0, 3.0], [200.0, 200.0]],
    ])
    return [
        {"obs": {"prediction": first}, "action": torch.zeros_like(first), "loss": torch.tensor([1.0])},
        {"obs": {"prediction": second}, "action": torch.zeros_like(second), "loss": torch.tensor([3.0, 5.0])},
    ]


def load_payload(path):
    return torch.load(path, map_location="cpu", pickle_module=dill)


def test_validation_runs_inference_and_aligns_future_action_chunk(tmp_path):
    workspace = make_workspace(tmp_path)
    policy = workspace.model.eval()
    batches = make_batches()
    metrics = workspace._validate_metrics(policy, batches, torch.device("cpu"), None, workspace.cfg)
    prediction = torch.cat([batch["obs"]["prediction"][:, 1:3] for batch in batches])
    expected_raw = prediction.square().mean().item()
    expected_normalized = policy.normalizer["action"].normalize(prediction).square().mean().item()
    assert metrics["val_action_mse_raw"] == pytest.approx(expected_raw)
    assert metrics["val_action_mse"] == pytest.approx(expected_normalized)
    assert metrics["val_loss"] == pytest.approx(3.0)  # Weight the short final batch correctly.
    seeds = list(policy.prediction_calls)
    repeated = workspace._validate_metrics(policy, batches, torch.device("cpu"), None, workspace.cfg)
    assert repeated == metrics
    assert policy.prediction_calls[2:] == seeds
    limited = workspace._validate_metrics(policy, batches, torch.device("cpu"), 1, workspace.cfg)
    assert limited["val_action_mse_raw"] == pytest.approx(7.5)


def test_validation_uses_ema_restores_modes_rng_and_caches_only_same_weights(tmp_path):
    workspace = make_workspace(tmp_path)
    workspace.ema_model = _Policy().train()
    workspace._update_validation_plot = Mock(return_value=None)
    wandb_logger, json_logger = _Logger(), _Logger()
    rng_before = torch.get_rng_state().clone()
    metrics = workspace._validate_and_log(
        workspace.cfg, make_batches(), torch.device("cpu"), wandb_logger, json_logger,
    )
    assert workspace.ema_model.training
    assert workspace.model.training
    assert not workspace.model.prediction_calls
    assert len(workspace.ema_model.prediction_calls) == 2
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    assert json_logger.records[-1]["val_action_mse_raw"] == metrics["val_action_mse_raw"]
    assert workspace._validate_and_log(
        workspace.cfg, make_batches(), torch.device("cpu"), wandb_logger, json_logger,
    ) == metrics
    assert len(workspace.ema_model.prediction_calls) == 2
    workspace.optimizer_step += 1
    workspace._validate_and_log(
        workspace.cfg, make_batches(), torch.device("cpu"), wandb_logger, json_logger,
    )
    assert len(workspace.ema_model.prediction_calls) == 4


def test_validation_checkpoint_refuses_empty_split_and_restores_training_mode(tmp_path):
    workspace = make_workspace(tmp_path)
    with pytest.raises(ValueError, match="non-empty validation split"):
        workspace._validate_and_log(workspace.cfg, [], torch.device("cpu"), _Logger(), _Logger())
    assert workspace.model.training


def test_step_checkpoint_gets_fresh_validation_before_ranking(tmp_path):
    workspace = make_workspace(tmp_path)
    seen_steps = []

    def validate(*args):
        seen_steps.append(workspace.optimizer_step)
        return {"val_action_mse_raw": float(workspace.optimizer_step)}

    workspace._validate_and_log = Mock(side_effect=validate)
    workspace._save_optimizer_step_checkpoint = Mock()
    scheduler = torch.optim.lr_scheduler.LambdaLR(workspace.optimizer, lambda _: 1.0)
    workspace._train_epoch(
        cfg=workspace.cfg, dataloader=make_batches() * 3, device=torch.device("cpu"),
        lr_scheduler=scheduler, ema=None, wandb_run=_Logger(), json_logger=_Logger(),
        target_optimizer_steps=5, val_dataloader=make_batches(),
    )
    assert seen_steps == [2, 4]
    saved = workspace._save_optimizer_step_checkpoint.call_args_list
    assert [call.kwargs["metric_dict"]["val_action_mse_raw"] for call in saved] == [2.0, 4.0]
    assert workspace.model.training


def test_topk_uses_rollout_mse_and_latest_does_not_overwrite_best(tmp_path):
    workspace = make_workspace(tmp_path)
    manager = workspace._make_topk_manager(workspace.cfg)

    def save(step, mse):
        workspace.optimizer_step = workspace.global_step = step
        with torch.no_grad():
            workspace.model.weight.fill_(step)
        workspace._save_optimizer_step_checkpoint(
            workspace.cfg, manager,
            {"optimizer_step": step, "val_action_mse_raw": mse, "train_loss": -float(step)},
        )

    save(1, 1.0)
    save(2, 0.2)
    save(3, 0.8)
    directory = tmp_path / "checkpoints"
    assert sorted(manager.path_value_map.values()) == [0.2, 0.8]
    assert len(list(directory.glob("optimizer_step=*.ckpt"))) == 2
    assert "00000002" in (directory / "best.ckpt").resolve().name
    third = (directory / "latest.ckpt").resolve()
    assert "00000003" in third.name

    save(4, 0.9)  # Reject this candidate but keep it for resume.
    assert not (directory / "latest.ckpt").is_symlink()
    assert load_payload(directory / "latest.ckpt")["state_dicts"]["model"]["weight"].item() == 4
    assert load_payload(third)["state_dicts"]["model"]["weight"].item() == 3
    assert load_payload(directory / "best.ckpt")["state_dicts"]["model"]["weight"].item() == 2
    save(5, float("nan"))
    save(6, float("inf"))
    assert sorted(manager.path_value_map.values()) == [0.2, 0.8]

    # The ranking is part of the checkpoint, not an empty new list on resume.
    resumed = make_workspace(tmp_path)
    resumed.load_payload(load_payload(directory / "latest.ckpt"))
    restored_manager = resumed._make_topk_manager(resumed.cfg)
    assert restored_manager.path_value_map == manager.path_value_map
    resumed.optimizer_step = 7
    resumed._save_optimizer_step_checkpoint(
        resumed.cfg, restored_manager, {"optimizer_step": 7, "val_action_mse_raw": 0.1},
    )
    assert sorted(restored_manager.path_value_map.values()) == [0.1, 0.2]
    assert "00000007" in (directory / "best.ckpt").resolve().name
    assert not third.exists()


def test_duplicate_and_missing_validation_scores_do_not_evict_topk(tmp_path):
    workspace = make_workspace(tmp_path)
    manager = workspace._make_topk_manager(workspace.cfg)
    workspace.optimizer_step = 1
    metrics = {"optimizer_step": 1, "val_action_mse_raw": 0.5}
    workspace._save_optimizer_step_checkpoint(workspace.cfg, manager, metrics)
    original = dict(manager.path_value_map)
    workspace._save_checkpoints(workspace.cfg, metrics, manager)
    workspace._save_optimizer_step_checkpoint(workspace.cfg, manager, {"optimizer_step": 2})
    assert manager.path_value_map == original
    assert load_payload(tmp_path / "checkpoints/best.ckpt")["pickles"]


def test_self_cross_config_ranks_raw_inference_mse_on_full_validation_split():
    root = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(config_name="baseline_train/train_dp_self_crossattn_insert_usb")
        original = compose(config_name="train_dp_insert_usb_force_aware")
    assert cfg.checkpoint.topk.monitor_key == "val_action_mse_raw"
    assert cfg.checkpoint.topk.mode == "min"
    assert cfg.checkpoint.topk.k == 5
    assert cfg.training.max_val_steps is None
    assert cfg.training.checkpoint_every_optimizer_steps == 20000
    assert original.checkpoint.topk.monitor_key == "train_loss"
