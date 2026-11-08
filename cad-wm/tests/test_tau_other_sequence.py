import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.tau_other_gru import TauOtherGRURegressor
from model.tau_other_lstm import TauOtherLSTMRegressor
from model.tau_other_sequence import (
    TauOtherSequenceRegressor,
    build_tau_other_sequence_model,
)
from model.tau_other_tcn import TauOtherTCNRegressor
from train.base_trainer import BaseTrainer, ModelEMA
from train.nomalizer import Normalizer
from train.torque_sequence_peak_loss import TorqueSequencePeakLoss
from train.trainer.tau_other_sequence_train import (
    SampleIndexDataset,
    TauOtherTrainer,
    run_purged_kfold_workflow,
)


def make_config(architecture="lstm", inputs=None):
    return {
        "dataloader": {"horizon": 8},
        "model": {
            "architecture": architecture,
            "inputs": inputs or ["q", "dq", "ddq", "tau"],
            "input_dims": {"q": 7, "dq": 7, "ddq": 7, "tau": 7},
            "output_dim": 7,
            "target_key": "tau_other",
            "hidden_dim": 16,
            "num_layers": 2,
            "head_hidden_dim": 24,
            "head_num_layers": 2,
            "dropout": 0.0,
        },
        "loss": {
            "type": "peak_cvar",
            "tail_fraction": 0.02,
            "peak_weight": 0.9,
            "mean_weight": 0.1,
            "joint_weights": None,
        },
        "train": {},
    }


def make_batch(batch_size=4, horizon=8):
    return {
        key: torch.randn(batch_size, horizon, 7)
        for key in ("q", "dq", "ddq", "tau", "tau_other")
    }


class TauOtherSequenceRegressorTest(unittest.TestCase):
    def test_trainer_rejects_conflicting_cudnn_modes(self):
        config = make_config()
        config["train"].update(deterministic=True, cudnn_benchmark=True)

        with self.assertRaisesRegex(ValueError, "incompatible"):
            TauOtherTrainer(config)

    def test_trainer_accepts_fast_seeded_cudnn_mode(self):
        config = make_config()
        config["train"].update(deterministic=False, cudnn_benchmark=True)

        trainer = TauOtherTrainer(config)

        self.assertFalse(trainer.deterministic)
        self.assertTrue(trainer.cudnn_benchmark)

    def test_early_stopping_can_use_a_different_metric_than_checkpoints(self):
        config = make_config()
        config["train"].update(
            monitor_key="val_wrench_force_norm_p95_n",
            early_stopping_monitor_key="val_loss",
            early_stopping={
                "enabled": True,
                "patience": 1,
                "warmup_epochs": 0,
                "min_delta": 0.0,
            },
        )
        trainer = TauOtherTrainer(config)

        self.assertFalse(
            trainer.should_stop_early(
                0,
                {"val_loss": 1.0, "val_wrench_force_norm_p95_n": 10.0},
            )
        )
        self.assertTrue(
            trainer.should_stop_early(
                1,
                {"val_loss": 1.1, "val_wrench_force_norm_p95_n": 0.1},
            )
        )

    def test_model_ema_smooths_parameters_and_copies_buffers(self):
        model = torch.nn.BatchNorm1d(2)
        ema = ModelEMA(model, decay=0.5)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.running_mean.fill_(4.0)
        ema.update(model, step=1)

        torch.testing.assert_close(
            ema.model.weight,
            torch.full_like(ema.model.weight, 2.0),
        )
        torch.testing.assert_close(
            ema.model.running_mean,
            model.running_mean,
        )
        self.assertFalse(any(p.requires_grad for p in ema.model.parameters()))

    def test_supported_architectures_forward_backward(self):
        for architecture in ("lstm", "gru", "tcn"):
            with self.subTest(architecture=architecture):
                model = TauOtherSequenceRegressor(make_config(architecture))
                out = model(make_batch())

                self.assertEqual(out["tau_other_pred"].shape, (4, 7))
                self.assertEqual(out["tau_other_target"].shape, (4, 7))
                out["tau_other_pred"].square().mean().backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )

    def test_factory_dispatches_to_three_independent_model_types(self):
        expected_types = {
            "lstm": TauOtherLSTMRegressor,
            "gru": TauOtherGRURegressor,
            "tcn": TauOtherTCNRegressor,
        }

        for architecture, expected_type in expected_types.items():
            with self.subTest(architecture=architecture):
                model = build_tau_other_sequence_model(make_config(architecture))
                self.assertIs(type(model), expected_type)

    def test_recurrent_last_prediction_is_bitwise_legacy_compatible(self):
        batch = make_batch()
        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = TauOtherSequenceRegressor(make_config(architecture)).eval()
                sequence = torch.cat(
                    [batch[key] for key in model.active_inputs],
                    dim=-1,
                )

                with torch.no_grad():
                    recurrent_output, _ = model.recurrent(sequence)
                    legacy_prediction = model.head(recurrent_output[:, -1])
                    prediction = model(batch)["tau_other_pred"]

                self.assertTrue(torch.equal(prediction, legacy_prediction))

    def test_configurable_input_subset_changes_recurrent_input_size(self):
        model = TauOtherSequenceRegressor(make_config(inputs=["q", "tau"]))
        batch = make_batch()
        batch.pop("dq")
        batch.pop("ddq")

        out = model(batch)

        self.assertEqual(model.recurrent.input_size, 14)
        self.assertEqual(out["tau_other_pred"].shape, (4, 7))

    def test_target_is_last_window_frame(self):
        model = TauOtherSequenceRegressor(make_config())
        batch = make_batch(batch_size=2, horizon=5)
        batch["tau_other"] = torch.arange(70, dtype=torch.float32).reshape(2, 5, 7)

        out = model(batch)

        torch.testing.assert_close(out["tau_other_target"], batch["tau_other"][:, -1])

    def test_tcn_is_strictly_causal_and_covers_the_configured_history(self):
        config = make_config("tcn")
        model = TauOtherSequenceRegressor(config).eval()
        batch = make_batch(batch_size=2, horizon=8)
        changed = {key: value.clone() for key, value in batch.items()}
        for key in model.active_inputs:
            changed[key][:, 5:] += 100.0

        with torch.no_grad():
            baseline = model.forward_sequence(batch)
            perturbed = model.forward_sequence(changed)

        self.assertEqual(model.temporal_receptive_field, 8)
        self.assertIsInstance(model.current_state_skip, torch.nn.Linear)
        torch.testing.assert_close(baseline[:, :5], perturbed[:, :5])
        self.assertFalse(torch.allclose(baseline[:, 5:], perturbed[:, 5:]))

    def test_tcn_rejects_a_receptive_field_shorter_than_history(self):
        config = make_config("tcn")
        config["model"].update(tcn_kernel_size=2, tcn_dilations=[1, 2])

        with self.assertRaisesRegex(ValueError, "receptive field"):
            TauOtherSequenceRegressor(config)

    def test_tcn_can_add_causal_first_difference_to_current_skip(self):
        config = make_config("tcn")
        config["model"]["current_delta_skip"] = True
        model = TauOtherSequenceRegressor(config)

        out = model(make_batch())

        self.assertTrue(model.current_delta_skip)
        self.assertEqual(model.current_state_skip.in_features, 2 * model.input_dim)
        self.assertEqual(out["tau_other_pred"].shape, (4, 7))

    def test_trainer_computes_normalized_and_physical_mae(self):
        trainer = TauOtherTrainer(make_config())
        trainer.model = trainer.build_model()

        loss, out = trainer.compute_loss(make_batch())

        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertEqual(
            set(out["loss_dict"]),
            {
                "peak_objective_nm2",
                "mse",
                "mae",
                "mae_nm",
                "mae_nm_j1",
                "mae_nm_j2",
                "mae_nm_j3",
                "mae_nm_j4",
                "mae_nm_j5",
                "mae_nm_j6",
                "mae_nm_j7",
            },
        )

    def test_physical_mae_uses_target_denormalization(self):
        trainer = TauOtherTrainer(make_config())
        std = torch.arange(1, 8, dtype=torch.float32)
        trainer.dataset = SimpleNamespace(
            normalize_mode="gaussian",
            normalize_lowdim_keys=["tau_other"],
            normalizer=Normalizer(
                {
                    "tau_other": {
                        "mean": torch.arange(7, dtype=torch.float32),
                        "std": std,
                    }
                }
            ),
        )
        target = torch.zeros(2, 7)
        prediction = torch.ones(2, 7)

        metrics = trainer._physical_mae_metrics(prediction, target)

        expected_joint_mae = std + trainer.dataset.normalizer.eps
        for joint_index, expected in enumerate(expected_joint_mae, start=1):
            torch.testing.assert_close(
                metrics[f"mae_nm_j{joint_index}"],
                expected,
            )
        torch.testing.assert_close(metrics["mae_nm"], expected_joint_mae.mean())

    def test_validation_mae_is_weighted_by_sample_count(self):
        class FixedPredictionModel(torch.nn.Module):
            def forward(self, batch):
                return {
                    "tau_other_pred": batch["prediction"],
                    "tau_other_target": batch["target"],
                }

        samples = [
            {
                "prediction": torch.full((7,), error),
                "target": torch.zeros(7),
            }
            for error in (1.0, 1.0, 3.0)
        ]
        trainer = TauOtherTrainer(make_config())
        trainer.device = "cpu"
        trainer.dataset = SimpleNamespace(
            normalize_mode=None,
            normalize_lowdim_keys=[],
        )
        trainer.model = FixedPredictionModel()
        trainer.val_loader = torch.utils.data.DataLoader(samples, batch_size=2)

        val_loss = trainer.validate_one_epoch(epoch=0)

        self.assertAlmostEqual(val_loss, 11.0 / 3.0, places=6)
        self.assertAlmostEqual(
            trainer.last_val_epoch_metrics["mae_nm"],
            5.0 / 3.0,
        )
        for joint_index in range(1, 8):
            self.assertAlmostEqual(
                trainer.last_val_epoch_metrics[f"mae_nm_j{joint_index}"],
                5.0 / 3.0,
            )
        self.assertEqual(
            trainer.last_val_epoch_metrics["peak_cvar_rmse_nm"],
            3.0,
        )

    def test_validation_nmae_uses_complete_split_target_distribution(self):
        class FixedPredictionModel(torch.nn.Module):
            def forward(self, batch):
                return {
                    "tau_other_pred": batch["prediction"],
                    "tau_other_target": batch["target"],
                }

        target_levels = torch.tensor([0.0, 1.0, 2.0, 3.0])
        samples = [
            {
                "prediction": torch.full((7,), level + 1.0),
                "target": torch.full((7,), level),
            }
            for level in target_levels
        ]
        trainer = TauOtherTrainer(make_config())
        trainer.device = "cpu"
        trainer.dataset = SimpleNamespace(
            normalize_mode=None,
            normalize_lowdim_keys=[],
        )
        trainer.model = FixedPredictionModel()
        trainer.val_loader = torch.utils.data.DataLoader(samples, batch_size=2)

        trainer.validate_one_epoch(epoch=0)

        target_std = target_levels.std().item()
        robust_range = (
            torch.quantile(target_levels, 0.95)
            - torch.quantile(target_levels, 0.05)
        ).item()
        metrics = trainer.last_val_epoch_metrics
        for joint_index in range(1, 8):
            self.assertAlmostEqual(metrics[f"mae_nm_j{joint_index}"], 1.0)
            self.assertAlmostEqual(metrics[f"joint_mae_nm_j{joint_index}"], 1.0)
            self.assertAlmostEqual(
                metrics[f"joint_tau_std_nm_j{joint_index}"], target_std
            )
            self.assertAlmostEqual(
                metrics[f"joint_nmae_std_j{joint_index}"],
                1.0 / (target_std + 1.0e-8),
            )
            self.assertAlmostEqual(
                metrics[f"joint_tau_p90_range_nm_j{joint_index}"], robust_range
            )
            self.assertAlmostEqual(
                metrics[f"joint_nmae_p90_j{joint_index}"],
                1.0 / (robust_range + 1.0e-8),
            )

    def test_peak_cvar_targets_the_largest_errors(self):
        objective = TorqueSequencePeakLoss(
            {
                "type": "peak_cvar",
                "tail_fraction": 0.25,
                "peak_weight": 1.0,
                "mean_weight": 0.0,
            }
        )
        prediction = torch.tensor(
            [[0.0, 0.0], [0.0, 4.0]],
            requires_grad=True,
        )
        target = torch.zeros_like(prediction)

        loss = objective(prediction, target)
        loss.backward()

        self.assertEqual(loss.item(), 16.0)
        self.assertEqual(prediction.grad[1, 1].item(), 8.0)
        self.assertEqual(torch.count_nonzero(prediction.grad).item(), 1)

    def test_mse_mode_restores_the_weighted_mean_objective(self):
        objective = TorqueSequencePeakLoss({"type": "mse"})
        prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.zeros_like(prediction)

        loss = objective(
            prediction,
            target,
            joint_weights=[1.0, 0.5],
        )

        self.assertEqual(objective.loss_type, "mse")
        self.assertEqual(loss.item(), (1.0 + 2.0 + 9.0 + 8.0) / 4.0)

    def test_trainer_mse_mode_uses_normalized_values(self):
        config = make_config()
        config["loss"] = {"type": "mse", "joint_weights": None}
        trainer = TauOtherTrainer(config)
        trainer.model = trainer.build_model()

        loss, out = trainer.compute_loss(make_batch())

        self.assertEqual(trainer.peak_loss.loss_type, "mse")
        torch.testing.assert_close(loss, out["loss_dict"]["mse"])
        self.assertIn("mse_objective", out["loss_dict"])
        self.assertNotIn("peak_objective_nm2", out["loss_dict"])

    def test_peak_metrics_use_the_complete_error_population(self):
        objective = TorqueSequencePeakLoss(
            {
                "type": "peak_cvar",
                "tail_fraction": 0.25,
                "peak_weight": 1.0,
                "mean_weight": 0.0,
            }
        )
        metrics = objective.metrics_from_absolute_error(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )

        self.assertEqual(metrics["peak_cvar_rmse_nm"].item(), 4.0)
        self.assertEqual(metrics["peak_cvar_rmse_nm_j1"].item(), 3.0)
        self.assertEqual(metrics["peak_cvar_rmse_nm_j2"].item(), 4.0)
        self.assertEqual(metrics["peak_max_nm"].item(), 4.0)

    def test_train_eval_disables_training_mode_noise(self):
        class ModeAwareModel(torch.nn.Module):
            def forward(self, batch):
                prediction = (
                    torch.ones_like(batch["target"])
                    if self.training
                    else torch.zeros_like(batch["target"])
                )
                return {
                    "tau_other_pred": prediction,
                    "tau_other_target": batch["target"],
                }

        trainer = TauOtherTrainer(make_config())
        trainer.device = "cpu"
        trainer.train_eval_enabled = True
        trainer.dataset = SimpleNamespace(
            normalize_mode=None,
            normalize_lowdim_keys=[],
        )
        trainer.model = ModeAwareModel().train()
        trainer.loader = torch.utils.data.DataLoader(
            [{"target": torch.zeros(7)} for _ in range(3)],
            batch_size=2,
        )

        train_eval_loss = trainer.evaluate_train_one_epoch(epoch=0)

        self.assertEqual(train_eval_loss, 0.0)
        self.assertEqual(trainer.last_train_eval_epoch_metrics["mae_nm"], 0.0)

    def test_invalid_architecture_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            TauOtherSequenceRegressor(make_config(architecture="transformer"))

    def test_each_window_starts_from_a_fresh_recurrent_state(self):
        for architecture in ("lstm", "gru", "tcn"):
            with self.subTest(architecture=architecture):
                model = TauOtherSequenceRegressor(make_config(architecture))
                model.eval()
                batch = make_batch(batch_size=3, horizon=6)

                first = model(batch)["tau_other_pred"]
                model(make_batch(batch_size=3, horizon=6))
                second = model(batch)["tau_other_pred"]

                torch.testing.assert_close(first, second)

    def test_model_does_not_expose_a_streaming_hidden_state_api(self):
        for architecture in ("lstm", "gru", "tcn"):
            with self.subTest(architecture=architecture):
                model = TauOtherSequenceRegressor(make_config(architecture))
                out = model(make_batch())

                self.assertFalse(hasattr(model, "forward_step"))
                self.assertFalse(hasattr(model, "init_recurrent_state"))
                self.assertNotIn("recurrent_state", out)
                self.assertNotIn("sequence_features", out)

    def test_next_topology_uses_two_recurrent_and_two_mlp_layers(self):
        config = make_config("lstm")
        config["model"].update(
            hidden_dim=128,
            num_layers=2,
            head_hidden_dim=256,
            head_num_layers=2,
        )

        model = TauOtherSequenceRegressor(config)

        self.assertIsInstance(model.recurrent, torch.nn.LSTM)
        self.assertEqual(model.recurrent.num_layers, 2)
        self.assertEqual(model.recurrent.hidden_size, 128)
        self.assertEqual(model.head[0].in_features, 128)
        self.assertEqual(model.head[0].out_features, 256)
        self.assertEqual(model.head[-1].in_features, 256)
        self.assertEqual(model.head[-1].out_features, 7)

    def test_invalid_history_mode_fails_fast(self):
        config = make_config()
        config["model"]["history_mode"] = "stateful_stream"

        with self.assertRaisesRegex(ValueError, "stateless_sliding_window"):
            TauOtherSequenceRegressor(config)


class FakeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, episode_length=30, horizon=5):
        self.horizon = horizon
        self.valid_indices = list(range(horizon - 1, episode_length))
        self.raw_idx_to_episode_start = {idx: 0 for idx in self.valid_indices}
        self.raw_idx_to_episode_end = {
            idx: episode_length for idx in self.valid_indices
        }
        self.dataset = SimpleNamespace(
            meta=SimpleNamespace(
                episodes=[
                    {
                        "dataset_from_index": 0,
                        "dataset_to_index": episode_length,
                    }
                ]
            )
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        return self.valid_indices[index]


class FakeMultiEpisodeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, episode_lengths=(36, 42), horizon=5):
        self.horizon = horizon
        self.valid_indices = []
        self.raw_idx_to_episode_start = {}
        self.raw_idx_to_episode_end = {}
        episodes = []
        episode_start = 0
        for episode_index, episode_length in enumerate(episode_lengths):
            episode_end = episode_start + episode_length
            episodes.append(
                {
                    "episode_index": episode_index,
                    "dataset_from_index": episode_start,
                    "dataset_to_index": episode_end,
                }
            )
            for raw_idx in range(episode_start + horizon - 1, episode_end):
                self.valid_indices.append(raw_idx)
                self.raw_idx_to_episode_start[raw_idx] = episode_start
                self.raw_idx_to_episode_end[raw_idx] = episode_end
            episode_start = episode_end
        self.dataset = SimpleNamespace(
            meta=SimpleNamespace(episodes=episodes)
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        return self.valid_indices[index]


class EpisodeSplitTest(unittest.TestCase):
    def test_explicit_validation_episodes_override_ratio_and_seed(self):
        dataset = FakeMultiEpisodeWindowDataset(
            episode_lengths=(12, 14, 16),
            horizon=5,
        )
        trainer = BaseTrainer(
            {
                "train": {
                    "val_ratio": 0.9,
                    "split_mode": "episode",
                    "seed": 999,
                    "val_episode_indices": [1],
                }
            }
        )
        trainer.dataset = dataset

        train_dataset, val_dataset = trainer.split_dataset_by_episode()

        def episode_indices(subset):
            result = set()
            for sample_idx in subset.indices:
                raw_idx = dataset.valid_indices[sample_idx]
                episode_start = dataset.raw_idx_to_episode_start[raw_idx]
                for episode in dataset.dataset.meta.episodes:
                    if int(episode["dataset_from_index"]) == episode_start:
                        result.add(int(episode["episode_index"]))
                        break
            return result

        self.assertEqual(episode_indices(val_dataset), {1})
        self.assertEqual(episode_indices(train_dataset), {0, 2})

    def test_unknown_explicit_validation_episode_fails(self):
        trainer = BaseTrainer(
            {
                "train": {
                    "split_mode": "episode",
                    "val_episode_indices": [7],
                }
            }
        )
        trainer.dataset = FakeMultiEpisodeWindowDataset()

        with self.assertRaisesRegex(ValueError, "unknown episode indices"):
            trainer.split_dataset_by_episode()


class PurgedTemporalSplitTest(unittest.TestCase):
    def test_train_and_validation_windows_share_no_raw_frames(self):
        trainer = BaseTrainer(
            {"train": {"val_ratio": 0.2, "split_mode": "purged_temporal"}}
        )
        trainer.dataset = FakeWindowDataset(episode_length=30, horizon=5)

        train_dataset, val_dataset = trainer.split_dataset_purged_temporal()

        def covered_frames(subset):
            frames = set()
            for sample_idx in subset.indices:
                raw_idx = trainer.dataset.valid_indices[sample_idx]
                frames.update(range(raw_idx - trainer.dataset.horizon + 1, raw_idx + 1))
            return frames

        train_frames = covered_frames(train_dataset)
        val_frames = covered_frames(val_dataset)
        self.assertTrue(train_frames)
        self.assertTrue(val_frames)
        self.assertTrue(train_frames.isdisjoint(val_frames))
        self.assertEqual(max(train_frames) + 1, min(val_frames))


class PurgedKFoldSplitTest(unittest.TestCase):
    @staticmethod
    def covered_frames(dataset, subset):
        frames = set()
        for sample_idx in subset.indices:
            raw_idx = dataset.valid_indices[sample_idx]
            episode_start = dataset.raw_idx_to_episode_start[raw_idx]
            frames.update(
                range(max(episode_start, raw_idx - dataset.horizon + 1), raw_idx + 1)
            )
        return frames

    def test_every_fold_has_disjoint_raw_history_and_covers_all_targets(self):
        dataset = FakeMultiEpisodeWindowDataset()
        validation_sample_sets = []

        for fold_index in range(3):
            trainer = BaseTrainer(
                {
                    "train": {
                        "split_mode": "purged_kfold",
                        "purged_kfold": {
                            "num_folds": 3,
                            "fold_index": fold_index,
                        },
                    }
                }
            )
            trainer.dataset = dataset
            train_dataset, val_dataset = trainer.split_dataset_purged_kfold()

            train_frames = self.covered_frames(dataset, train_dataset)
            val_frames = self.covered_frames(dataset, val_dataset)
            self.assertTrue(train_frames)
            self.assertTrue(val_frames)
            self.assertTrue(train_frames.isdisjoint(val_frames))
            self.assertEqual(
                len(trainer.current_split_metadata["episodes"]),
                2,
            )
            self.assertTrue(
                all(
                    episode["train_samples"] > 0
                    and episode["val_samples"] > 0
                    for episode in trainer.current_split_metadata["episodes"]
                )
            )
            validation_sample_sets.append(set(val_dataset.indices))

        for left_index, left in enumerate(validation_sample_sets):
            for right in validation_sample_sets[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(
            set().union(*validation_sample_sets),
            set(range(len(dataset))),
        )

    def test_short_episode_fails_instead_of_silently_disappearing(self):
        trainer = BaseTrainer(
            {
                "train": {
                    "split_mode": "purged_kfold",
                    "purged_kfold": {"num_folds": 3, "fold_index": 1},
                }
            }
        )
        trainer.dataset = FakeMultiEpisodeWindowDataset(
            episode_lengths=(11,),
            horizon=5,
        )

        with self.assertRaisesRegex(ValueError, "removed all training windows"):
            trainer.split_dataset_purged_kfold()


class PurgedKFoldWorkflowTest(unittest.TestCase):
    def test_workflow_aggregates_folds_and_retrains_all_data(self):
        class FakeTrainer:
            configs = []

            def __init__(self, config):
                self.config = copy.deepcopy(config)
                self.__class__.configs.append(self.config)

            def train(self):
                train_config = self.config["train"]
                if train_config["split_mode"] == "all":
                    return {
                        "num_epochs": train_config["num_epochs"],
                        "best_checkpoints": [],
                        "monitor_key": train_config["monitor_key"],
                    }

                fold_index = train_config["purged_kfold"]["fold_index"]
                best_epochs = (4, 8, 6)
                val_losses = (0.03, 0.02, 0.04)
                return {
                    "monitor_key": "val_loss",
                    "split": {"fold_index": fold_index},
                    "best_checkpoints": [
                        {
                            "epoch": best_epochs[fold_index],
                            "score": val_losses[fold_index],
                            "path": Path(train_config["output_dir"]) / "best.pt",
                            "metrics": {
                                "val_loss": val_losses[fold_index],
                                "val_mae_nm": 0.1 + 0.01 * fold_index,
                            },
                        }
                    ],
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "train": {
                    "split_mode": "purged_kfold",
                    "output_dir": tmp_dir,
                    "monitor_key": "val_loss",
                    "train_eval": {"enabled": True},
                    "early_stopping": {"enabled": True},
                    "purged_kfold": {
                        "num_folds": 3,
                        "production_retrain": True,
                    },
                }
            }

            report = run_purged_kfold_workflow(
                config,
                trainer_class=FakeTrainer,
            )

            self.assertEqual(len(report["folds"]), 3)
            self.assertEqual(report["production"]["num_epochs"], 7)
            self.assertAlmostEqual(
                report["aggregate"]["val_loss"]["mean"],
                0.03,
            )
            self.assertEqual(
                report["aggregate"]["val_loss"]["worst"],
                0.04,
            )
            production_config = FakeTrainer.configs[-1]["train"]
            self.assertEqual(production_config["split_mode"], "all")
            self.assertEqual(production_config["val_ratio"], 0.0)
            self.assertEqual(production_config["monitor_key"], "train_eval_loss")
            self.assertFalse(production_config["early_stopping"]["enabled"])
            self.assertTrue(Path(report["report_path"]).is_file())


class DeviceCacheTest(unittest.TestCase):
    def test_index_dataset_returns_only_sample_indices(self):
        dataset = SampleIndexDataset([4, 9])

        self.assertEqual(dataset[0], {"sample_idx": 4})
        self.assertEqual(dataset[1], {"sample_idx": 9})

    def test_cached_batch_gathers_windows_and_current_target(self):
        config = make_config(inputs=["q", "tau"])
        config["dataloader"] = {"horizon": 3}
        trainer = TauOtherTrainer(config)
        trainer.dataset = SimpleNamespace(horizon=3)

        values = torch.arange(70, dtype=torch.float32).reshape(10, 7)
        trainer.tensor_cache = {
            "q": values,
            "tau": values + 100,
            "tau_other": values + 200,
        }
        trainer.valid_raw_indices_device = torch.tensor([2, 3, 4])
        trainer.episode_starts_device = torch.tensor([0, 0, 0])

        batch = trainer._cached_batch(torch.tensor([0, 2]))

        torch.testing.assert_close(batch["q"][0], values[0:3])
        torch.testing.assert_close(batch["q"][1], values[2:5])
        torch.testing.assert_close(
            batch["tau_other"],
            torch.stack([values[2] + 200, values[4] + 200]),
        )

    def test_cached_batch_left_pads_early_history_with_zeros(self):
        config = make_config(inputs=["q"])
        config["dataloader"] = {"horizon": 3}
        trainer = TauOtherTrainer(config)
        trainer.dataset = SimpleNamespace(horizon=3)
        values = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        trainer.tensor_cache = {"q": values, "tau_other": values + 100}
        trainer.valid_raw_indices_device = torch.tensor([0, 1])
        trainer.episode_starts_device = torch.tensor([0, 0])

        batch = trainer._cached_batch(torch.tensor([0, 1]))

        torch.testing.assert_close(
            batch["q"][0],
            torch.stack([torch.zeros(7), torch.zeros(7), values[0]]),
        )
        torch.testing.assert_close(
            batch["q"][1],
            torch.stack([torch.zeros(7), values[0], values[1]]),
        )
        torch.testing.assert_close(
            batch["tau_other"],
            torch.stack([values[0] + 100, values[1] + 100]),
        )

    def test_normalizer_stats_use_training_frames_only(self):
        trainer = TauOtherTrainer(make_config(inputs=["q"]))
        trainer.dataset = SimpleNamespace(
            normalize_mode="gaussian",
            normalize_lowdim_keys=["q"],
        )
        values = torch.tensor([[0.0], [2.0], [100.0], [200.0]])

        stats = trainer._normalizer_stats({"q": values}, training_frames=[0, 1])

        torch.testing.assert_close(stats["q"]["mean"], torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
