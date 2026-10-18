import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusion_policy.dataset.hirol_lerobot_v3_dataset import HirolLeRobotV3Dataset
from diffusion_policy.dataset.img_randomer import Image_randomer


class _FakeLeRobotV3Dataset:
    columns = {}
    episode_data_index = {"from": [0], "to": [4]}

    def __init__(self, *args, **kwargs):
        self.episode_data_index = type(self).episode_data_index

    def __len__(self):
        return len(type(self).columns["timestamp"])

    def get_column(self, name):
        return type(self).columns[name]

    def close(self):
        return None


def _columns(action):
    return {
        "timestamp": [float(index) / 10.0 for index in range(4)],
        "observation.wrench_ext": [
            np.full((8, 6), index, dtype=np.float32)
            for index in range(4)
        ],
        "action.ee_pose": [value for value in action],
    }


def _shape_meta():
    return {
        "obs": {
            "wrench_ext": {
                "shape": [8, 6],
                "type": "low_dim",
            }
        },
        "action": {"shape": [7]},
    }


class HirolLeRobotV3DatasetActionLayoutTest(unittest.TestCase):
    def _create_dataset(
        self,
        action,
        *,
        action_layout,
        horizon,
        n_latency_steps=0,
        relative_pose_actions=False,
    ):
        _FakeLeRobotV3Dataset.columns = _columns(action)
        with patch(
            "diffusion_policy.dataset.hirol_lerobot_v3_dataset.LeRobotV3Dataset",
            _FakeLeRobotV3Dataset,
        ):
            return HirolLeRobotV3Dataset(
                shape_meta=_shape_meta(),
                dataset_path="unused",
                horizon=horizon,
                pad_before=1,
                pad_after=max(0, horizon - 1),
                n_obs_steps=2,
                n_latency_steps=n_latency_steps,
                val_ratio=0.0,
                action_feature_fields=["action.ee_pose"],
                action_layout=action_layout,
                relative_pose_actions=relative_pose_actions,
                preload_images=False,
            )

    def test_prechunked_action_uses_last_observation_as_anchor(self):
        action = np.zeros((4, 8, 7), dtype=np.float32)
        for frame_idx in range(4):
            action[frame_idx, :, 0] = frame_idx * 10 + np.arange(8)

        dataset = self._create_dataset(
            action,
            action_layout="prechunked",
            horizon=8,
        )

        self.assertEqual(len(dataset), 4)
        first = dataset[0]
        last = dataset[3]
        self.assertEqual(tuple(first["obs"]["wrench_ext"].shape), (2, 8, 6))
        self.assertEqual(tuple(first["action"].shape), (8, 7))
        self.assertNotIn("action_reference", first["obs"])
        np.testing.assert_array_equal(first["action"][:, 0].numpy(), np.arange(8))
        np.testing.assert_array_equal(last["action"][:, 0].numpy(), 30 + np.arange(8))

        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)))
        self.assertEqual(tuple(batch["obs"]["wrench_ext"].shape), (2, 2, 8, 6))
        self.assertEqual(tuple(batch["action"].shape), (2, 8, 7))

    def test_prechunked_action_applies_latency_inside_chunk(self):
        action = np.zeros((4, 5, 7), dtype=np.float32)
        action[:, :, 0] = np.arange(5)
        dataset = self._create_dataset(
            action,
            action_layout="prechunked",
            horizon=3,
            n_latency_steps=2,
        )

        sample = dataset[0]
        np.testing.assert_array_equal(sample["action"][:, 0].numpy(), [2, 3, 4])

    def test_per_step_action_behavior_is_unchanged(self):
        action = np.zeros((4, 7), dtype=np.float32)
        action[:, 0] = np.arange(4)
        dataset = self._create_dataset(
            action,
            action_layout="per_step",
            horizon=3,
        )

        sample = dataset[0]
        self.assertEqual(tuple(sample["action"].shape), (3, 7))
        np.testing.assert_array_equal(sample["action"][:, 0].numpy(), [0, 0, 1])

    def test_prechunked_action_rejects_short_chunk(self):
        action = np.zeros((4, 7, 7), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "require at least 8"):
            self._create_dataset(
                action,
                action_layout="prechunked",
                horizon=8,
            )

    def test_prechunked_relative_pose_actions_use_chunk_step_zero_as_reference(self):
        action = np.zeros((4, 3, 7), dtype=np.float32)
        action[..., 6] = 1.0
        for frame_idx in range(4):
            action[frame_idx, :, 0] = frame_idx + np.array([0.0, 0.1, 0.2])

        dataset = self._create_dataset(
            action,
            action_layout="prechunked",
            horizon=3,
            relative_pose_actions=True,
        )

        sample = dataset[2]
        np.testing.assert_allclose(
            sample["obs"]["action_reference"].numpy(),
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sample["action"][:, 0].numpy(),
            [0.0, 0.1, 0.2],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sample["action"][:, 3:7].numpy(),
            np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
            atol=1e-6,
        )

    def test_relative_pose_actions_reject_per_step_layout(self):
        action = np.zeros((4, 7), dtype=np.float32)
        action[..., 6] = 1.0
        with self.assertRaisesRegex(ValueError, "requires action_layout='prechunked'"):
            self._create_dataset(
                action,
                action_layout="per_step",
                horizon=3,
                relative_pose_actions=True,
            )

    def test_shared_image_augmentation_is_consistent_within_window(self):
        dataset = object.__new__(HirolLeRobotV3Dataset)
        dataset.image_randomer_shared = True
        dataset.image_randomer = Image_randomer(
            {
                "resize": False,
                "random_crop": False,
                "rotation": False,
                "color_jitter": True,
                "color_jitter_params": {
                    "brightness": 0.1,
                    "contrast": 0.15,
                    "saturation": 0.1,
                    "hue": 0.03,
                },
            }
        )
        image = np.linspace(0.0, 1.0, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
        augmented = dataset._augment_image_sequence(
            np.stack([image, image], axis=0)
        )
        np.testing.assert_array_equal(augmented[0], augmented[1])


class HirolLeRobotV3DatasetTimestampValidityTest(unittest.TestCase):
    def _create_dataset(
        self, *, warmup=0, invalid=(), val_ratio=0.0,
        source_timestamp=True, source_seconds=False, offsets=(0.04, 0.08),
        nonmonotonic_source=False,
    ):
        length = 32
        count = 2 * length
        local_rows = np.tile(np.arange(length), 2)
        state = np.concatenate([
            1_700_000_000_000_000_000 + episode * 10_000_000_000
            + np.arange(length, dtype=np.int64) * 20_000_000
            for episode in range(2)
        ])
        camera = state - 20_000_000
        if nonmonotonic_source:
            state[4] = state[3]
        valid = np.ones(count, dtype=np.uint8)
        valid[list(invalid)] = 0
        force = np.arange(count, dtype=np.float32)[:, None]
        # A valid zero force later in an episode must not be treated as warmup.
        force[12] = 0
        actions = np.zeros((count, 7), dtype=np.float32)
        actions[:, 0] = np.arange(count)
        actions[:, 6] = 1.0
        columns = {
            "timestamp": local_rows.astype(np.float64) * 0.02,
            "timing.state_timestamp_ns": state * 1e-9 if source_seconds else state,
            "timing.camera_timestamp_ns": camera,
            "timing.camera_valid": valid,
            "observation.tau_ext": force,
            "observation.ee_pose": actions,
        }
        future = {
            "source_key": "observation.tau_ext", "steps": 2,
            "start_offset": 2, "stride": 2,
            "anchor_timestamp_key": "timing.camera_timestamp_ns",
            "offsets_sec": list(offsets),
        }
        if source_timestamp:
            future["source_timestamp_key"] = "timing.state_timestamp_ns"
        with patch(
            "diffusion_policy.dataset.hirol_lerobot_v3_dataset.LeRobotV3Dataset",
            _FakeLeRobotV3Dataset,
        ), patch.object(_FakeLeRobotV3Dataset, "columns", columns), patch.object(
            _FakeLeRobotV3Dataset, "episode_data_index",
            {"from": [0, length], "to": [length, count]},
        ):
            return HirolLeRobotV3Dataset(
                shape_meta={"obs": {
                    "tau_ext": {"type": "low_dim", "shape": [3, 1]},
                    "future_tau_ext": {"type": "low_dim", "shape": [2, 1]},
                }, "action": {"shape": [7]}},
                dataset_path="unused", horizon=3, n_obs_steps=2, pad_before=1,
                window_sampling_strategy="timestamp", timestamp_step_sec=0.04,
                windowed_lowdim_feature_groups={
                    "tau_ext": {"source_key": "observation.tau_ext", "steps": 3,
                                "start_offset": -2, "stride": 1},
                    "future_tau_ext": future,
                },
                action_feature_fields=["observation.ee_pose"],
                observation_validity_key="timing.camera_valid",
                episode_warmup_steps=warmup, val_ratio=val_ratio, seed=42,
                preload_images=False,
            )

    def test_future_force_matches_absolute_state_clock_in_each_episode(self):
        for source_seconds in (False, True):
            with self.subTest(source_seconds=source_seconds):
                dataset = self._create_dataset(source_seconds=source_seconds)
                for anchor in (4, 36):
                    np.testing.assert_array_equal(
                        dataset.windowed_lowdim_data["future_tau_ext"][anchor, :, 0],
                        [anchor + 1, anchor + 3],
                    )

    def test_mixed_time_origins_fail_instead_of_selecting_episode_end(self):
        with self.assertRaisesRegex(ValueError, "different time origins.*source_timestamp_key"):
            self._create_dataset(source_timestamp=False)

    def test_nonmonotonic_force_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "source timestamps must increase"):
            self._create_dataset(nonmonotonic_source=True)

    def test_invalid_observation_history_is_filtered_in_train_and_validation(self):
        dataset = self._create_dataset(invalid=(10, 42), val_ratio=0.5)
        for view in (dataset, dataset.get_validation_dataset()):
            sequences = view._sequence_indices_from_sampler_rows(view.indices)
            observations = sequences[:, :2]
            self.assertTrue(view._observation_valid[observations].all())
            anchors = sequences[:, 1] % 32
            self.assertNotIn(10, anchors)
            self.assertNotIn(12, anchors)  # previous image would be row 10
            self.assertIn(11, anchors)

    def test_warmup_covers_earliest_force_history_without_removing_valid_zeros(self):
        dataset = self._create_dataset(warmup=4)
        sequences = dataset._sequence_indices_from_sampler_rows(dataset.indices)
        anchors = sequences[:, 1]
        # Two observation rows are 40 ms apart; each has three 50 Hz tau rows.
        self.assertEqual(int((anchors % 32).min()), 8)
        self.assertIn(12, anchors)
        self.assertEqual(float(dataset.windowed_lowdim_data["tau_ext"][12, -1, 0]), 0.0)
        self.assertTrue(((sequences[:, :2] % 32) - 2 >= 4).all())

    def test_future_time_bounds_use_offsets_instead_of_nominal_row_margin(self):
        dataset = self._create_dataset(offsets=(0.04, 0.12))
        sequences = dataset._sequence_indices_from_sampler_rows(dataset.indices)
        # Camera is one state row behind, so +120 ms needs five future rows.
        self.assertEqual(int((sequences[:, 1] % 32).max()), 26)

    def test_batched_filtering_matches_getitem_indices_and_pose_targets(self):
        dataset = self._create_dataset(warmup=4, invalid=(15,))
        sequences = dataset._sequence_indices_from_sampler_rows(dataset.indices)
        for index, expected in enumerate(sequences):
            actual = dataset._retime_sequence_indices(dataset._sample_indices_to_sequence(index))
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(dataset[index]["action"].numpy(), dataset.action_data[expected])
            self.assertEqual(len(np.unique(dataset.episode_index[expected])), 1)

    def test_normalizer_uses_valid_training_rows_and_excludes_validation(self):
        dataset = self._create_dataset(warmup=4, invalid=(15, 47), val_ratio=0.5)
        sequences = dataset._sequence_indices_from_sampler_rows(dataset.indices)
        actions = np.unique(sequences)
        observations = np.unique(sequences[:, :2])
        validation_rows = np.flatnonzero(dataset.val_mask[dataset.episode_index])
        dataset.action_data[validation_rows, 0] = 1e6
        dataset.windowed_lowdim_data["tau_ext"][validation_rows] = 1e6
        normalizer = dataset.get_normalizer(mode="gaussian")
        np.testing.assert_allclose(
            normalizer["action"].get_input_stats()["max"].numpy(),
            dataset.action_data[actions].max(axis=0),
        )
        np.testing.assert_allclose(
            normalizer["tau_ext"].get_input_stats()["max"].numpy(),
            dataset.windowed_lowdim_data["tau_ext"][observations].max(axis=(0, 1)),
        )
        self.assertLess(float(normalizer["action"].get_input_stats()["max"][0]), 1e6)


if __name__ == "__main__":
    unittest.main()
