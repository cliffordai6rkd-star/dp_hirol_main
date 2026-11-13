"""Teacher and student share causal, episode-filtered force/torque observations."""

import copy
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.common.wrench_filter import first_order_lowpass, relative_time_seconds
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector, VectorThresholdGate
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.dataset import lerobot_v3_dataset as v3_dataset
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[("wrench", 6, (0, 1, 2)), ("tau_ext", 7, tuple(range(7)))],
                ids=["wrench", "tau_ext"])
def aligned_datasets(monkeypatch, request):
    key, dimensions, contact_dims = request.param
    source_key = "observation." + key
    size = 24
    clocks = np.rint((0.25 + np.arange(size) * .05) * 1e9).astype(np.int64)
    first = np.zeros((size, dimensions), dtype=np.float32)
    first[4:8, :3] = [20, 1, -1]
    first[7, 3:6] = [30, -20, 10]
    if key == "tau_ext":
        first[4:8, 6] = 20
    resting = np.array([2, -1, 3, -2, 1, -3, 2][:dimensions], dtype=np.float32)
    second = np.tile(resting, (size, 1))
    raw = np.concatenate((first, second))
    columns = {"timestamp": np.tile(np.arange(size) * .05, 2),
               "timing.state_timestamp_ns": np.tile(clocks, 2),
               source_key: raw,
               "action": np.zeros((size * 2, 7), dtype=np.float32)}

    class FakeV3:
        episode_data_index = {"from": [0, size], "to": [size, size * 2]}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return size * 2

        def get_column(self, key):
            return columns[key]

        def close(self):
            pass

    monkeypatch.setattr(v3_dataset, "LeRobotV3Dataset", FakeV3)
    config = dict(
        shape_meta={"obs": {key: {"shape": [8, dimensions], "type": "low_dim"}},
                    "action": {"shape": [7]}},
        dataset_path="unused", horizon=9, n_obs_steps=2, pad_before=1, val_ratio=.5,
        windowed_lowdim_feature_groups={
            key: {"source_key": source_key, "steps": 8, "start_offset": -7},
        },
        phase_resampling_enabled=True, phase_source_key=source_key,
        contact_threshold=8, contact_enter_frames=5, contact_exit_frames=5, contact_norm="l2",
        contact_force_dims=contact_dims,
        wrench_filter={"enabled": True, "cutoff_hz": 2., "source_key": source_key,
                       "timestamp_key": "timing.state_timestamp_ns"},
    )
    teacher = v3_dataset.LeRobotV3ImageDataset(**config)
    workspace = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.load(ROOT / "diffusion_policy/config/train_distill_render_net_student.yaml")
    workspace.cfg.dataset.val_ratio = .5
    workspace.teacher_cfg = OmegaConf.create({"task": {"dataset": config}})
    workspace.teacher_source_cfg = None
    student_config = OmegaConf.to_container(workspace._dataset_config(), resolve=True)
    student = v3_dataset.LeRobotV3ImageDataset(**student_config)

    return teacher, student, raw, config


def test_teacher_student_filter_matches_and_reuses_current_observation_column(aligned_datasets):
    teacher, student, raw, config = aligned_datasets
    key = next(iter(config["windowed_lowdim_feature_groups"]))
    source_key = config["phase_source_key"]
    expected = np.concatenate([
        first_order_lowpass(np.arange(24) * .05, raw[start:start + 24], 2).astype(np.float32)
        for start in (0, 24)
    ])
    for dataset in (teacher, student):
        actual = dataset._load_column(source_key, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(dataset._phase_source_data, actual)
        # Each history ends at the observed row and retains its temporal order.
        np.testing.assert_array_equal(dataset.windowed_lowdim_data[key][6, -1], actual[6])
        np.testing.assert_array_equal(dataset.windowed_lowdim_data[key][9], actual[2:10])
        np.testing.assert_array_equal(dataset._load_column(source_key, np.float32), actual)
        validation = dataset.get_validation_dataset()
        assert validation._filtered_columns is dataset._filtered_columns
        assert validation.windowed_lowdim_data is dataset.windowed_lowdim_data
    np.testing.assert_array_equal(teacher._filtered_columns[source_key],
                                  student._filtered_columns[source_key])
    # State resets at the second demo, rather than inheriting the first demo's tail.
    np.testing.assert_array_equal(expected[24:], raw[24:])
    assert not np.array_equal(expected, raw)


def test_legacy_data_config_without_filter_keeps_original_values(aligned_datasets):
    _, _, raw, config = aligned_datasets
    legacy_config = copy.deepcopy(config)
    del legacy_config["wrench_filter"]
    dataset = v3_dataset.LeRobotV3ImageDataset(**legacy_config)
    np.testing.assert_array_equal(dataset._load_column(config["phase_source_key"], np.float32), raw)


@pytest.mark.parametrize("name,teacher_name,source_key,filter_enabled", [
    ("train_distill_render_net", "train_civi_teacher_libero", "observation.tau_ext", True),
    ("train_distill_render_net_student", "train_civi_teacher", "observation.tau_ext_unthresholded", False),
    ("train_distill_render_net_student_test", "train_civi_teacher_libero", "observation.wrench", True),
])
def test_student_configs_inherit_matching_teacher_torque_preprocessing(
        name, teacher_name, source_key, filter_enabled):
    teacher = OmegaConf.load(ROOT / f"diffusion_policy/config/{teacher_name}.yaml")
    workspace = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.load(ROOT / f"diffusion_policy/config/{name}.yaml")
    workspace.teacher_cfg = teacher
    workspace.teacher_source_cfg = None
    if source_key == "observation.tau_ext":
        assert "wrench_filter" not in workspace.cfg.dataset
        assert workspace._dataset_config().wrench_filter == teacher.task.dataset.wrench_filter
    elif name == "train_distill_render_net_student":
        assert "wrench_filter" not in workspace.cfg.dataset
        assert workspace._dataset_config().wrench_filter == teacher.task.dataset.wrench_filter
    else:
        # The historical student keeps its explicit filter for its wrench checkpoint.
        assert workspace.cfg.dataset.wrench_filter.source_key == source_key
    effective_filter = workspace._dataset_config().wrench_filter
    assert effective_filter.enabled is filter_enabled
    if filter_enabled:
        assert effective_filter.cutoff_hz == 2
    assert effective_filter.source_key == source_key


def test_future_measurements_cannot_change_current_observation_or_contact(aligned_datasets):
    teacher, _, raw, config = aligned_datasets
    key = next(iter(config["windowed_lowdim_feature_groups"]))
    raw[10:24] += 100.0
    changed = v3_dataset.LeRobotV3ImageDataset(**config)
    np.testing.assert_array_equal(
        changed.windowed_lowdim_data[key][:10],
        teacher.windowed_lowdim_data[key][:10],
    )
    assert not np.array_equal(
        changed.windowed_lowdim_data[key][10:24],
        teacher.windowed_lowdim_data[key][10:24],
    )
    preprocessor = WrenchPreprocessor(
        2, key, key, None,
        HysteresisContactDetector(8., 8., enter_frames=5, exit_frames=5, norm="l2",
                                  force_dims=config["contact_force_dims"]),
        VectorThresholdGate(0., enabled=False), LinearNormalizer(),
    )
    labels = [preprocessor.contact_alpha({
        key: torch.from_numpy(dataset.windowed_lowdim_data[key][8:10]).unsqueeze(0),
    }) for dataset in (teacher, changed)]
    torch.testing.assert_close(labels[0], labels[1], rtol=0, atol=0)


def test_integer_epoch_nanoseconds_keep_interval_precision():
    clocks = np.array([1_787_380_170_640_629_000, 1_787_380_170_690_629_001], dtype=np.int64)
    np.testing.assert_allclose(relative_time_seconds(clocks, "ns"), [0, .050000001], rtol=0, atol=1e-16)


def test_dataloader_returns_cached_alpha_from_the_teacher_contact_rules(aligned_datasets):
    teacher, student, _, config = aligned_datasets
    key = next(iter(config["windowed_lowdim_feature_groups"]))
    preprocessor = WrenchPreprocessor(
        2, key, key, None,
        HysteresisContactDetector(8., 8., enter_frames=5, exit_frames=5, norm="l2",
                                  force_dims=config["contact_force_dims"]),
        VectorThresholdGate(0., enabled=False), LinearNormalizer(),
    )
    labels = []
    for dataset in (teacher, student):
        dataset.set_contact_label_config(preprocessor.get_config())
        evaluation = dataset.get_all_episodes_dataset()
        batches = []
        for batch in torch.utils.data.DataLoader(evaluation, batch_size=3, shuffle=False):
            obs = batch["obs"]
            assert obs["contact_alpha"].shape == (len(batch["action"]), 2)
            torch.testing.assert_close(obs["contact_alpha"], preprocessor.contact_alpha(obs), rtol=0, atol=0)
            batches.append(obs["contact_alpha"])
        labels.append(torch.cat(batches))
        assert (labels[-1] == 0).any()
        assert (labels[-1] == 1).any()
        assert torch.all((labels[-1] == 0) | (labels[-1] == 1))
        cache_size = len(dataset._contact_alpha_cache)
        evaluation[0]
        assert len(dataset._contact_alpha_cache) == cache_size
        assert "contact_alpha" not in dataset.lowdim_keys
    torch.testing.assert_close(labels[0], labels[1], rtol=0, atol=0)
