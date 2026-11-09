"""DP teacher/student and CAD-WM must see the same episode-filtered wrench."""

import copy
import importlib
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import yaml

from diffusion_policy.common.wrench_filter import first_order_lowpass, relative_time_seconds
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector, VectorThresholdGate
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.dataset import lerobot_v3_dataset as v3_dataset
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


ROOT = Path(__file__).resolve().parents[1]
cad_data = importlib.import_module("cad-wm.data_process.libero_wrench_dataset")


@pytest.fixture
def aligned_datasets(tmp_path, monkeypatch):
    size = 24
    clocks = np.rint((0.25 + np.arange(size) * .05) * 1e9).astype(np.int64)
    first = np.zeros((size, 6), dtype=np.float32)
    first[4:8, :3] = [20, 1, -1]
    first[7, 3:] = [30, -20, 10]
    second = np.tile(np.array([2, -1, 3, -2, 1, -3], dtype=np.float32), (size, 1))
    raw = np.concatenate((first, second))
    columns = {"timestamp": np.tile(np.arange(size) * .05, 2),
               "timing.state_timestamp_ns": np.tile(clocks, 2),
               "observation.wrench": raw,
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
        shape_meta={"obs": {key: {"shape": [8, 6], "type": "low_dim"}
                            for key in ("wrench", "future_wrench")}, "action": {"shape": [7]}},
        dataset_path="unused", horizon=9, n_obs_steps=2, pad_before=1, val_ratio=.5,
        windowed_lowdim_feature_groups={
            "wrench": {"source_key": "observation.wrench", "steps": 8, "start_offset": -7},
            "future_wrench": {"source_key": "observation.wrench", "steps": 8, "start_offset": 1},
        },
        phase_resampling_enabled=True, phase_source_key="observation.wrench",
        contact_threshold=8, contact_enter_frames=5, contact_exit_frames=5, contact_norm="l2",
        wrench_filter={"enabled": True, "cutoff_hz": 2., "source_key": "observation.wrench",
                       "timestamp_key": "timing.state_timestamp_ns"},
    )
    teacher = v3_dataset.LeRobotV3ImageDataset(**config)
    workspace = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.load(ROOT / "diffusion_policy/config/train_distill_render_net_student.yaml")
    workspace.cfg.dataset = {"val_ratio": .5}
    workspace.teacher_cfg = OmegaConf.create({"task": {"dataset": config}})
    workspace.teacher_source_cfg = None
    student_config = OmegaConf.to_container(workspace._dataset_config(), resolve=True)
    student = v3_dataset.LeRobotV3ImageDataset(**student_config)

    h5_path = tmp_path / "demos.hdf5"
    with h5py.File(h5_path, "w") as file:
        for episode, values in enumerate((first, second)):
            demo = file.create_group(f"data/demo_{episode}")
            demo.create_dataset("states", data=(clocks * 1e-9)[:, None])
            for key, width in (("joint_states", 7), ("dq", 7), ("tau", 7)):
                demo.create_dataset(f"obs/{key}", data=np.zeros((size, width), dtype=np.float32))
            demo.create_dataset("obs/wrenches", data=values)
    cad_config = yaml.safe_load((ROOT / "cad-wm/config/bench_train_config/libero_drawer_wrench_h5.yaml").read_text())
    cad_config["dataloader"]["root"] = str(h5_path)
    cad = cad_data.LiberoWrenchDataset(cad_config)
    return teacher, student, cad, raw, config


def test_full_episode_filter_matches_all_three_and_reuses_one_column(aligned_datasets):
    teacher, student, cad, raw, _ = aligned_datasets
    expected = np.concatenate([
        first_order_lowpass(np.arange(24) * .05, raw[start:start + 24], 2).astype(np.float32)
        for start in (0, 24)
    ])
    for dataset in (teacher, student):
        actual = dataset._load_column("observation.wrench", np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(dataset._phase_source_data, actual)
        # Both directions use the same filtered native row, even in overlapping windows.
        np.testing.assert_array_equal(dataset.windowed_lowdim_data["future_wrench"][5, 0], actual[6])
        np.testing.assert_array_equal(dataset.windowed_lowdim_data["wrench"][6, -1], actual[6])
        np.testing.assert_array_equal(dataset._load_column("observation.wrench", np.float32), actual)
        validation = dataset.get_validation_dataset()
        assert validation._filtered_columns is dataset._filtered_columns
        assert validation.windowed_lowdim_data is dataset.windowed_lowdim_data
    np.testing.assert_array_equal(teacher._filtered_columns["observation.wrench"],
                                  student._filtered_columns["observation.wrench"])
    np.testing.assert_allclose(cad.high_tensors["wrench"].numpy(), expected, rtol=1e-6, atol=1e-6)
    # State resets at the second demo, rather than inheriting the first demo's tail.
    np.testing.assert_array_equal(expected[24:], raw[24:])
    assert not np.array_equal(expected, raw)
    cad.fit_normalizer(range(len(cad)))
    torch.testing.assert_close(cad.normalizer.stats["wrench"]["mean"], cad.high_tensors["wrench"].mean(0))
    expected_labels = torch.cat([cad_data.wrench_phase_labels(cad.high_tensors["wrench"][start:start + 24],
                                                           cad.config, 20) for start in (0, 24)])
    torch.testing.assert_close(cad.contact, expected_labels)


def test_legacy_data_config_without_filter_keeps_original_values(aligned_datasets):
    _, _, _, raw, config = aligned_datasets
    legacy_config = copy.deepcopy(config)
    del legacy_config["wrench_filter"]
    dataset = v3_dataset.LeRobotV3ImageDataset(**legacy_config)
    np.testing.assert_array_equal(dataset._load_column("observation.wrench", np.float32), raw)


def test_active_configs_all_select_first_order_two_hz():
    teacher = OmegaConf.load(ROOT / "diffusion_policy/config/train_civi_teacher.yaml")
    assert teacher.task.dataset.wrench_filter.enabled
    assert teacher.task.dataset.wrench_filter.cutoff_hz == 2
    for name in ("train_distill_render_net", "train_distill_render_net_student",
                 "train_distill_render_net_student_test"):
        config = OmegaConf.load(ROOT / f"diffusion_policy/config/{name}.yaml")
        assert config.dataset.wrench_filter == teacher.task.dataset.wrench_filter
    for name in ("libero_drawer_wrench_h5", "libero_drawer_wrench_h5_median3"):
        config = yaml.safe_load((ROOT / f"cad-wm/config/bench_train_config/{name}.yaml").read_text())
        assert config["dataloader"]["wrench_filter"]["enabled"]
        assert config["dataloader"]["wrench_filter"]["cutoff_hz"] == 2
        assert "median3" not in config["dataloader"]["root"]


def test_integer_epoch_nanoseconds_keep_interval_precision():
    clocks = np.array([1_787_380_170_640_629_000, 1_787_380_170_690_629_001], dtype=np.int64)
    np.testing.assert_allclose(relative_time_seconds(clocks, "ns"), [0, .050000001], rtol=0, atol=1e-16)


def test_dataloader_returns_cached_alpha_from_the_teacher_contact_rules(aligned_datasets):
    teacher, student, _, _, _ = aligned_datasets
    preprocessor = WrenchPreprocessor(
        2, "future_wrench", "wrench", "future_wrench",
        HysteresisContactDetector(8., 8., enter_frames=5, exit_frames=5, norm="l2"),
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
        assert ((labels[-1] > 0) & (labels[-1] < 1)).any()
        cache_size = len(dataset._contact_alpha_cache)
        evaluation[0]
        assert len(dataset._contact_alpha_cache) == cache_size
        assert "contact_alpha" not in dataset.lowdim_keys
    torch.testing.assert_close(labels[0], labels[1], rtol=0, atol=0)
