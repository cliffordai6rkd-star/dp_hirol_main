import csv
import json
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.scripts import compare_wrench_action_mse as comparison


class WrenchPolicy(torch.nn.Module):
    wrench_key = "future_wrench"
    action_start_index = 1
    n_action_steps = 2

    def __init__(self):
        super().__init__()
        self.calls = []
        self.normalizer = LinearNormalizer()
        self.normalizer[self.wrench_key] = SingleFieldLinearNormalizer.create_identity()

    def predict_action(self, obs, generator=None):
        assert not torch.is_grad_enabled()
        wrench = obs[self.wrench_key]
        noise = torch.randn(len(wrench), generator=generator, device=wrench.device)
        self.calls.append((wrench.clone(), noise.clone(), obs["phase"]))
        value = wrench.flatten(1).mean(1) + noise
        prediction = value[:, None, None].expand(-1, 4, 2).clone()
        prediction[:, 0] = 1e4  # Context and unused action positions must be excluded.
        prediction[:, 3] = 2e4
        return {"model_action_pred": prediction}


class Dataset:
    dataset_path = "test-dataset"
    n_obs_steps = 2
    phase_source_key = "observation.tau_ext"
    contact_threshold = 1.5
    contact_force_dims = (0, 1, 2)
    contact_norm = "l1"
    precontact_seconds = 1.0

    def __init__(self):
        self.indices = np.arange(5, dtype=np.int64)[:, None]
        self.windowed_lowdim_data = {"future_wrench": np.arange(16, dtype=np.float32).reshape(8, 2, 1)}
        self.lowdim_data = {}
        self.episode_index = np.zeros(8, dtype=np.int64)
        self.episode_ranges = [range(8)]
        self.timestamps = np.arange(8) / 50.0
        self.phase_labels = np.array([0, 0, 0, 1, 2], dtype=np.int8)
        self.lerobot_dataset = SimpleNamespace(close=self.close)
        self.closed = False
        self.image_reads = 0

    def close(self):
        self.closed = True

    def __len__(self):
        return len(self.indices)

    def get_phase_labels(self):
        return self.phase_labels.copy()

    @staticmethod
    def _timestamp_seconds(values):
        return values

    def _anchor_index_from_sampler_row(self, row):
        return int(row[0])

    def _sequence_indices_from_sampler_rows(self, rows):
        # A non-contiguous window represents timestamp-based resampling.
        return np.concatenate([rows, rows + 2], axis=1)

    def __getitem__(self, index):
        self.image_reads += 1
        rows = self._sequence_indices_from_sampler_rows(self.indices[index:index + 1])[0]
        return {"obs": {"future_wrench": torch.from_numpy(self.windowed_lowdim_data["future_wrench"][rows]),
                        "image": torch.full((2, 3, 2, 2), float(index)),
                        "phase": torch.ones(2, 2, 1)},
                "action": torch.zeros(4, 2)}


def test_phase_spans_use_original_dense_windows_and_episode_relative_time():
    dataset = Dataset()
    dataset.timestamps += 20
    # The episode can begin before its first valid inference window.
    dataset.indices = np.arange(1, 5)[:, None]
    dataset.phase_labels = np.array([0, 1, 2, 2], dtype=np.int8)
    spans = comparison.episode_phase_spans(dataset, 0, np.arange(4))
    assert [span["phase"] for span in spans] == ["free", "precontact", "contact"]
    np.testing.assert_allclose([span["start_seconds"] for span in spans], [0.02, 0.04, 0.06])
    np.testing.assert_allclose([span["end_seconds"] for span in spans], [0.04, 0.06, 0.08])
    dataset.phase_labels[:] = 0  # No contact: do not invent other stages.
    assert len(comparison.episode_phase_spans(dataset, 0, np.arange(4))) == 1
    assert dataset.image_reads == 0


def test_conditions_share_diffusion_noise_and_only_change_raw_wrench():
    policy = WrenchPolicy().eval()
    obs = {"future_wrench": torch.tensor([2.0, 5.0])[:, None, None, None].expand(2, 2, 2, 1),
           "phase": torch.ones(2, 2, 2, 1), "image": torch.rand(2, 2, 3, 2, 2)}
    before = {key: value.clone() for key, value in obs.items()}
    expected_random = torch.rand(obs["future_wrench"].shape, generator=torch.Generator().manual_seed(10)) * 40 - 20
    scores = comparison.evaluate_batch(policy, obs, torch.zeros(2, 4, 2), seed=9,
                                       noise_generator=torch.Generator().manual_seed(10))
    noise = policy.calls[0][1]
    for wrench, sampled_noise, phase in policy.calls:
        torch.testing.assert_close(sampled_noise, noise, rtol=0, atol=0)
        assert phase is obs["phase"]
    torch.testing.assert_close(scores["original"], (torch.tensor([2.0, 5.0]) + noise).square())
    torch.testing.assert_close(scores["zero"], noise.square())
    torch.testing.assert_close(policy.calls[0][0], expected_random)
    torch.testing.assert_close(policy.calls[1][0], torch.zeros_like(expected_random))
    torch.testing.assert_close(scores["random"], (expected_random.flatten(1).mean(1) + noise).square())
    for key in obs:
        torch.testing.assert_close(obs[key], before[key], rtol=0, atol=0)

    repeat = comparison.evaluate_batch(policy, obs, torch.zeros(2, 4, 2), seed=9,
                                       noise_generator=torch.Generator().manual_seed(10))
    for key in scores:
        torch.testing.assert_close(scores[key], repeat[key], rtol=0, atol=0)


def test_uniform_wrench_uses_training_stats_and_configurable_scale_for_each_dimension():
    policy = WrenchPolicy().eval()
    means = torch.tensor([-2.0, 0.1])
    stds = torch.tensor([3.0, 0.02])
    policy.normalizer[policy.wrench_key] = SingleFieldLinearNormalizer.create_manual(
        torch.ones(2), torch.zeros(2),
        {"mean": means, "std": stds, "min": means - 3 * stds, "max": means + 3 * stds},
    )
    obs = {"future_wrench": torch.zeros(1024, 2, 8, 2), "phase": torch.ones(1024, 2, 8, 1)}
    comparison.evaluate_batch(
        policy, obs, torch.zeros(1024, 4, 2), seed=9, noise_scale=4,
        noise_generator=torch.Generator().manual_seed(10),
    )
    noise = policy.calls[0][0]
    standardized = (noise - means) / stds
    assert standardized.min() >= -4 and standardized.max() <= 4
    # Check coverage and spread rather than just comparing to a second rand call.
    torch.testing.assert_close(standardized.flatten(0, -2).mean(0), torch.zeros(2), atol=0.05, rtol=0)
    torch.testing.assert_close(standardized.flatten(0, -2).std(0),
                               torch.full((2,), 4 / np.sqrt(3)), atol=0.05, rtol=0)
    assert not torch.equal(noise[:, :, 0], noise[:, :, 1])
    assert torch.count_nonzero(policy.calls[1][0]) == 0


def test_action_mse_rejects_incomplete_action_chunk():
    with pytest.raises(ValueError, match="matching shapes"):
        comparison.action_mse_per_sample(WrenchPolicy(), torch.zeros(2, 2, 2), torch.zeros(2, 2, 2))


def test_pose_errors_use_3d_distance_and_sign_invariant_rotation_on_executed_steps():
    policy = WrenchPolicy()
    target = torch.zeros(1, 4, 7, dtype=torch.float64)
    target[..., -1] = 1
    prediction = target.clone()
    prediction[:, 0, :3] = 1000  # Nonexecuted poses must be excluded.
    prediction[:, 3, :3] = 2000
    prediction[:, 1, :3] = torch.tensor([0.003, 0.004, 0])
    prediction[:, 2, :3] = torch.tensor([0.006, 0.008, 0])
    prediction[:, 1, 3:] = torch.tensor([0, 0, 0, -2])  # q and -q are the same pose.
    prediction[:, 2, 3:] = torch.tensor([0, 0, 2 ** -0.5, 2 ** -0.5])
    values = comparison.pose_errors_per_sample(policy, prediction, target)
    assert values["position_rmse_mm"].item() == pytest.approx(np.sqrt((5 ** 2 + 10 ** 2) / 2))
    assert values["rotation_mean_deg"].item() == pytest.approx(45)
    assert values["rotation_mse_deg2"].item() == pytest.approx(90 ** 2 / 2)
    with pytest.raises(ValueError, match="xyz"):
        comparison.pose_errors_per_sample(policy, prediction[..., :6], target[..., :6])


def test_pose_summary_aggregates_squared_errors_before_root_and_exports_phases(tmp_path):
    records = []
    for index, position_mm in enumerate((0, 10)):
        row = {"sample_index": index, "episode_time_seconds": float(index),
               "phase": "free" if index == 0 else "contact"}
        for name, _, _ in comparison.CURVES:
            row.update({f"mse_{name}": 0.0,
                        f"position_mse_m2_{name}": (position_mm / 1000) ** 2,
                        f"position_rmse_mm_{name}": position_mm,
                        f"rotation_mean_deg_{name}": float(index * 90),
                        f"rotation_mse_deg2_{name}": float(index * 90 ** 2)})
        records.append(row)
    metadata = {"episode": 50, "n_action_steps": 8, "phase_spans": []}
    paths = comparison.save_results(tmp_path / "pose.png", records, metadata)
    summary = metadata["pose_metrics"]["summary"]
    overall = next(row for row in summary if row["phase"] == "overall" and row["condition"] == "original")
    assert overall["position_rmse_mm"] == pytest.approx(np.sqrt(50))
    assert overall["rotation_mean_deg"] == 45
    assert overall["rotation_rmse_deg"] == pytest.approx(np.sqrt(90 ** 2 / 2))
    assert overall["windows"] == 2 and overall["predicted_poses"] == 16
    assert {row["phase"] for row in summary} == {"overall", "free", "contact"}
    assert paths["pose_plot"].is_file() and paths["pose_summary"].is_file()


def test_export_has_time_axis_three_curve_colors_and_phase_backgrounds(monkeypatch, tmp_path):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    axes = []
    subplots = plt.subplots

    def capture_axes(*args, **kwargs):
        fig, ax = subplots(*args, **kwargs)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", capture_axes)
    records = [{"sample_index": i, "episode_time_seconds": 2 + i / 5,
                "mse_random": i + 2, "mse_zero": i + 1, "mse_original": i} for i in range(3)]
    # Boundaries come from the dense episode, not the sparse evaluation points.
    metadata = {"episode": 7, "phase_spans": [
        {"phase_id": phase, "start_seconds": start, "end_seconds": stop}
        for phase, start, stop in [(0, 2.0, 2.16), (1, 2.16, 2.26), (2, 2.26, 2.4)]
    ]}
    paths = comparison.save_results(tmp_path / "comparison.png", records, metadata)
    ax = axes[0]
    curves = [line for line in ax.lines if line.get_label().startswith(("Uniform", "Zero", "Dataset"))]
    boundaries = [line for line in ax.lines if line not in curves]
    assert len(curves) == 3 and len(boundaries) == 2
    assert [line.get_color() for line in curves] == ["red", "yellow", "blue"]
    assert all("mean action MSE=" in line.get_label() for line in curves)
    np.testing.assert_allclose(curves[0].get_xdata(), [2.0, 2.2, 2.4])
    np.testing.assert_allclose([line.get_xdata()[0] for line in boundaries], [2.16, 2.26])
    assert [patch.get_facecolor() for patch in ax.patches] == [
        matplotlib.colors.to_rgba(color) for color in ["white", "#ffe0b2", "#ffcdd2"]
    ]
    assert ax.get_xlabel() == "Episode time (s)"
    assert paths["plot"].is_file()
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["mean_action_mse_raw"] == {"random": 3, "zero": 2, "original": 1}
    with paths["csv"].open() as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_run_evaluates_all_three_conditions_and_final_single_sample_batch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    cfg = OmegaConf.create({"training": {"device": "cpu", "amp": {"enabled": False, "dtype": "bfloat16"}}})
    dataset, policy = Dataset(), WrenchPolicy().eval()
    # Interleave another episode and scramble indices to exercise filtering/sorting.
    dataset.indices = np.array([[5], [2], [0], [4], [1], [3]])
    dataset.episode_index[5:] = 1
    dataset.episode_ranges = [range(5), range(5, 8)]
    dataset.timestamps += 17
    dataset.phase_labels = np.array([2, 0, 0, 2, 0, 1], dtype=np.int8)
    monkeypatch.setattr(comparison.common, "load_checkpoint_payload",
                        lambda path: ({"state_dicts": {"ema_model": {}}}, cfg))
    monkeypatch.setattr(comparison.common, "instantiate_policy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(comparison, "instantiate_dataset", lambda *args: dataset)
    args = comparison.parse_args(["--checkpoint", str(checkpoint), "--episode", "0",
                                  "--batch-size", "2", "--num-workers", "0", "--start-time", "0"])
    paths = comparison.run(args)
    assert dataset.closed and dataset.image_reads == 5
    assert len(policy.calls) == 9  # Three conditions for each of the 2+2+1 batches.
    with paths["csv"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert [int(row["anchor_frame"]) for row in rows] == [0, 1, 2, 3, 4]
    assert all(row["episode"] == "0" for row in rows)
    assert [row["phase"] for row in rows] == ["free", "free", "free", "precontact", "contact"]
    np.testing.assert_allclose([float(row["episode_time_seconds"]) for row in rows], np.arange(5) / 50)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["samples"] == 5 and metadata["state_key"] == "ema_model"
    assert metadata["paired_diffusion_noise"] and metadata["phase_signals"] == "unchanged"
    assert metadata["episode"] == 0
    assert metadata["random_noise"]["distribution"] == "uniform"
    assert metadata["random_noise"]["low"] == [-20.0]
    assert metadata["random_noise"]["high"] == [20.0]
    assert metadata["zero_wrench"]["raw_value"] == 0
    assert "action prediction MSE" in metadata["legend_mean"]


def test_episode_is_required():
    with pytest.raises(SystemExit):
        comparison.parse_args([])


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
def test_noise_scale_must_be_positive_and_finite(scale):
    with pytest.raises(SystemExit):
        comparison.parse_args(["--episode", "0", "--noise-scale", scale])
