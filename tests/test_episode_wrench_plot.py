"""Recorded wrench and contact must share the attention anchor's clock."""

from types import SimpleNamespace

import numpy as np
import pytest

from diffusion_policy.dataset.hirol_lerobot_v3_dataset import HirolLeRobotV3Dataset
from diffusion_policy.model.vision.contact_curriculum import ContactDetector, HysteresisContactDetector
from diffusion_policy.scripts.episode_wrench_plot import WrenchEpisodePlot, load_episode_wrench


def episode_dataset(values, *, start=5, timestamp_key="timestamp"):
    values = np.asarray(values, dtype=np.float32)
    raw = np.concatenate((np.zeros((start, 6), dtype=np.float32), values))
    times = 120.0 + np.arange(len(raw), dtype=np.float64) * .05
    if timestamp_key.endswith("_ns"):
        times *= 1e9

    def column(key, dtype):
        assert key == "observation.wrench"
        return raw.astype(dtype)

    return SimpleNamespace(
        phase_source_key="observation.wrench", _phase_source_data=None,
        _load_column=column, episode_ranges=[range(start), range(start, len(raw))],
        timestamps=times, timestamp_key=timestamp_key,
        _timestamp_seconds=HirolLeRobotV3Dataset._timestamp_seconds,
        contact_threshold=100., contact_force_dims=(0, 1, 2), contact_norm="l2",
        contact_enter_frames=1, contact_exit_frames=1,
    )


def contact_policy(*, enter=3, exit=2):
    return SimpleNamespace(contact_detector=HysteresisContactDetector(
        free_threshold=8., contact_threshold=8., force_dims=(0, 1, 2),
        norm="l2", enter_frames=enter, exit_frames=exit,
    ))


@pytest.mark.parametrize("timestamp_key", ["timestamp", "timing.state_timestamp_ns"])
def test_recorded_six_channels_and_anchor_time_have_no_future_shift(timestamp_key):
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    dataset = episode_dataset(values, timestamp_key=timestamp_key)
    timeline = load_episode_wrench(dataset, contact_policy(), 1, [7, 11], 20)
    np.testing.assert_array_equal(timeline["values"], values)
    np.testing.assert_allclose(timeline["times"], np.arange(10) * .05)
    np.testing.assert_array_equal(timeline["anchor_offsets"], [2, 6])
    np.testing.assert_allclose(timeline["anchor_times"], [.1, .3])
    np.testing.assert_array_equal(timeline["values"][timeline["anchor_offsets"]], values[[2, 6]])


def test_contact_scans_full_episode_before_anchor_stride_and_respects_debounce():
    values = np.zeros((10, 6), dtype=np.float32)
    # Exact-threshold values hold the state but interrupt a consecutive streak.
    values[:, 0] = [0, 9, 9, 8, 9, 9, 9, 8, 0, 0]
    values[:, 3:] = 1000  # Torque must not count as force contact.
    dataset = episode_dataset(values)
    policy = contact_policy(enter=3, exit=2)
    all_frames = load_episode_wrench(dataset, policy, 1, range(5, 15), 20)
    sparse_frames = load_episode_wrench(dataset, policy, 1, [5, 11, 14], 20)
    causal_expected = [False, False, False, False, False, False, True, True, True, False]
    offline_expected = [False, False, False, False, True, True, True, True, False, False]
    np.testing.assert_array_equal(all_frames["contact"], offline_expected)
    np.testing.assert_array_equal(sparse_frames["contact"], offline_expected)
    np.testing.assert_array_equal(all_frames["causal_contact"], causal_expected)
    np.testing.assert_array_equal(sparse_frames["causal_contact"], causal_expected)
    np.testing.assert_allclose(all_frames["contact_magnitude"], values[:, 0])
    np.testing.assert_allclose(all_frames["contact_intervals_sec"], [[.2, .4]])
    assert all_frames["metadata"]["contact"]["annotation_mode"] == "offline_backdated_confirmation_windows"
    # Dataset threshold=100 is intentionally different: the teacher's rule wins.
    assert bool(sparse_frames["contact"][sparse_frames["anchor_offsets"][1]])


def test_five_frame_confirmation_backdates_entry_and_exit_by_point_two_seconds():
    values = np.zeros((18, 6), dtype=np.float32)
    values[2:10, 0] = 9.
    timeline = load_episode_wrench(
        episode_dataset(values), contact_policy(enter=5, exit=5), 1, range(5, 23), 20,
    )
    # At 20 Hz the fifth confirming sample arrives 4 * .05 s after
    # the first sample of the confirmed contact/free window.
    offline_indices = np.flatnonzero(timeline["contact"])
    causal_indices = np.flatnonzero(timeline["causal_contact"])
    np.testing.assert_array_equal(offline_indices, np.arange(2, 10))
    np.testing.assert_array_equal(causal_indices, np.arange(6, 14))
    np.testing.assert_allclose(timeline["contact_intervals_sec"], [[.1, .5]])
    np.testing.assert_allclose(
        timeline["times"][causal_indices[[0, -1]]] - timeline["times"][offline_indices[[0, -1]]],
        [.2, .2],
    )


def test_shorter_than_five_samples_does_not_confirm_a_contact_interval():
    values = np.zeros((6, 6), dtype=np.float32)
    values[1:5, 0] = 9.
    timeline = load_episode_wrench(
        episode_dataset(values), contact_policy(enter=5, exit=5), 1, [5, 10], 20,
    )
    assert not timeline["contact"].any()
    assert not timeline["causal_contact"].any()
    assert timeline["contact_intervals_sec"].shape == (0, 2)


def test_episode_end_does_not_backdate_an_unconfirmed_exit():
    values = np.zeros((10, 6), dtype=np.float32)
    values[1:6, 0] = 9.
    timeline = load_episode_wrench(
        episode_dataset(values), contact_policy(enter=5, exit=5), 1, [5, 14], 20,
    )
    # Four below-threshold samples at the tail cannot confirm an exit.
    np.testing.assert_array_equal(timeline["contact"], [False] + [True] * 9)
    np.testing.assert_array_equal(timeline["causal_contact"], [False] * 5 + [True] * 5)
    np.testing.assert_allclose(timeline["contact_intervals_sec"], [[.05, .5]])


def test_plain_contact_detector_has_no_confirmation_window_to_backdate():
    values = np.zeros((6, 6), dtype=np.float32)
    values[:, 0] = [0., 9., 9., 8., 9., 0.]
    dataset = episode_dataset(values)
    dataset.contact_enter_frames = dataset.contact_exit_frames = 5
    policy = SimpleNamespace(contact_detector=ContactDetector(threshold=8., norm="l2"))
    timeline = load_episode_wrench(dataset, policy, 1, range(5, 11), 20)
    expected = [False, True, True, False, True, False]
    np.testing.assert_array_equal(timeline["contact"], expected)
    np.testing.assert_array_equal(timeline["causal_contact"], expected)
    np.testing.assert_allclose(timeline["contact_intervals_sec"], [[.05, .15], [.2, .25]])


def test_contact_at_end_is_shaded_through_last_sample_duration():
    values = np.zeros((4, 6), dtype=np.float32)
    values[1:, 2] = 9.
    timeline = load_episode_wrench(episode_dataset(values), contact_policy(enter=1), 1, [8], 20)
    np.testing.assert_allclose(timeline["contact_intervals_sec"], [[.05, .2]])


def test_plot_updates_synchronized_cursor_and_exports_static_six_channel_chart(tmp_path):
    values = np.arange(60, dtype=np.float32).reshape(10, 6) / 10
    timeline = load_episode_wrench(episode_dataset(values), contact_policy(), 1, [5, 12], 20)
    chart = WrenchEpisodePlot(timeline, width=512, height=420)
    first = chart.render(timeline["anchor_times"][0])
    last = chart.render(timeline["anchor_times"][1])
    assert first.shape == last.shape == (420, 512, 3)
    assert first.dtype == np.uint8 and first.flags.c_contiguous
    assert not np.array_equal(first, last)
    target = tmp_path / "wrench.png"
    chart.save(target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
