import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

av = pytest.importorskip("av")

from data_process.tau_ext_video import (
    EpisodeVideoCursor, TauExtNormPlot, episode_video_spec, export_tau_ext_video,
)


def _write_video(path, count=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for index in range(count):
            image = np.full((64, 64, 3), 20 + index * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_episode_video_spec_uses_shard_offset_and_actual_timestamps():
    key = "observation.images.front"
    info = {"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    metadata = {f"videos/{key}/chunk_index": 2, f"videos/{key}/file_index": 4,
                f"videos/{key}/from_timestamp": 8.4}
    path, targets = episode_video_spec(info, metadata, key, [0, 0.039, 0.081])
    assert path == "videos/observation.images.front/chunk-002/file-004.mp4"
    np.testing.assert_allclose(targets, [8.4, 8.439, 8.481])


def test_video_cursor_seeks_to_later_episode_without_frame_offset(tmp_path):
    path = tmp_path / "source.mp4"
    _write_video(path)
    cursor = EpisodeVideoCursor(path, 0.2, 25)
    try:
        for index in (5, 6, 7):
            rgb = cursor.rgb_at(index / 25 + 1e-8)
            assert abs(float(rgb.mean()) - (20 + index * 20)) < 3
        with pytest.raises(ValueError, match="timestamp mismatch"):
            cursor.rgb_at(2)
    finally:
        cursor.close()


def test_norm_plot_changes_cursor_and_current_readout():
    plot = TauExtNormPlot([0, 0.04, 0.08], [5, 15, 10], [0, 15, 0],
                         [False, True, False], 12.5, 896, 280)
    free, contact = plot.render(0), plot.render(1)
    assert free.shape == contact.shape == (280, 896, 3)
    assert not np.array_equal(free, contact)
    assert np.array_equal(plot.background, plot.render(0)) is False


def test_export_uses_episode_video_offset_and_keeps_all_source_frames(tmp_path):
    root = tmp_path / "dataset"
    key = "observation.images.front"
    video_template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    _write_video(root / video_template.format(video_key=key, chunk_index=0, file_index=0))
    meta = root / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "fps": 25, "video_path": video_template, "features": {key: {"dtype": "video"}},
    }))
    (meta / "tau_ext_processing.json").write_text(json.dumps({
        "config": {"contact_threshold_l1": 12.5},
    }))
    pq.write_table(pa.table({
        "episode_index": [3], f"videos/{key}/chunk_index": [0],
        f"videos/{key}/file_index": [0], f"videos/{key}/from_timestamp": [0.2],
    }), meta / "episodes/chunk-000/file-000.parquet")
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [3, 3, 3], "frame_index": [0, 1, 2], "timestamp": [0., .04, .08],
        "observation.tau_ext": [[0.] * 7, [2.] * 7, [0.] * 7],
        "observation.tau_ext_l1": [[5.], [14.], [10.]],
        "observation.contact": [[0.], [1.], [0.]],
    }), data / "file-000.parquet")
    result = export_tau_ext_video(root, 3, [key], width_per_camera=448)
    with av.open(result["video"]) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 3
        assert float(container.streams.video[0].average_rate) == 25
        # The first state belongs to source frame 5 (gray 120), not frame 0.
        first = frames[0].to_ndarray(format="rgb24")
        assert abs(float(first[100:300, 50:400].mean()) - 120) < 4
    assert result["threshold_l1_Nm"] == 12.5
    assert result["norm"] == "L1 over seven joint torques"
