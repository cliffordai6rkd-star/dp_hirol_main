"""Decode the ablation's two videos sequentially into a shared read-only cache."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = [pq.read_table(path, columns=["timestamp", "episode_index"]).to_pydict()
            for path in sorted((args.dataset / "data").rglob("*.parquet"))]
    timestamps = np.concatenate([np.asarray(item["timestamp"]).reshape(-1) for item in data])
    episode_index = np.concatenate([item["episode_index"] for item in data])
    columns = ["episode_index"] + [f"videos/{key}/{field}" for key in (
        "observation.images.front", "observation.images.right_wrist") for field in (
            "chunk_index", "file_index", "from_timestamp")]
    episodes = [row for path in sorted((args.dataset / "meta/episodes").rglob("*.parquet"))
                for row in pq.read_table(path, columns=columns).to_pylist()]
    camera_map = {"side": "observation.images.front", "wrist": "observation.images.right_wrist"}
    for key, feature in camera_map.items():
        output = np.lib.format.open_memmap(args.output / f"{key}.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(timestamps), 3, 224, 224))
        files = {}
        for episode in episodes:
            rows = np.flatnonzero(episode_index == episode["episode_index"])
            times = timestamps[rows] + episode[f"videos/{feature}/from_timestamp"]
            video = args.dataset / f'videos/{feature}/chunk-{episode[f"videos/{feature}/chunk_index"]:03d}/file-{episode[f"videos/{feature}/file_index"]:03d}.mp4'
            files.setdefault(video, []).extend(zip(times.tolist(), rows.tolist()))
        assigned = np.zeros(len(timestamps), dtype=bool)
        for video, requested in files.items():
            requested.sort()
            cursor = 0
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for frame in container.decode(stream):
                    moment = float(frame.pts * frame.time_base)
                    while cursor < len(requested) and requested[cursor][0] <= moment + .5 / 25 + 1e-6:
                        timestamp, row = requested[cursor]
                        if abs(moment - timestamp) > .5 / 25 + 1e-5:
                            raise ValueError(f"Missing video frame {row}: requested {timestamp}, found {moment}")
                        output[row] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                        assigned[row] = True
                        cursor += 1
            assert cursor == len(requested)
        assert assigned.all()
        output.flush()
        print(f"Decoded {key}: {len(timestamps)} frames", flush=True)
    metadata = dict(dataset_length=len(timestamps), rgb_keys=list(camera_map), dtype="uint8",
                    extra={"image_feature_map": camera_map}, decoder="PyAV sequential, rgb24, nearest timestamp")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
