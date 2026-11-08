import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location('v21_converter', Path(__file__).resolve().parents[1] / 'data_process/tool/lerobotv3_to_v21.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    features = {k: {'dtype': 'float32', 'shape': [7]} for k in c.ALIASES.values()}
    features.update({k: {'dtype': 'int64', 'shape': [1]} for k in ['index', 'frame_index', 'episode_index', 'task_index']})
    features['timestamp'] = {'dtype': 'float32', 'shape': [1]}
    features['observation.images.wrist'] = {'dtype': 'video', 'shape': [16, 16, 3], 'info': {'video.pix_fmt': 'yuv420p', 'video.codec': 'h264'}}
    c.write_json(root / 'meta/info.json', {
        'codebase_version': 'v3.0', 'fps': 5, 'total_episodes': 2, 'total_frames': 10,
        'total_tasks': 1, 'features': features, 'splits': {'train': '0:2'},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
    })
    stats = {k: {'min': [0], 'max': [1], 'mean': [0.5], 'std': [0.5], 'count': [5]} for k in features}
    c.write_json(root / 'meta/stats.json', stats)
    pq.write_table(pa.table({'task_index': [0], 'task': ['test']}), root / 'meta/tasks.parquet')
    episodes = []
    for i in range(2):
        e = {'episode_index': i, 'length': 5, 'tasks': ['test'], 'dataset_from_index': i*5, 'dataset_to_index': i*5+5,
             'data/chunk_index': 0, 'data/file_index': 0}
        for k, v in {'chunk_index': 0, 'file_index': 0, 'from_timestamp': float(i), 'to_timestamp': float(i+1)}.items():
            e[f'videos/observation.images.wrist/{k}'] = v
        for key, values in stats.items():
            for name, value in values.items():
                e[f'stats/{key}/{name}'] = value
        episodes.append(e)
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), root / 'meta/episodes/chunk-000/file-000.parquet')
    rows = {'index': range(10), 'frame_index': list(range(5))*2, 'episode_index': [0]*5+[1]*5,
            'task_index': [0]*10, 'timestamp': list(np.arange(5)/5)*2}
    rows.update({k: [[float(i)]*7 for i in range(10)] for k in c.ALIASES.values()})
    (root / 'data/chunk-000').mkdir(parents=True)
    pq.write_table(pa.table(rows), root / 'data/chunk-000/file-000.parquet')
    video = root / 'videos/observation.images.wrist/chunk-000/file-000.mp4'
    video.parent.mkdir(parents=True)
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=16x16:r=5:d=1',
                    '-f', 'lavfi', '-i', 'color=c=blue:s=16x16:r=5:d=1',
                    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0', '-r', '5', '-c:v', 'libx264', '-crf', '0', str(video)], check=True)
    return root


def decode(path):
    return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                          check=True, capture_output=True).stdout


def test_conversion_aliases_and_video_boundaries(source, tmp_path):
    output = tmp_path / 'output'
    c.convert(source, output)
    info = c.read_json(output / 'meta/info.json')
    assert info['codebase_version'] == 'v2.1'
    assert info['total_videos'] == 2
    original = pq.read_table(source / 'data/chunk-000/file-000.parquet')
    original_video = decode(source / 'videos/observation.images.wrist/chunk-000/file-000.mp4')
    for i in range(2):
        table = pq.read_table(output / c.DATA_PATH.format(episode_chunk=0, episode_index=i))
        assert table.select(original.column_names).equals(original.slice(i*5, 5))
        for alias, key in c.ALIASES.items():
            assert table[alias].equals(table[key])
        video = output / c.VIDEO_PATH.format(episode_chunk=0, episode_index=i, video_key='observation.images.wrist')
        assert decode(video) == original_video[i*5*16*16*3:(i+1)*5*16*16*3]
    es = json.loads((output / 'meta/episodes_stats.jsonl').read_text().splitlines()[1])
    assert es['stats']['action'] == es['stats']['action.ee_pose']
    with pytest.raises(FileExistsError):
        c.convert(source, output)


def test_invalid_rows_never_publish_partial_dataset(source, tmp_path):
    path = source / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).slice(0, 9)
    pq.write_table(table, path)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='missing rows'):
        c.convert(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.output.partial-*'))


def test_huggingface_alias_metadata():
    table = pa.table({k: [[1., 2.]] for k in c.ALIASES.values()})
    hf = {'info': {'features': {k: {'_type': 'List', 'feature': {'_type': 'Value', 'dtype': 'float32'}, 'length': 2} for k in c.ALIASES.values()}}}
    table = table.replace_schema_metadata({b'huggingface': json.dumps(hf).encode()})
    result = c.add_aliases(table)
    features = json.loads(result.schema.metadata[b'huggingface'])['info']['features']
    assert features['action'] == features['action.ee_pose']

    assert features['action']['_type'] == 'Sequence'
