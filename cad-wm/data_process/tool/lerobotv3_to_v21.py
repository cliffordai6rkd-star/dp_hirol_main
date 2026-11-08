#!/usr/bin/env python3
"""Convert complete local LeRobot v3 datasets to v2.1 without changing controls.

Dependencies: numpy, pandas, pyarrow; ffmpeg/ffprobe with libx264.
Video is decoded and losslessly encoded as H.264 (CRF 0), with accurate seeking.
Original numerical columns, task IDs, episode IDs and statistics are preserved.
Adds observation.state / action as exact aliases of the ee_pose fields.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

WORKSPACE = Path(__file__).resolve().parents[3]
DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
ALIASES = {'observation.state': 'observation.ee_pose', 'action': 'action.ee_pose'}
VIDEO_PATH = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def checked_path(root, template, **values):
    path = (root / template.format(**values)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes dataset: {path}')
    return path


def discover(root):
    if not root.is_dir():
        raise FileNotFoundError(f'Source directory does not exist: {root}')
    result = []
    for path in sorted(root.rglob('meta/info.json')):
        if read_json(path).get('codebase_version') == 'v3.0':
            result.append(path.parent.parent)
    if not result:
        raise ValueError(f'No v3.0 datasets in {root}')
    return result


def metadata(source):
    info = read_json(source / 'meta/info.json')
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('Expected codebase_version v3.0')
    episodes = []
    for path in sorted((source / 'meta/episodes').rglob('*.parquet')):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda e: e['episode_index'])
    if [e['episode_index'] for e in episodes] != list(range(info['total_episodes'])):
        raise ValueError('Episode metadata must contain every episode exactly once, starting at zero')
    if sum(e['length'] for e in episodes) != info['total_frames']:
        raise ValueError('Episode lengths do not match total_frames')
    tasks_df = pq.read_table(source / 'meta/tasks.parquet').to_pandas()
    tasks = []
    for label, row in tasks_df.iterrows():
        task = row['task'] if 'task' in tasks_df.columns else label
        if not isinstance(task, str):
            raise ValueError('Task text must be a string')
        tasks.append({'task_index': int(row['task_index']), 'task': task})
    tasks.sort(key=lambda t: t['task_index'])
    if len(tasks) != info['total_tasks'] or len({t['task_index'] for t in tasks}) != len(tasks):
        raise ValueError('Task metadata is inconsistent')
    return info, episodes, tasks


def episode_stats(episode, features):
    stats = {}
    for key, value in episode.items():
        if key.startswith('stats/') and value is not None:
            feature, stat = key[6:].rsplit('/', 1)
            stats.setdefault(feature, {})[stat] = value
    for key in features:
        if not {'min', 'max', 'mean', 'std', 'count'} <= stats.get(key, {}).keys():
            raise ValueError(f'Episode {episode["episode_index"]}: missing statistics for {key}')
    return stats


def split_video(source, destination, start, length, fps):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', f'{start:.12f}',
        '-i', str(source), '-map', '0:v:0', '-an', '-frames:v', str(length),
        '-vf', f'setpts=N/({fps}*TB)', '-r', str(fps),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination),
    ], check=True)
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames,avg_frame_rate,start_time',
        '-of', 'json', str(destination),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    num, den = map(int, stream['avg_frame_rate'].split('/'))
    if int(stream['nb_read_frames']) != length or not math.isclose(num / den, fps, abs_tol=1e-6):
        raise ValueError(f'Video frame count or FPS mismatch: {destination}')
    if abs(float(stream.get('start_time', 0))) > 1e-6:
        raise ValueError(f'Video timestamp does not start at zero: {destination}')


def legacy_hf_features(value):
    """datasets 3.x (openpi lockfile) calls primitive Lists Sequences."""
    if isinstance(value, list):
        return [legacy_hf_features(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: legacy_hf_features(v) for k, v in value.items()}
    if result.get('_type') == 'List':
        if 'dtype' not in result.get('feature', {}):
            raise ValueError('Nested List features need explicit v2.1 schema conversion')
        result['_type'] = 'Sequence'
    return result


def add_aliases(table):
    # Update the Hugging Face schema together with the Arrow columns.
    schema_meta = dict(table.schema.metadata or {})
    hf = json.loads(schema_meta[b'huggingface']) if b'huggingface' in schema_meta else None
    for target, original in ALIASES.items():
        if target in table.column_names:
            raise ValueError(f'Alias already exists: {target}')
        table = table.append_column(target, table[original])
        if hf is not None:
            hf['info']['features'][target] = copy.deepcopy(hf['info']['features'][original])
    if hf is not None:
        hf = legacy_hf_features(hf)
        hf.pop('fingerprint', None)
        schema_meta[b'huggingface'] = json.dumps(hf).encode()
    return table.replace_schema_metadata(schema_meta)


def convert(source, destination):
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite: {destination}')
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError('Output must be outside source dataset')
    info, episodes, tasks = metadata(source)
    features = info['features']
    for target, original in ALIASES.items():
        if target in features or original not in features:
            raise ValueError(f'Cannot add {target} from {original}')
    cameras = [k for k, v in features.items() if v['dtype'] == 'video']
    if any(v['dtype'] == 'image' for v in features.values()):
        raise ValueError('Image-backed datasets are not supported; expected video-backed inputs')
    if cameras and (not shutil.which('ffmpeg') or not shutil.which('ffprobe')):
        raise RuntimeError('ffmpeg and ffprobe are required')
    for key in cameras:
        vi = features[key].get('info', {})
        if vi.get('video.pix_fmt') != 'yuv420p' or vi.get('video.is_depth_map', False):
            raise ValueError(f'Only RGB yuv420p videos supported: {key}')
    # Validate all per-episode statistics before expensive video conversion.
    stats = [episode_stats(e, features) for e in episodes]
    global_stats = read_json(source / 'meta/stats.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.partial-', dir=destination.parent))
    try:
        cached_path, cached_table = None, None
        cursor = 0
        task_ids = {t['task_index'] for t in tasks}
        episode_rows, stats_rows = [], []
        for e, es in zip(episodes, stats):
            idx, length = e['episode_index'], e['length']
            if length <= 0 or e['dataset_from_index'] != cursor or e['dataset_to_index'] != cursor + length:
                raise ValueError(f'Invalid episode boundaries: {idx}')
            path = checked_path(source, info['data_path'], chunk_index=e['data/chunk_index'], file_index=e['data/file_index'])
            if path != cached_path:
                cached_table = pq.read_table(path)
                cached_path = path
            table = cached_table.filter(pc.equal(cached_table['episode_index'], idx))
            if len(table) != length:
                raise ValueError(f'Episode {idx} is missing rows or spans unsupported data files')
            for key, expected in [('frame_index', np.arange(length)), ('index', np.arange(cursor, cursor + length))]:
                if not np.array_equal(table[key].to_numpy(), expected):
                    raise ValueError(f'Episode {idx}: invalid {key}')
            timestamps = table['timestamp'].to_numpy()
            if not np.allclose(timestamps, np.arange(length) / info['fps'], atol=1e-4, rtol=0):
                raise ValueError(f'Episode {idx}: timestamps must start at zero and match FPS')
            if not set(table['task_index'].to_pylist()) <= task_ids:
                raise ValueError(f'Episode {idx}: unknown task ID')
            values = {'episode_chunk': idx // 1000, 'episode_index': idx}
            output = stage / DATA_PATH.format(**values)
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(add_aliases(table), output)
            for key in cameras:
                prefix = f'videos/{key}'
                video = checked_path(source, info['video_path'], video_key=key,
                                     chunk_index=e[f'{prefix}/chunk_index'], file_index=e[f'{prefix}/file_index'])
                start, end = e[f'{prefix}/from_timestamp'], e[f'{prefix}/to_timestamp']
                if start < 0 or not math.isclose(end - start, length / info['fps'], abs_tol=1e-4):
                    raise ValueError(f'Episode {idx}: invalid video boundaries for {key}')
                split_video(video, stage / VIDEO_PATH.format(video_key=key, **values), start, length, info['fps'])
            episode_rows.append({k: e[k] for k in ('episode_index', 'tasks', 'length')})
            for target, original in ALIASES.items():
                es[target] = copy.deepcopy(es[original])
            stats_rows.append({'episode_index': idx, 'stats': es})
            cursor += length
            print(f'{source.name}: {idx + 1}/{len(episodes)} episodes', flush=True)
        result = copy.deepcopy(info)
        for key in ('data_files_size_in_mb', 'video_files_size_in_mb'):
            result.pop(key, None)
        result.update(codebase_version='v2.1', chunks_size=1000,
                      total_chunks=math.ceil(len(episodes) / 1000),
                      total_videos=len(episodes) * len(cameras), data_path=DATA_PATH,
                      video_path=VIDEO_PATH if cameras else None)
        for key in cameras:
            result['features'][key]['info'].update({'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'has_audio': False})
        for target, original in ALIASES.items():
            result['features'][target] = copy.deepcopy(features[original])
            global_stats[target] = copy.deepcopy(global_stats[original])
        write_json(stage / 'meta/info.json', result)
        write_json(stage / 'meta/stats.json', global_stats)
        write_jsonl(stage / 'meta/tasks.jsonl', tasks)
        write_jsonl(stage / 'meta/episodes.jsonl', episode_rows)
        write_jsonl(stage / 'meta/episodes_stats.jsonl', stats_rows)
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=WORKSPACE / 'nero_ws/runs/demosrtate')
    parser.add_argument('--output-dir', type=Path, default=WORKSPACE / 'nero_ws/runs/demostrate_v21')
    parser.add_argument('--dry-run', action='store_true', help='Validate metadata and list conversions without writing')
    args = parser.parse_args()
    root, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == root or output.is_relative_to(root):
        parser.error('Output directory must be outside input directory')
    sources = discover(root)
    jobs = [(s, output / (s.relative_to(root) if s != root else Path(s.name))) for s in sources]
    for source, destination in jobs:
        info, episodes, _ = metadata(source)
        for e in episodes:
            episode_stats(e, info['features'])
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite: {destination}')
        print(f'{source} -> {destination} ({len(episodes)} episodes, {info["total_frames"]} frames)')
    if not args.dry_run:
        for source, destination in jobs:
            convert(source, destination)


if __name__ == '__main__':
    main()
