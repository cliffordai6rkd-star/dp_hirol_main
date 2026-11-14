"""Measure the production input pipeline without changing a running model.

Run from the same source directory as training. This CPU-only probe reuses
decoded images and completed teacher targets; it does not build teacher labels.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusion_policy.common.dataset_config import canonical_dataset_config
from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Student output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 2, 4])
    parser.add_argument('--batches', type=int, default=120)
    parser.add_argument('--warmup-batches', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    cfg = OmegaConf.load(args.run_dir / 'config.yaml')
    teacher = OmegaConf.load(args.run_dir / 'teacher_config.yaml')
    dataset_cfg = copy.deepcopy(teacher.task.dataset)
    with open_dict(dataset_cfg):
        dataset_cfg.update(OmegaConf.to_container(cfg.dataset, resolve=True))
        dataset_cfg.video_decode_device = 'cpu'
    # Use the same shared image cache as the two training ranks.
    os.environ['WORLD_SIZE'] = '2'
    dataset = hydra.utils.instantiate(canonical_dataset_config(dataset_cfg))
    manifest = json.loads((args.run_dir / 'teacher_cache.json').read_text())
    directory = Path(manifest['splits']['train'])
    metadata = json.loads((directory / 'metadata.json').read_text())
    cache = TeacherTargetCache(directory, metadata['identity'], metadata['length'],
                               token_dtype=metadata['dtype'])
    cached = CachedTeacherDataset(dataset, cache)
    batch_size = int(cfg.dataloader.batch_size) // 2
    sampler = DistributedSampler(cached, num_replicas=2, rank=0,
                                  shuffle=True, seed=int(cfg.training.seed))
    reference = next(iter(DataLoader(cached, batch_size=batch_size, sampler=sampler)))
    results = []
    for workers in args.workers:
        options = dict(num_workers=workers, pin_memory=False)
        if workers:
            options.update(multiprocessing_context='spawn', persistent_workers=True,
                           prefetch_factor=2)
        loader = DataLoader(cached, batch_size=batch_size, sampler=sampler, **options)
        iterator = None
        try:
            started = time.perf_counter()
            iterator = iter(loader)
            first = next(iterator)
            first_batch_seconds = time.perf_counter() - started
            torch.testing.assert_close(first, reference, rtol=0, atol=0)
            for _ in range(args.warmup_batches):
                next(iterator)
            latencies = []
            started = time.perf_counter()
            for _ in range(args.batches):
                batch_started = time.perf_counter()
                next(iterator)
                latencies.append(time.perf_counter() - batch_started)
            seconds = time.perf_counter() - started
            result = dict(workers=workers, batch_size=batch_size, batches=args.batches,
                          seconds=seconds, first_batch_seconds=first_batch_seconds,
                          median_batch_seconds=statistics.median(latencies),
                          samples_per_second=batch_size * args.batches / seconds,
                          first_batch_matches=True)
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            if iterator is not None and workers:
                iterator._shutdown_workers()
            del iterator, loader
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    cache.close()
    dataset.lerobot_dataset.close()


if __name__ == '__main__':
    main()
