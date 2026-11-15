"""Audited historical windows + shared uint8 mmap RGB + static frozen targets."""
import copy
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from diffusion_policy.dataset.render_net_teacher_cache import TeacherTargetCache
from diffusion_policy.distillation.dit_rgb import file_sha


class CiviDiTDataset(Dataset):
    def __init__(self, dataset, split, image_cache, cache=None):
        self.dataset, self.split = dataset, split
        self.image_cache = str(image_cache)
        self.cache = cache
        self.phases = np.asarray(dataset.phase_labels)
        anchors = [dataset._anchor_index_from_sampler_row(row) for row in dataset.indices]
        self.episodes = np.asarray(dataset.episode_index)[anchors].astype(np.int64)
        self.anchors = np.asarray(anchors, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item.update(index=index, sample_id=int(self.episodes[index])*1_000_000+int(self.anchors[index]),
                    episode=int(self.episodes[index]), phase=int(self.phases[index]))
        if self.cache is not None:
            item['static'] = self.cache.read(index)
        return item

    def __getstate__(self):
        state = self.__dict__.copy()
        state['dataset'] = copy.copy(self.dataset)
        state['dataset'].image_data = {}
        state['dataset']._video_decoder = None
        state['dataset'].lerobot_dataset = None
        # Reopen rather than pickle several GB of numpy memmap payload per worker.
        if self.cache is not None:
            state['cache'] = copy.copy(self.cache)
            state['cache']._arrays = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataset.image_data = {k: np.load(Path(self.image_cache)/f'{k}.npy', mmap_mode='r')
                                   for k in self.dataset.rgb_keys}


def historical_datasets(cfg, preprocessing):
    """Reconstruct saved indices, then verify pixels against the original decoder.

    This cache is a historical TorchCodec CUDA uint8 cache, not a new PyAV decode.
    Only raw RGB is reused; teacher targets are separately hash-bound.
    """
    reference = Path(cfg.reference_inputs)
    metadata = json.loads((reference/'reference_train_cache.json').read_text())
    config = OmegaConf.create(metadata['identity']['dataset'])
    config.dataset_path = cfg.dataset_path
    config.preload_images = False
    config.image_randomer_config = None
    config.video_decode_device = 'cuda:0'
    dataset = hydra.utils.instantiate(config)
    dataset.set_contact_label_config(preprocessing)
    images = Path(cfg.decoded_rgb)
    meta = json.loads((images/'metadata.json').read_text())
    if meta['decoder'] != 'TorchCodec CUDA' or meta['dataset_length'] != dataset.dataset_length:
        raise ValueError('Historical RGB decoder/pixel definition mismatch')
    arrays = {k: np.load(images/f'{k}.npy', mmap_mode='r') for k in dataset.rgb_keys}
    # Native historical backend probes; exact uint8 pixel equality is required.
    dataset._video_decoder = dataset._make_video_decoder()
    probes = [0, dataset.dataset_length//2, dataset.dataset_length-1]
    decoded = dataset._build_image_chunk(probes, {k: tuple(v.shape[1:]) for k,v in arrays.items()})
    for key in arrays:
        np.testing.assert_array_equal(np.rint(np.clip(decoded[key], 0, 1)*255).astype(np.uint8), arrays[key][probes])
    dataset.image_data = arrays
    dataset.lerobot_dataset.close(); dataset._video_decoder = None
    validation = dataset.get_validation_dataset()
    splits, identity = {}, {}
    expected = json.loads(Path(cfg.split_manifest).read_text())
    for name, raw in [('train', dataset), ('validation', validation)]:
        sha = hashlib.sha256(np.ascontiguousarray(raw.indices).tobytes()).hexdigest()
        ref = json.loads((reference/f'reference_{name}_cache.json').read_text())['identity'][name]
        if sha != ref['indices_sha256']:
            raise ValueError(f'{name}: saved window indices changed')
        wrapped = CiviDiTDataset(raw, name, images)
        episodes = np.unique(wrapped.episodes).tolist()
        if episodes != expected[name]['episodes']:
            raise ValueError(f'{name}: episode split changed')
        counts = np.bincount(wrapped.phases, minlength=3).tolist()
        if counts != expected[name]['phase_counts']:
            raise ValueError(f'{name}: phase definition changed')
        identity[name] = dict(samples=len(raw), episodes=episodes, phase_counts=counts, indices_sha256=sha)
        splits[name] = wrapped
    # Hash once; this binds the shared mmap content, not mutable path/mtime alone.
    identity['rgb'] = dict(dtype='uint8', decoder='TorchCodec CUDA', cameras=list(dataset.rgb_keys),
                           hashes={k:file_sha(images/f'{k}.npy') for k in dataset.rgb_keys})
    return splits, identity
