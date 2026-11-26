import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)


class AugmentedImageWindows(Dataset):
    def __init__(self, dataset, camera_keys):
        self.dataset = dataset
        self.camera_keys = camera_keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        images = self.dataset.get_rgb_window(index)
        # Torchvision's PIL transforms end in ToTensor. Keep the same pixels
        # in uint8 until they are uploaded to DINO.
        return torch.stack([images[key] for key in self.camera_keys], dim=1).mul_(255).round_().byte()


@torch.no_grad()
def precompute_dino_patches(dataset, encoder, path, *, device, batch_size, num_workers,
                            seed, amp_dtype, preprocessing, dino_path):
    """Freeze one augmentation draw per observation window before optimization."""
    if not encoder.freeze_backbone or any(p.requires_grad for p in encoder.backbone.parameters()):
        raise ValueError("DINO patch caching requires a frozen backbone")
    path = Path(path)
    camera_keys = tuple(encoder.image_keys)
    rows = torch.from_numpy(dataset.get_window_indices()[:, :dataset.n_obs_steps].copy())
    metadata = {
        "dataset_path": str(Path(dataset.dataset_path).resolve()),
        "dino_path": str(Path(dino_path).resolve()),
        "camera_keys": list(camera_keys),
        "rgb": preprocessing,
        "seed": seed,
        "amp_dtype": str(amp_dtype),
    }
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if cached["metadata"] != metadata or not torch.equal(cached["obs_windows"], rows):
            raise ValueError(f"DINO cache settings or observation windows changed: {path}")
        log.info("Loaded DINO patch cache %s, shape=%s", path, tuple(cached["patches"].shape))
        return cached["patches"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dino-", dir=path.parent) as scratch:
        _build_cache(dataset, encoder, Path(scratch), path, rows, metadata,
                     device, batch_size, num_workers, seed, amp_dtype)
    patches = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["patches"]
    log.info("DINO patch cache ready at %s, shape=%s, dtype=%s", path, tuple(patches.shape), patches.dtype)
    return patches


def _build_cache(dataset, encoder, scratch, path, rows, metadata,
                 device, batch_size, num_workers, seed, amp_dtype):
    images = AugmentedImageWindows(dataset, tuple(encoder.image_keys))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(images, batch_size=batch_size, num_workers=num_workers, generator=generator)
    # Isolate augmentation RNG from model initialization and diffusion noise.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        offset = 0
        for batch in tqdm(loader, desc="Augment RGB windows"):
            if offset == 0:
                shape = (len(images), *batch.shape[1:])
                augmented = torch.from_file(
                    str(scratch / "rgb"), shared=True, size=math.prod(shape), dtype=torch.uint8,
                ).view(shape)
            augmented[offset:offset + len(batch)].copy_(batch)
            offset += len(batch)

    encoder.backbone.to(device).eval()
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for start in tqdm(range(0, len(images), batch_size), desc="Precompute frozen DINO patches"):
            batch = augmented[start:start + batch_size]
            pixels = batch.flatten(0, 2).to(device=device, dtype=torch.float32).div_(255)
            _, tokens = encoder._encode_dino(pixels)
            tokens = tokens.to(dtype=amp_dtype or tokens.dtype).reshape(*batch.shape[:3], *tokens.shape[1:])
            if start == 0:
                shape = (len(images), *tokens.shape[1:])
                patches = torch.from_file(
                    str(scratch / "patches"), shared=True, size=math.prod(shape), dtype=tokens.dtype,
                ).view(shape)
            patches[start:start + len(batch)].copy_(tokens)
    del augmented

    temporary = scratch / path.name
    # A buffered stream completes writes larger than the OS limit for one
    # syscall, including on a data disk with a non-ASCII path.
    with temporary.open("wb") as handle:
        torch.save({"metadata": metadata, "obs_windows": rows, "patches": patches}, handle)
    temporary.replace(path)
