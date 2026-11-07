"""Exercise real DDP collectives without requiring multiple CUDA devices."""

from datetime import timedelta
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from omegaconf import OmegaConf

from diffusion_policy.common.render_net_training_logger import RenderNetTrainingLogger
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_query = "image"
        self.wrench_projection = nn.Linear(1, 4)
        self.backbone = nn.Conv2d(3, 4, 1)
        self.image_keys = ("side",)
        self.patch_counts = {"side": 4}
        self.wrench_history_steps = 1
        self.max_obs_steps = 1

    def encode_image_context(self, images, *, input_grad=False):
        image = images["side"][:, 0]
        return self.backbone(image).flatten(2).transpose(1, 2)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = TinyEncoder()
        self.normalizer = LinearNormalizer()
        self.image_keys = ("side",)
        self.n_obs_steps = 1
        self.wrench_key = "wrench"
        self.target_calls = 0

    def prepare_context_inputs(self, obs):
        return {"side": obs["side"]}, obs["wrench"], None, None

    def _encode_context(self, obs, apply_curriculum_mask=False):
        self.target_calls += 1
        tokens = self.obs_encoder.encode_image_context({"side": obs["side"]})
        return tokens + obs["wrench"][:, :1]


class TinyPreprocessor(nn.Module):
    def get_config(self):
        return {"version": 1}


class TinyRenderer(nn.Module):
    backbone_name = "test"
    image_encoder_name = "test"
    wrench_encoder_name = "test"

    def __init__(self):
        super().__init__()
        self.offset = nn.Linear(1, 3)

    def forward(self, images, wrench, *, return_aux=False):
        output = {"side": images["side"] + self.offset(wrench).reshape(-1, 1, 3, 1, 1)}
        return (output, {}) if return_aux else output


class TinyDataset:
    image_randomer = None

    def __init__(self, root, indices=range(8)):
        self.dataset_path = str(root)
        self.indices = np.asarray(list(indices), dtype=np.int64).reshape(-1, 1)
        self.lerobot_dataset = SimpleNamespace(close=lambda: None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        value = float(self.indices[index, 0])
        return {"obs": {"side": torch.full((1, 3, 2, 2), 0.2 + value * 0.03),
                        "wrench": torch.tensor([[value * 0.02]])}}

    def get_validation_dataset(self):
        return TinyDataset(self.dataset_path, range(8, 13))


def _config(root):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         "diffusion_policy/config/train_distill_render_net.yaml")
    del cfg["hydra"]
    cfg.distillation.objective = "token_mse" # This fixture exercises the legacy obs-only teacher.
    cfg.teacher_checkpoint = str(root / "teacher.ckpt")
    cfg.teacher_cache.directory = str(root / "teacher_cache")
    cfg.teacher_cache.batch_size = 2
    cfg.teacher_cache.num_workers = 0
    # This test explicitly checks cache reuse across independent runs.
    cfg.teacher_cache.cleanup_on_exit = False
    cfg.training.device = "cpu"
    cfg.training.amp.enabled = False
    cfg.training.compile.enabled = False
    cfg.training.num_epochs = 1
    cfg.training.max_grad_norm = 1e6
    # This fixture replaces both networks; its teacher has no LSTM to copy.
    cfg.render_net.initialize_wrench_from_teacher = False
    cfg.render_net.high_attention_render.enabled = False
    cfg.logging.progress_bar = False
    cfg.logging.plot_every_steps = 0
    for loader in (cfg.dataloader, cfg.val_dataloader):
        loader.batch_size = 4
        loader.num_workers = 0
        loader.persistent_workers = False
        loader.pin_memory = False
        loader.shuffle = False
    return cfg


def _exercise_real_renderer(rank):
    # Exercise the actual LSTM, camera embedding, attention and mask parameters
    # through DDP, including compiled backbone calls and an external frozen net.
    for rendering_mode, compile_backbone in product(("residual", "gaussian_glow"), (False, True)):
        torch.manual_seed(17)
        renderer = RenderNet(
            wrench_dim=6, base_channels=4, conditioning_dim=8,
            wrench_history_steps=3, max_obs_steps=2,
            image_keys=("side", "wrist"), conv_block="bottleneck",
            rendering_mode=rendering_mode, glow={"num_blobs": 3},
            patch_size=2, embed_dim=8, image_self_attn_layers=1, cross_attn_layers=1, num_heads=2,
            decoder_layers=1,
            high_attention_render={"enabled": compile_backbone or rendering_mode == "gaussian_glow"},
        )
        encoder = nn.Conv2d(3, 4, 1).eval().requires_grad_(False)
        if compile_backbone:
            renderer.compile_backbone(backend="aot_eager", fullgraph=True, dynamic=False)
        model = torch.nn.parallel.DistributedDataParallel(renderer, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(renderer.parameters(), lr=0.001)
        torch.manual_seed(101 + rank)
        images = {key: torch.rand(2, 2, 3, 8, 8) for key in ("side", "wrist")}
        wrench = torch.randn(2, 2, 3, 6)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            rendered, masks = model(images, wrench, return_aux=True)
            loss = sum((encoder(value.flatten(0, 1)) - 0.2).square().mean()
                       for value in rendered.values())
            loss = loss + sum((mask - 0.3).square().mean() for mask in masks.values())
            loss.backward()
            assert all(parameter.grad is not None for parameter in renderer.parameters())
            assert all(parameter.grad is None for parameter in encoder.parameters())
            optimizer.step()
        state = torch.cat([parameter.detach().flatten() for parameter in renderer.parameters()])
        replicas = [torch.empty_like(state) for _ in range(2)]
        distributed.all_gather(replicas, state)
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)


def _worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    distributed.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}",
                                   rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        # Keep the actual workspace, loss, cache, DDP forward/backward, optimizer,
        # validation and checkpoint code. Replace only the expensive networks/IO.
        teacher_cfg = OmegaConf.create({"policy": {"test": True}, "task": {"dataset": {}}})
        distill.load_checkpoint_payload = lambda _: ({"state_dicts": {}}, teacher_cfg)
        distill.instantiate_policy = lambda *args, **kwargs: TinyTeacher()
        distill.build_render_net = lambda *args: TinyRenderer()
        distill.WrenchPreprocessor.from_policy = lambda policy: TinyPreprocessor()
        distill.hydra.utils.instantiate = lambda cfg: TinyDataset(root / "data")

        class MainLogger(RenderNetTrainingLogger):
            def __init__(self, *args):
                assert rank == 0, "Only rank 0 may open the shared log"
                super().__init__(*args)

            def save_plot(self):
                pass

        distill.RenderNetTrainingLogger = MainLogger
        context = DistributedContext(rank=rank, local_rank=rank, world_size=2)
        cfg = _config(root)
        ws = distill.TrainDistillRenderNetWorkspace(cfg, str(root / "ddp"), context)
        original_save = ws.save_checkpoint

        def main_save(*args, **kwargs):
            assert rank == 0, "Only rank 0 may save a checkpoint"
            return original_save(*args, **kwargs)

        ws.save_checkpoint = main_save
        ws.run()
        assert ws.global_step == 2 and ws.epoch == 1
        assert ws.validation_samples == 5  # Unequal shards: 3 vs 2 samples.
        assert ws.teacher_policy.target_calls == (7 if rank == 0 else 0)
        assert all(p.grad is None and not p.requires_grad for p in ws.teacher_policy.parameters())
        distributed.barrier()

        resumed_cfg = _config(root)
        resumed_cfg.training.resume_from = str(root / "ddp/checkpoints/latest.ckpt")
        resumed_cfg.training.num_epochs = 2
        resumed = distill.TrainDistillRenderNetWorkspace(resumed_cfg, str(root / "resumed"), context)
        resumed.run()
        assert resumed.global_step == 4 and resumed.epoch == 2
        assert resumed.teacher_policy.target_calls == 0
        state = torch.cat([p.detach().flatten() for p in resumed.model.parameters()])
        gathered = [torch.empty_like(state) for _ in range(2)]
        distributed.all_gather(gathered, state)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)

        # A capped validation evaluates one GLOBAL batch, without duplication.
        resumed.max_val_batches = 1
        capped = resumed._loader(TinyDataset(root / "data", range(8, 13)),
                                 resumed.cfg.val_dataloader, validation=True)
        resumed._validate(capped)
        assert resumed.validation_samples == 4
        resumed.max_val_batches = None
        empty_rank = resumed._loader(TinyDataset(root / "data", [8]),
                                     resumed.cfg.val_dataloader, validation=True)
        resumed._validate(empty_rank)
        assert resumed.validation_samples == 1  # Rank 1 has no local batches.

        # Preparation failures reach all ranks instead of leaving readers stuck.
        resumed.cfg.teacher_checkpoint = str(root / "missing.ckpt")
        with pytest.raises(RuntimeError, match="Preparation failed on rank 0"):
            resumed._prepare_cached_datasets(TinyDataset(root / "data"),
                                             TinyDataset(root / "data", range(8, 13)),
                                             OmegaConf.create({}))

        _exercise_real_renderer(rank)

        if rank == 0:
            # Same global batch and order must match a single-process reference,
            # including optimizer state restored at the epoch boundary.
            reference_cfg = _config(root)
            reference_cfg.training.num_epochs = 2
            reference = distill.TrainDistillRenderNetWorkspace(reference_cfg, str(root / "single"))
            reference.run()
            reference_state = torch.cat([p.detach().flatten() for p in reference.model.parameters()])
            torch.testing.assert_close(state, reference_state, rtol=1e-5, atol=1e-6)
            assert reference.latest_val_loss == pytest.approx(resumed.latest_val_loss, rel=1e-5)
            payload = torch.load(root / "resumed/checkpoints/latest.ckpt", pickle_module=dill)
            assert "teacher_policy" not in payload["state_dicts"]
            assert "_training_model" not in payload["state_dicts"]
            assert all(not key.startswith("module.") for key in payload["state_dicts"]["model"])
            summary = json.loads((root / "resumed/training_summary.json").read_text())
            assert summary["world_size"] == 2 and summary["validation_samples"] == 5
            records = [json.loads(line) for line in (root / "resumed/logs.json.txt").read_text().splitlines()]
            assert [r["global_step"] for r in records if "train_mse" in r] == [3, 4]
            assert len(list((root / "teacher_cache").glob("*/train/metadata.json"))) == 1
        distributed.barrier()

        # Default cleanup must also work across all DDP readers, including a
        # faster rank returning from run() before the last lease is released.
        cleanup_cfg = _config(root)
        cleanup_cfg.teacher_cache.cleanup_on_exit = True
        cleanup = distill.TrainDistillRenderNetWorkspace(cleanup_cfg, str(root / "cleanup"), context)
        cleanup.run()
        distributed.barrier()
        assert not list((root / "teacher_cache").glob("*/train/metadata.json"))
        assert (root / "cleanup/checkpoints/latest.ckpt").is_file()

        # A rank-0 failure after writing the first batch must reach all ranks
        # and remove the partial cache without a cleanup barrier/deadlock.
        failing = distill.TrainDistillRenderNetWorkspace(cleanup_cfg, str(root / "failed"), context)
        encode = failing.teacher_policy._encode_context

        def fail_after_first_batch(obs, **kwargs):
            if failing.teacher_policy.target_calls:
                raise RuntimeError("cache build failed")
            return encode(obs, **kwargs)

        failing.teacher_policy._encode_context = fail_after_first_batch
        with pytest.raises(RuntimeError, match="Preparation failed on rank 0.*cache build failed"):
            failing.run()
        distributed.barrier()
        assert not list((root / "teacher_cache").glob("*/train"))
    finally:
        distributed.destroy_process_group()


def test_ddp_frozen_encoder_gradients_ssd_targets_validation_and_resume(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "teacher.ckpt").write_bytes(b"fixed teacher")
    multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_four_ranks_preserve_global_batch_and_shard_validation_without_padding():
    dataset = TinyDataset("unused", range(9))
    shards = []
    for rank in range(4):
        ws = object.__new__(distill.TrainDistillRenderNetWorkspace)
        ws.distributed = DistributedContext(rank=rank, local_rank=rank, world_size=4)
        ws.cfg = OmegaConf.create({"training": {"seed": 42}})
        ws.max_val_batches = None
        loader_cfg = OmegaConf.create({"batch_size": 32, "shuffle": True, "num_workers": 0})
        train = ws._loader(dataset, loader_cfg)
        assert train.batch_size == 8 and train.sampler.num_replicas == 4
        worker_cfg = OmegaConf.create({"batch_size": 32, "num_workers": 1})
        workers = ws._loader(dataset, worker_cfg)
        assert workers.multiprocessing_context.get_start_method() == "spawn"
        val = ws._loader(dataset, loader_cfg, validation=True)
        shards.extend(val.sampler)
    assert sorted(shards) == list(range(9))
