"""Check GPU collectives, synchronized updates and local assets; never train a policy."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def check_assets(device):
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3"
    info = json.loads((dataset / "meta/info.json").read_text())
    for name in ("data", "videos"):
        if not any((dataset / name).rglob("*")):
            raise RuntimeError(f"Missing dataset files: {dataset / name}")
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(
        root / "model/model/dinov3-vitb16-pretrain-lvd1689m", local_files_only=True,
    ).to(device).eval()
    with torch.inference_mode():
        output = backbone(pixel_values=torch.zeros(1, 3, 192, 256, device=device),
                          interpolate_pos_encoding=True).last_hidden_state
    if not torch.isfinite(output).all():
        raise RuntimeError("DINO produced non-finite output")
    print(f"Local dataset episodes={info.get('total_episodes')}; DINO output={tuple(output.shape)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Use Gloo for local CPU verification")
    parser.add_argument("--skip-assets", action="store_true", help="Only check collectives/updates")
    args = parser.parse_args()
    torch.set_num_threads(1)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check the host NVIDIA driver and NVIDIA Container Toolkit")
    device = torch.device("cpu" if args.cpu else f"cuda:{local_rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backend = "gloo" if args.cpu else "nccl"
    try:
        if world > 1:
            dist.init_process_group(backend, timeout=timedelta(seconds=180))
            total = torch.tensor(float(rank + 1), device=device)
            dist.all_reduce(total)
            if total.item() != world * (world + 1) / 2:
                raise RuntimeError("Unexpected all_reduce result")
        torch.manual_seed(42)
        model = nn.Linear(4, 2).to(device)
        wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
                   if world > 1 else model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            wrapped(torch.ones(2, 4, device=device) * (rank + 1)).square().mean().backward()
            optimizer.step()
        if world > 1:
            state = torch.cat([p.detach().flatten() for p in model.parameters()])
            replicas = [torch.empty_like(state) for _ in range(world)]
            dist.all_gather(replicas, state)
            for replica in replicas:
                torch.testing.assert_close(replica, state, rtol=0, atol=0)
        name = "CPU" if args.cpu else torch.cuda.get_device_name(device)
        print(f"[check] rank={rank}/{world} device={device} {name}: synchronized updates OK", flush=True)
        if not args.skip_assets:
            check_assets(device)
        if world > 1:
            dist.barrier()
        if rank == 0:
            print(f"Ready: {world} process(es); {'DDP ' + backend if world > 1 else 'single device only'}. "
                  "No teacher/student training started.", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
