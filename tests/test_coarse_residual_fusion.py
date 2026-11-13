"""CR-VTF pools only the fusion path and restores its update onto dense patches."""

import io
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from test_civi_action_heads import _batch, _make_policy


class _GridBackbone(nn.Module):
    def __init__(self, patch_size=8):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16, patch_size=patch_size, num_register_tokens=2,
        )
        self.projection = nn.Conv2d(3, 16, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        prefix = patches.new_zeros(patches.shape[0], 3, patches.shape[-1])
        return SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _encoder(*, image_shapes=None, factor=2, patch_size=8, n_emb=16, **kwargs):
    torch.manual_seed(17)
    return ForceAwareObsEncoder(
        pretrained_model_name_or_path="unused",
        backbone=_GridBackbone(patch_size),
        image_shapes=image_shapes or {"side": (3, 32, 48), "wrist": (3, 48, 32)},
        n_emb=n_emb,
        n_head=4,
        max_obs_steps=2,
        wrench_dim=7,
        wrench_history_steps=8,
        force_temporal_encoder="lstm",
        force_encoder_layers=2,
        pre_cross_self_attn_layers=kwargs.pop("pre_cross_self_attn_layers", 1),
        cross_attn_layers=kwargs.pop("cross_attn_layers", 0),
        cross_attn_query="image",
        dropout=0.0,
        fusion_downsample_factor=factor,
        **kwargs,
    ).eval()


def _observations(encoder, batch_size=2):
    generator = torch.Generator().manual_seed(29)
    images = {
        key: torch.rand(batch_size, 2, *shape, generator=generator)
        for key, shape in encoder.image_shapes.items()
    }
    wrench = torch.randn(batch_size, 2, 8, 7, generator=generator)
    return images, wrench


def test_coarse_fusion_matches_spatial_pooling_and_residual_delta_reference():
    encoder = _encoder()
    images, wrench = _observations(encoder)
    batch_size, obs_steps = wrench.shape[:2]
    _, dense_cameras = encoder._encode_image_streams(
        images, encoder.image_keys, batch_size, obs_steps,
    )
    references = []
    for key, dense in zip(encoder.image_keys, dense_cameras):
        height, width = (value // 8 for value in encoder.image_shapes[key][1:])
        grid = dense.reshape(batch_size * obs_steps, height, width, encoder.n_emb)
        grid = grid.permute(0, 3, 1, 2)
        coarse_grid = F.avg_pool2d(grid, 2, 2)
        references.append((dense, grid, coarse_grid))

    seen = {}
    handles = [
        encoder.pre_cross_self_attention.register_forward_hook(
            lambda module, args, output: seen.update(input=args[0], output=output)
        ),
        encoder.force_encoder.register_forward_hook(
            lambda module, args, output: seen.update(force=output[0])
        ),
    ]
    try:
        context = encoder(images=images, wrench_history=wrench)
    finally:
        for handle in handles:
            handle.remove()

    coarse_inputs = [grid.flatten(2).transpose(1, 2) for _, _, grid in references]
    expected_input = torch.cat([*coarse_inputs, seen["force"]], dim=1)
    torch.testing.assert_close(seen["input"], expected_input)
    # The two rectangular grids stay separate and occupy six coarse tokens each.
    assert seen["input"].shape == (4, 6 + 6 + 8, 16)
    assert encoder.fusion_tokens_per_observation == 20

    expected_images = []
    offset = 0
    for coarse, (_, original_grid, coarse_grid) in zip(coarse_inputs, references):
        count = coarse.shape[1]
        update = seen["output"][:, offset:offset + count] - coarse
        update = update.transpose(1, 2).reshape_as(coarse_grid)
        restored = F.interpolate(
            update, size=original_grid.shape[-2:], mode="bilinear", align_corners=False,
        )
        expected_images.append((original_grid + restored).flatten(2).transpose(1, 2))
        offset += count
    expected = torch.cat([*expected_images, seen["output"][:, offset:]], dim=1)
    expected = encoder.context_norm(expected.reshape(batch_size, -1, encoder.n_emb))
    torch.testing.assert_close(context, expected)
    assert context.shape == (2, 2 * (24 + 24 + 8), 16)


def test_identity_fusion_preserves_dense_skip_without_adding_the_base_twice():
    dense = _encoder(factor=1)
    coarse = _encoder(factor=2)
    coarse.load_state_dict(dense.state_dict(), strict=True)
    dense.pre_cross_self_attention = nn.Identity()
    coarse.pre_cross_self_attention = nn.Identity()
    images, wrench = _observations(coarse)
    torch.testing.assert_close(
        coarse(images=images, wrench_history=wrench),
        dense(images=images, wrench_history=wrench),
        rtol=0, atol=0,
    )


def test_coarse_fusion_does_not_pool_across_observation_times():
    encoder = _encoder()
    images, wrench = _observations(encoder)
    reference = encoder(images=images, wrench_history=wrench).reshape(2, 2, -1, 16)
    changed_images = {key: value.clone() for key, value in images.items()}
    changed_images["side"][:, 0].zero_()
    changed = encoder(images=changed_images, wrench_history=wrench).reshape_as(reference)
    assert not torch.allclose(changed[:, 0], reference[:, 0])
    torch.testing.assert_close(changed[:, 1], reference[:, 1], rtol=0, atol=0)


def test_full_size_cameras_use_106_fusion_tokens_and_keep_800_context_tokens():
    encoder = _encoder(
        image_shapes={"side": (3, 224, 224), "wrist": (3, 224, 224)},
        patch_size=16,
        n_emb=256,
    )
    images, wrench = _observations(encoder, batch_size=1)
    seen = []
    handle = encoder.pre_cross_self_attention.register_forward_pre_hook(
        lambda module, args: seen.append(tuple(args[0].shape))
    )
    try:
        with torch.no_grad():
            context = encoder(images=images, wrench_history=wrench)
    finally:
        handle.remove()
    assert seen == [(2, 49 + 49 + 8, 256)]
    assert encoder.fusion_tokens_per_observation == 106
    assert encoder.context_tokens_per_observation == 400
    assert context.shape == (1, 800, 256)


@pytest.mark.parametrize("head", ["act", "dit"])
def test_both_action_heads_backpropagate_through_coarse_fusion_and_all_tau_steps(head):
    policy = _make_policy(head, fusion_downsample_factor=2).train()
    batch = _batch()
    batch["obs"]["tau_ext"].requires_grad_()
    seen = []
    handle = policy.obs_encoder.pre_cross_self_attention.register_forward_pre_hook(
        lambda module, args: seen.append(tuple(args[0].shape))
    )
    try:
        loss = policy.compute_loss(batch, generator=torch.Generator().manual_seed(31))
        loss.backward()
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert seen == [(4, 1 + 1 + 8, 16)]
    for module in (
        policy.obs_encoder.image_projection,
        policy.obs_encoder.wrench_projection,
        policy.obs_encoder.force_encoder,
        policy.obs_encoder.pre_cross_self_attention,
        policy.model.decoder,
    ):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum().item() for value in gradients) > 0
    per_history_step = batch["obs"]["tau_ext"].grad.abs().sum(dim=(0, 1, 3))
    assert (per_history_step > 0).all()
    assert all(parameter.grad is None for parameter in policy.obs_encoder.backbone.parameters())


def test_coarse_policy_checkpoint_round_trip_preserves_action_predictions():
    policy = _make_policy(fusion_downsample_factor=2).eval()
    obs = _batch()["obs"]
    expected = policy.predict_action(obs)
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    buffer.seek(0)
    restored = _make_policy(fusion_downsample_factor=2).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert restored.obs_encoder.fusion_downsample_factor == 2
    actual = restored.predict_action(obs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"factor": 0},
    {"factor": 1.5},
    {"factor": 2, "pre_cross_self_attn_layers": 0},
    {"factor": 2, "cross_attn_layers": 1},
    {"factor": 2, "image_shapes": {"side": (3, 16, 24)}},
])
def test_invalid_coarse_fusion_configurations_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        _encoder(**kwargs)


def _coarse_ddp_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{Path(root) / 'coarse_rendezvous'}",
        rank=rank, world_size=2, timeout=timedelta(seconds=45),
    )
    try:
        policy = _make_policy(
            fusion_downsample_factor=2, blur=True, blur_scope="all",
        ).train()
        wrapped = DistributedDataParallel(policy, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        batch = _batch()
        batch["obs"]["tau_ext"] += rank
        batch["action"] += rank * 0.1
        for step in (0, 45):
            optimizer.zero_grad(set_to_none=True)
            loss = wrapped(batch, optimizer_step=step)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            assert policy.last_curriculum_metrics["curriculum/blurred_image_fraction"] == 1.0
        state = torch.cat([
            parameter.detach().flatten() for parameter in policy.parameters() if parameter.numel()
        ])
        replicas = [torch.empty_like(state) for _ in range(2)]
        dist.all_gather(replicas, state)
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_coarse_act_and_global_curriculum_train_on_two_ddp_replicas(tmp_path):
    mp.spawn(_coarse_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
