"""Selected-observation execution preserves the real shared student's updates."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from test_render_net_condition_training import _targets, _workspace
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _shared_workspace(*, joint=True):
    workspace, obs = _workspace()
    workspace.cfg.distillation.token_region_phases = [1, 2]
    workspace.cfg.distillation.skip_unselected_observations = True
    workspace.cfg.render_net = {"phase_prediction": {"enabled": True}}
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        patch_size=8, decoder_layers=3, num_heads=4, conv_block="bottleneck",
        shared_condition_backbone=True, separate_gate_path=False,
        high_attention_render={"enabled": True, "threshold": 0., "detach_for_rgb": False},
        phase_prediction={"enabled": True, "hidden_dim": 8},
    )
    workspace.model.freeze_wrench_encoder()
    # Exercise the entire RGB graph immediately with a nonzero output layer.
    with torch.no_grad():
        workspace.model.output.weight.normal_(std=.03)
    if joint:
        workspace.epoch = 1
        workspace._maybe_transition_stage()
    else:
        workspace.model.set_training_stage("condition_warmup")
    workspace.model.set_stage_mode(True)
    return workspace, obs


def _execution_counts(workspace):
    counts = {"dino": [], "decoder": []}
    handles = [
        workspace.teacher_policy.obs_encoder.backbone.register_forward_pre_hook(
            lambda module, args, kwargs: counts["dino"].append(kwargs["pixel_values"].shape[0]),
            with_kwargs=True,
        ),
        workspace.model.backbone.decoders[0].register_forward_pre_hook(
            lambda module, args: counts["decoder"].append(args[0].shape[0]),
        ),
    ]
    return counts, handles


def _assert_gradients_match(first, second):
    other = dict(second.named_parameters())
    for name, parameter in first.named_parameters():
        comparison = other[name]
        assert (parameter.grad is None) == (comparison.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, comparison.grad, rtol=2e-4, atol=2e-6,
                                       msg=lambda message: f"{name}: {message}")


def test_sparse_shared_loss_gradients_and_selected_metrics_match_full_execution():
    sparse, obs = _shared_workspace()
    full = copy.deepcopy(sparse)
    full.cfg.distillation.skip_unselected_observations = False
    targets = _targets(sparse, obs)
    counts = []
    for workspace in (full, sparse):
        observed, handles = _execution_counts(workspace)
        try:
            workspace._loss(obs, cached_targets=targets).backward()
        finally:
            for handle in handles:
                handle.remove()
        counts.append(observed)

    for key in ("loss", "token_mse", "mask_mse", "context_patch_mse", "intensity_loss",
                "intensity_smoothness", "phase_ce", "alpha_mse", "alpha_mae", "phase_accuracy", "open_token_mse",
                "token_region_observation_count"):
        assert sparse.loss_metrics[key] == pytest.approx(full.loss_metrics[key], abs=2e-7), key
    _assert_gradients_match(full.model, sparse.model)
    assert counts[0] == {"dino": [4, 4], "decoder": [4, 4]}
    assert counts[1] == {"dino": [3, 3], "decoder": [3, 3]}
    assert sparse.loss_metrics["token_region_observation_count"] == 3


def test_all_free_skips_rgb_and_dino_but_preserves_adamw_momentum_update():
    sparse, contact_obs = _shared_workspace()
    free_obs = {key: value.clone() for key, value in contact_obs.items()}
    free_obs["contact_phase"].zero_()
    free_obs["contact_alpha"].zero_()
    targets = _targets(sparse, contact_obs)

    # Give RGB parameters nonzero optimizer momentum before the skipped step.
    sparse_optimizer = torch.optim.AdamW(sparse.model.parameters(), lr=.001, weight_decay=.01)
    sparse._loss(contact_obs, cached_targets=targets).backward()
    sparse_optimizer.step()
    assert sparse_optimizer.state[sparse.model.output.weight]["exp_avg"].abs().sum() > 0
    # Start the comparison from precisely the same parameters and moments.
    # Independent first steps can amplify roundoff in normalization-invariant
    # convolution biases, obscuring the behavior of the all-free step itself.
    full = copy.deepcopy(sparse)
    full.cfg.distillation.skip_unselected_observations = False
    full_optimizer = torch.optim.AdamW(full.model.parameters(), lr=.001, weight_decay=.01)
    full_optimizer.load_state_dict(copy.deepcopy(sparse_optimizer.state_dict()))
    optimizers = (full_optimizer, sparse_optimizer)

    for workspace, optimizer in zip((full, sparse), optimizers):
        optimizer.zero_grad(set_to_none=True)
        counts, handles = _execution_counts(workspace)
        try:
            workspace._loss(free_obs, cached_targets=targets).backward()
        finally:
            for handle in handles:
                handle.remove()
        if workspace is sparse:
            assert counts == {"dino": [], "decoder": []}
            assert workspace.loss_metrics["token_region_observation_count"] == 0
            assert not workspace.loss_metrics["token_mse_measured"]
            assert workspace.model.mask_head.weight.grad.abs().sum() > 0
            assert workspace.model.backbone.stem.body[0].weight.grad.abs().sum() > 0
            for parameter in workspace.model.rgb_parameters():
                assert parameter.grad is not None
                assert not parameter.grad.count_nonzero()
        else:
            assert counts == {"dino": [4, 4], "decoder": [4, 4]}
        optimizer.step()

    _assert_gradients_match(full.model, sparse.model)
    for name, parameter in full.model.named_parameters():
        torch.testing.assert_close(parameter, dict(sparse.model.named_parameters())[name],
                                   rtol=2e-5, atol=2e-7,
                                   msg=lambda message: f"{name}: {message}")
    torch.testing.assert_close(
        optimizers[0].state[full.model.output.weight]["exp_avg"],
        optimizers[1].state[sparse.model.output.weight]["exp_avg"], rtol=2e-4, atol=2e-7,
    )


def test_validation_encodes_all_observations_and_measures_closed_tokens():
    workspace, obs = _shared_workspace()
    targets = _targets(workspace, obs)
    workspace.model.eval()
    counts, handles = _execution_counts(workspace)
    try:
        with torch.no_grad():
            workspace._loss(obs, cached_targets=targets)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"dino": [4, 4], "decoder": [4, 4]}
    assert workspace.loss_metrics["closed_token_count"] > 0
    assert workspace.loss_metrics["closed_token_mse"] >= 0
    assert workspace.loss_metrics["token_region_observation_count"] == 3


def _sparse_ddp_worker(rank, init_file):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                                   world_size=2, timeout=timedelta(seconds=60))
    try:
        workspace, obs = _shared_workspace(joint=False)
        workspace.distributed = DistributedContext(rank=rank, local_rank=rank, world_size=2)
        workspace._setup_distributed_model()
        obs = {key: value.clone() for key, value in obs.items()}
        obs["side"].add_(rank * .1)
        if rank == 0:
            obs["contact_phase"].zero_()
            obs["contact_alpha"].zero_()
        targets = _targets(workspace, obs)
        optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
        for step in range(3):
            if step == 1:
                workspace.epoch = 1
                assert workspace._maybe_transition_stage()
            workspace.model.set_stage_mode(True)
            optimizer.zero_grad(set_to_none=True)
            workspace._loss(obs, cached_targets=targets).backward()
            optimizer.step()
            state = torch.cat([parameter.detach().flatten()
                               for parameter in workspace.model.parameters()])
            replicas = [torch.empty_like(state) for _ in range(2)]
            distributed.all_gather(replicas, state)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    finally:
        distributed.destroy_process_group()


def test_sparse_ddp_warmup_to_joint_with_one_all_free_rank_stays_synchronized(tmp_path):
    multiprocessing.spawn(_sparse_ddp_worker, args=(str(tmp_path / "sparse_ddp_init"),),
                          nprocs=2, join=True)
