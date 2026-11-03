import json
import copy
from pathlib import Path
from types import SimpleNamespace, MethodType

import dill
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers.modeling_outputs import BaseModelOutput

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.render_net_attention import teacher_attention_targets
from diffusion_policy.model.diffusion.context_transformer_for_diffusion import ContextTransformerForDiffusion
from diffusion_policy.model.vision.contact_curriculum import ContactDetector, VectorThresholdGate
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from diffusion_policy.scripts.dit_action_attention_capture import DitActionAttentionCapture
from diffusion_policy.scripts.fusion_visual_attention_trace import FusionVisualAttentionTrace


class PatchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12, patch_size=8, num_register_tokens=2)
        self.patch = nn.Conv2d(3, 12, 8, stride=8)

    def forward(self, pixel_values):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        return BaseModelOutput(last_hidden_state=torch.cat([
            patches.new_zeros(patches.shape[0], 3, 12), patches,
        ], dim=1))


def attach_preprocessing(teacher):
    teacher.n_obs_steps = 2
    teacher.wrench_key = "wrench"
    teacher.phase_wrench_key = "wrench"
    teacher.future_phase_wrench_key = None
    teacher.contact_detector = ContactDetector(threshold=0.0)
    teacher.input_gate = VectorThresholdGate(threshold=0.0, enabled=False)
    for name in ("_validate_obs", "prepare_context_inputs"):
        setattr(teacher, name, MethodType(getattr(ForceAwareDiffusionTransformerPolicy, name), teacher))


@pytest.mark.parametrize("backbone", ["unet", "transformer"])
@pytest.mark.parametrize("region_enabled", [False, True])
@pytest.mark.parametrize("conv_block", ["standard", "bottleneck"])
def test_render_net_starts_as_identity_and_handles_different_image_sizes(backbone, region_enabled, conv_block):
    model = RenderNet(wrench_dim=6, base_channels=4, conditioning_dim=8, backbone=backbone,
                      conv_block=conv_block,
                      wrench_history_steps=7,
                      transformer={"embed_dim": 16, "depth": 2, "patch_size": 8, "num_heads": 4},
                      high_attention_render={"enabled": region_enabled})
    images = {"side": torch.rand(2, 2, 3, 17, 25), "wrist": torch.rand(2, 2, 3, 20, 28)}
    rendered = model(images, torch.randn(2, 2, 7, 6))
    for key in images:
        torch.testing.assert_close(rendered[key], images[key], rtol=0, atol=0)


def test_teacher_lstm_initialization_preserves_all_temporal_outputs_without_sharing_weights():
    teacher = ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=16, n_head=4, max_obs_steps=2,
        wrench_history_steps=7, force_temporal_encoder="lstm", force_encoder_layers=2,
        cross_attn_query="image", dropout=0.0, image_shapes={"wrist": (3, 16, 16)},
    ).eval().requires_grad_(False)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=16, wrench_history_steps=7, wrench_layers=2,
        image_keys=["wrist"], transformer={"embed_dim": 16, "depth": 1, "patch_size": 8},
    ).eval()
    renderer.initialize_wrench_encoder(teacher)
    wrench = torch.randn(2, 2, 7, 6)
    with torch.no_grad():
        force = (teacher.wrench_projection(wrench) + teacher.wrench_position
                 + teacher.obs_position[:, :2] + teacher.force_modality)
        expected, _ = teacher.force_encoder(force.reshape(4, 7, 16))
        actual = renderer.encode_wrench_history(wrench)
    assert actual.shape == (4, 7, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert renderer.wrench_encoder.weight_ih_l0.requires_grad
    assert renderer.wrench_encoder.weight_ih_l0.data_ptr() != teacher.force_encoder.weight_ih_l0.data_ptr()
    original = teacher.force_encoder.weight_ih_l0.clone()
    with torch.no_grad():
        renderer.wrench_encoder.weight_ih_l0.add_(1)
    torch.testing.assert_close(teacher.force_encoder.weight_ih_l0, original, rtol=0, atol=0)


@pytest.mark.parametrize("backbone", ["unet", "transformer"])
def test_patch_queries_read_full_wrench_sequence_and_backpropagate_to_early_timesteps(backbone):
    torch.manual_seed(12)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7, backbone=backbone,
        transformer={"embed_dim": 16, "depth": 1, "patch_size": 8, "num_heads": 4},
    )
    with torch.no_grad():
        renderer.output.weight.normal_(std=0.03)
    wrench = torch.randn(2, 2, 7, 6, requires_grad=True)
    images = {"wrist": torch.rand(2, 2, 3, 17, 25) * 0.5 + 0.25}
    captured = {}

    def capture(module, args, kwargs):
        captured.update(query=kwargs["query"], value=kwargs["value"])

    handle = renderer.backbone.blocks[0].cross_attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        rendered = renderer(images, wrench)
        assert captured["query"].shape == (4, 12, 16)  # ceil(17/8) * ceil(25/8)
        assert captured["value"].shape == (4, 7, 8)
        torch.testing.assert_close(captured["value"], renderer.encode_wrench_history(wrench))
        rendered["wrist"].square().mean().backward()
    finally:
        handle.remove()
    assert torch.all(wrench.grad.abs().sum(dim=(0, 1, 3)) > 0)
    assert renderer.backbone.decoders[-1].body[0].weight.grad.abs().sum() > 0


def test_camera_identity_changes_rendering_without_depending_on_dictionary_order():
    torch.manual_seed(13)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8,
        transformer={"embed_dim": 16, "depth": 1, "patch_size": 8},
    ).eval()
    with torch.no_grad():
        renderer.output.weight.normal_(std=0.03)
        renderer.camera_embedding.weight[0].zero_()
        renderer.camera_embedding.weight[1].copy_(torch.linspace(-2, 2, 16))
        image = torch.rand(1, 2, 3, 17, 25) * 0.5 + 0.25
        wrench = torch.randn(1, 2, 8, 6)
        first = renderer({"side": image, "wrist": image}, wrench)
        reversed_order = renderer({"wrist": image, "side": image}, wrench)
    assert (first["side"] - first["wrist"]).abs().mean() > 1e-7
    for key in first:
        torch.testing.assert_close(first[key], reversed_order[key], rtol=0, atol=0)


@pytest.mark.parametrize("backbone", ["unet", "transformer"])
@pytest.mark.parametrize("region_enabled", [False, True])
@pytest.mark.parametrize("conv_block", ["standard", "bottleneck"])
def test_frozen_visual_path_trains_renderer_and_conditioner_without_teacher_fusion(backbone, region_enabled, conv_block):
    torch.manual_seed(42)
    teacher = nn.Module()
    attach_preprocessing(teacher)
    teacher.n_obs_steps = 2
    teacher.image_keys = ("side", "wrist")
    teacher.wrench_key = "wrench"
    teacher.normalizer = LinearNormalizer()
    teacher.normalizer.fit({"wrench": torch.randn(32, 6)})
    teacher.obs_encoder = ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=16, n_head=4, max_obs_steps=2,
        wrench_history_steps=7, cross_attn_query="image", dropout=0.0,
        force_temporal_encoder="lstm",
        image_shapes={"side": (3, 16, 16), "wrist": (3, 16, 24)},
    )
    teacher.model = nn.Linear(16, 7)  # A DiT stand-in that must never be invoked.
    teacher.eval().requires_grad_(False)
    images = {"side": torch.rand(2, 2, 3, 16, 16), "wrist": torch.rand(2, 2, 3, 16, 24)}
    obs = {**images, "wrench": torch.randn(2, 2, 7, 6)}
    with torch.no_grad():
        target = teacher.obs_encoder(
            images=images, wrench_history=teacher.normalizer["wrench"].normalize(obs["wrench"]),
        ).detach()
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}

    def forbidden(*args):
        raise AssertionError("Student executed a teacher force, cross-attention, or DiT module")

    handles = [module.register_forward_pre_hook(forbidden) for module in (
        teacher.obs_encoder.wrench_projection, teacher.obs_encoder.force_encoder,
        teacher.obs_encoder.cross_attention, teacher.model,
    )]
    try:
        renderer = RenderNet(wrench_dim=6, base_channels=4, conditioning_dim=16, backbone=backbone,
                             conv_block=conv_block,
                             wrench_history_steps=7,
                             transformer={"embed_dim": 16, "depth": 2, "patch_size": 8, "num_heads": 4},
                             high_attention_render={"enabled": region_enabled}).train()
        renderer.initialize_wrench_encoder(teacher.obs_encoder)
        optimizer = torch.optim.AdamW(renderer.parameters(), lr=1e-3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            tokens = student_context_tokens(teacher, renderer, obs)
            assert tokens.shape == target.shape == (2, 20, 16)
            F.mse_loss(tokens, target).backward()
            assert renderer.output.weight.grad.abs().sum() > 0
            if step:
                assert renderer.wrench_encoder.weight_ih_l0.grad.abs().sum() > 0
                assert renderer.backbone.blocks[0].cross_attention.out_proj.weight.grad.abs().sum() > 0
                assert renderer.camera_embedding.weight.grad.abs().sum() > 0
            optimizer.step()
        assert not teacher.training and not teacher.obs_encoder.backbone.training
        assert teacher.obs_encoder.freeze_backbone
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()


def test_compiled_student_preserves_context_gradients_and_portable_state_dict():
    torch.manual_seed(13)
    teacher = nn.Module()
    attach_preprocessing(teacher)
    teacher.image_keys = ("side", "wrist")
    teacher.normalizer = LinearNormalizer()
    teacher.normalizer.fit({"wrench": torch.randn(32, 6)})
    teacher.obs_encoder = ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=16, n_head=4, max_obs_steps=2,
        wrench_history_steps=7, cross_attn_query="image", dropout=0.0,
        image_shapes={"side": (3, 16, 16), "wrist": (3, 16, 24)},
    )
    teacher.eval().requires_grad_(False)
    eager = RenderNet(
        6, base_channels=4, conditioning_dim=16, wrench_history_steps=7, conv_block="bottleneck",
        transformer={"embed_dim": 16, "depth": 1, "patch_size": 8},
        high_attention_render={"enabled": True},
    )
    with torch.no_grad():
        eager.output.weight.normal_(std=0.03)
    compiled = copy.deepcopy(eager)
    compiled.compile_backbone(backend="aot_eager", fullgraph=True, dynamic=False)
    encode = torch.compile(teacher.obs_encoder.encode_image_context,
                           backend="aot_eager", fullgraph=True, dynamic=False)
    obs = {"side": torch.rand(2, 2, 3, 16, 16), "wrist": torch.rand(2, 2, 3, 16, 24),
           "wrench": torch.randn(2, 2, 7, 6)}
    expected = student_context_tokens(teacher, eager, obs)
    actual = student_context_tokens(teacher, compiled, obs, image_context_encoder=encode)
    torch.testing.assert_close(actual, expected)
    target = torch.randn_like(expected)
    F.mse_loss(expected, target).backward()
    F.mse_loss(actual, target).backward()
    assert compiled.state_dict().keys() == eager.state_dict().keys()
    for name, parameter in compiled.named_parameters():
        reference = dict(eager.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-4, atol=1e-6)
    assert compiled.wrench_encoder.weight_ih_l0.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())

    def forbid_compilation(*args):
        raise AssertionError("Evaluation must use the original backbone")

    compiled._compiled_backbone_forward = forbid_compilation
    compiled.eval()
    restored = copy.deepcopy(eager).eval()
    restored.load_state_dict(compiled.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(student_context_tokens(teacher, compiled, obs),
                                   student_context_tokens(teacher, restored, obs))


@pytest.mark.parametrize("backbone", ["unet", "transformer"])
def test_region_gate_scales_residual_and_can_be_disabled(backbone):
    model = RenderNet(6, base_channels=4, conditioning_dim=8, backbone=backbone,
                      transformer={"embed_dim": 16, "depth": 1, "patch_size": 8, "num_heads": 4},
                      high_attention_render={"enabled": True}).eval()
    with torch.no_grad():
        model.output.bias.fill_(0.5)
        model.mask_head.weight.zero_()
        model.mask_head.bias.zero_()  # sigmoid=0.5 everywhere
    images = {"wrist": torch.full((1, 2, 3, 17, 25), 0.5)}
    wrench = torch.randn(1, 2, 8, 6)
    gated, masks = model(images, wrench, return_aux=True)
    full = model(images, wrench, use_attention_mask=False)
    torch.testing.assert_close(gated["wrist"] - images["wrist"],
                               0.5 * (full["wrist"] - images["wrist"]))
    assert masks["wrist"].shape == (1, 2, 1, 17, 25)
    F.mse_loss(masks["wrist"], torch.zeros_like(masks["wrist"])).backward()
    assert model.mask_head.weight.grad.abs().sum() > 0


def test_validation_metric_excludes_auxiliary_region_loss():
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.model = nn.Identity()
    workspace.device = torch.device("cpu")
    workspace.max_val_batches = None
    workspace.global_step = workspace.epoch = 0

    def loss(obs):
        workspace.loss_metrics = {"token_mse": 0.25, "mask_mse": 9.0, "loss": 9.25}
        return torch.tensor(9.25)

    workspace._loss = loss
    assert workspace._validate([{"obs": {"value": torch.zeros(2, 1)}}]) == 0.25


def make_attention_teacher(scheduler_class=DDIMScheduler):
    torch.manual_seed(7)
    shapes = {"side": (3, 16, 16), "wrist": (3, 16, 24)}
    encoder = ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=16, n_head=4, max_obs_steps=2,
        wrench_history_steps=7, cross_attn_query="image", dropout=0.0, image_shapes=shapes,
    )
    teacher = ForceAwareDiffusionTransformerPolicy(
        shape_meta={"obs": {**{key: {"type": "rgb", "shape": shape} for key, shape in shapes.items()},
                            "wrench": {"shape": (7, 6), "type": "low_dim"}}, "action": {"shape": (7,)}},
        noise_scheduler=scheduler_class(num_train_timesteps=10), horizon=4, n_action_steps=2,
        n_obs_steps=2, wrench_key="wrench", wrench_history_steps=7, n_emb=16,
        obs_encoder=encoder, num_inference_steps=2,
        action_expert=ContextTransformerForDiffusion(
            input_dim=7, output_dim=7, horizon=4, n_emb=16, n_head=4, n_layer=1,
            n_cond_layers=1, max_context_tokens=20, p_drop_attn=0.0,
        ),
    ).eval().requires_grad_(False)
    teacher.normalizer.fit({"wrench": torch.randn(32, 6)})
    obs = {key: torch.rand(2, 2, *shape) for key, shape in shapes.items()}
    obs["wrench"] = torch.randn(2, 2, 7, 6)
    return teacher, obs, shapes


@pytest.mark.parametrize("scheduler_class,step_kwargs", [
    (DDIMScheduler, {}), (DDIMScheduler, {"eta": 0.3}), (DDPMScheduler, {}),
])
@pytest.mark.parametrize("seed", [3, 42])
def test_batched_teacher_targets_match_serial_rollout_and_noise(scheduler_class, step_kwargs, seed):
    teacher, obs, _ = make_attention_teacher(scheduler_class)
    teacher.scheduler_step_kwargs = step_kwargs
    expected = {key: [] for key in teacher.image_keys}
    with torch.no_grad():
        expected_tokens = teacher._encode_context(obs, apply_curriculum_mask=False)
        for index in range(2):
            sample = {key: value[index:index + 1] for key, value in obs.items()}
            with FusionVisualAttentionTrace(teacher) as fusion, DitActionAttentionCapture(teacher) as dit:
                tokens = teacher._encode_context(sample, apply_curriculum_mask=False)
                teacher.conditional_sample((1, teacher.horizon, teacher.action_dim), context=tokens,
                                           generator=torch.Generator().manual_seed(seed))
                maps, diagnostics = fusion.project(dit.step_context_relevance())
            for key in expected:
                expected[key].append((100 * maps[key] / diagnostics["input_mass"][:, None, None]).mean(0))
    calls, progress = [], []
    rng_state = torch.get_rng_state()
    handle = teacher.model.register_forward_pre_hook(lambda module, args: calls.append(args[0].clone()))
    try:
        tokens, patches = teacher_attention_targets(
            teacher, obs, percentiles=[60, 99], seed=seed, return_patch_attention=True,
            progress_callback=lambda done, total, elapsed: progress.append((done, total)),
        )
    finally:
        handle.remove()
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    for key in expected:
        torch.testing.assert_close(patches[key], torch.stack(expected[key]), rtol=2e-5, atol=2e-6)
    assert len(calls) == teacher.num_inference_steps
    assert all(value.shape[0] == 2 for value in calls)
    torch.testing.assert_close(calls[0][0], calls[0][1], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)
    assert progress == [(0, 2), (2, 2)]


def test_residual_free_image_query_uses_query_slot_region_targets():
    """The old teacher layout must not collapse spatial targets to all ones."""
    teacher, obs, shapes = make_attention_teacher()
    teacher.obs_encoder.cross_attn_query_residual = False
    _, patches = teacher_attention_targets(
        teacher, obs, percentiles=[80, 99], seed=42, return_patch_attention=True,
    )
    for key, shape in shapes.items():
        values = patches[key]
        assert values.shape == (2, 2, shape[-2] // 8 * (shape[-1] // 8))
        assert torch.isfinite(values).all() and (values >= 0).all()
        assert values.max() > values.min()


@pytest.mark.parametrize("region_enabled", [False, True])
@pytest.mark.parametrize("mode", ["train", "eval", "no_grad", "cached_train"])
def test_region_supervision_runs_only_on_frozen_teacher_branch(region_enabled, mode):
    teacher, obs, shapes = make_attention_teacher()
    config = {"enabled": region_enabled, "percentiles": [82.5, 97.3], "teacher_seed": 42}
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.teacher_policy = teacher
    workspace.device = torch.device("cpu")
    workspace.model = RenderNet(6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
                               transformer={"embed_dim": 16, "depth": 1, "num_heads": 4},
                               high_attention_render=config)
    workspace.model.train(mode != "eval")
    workspace.attention_config = config
    workspace.attention_enabled = region_enabled
    workspace.mask_loss_weight = 2.0
    workspace.amp_enabled = False
    workspace.shape_info = {}  # Shape logging covered by real checkpoint forward.
    cached_targets = None
    if mode == "cached_train":
        if region_enabled:
            tokens, raw_attention = teacher_attention_targets(
                teacher, obs, percentiles=[82.5, 97.3], return_patch_attention=True,
            )
        else:
            with torch.no_grad():
                tokens = teacher._encode_context(obs, apply_curriculum_mask=False).detach()
            raw_attention = {}
        cached_targets = {"tokens": tokens, "patch_attention": raw_attention}
    calls = []
    handle = teacher.model.register_forward_hook(lambda *args: calls.append(1))
    encoder_handle = None
    if mode == "cached_train":
        def forbidden(*args):
            raise AssertionError("Cached training recomputed the teacher encoder targets")
        encoder_handle = teacher.obs_encoder.register_forward_pre_hook(forbidden)
    try:
        with torch.set_grad_enabled(mode != "no_grad"):
            loss = workspace._loss(obs, cached_targets=cached_targets)
        supervised = region_enabled and mode in {"train", "cached_train"}
        assert len(calls) == (2 if supervised and mode != "cached_train" else 0)
        assert workspace.loss_metrics["mask_supervised"] == supervised
        assert loss.item() == pytest.approx(workspace.loss_metrics["token_mse"]
                                          + 2 * workspace.loss_metrics["mask_mse"])
        if mode in {"train", "cached_train"}:
            loss.backward()
            assert workspace.model.output.weight.grad.abs().sum() > 0
        if supervised and mode != "cached_train":
            assert workspace.model.mask_head.weight.grad.abs().sum() > 0
            tokens, masks = teacher_attention_targets(teacher, obs, percentiles=[82.5, 97.3])
            assert tokens.shape == (2, 20, 16) and not tokens.requires_grad
            for key, shape in shapes.items():
                assert masks[key].shape == (2, 2, 1, *shape[-2:])
                assert not masks[key].requires_grad and torch.isfinite(masks[key]).all()
                assert masks[key].min() >= 0 and masks[key].max() <= 1
            with torch.no_grad():
                torch.testing.assert_close(tokens, teacher._encode_context(obs, apply_curriculum_mask=False),
                                           atol=0, rtol=0)
        elif not supervised:
            assert workspace.loss_metrics["mask_mse"] == 0
        assert not teacher.training
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    finally:
        handle.remove()
        if encoder_handle is not None:
            encoder_handle.remove()


@pytest.mark.parametrize("losses,epochs,patience,min_delta,max_steps,expected_epoch,reason,best_epoch", [
    ([1.0, 0.9, 0.8, 0.7], 3, 2, 0.0, None, 3, "max_epochs", 3),
    ([1.0, 0.9, 0.95, 0.85, 0.86, 0.87], 1000, 2, 0.0, None, 5, "early_stopping", 3),
    ([1.0, 0.95, 0.94], 1000, 2, 0.1, None, 2, "early_stopping", 2),
    ([1.0, 0.9, 0.8], 1000, 30, 0.0, 5, 2, "max_steps", 2),
    ([], 2, 2, 0.0, None, 2, "max_epochs", 0),
    ([], 1000, 30, 0.0, 5, 2, "max_steps", 0),
])
def test_epoch_training_limits_early_stopping_and_best_checkpoint(
    monkeypatch, tmp_path, losses, epochs, patience, min_delta, max_steps,
    expected_epoch, reason, best_epoch,
):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         "diffusion_policy/config/train_distill_render_net.yaml")
    del cfg["hydra"]  # Hydra removes its own section before invoking the workspace.
    cfg.distillation.objective = "token_mse" # The loop fixture mocks an obs-only loss.
    cfg.render_net.rendering_mode = "residual"
    cfg.render_net.high_attention_render.enabled = False
    cfg.training.device = "cpu"
    cfg.teacher_cache.enabled = False
    cfg.training.amp.enabled = False
    cfg.training.num_epochs = epochs
    cfg.training.max_steps = max_steps
    cfg.training.early_stopping.patience = patience
    cfg.training.early_stopping.min_delta = min_delta
    for loader in (cfg.dataloader, cfg.val_dataloader):
        loader.batch_size = 2
        loader.num_workers = 0
        loader.persistent_workers = False
        loader.pin_memory = False
    teacher = nn.Module()
    attach_preprocessing(teacher)
    teacher.obs_encoder = ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=8, n_head=4, max_obs_steps=2,
        wrench_history_steps=7, cross_attn_query="image", dropout=0.0,
        force_temporal_encoder="lstm", image_shapes={"side": (3, 16, 16), "wrist": (3, 16, 16)},
    )
    teacher.normalizer = LinearNormalizer()
    teacher_cfg = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *a, **kw: teacher)

    class Dataset:
        closed = False

        def __init__(self):
            self.lerobot_dataset = SimpleNamespace(close=self.close)

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return {"obs": {"value": torch.tensor([float(index)])}}

        def get_validation_dataset(self):
            return self if losses else []

        def close(self):
            self.closed = True

    dataset = Dataset()
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: dataset)
    monkeypatch.setattr(distill.TrainDistillRenderNetWorkspace, "_loss",
                        lambda self, obs: self.model.output.bias.square().mean() + 1.0)
    validation = iter(losses)

    def validate(self, loader):
        self.model.eval()
        self.validation_samples = 5
        return next(validation)

    monkeypatch.setattr(distill.TrainDistillRenderNetWorkspace, "_validate", validate)
    workspace = distill.TrainDistillRenderNetWorkspace(cfg, output_dir=str(tmp_path))
    workspace.run()
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert dataset.closed and summary["epoch"] == expected_epoch
    assert summary["stop_reason"] == reason and summary["best_epoch"] == best_epoch
    assert summary["global_step"] == (max_steps if max_steps is not None else expected_epoch * 3)
    assert summary["status"] == ("early_stopped" if reason == "early_stopping" else "completed")
    latest = torch.load(tmp_path / "checkpoints/latest.ckpt", pickle_module=dill)
    if losses:
        best = torch.load(tmp_path / "checkpoints/best.ckpt", pickle_module=dill)
        assert dill.loads(best["pickles"]["epoch"]) == best_epoch
        assert dill.loads(best["pickles"]["best_val_loss"]) == min(losses)
        assert "teacher_policy" not in best["state_dicts"]
    else:
        assert not (tmp_path / "checkpoints/best.ckpt").exists()
        assert summary["validation_enabled"] is False
        assert summary["early_stopping"]["enabled"] is False
        assert summary["initial_val_mse"] is summary["best_val_mse"] is summary["best_checkpoint"] is None
        assert not any("val_mse" in record for record in
                       map(json.loads, (tmp_path / "logs.json.txt").read_text().splitlines()))
    assert dill.loads(latest["pickles"]["stop_reason"]) == reason
    assert "teacher_policy" not in latest["state_dicts"]
    records = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    steps = [record for record in records if "train_mse" in record]
    assert len(steps) == summary["global_step"]
    assert all(record["lr"] == cfg.optimizer.lr and record["data_seconds"] >= 0
               and record["step_seconds"] >= 0 and record["samples_per_second"] > 0 for record in steps)
    epochs_logged = [record for record in records if "epoch_train_mse" in record]
    assert epochs_logged[-1]["epoch_samples"] == (4 if max_steps == 5 else 5)
    assert (tmp_path / "training_metrics.png").is_file()


@pytest.mark.parametrize("max_batches,expected,samples", [(None, 2.6, 5), (2, 2.0, 4)])
def test_validation_covers_full_split_and_weights_partial_batch(max_batches, expected, samples):
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.model = nn.Identity()
    workspace.device = torch.device("cpu")
    workspace.max_val_batches = max_batches
    workspace.global_step = 0
    workspace.epoch = 0
    workspace._loss = lambda obs: obs["value"].mean()
    loader = [{"obs": {"value": torch.full((size, 1), loss)}}
              for size, loss in ((2, 1.0), (2, 3.0), (1, 5.0))]
    result = workspace._validate(loader)
    assert result == pytest.approx(expected)
    assert workspace.validation_samples == samples
