"""Dense intensity/RGB heads learn from interpolated context targets."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens
from test_render_net_denoiser_distillation import make_workspace
from test_render_net_distillation import make_attention_teacher


def pixel_options(history=3):
    config_dir = Path(__file__).resolve().parents[1] / "diffusion_policy" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train_distill_render_net_patch_context")
    options = OmegaConf.to_container(cfg.render_net, resolve=True)
    assert options["rendering_mode"] == "residual"
    assert options["high_attention_render"]["normalization"] == "image_minmax"
    options.update(base_channels=4, conditioning_dim=8, wrench_history_steps=history,
                   initialize_wrench_from_teacher=False)
    options["transformer"].update(embed_dim=16, depth=1, patch_size=8, num_heads=4)
    return options


def test_active_config_predicts_both_fields_at_pixel_resolution_and_thresholds_pixels():
    torch.manual_seed(41)
    renderer = build_render_net(6, pixel_options()).eval()
    with torch.no_grad():
        renderer.output.weight.normal_(std=.03)
        renderer.mask_head.weight.normal_(std=.1)
    images = {"side": torch.rand(1, 2, 3, 17, 25) * .6 + .2,
              "wrist": torch.rand(1, 2, 3, 20, 28) * .6 + .2}
    wrench = torch.randn(1, 2, 3, 6)
    masks_seen, residuals_seen = [], []
    handles = [
        renderer.mask_head.register_forward_hook(
            lambda module, args, output: masks_seen.append((args[0].shape, output.detach()))),
        renderer.output.register_forward_hook(
            lambda module, args, output: residuals_seen.append(output.detach())),
    ]
    try:
        with torch.no_grad():
            rendered, masks = renderer(images, wrench, return_aux=True)
    finally:
        for handle in handles:
            handle.remove()
    for index, (camera, image) in enumerate(images.items()):
        height, width = image.shape[-2:]
        feature_shape, logits = masks_seen[index]
        assert feature_shape[-2:] == (height, width)
        assert logits.shape == (2, 1, height, width)
        assert residuals_seen[index].shape == (2, 3, height, width)
        # The student head already emits every pixel; no patch mask resize follows it.
        torch.testing.assert_close(masks[camera], logits.sigmoid().reshape(1, 2, 1, height, width))
        assert masks[camera][..., :8, :8].std() > 0
        field = masks[camera].expand_as(image)
        closed, opened = field <= renderer.region_threshold, field > renderer.region_threshold
        assert closed.any() and opened.any()
        torch.testing.assert_close(rendered[camera][closed], image[closed], rtol=0, atol=0)
        expected = (image + renderer.residual_scale * field
                    * residuals_seen[index].tanh().reshape_as(image)).clamp(0, 1)
        torch.testing.assert_close(rendered[camera][opened], expected[opened])


def test_pixel_training_opens_initial_threshold_then_backpropagates_through_frozen_teacher():
    workspace, _ = make_workspace()
    teacher, obs, _ = make_attention_teacher()
    teacher.normalizer.fit({"action": torch.randn(32, 7)})
    workspace.teacher_policy = teacher
    options = pixel_options(history=7)
    workspace.model = build_render_net(6, options).train()
    workspace.attention_config = options["high_attention_render"]
    workspace.attention_enabled = True
    workspace.region_source = "context"
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    with torch.no_grad():
        teacher_tokens = teacher._encode_context(obs, apply_curriculum_mask=False)
        images, wrench, _, _ = teacher.prepare_context_inputs(obs)
        initial, fields = workspace.model(images, wrench, return_aux=True)
        for camera, image in images.items():
            torch.testing.assert_close(initial[camera], image, rtol=0, atol=0)
            assert (fields[camera] <= workspace.model.region_threshold).all()
    patches = {camera: torch.linspace(4., 12., count).reshape(1, 1, count).expand(2, 2, -1)
               for camera, count in teacher.obs_encoder.patch_counts.items()}
    targets = workspace._region_masks(patches, obs)
    for camera, target in targets.items():
        assert target.shape == fields[camera].shape
        assert target.min() == 0 and target.max() == 1
        assert ((target > 0) & (target < 1)).any()
    batch = {"obs": obs, "action": torch.randn(2, 4, 7),
             "teacher_targets": {"tokens": teacher_tokens, "patch_attention": patches}}
    optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.003)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = workspace._batch_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert workspace.model.mask_head.weight.grad.abs().sum() > 0
        if step == 0:
            # RGB starts closed, but raw-field supervision can open the sigmoid head.
            assert workspace.model.output.weight.grad.count_nonzero() == 0
            assert workspace.model.mask_head.bias.grad.item() < 0
        else:
            assert workspace.model.output.weight.grad.abs().sum() > 0
        optimizer.step()

    # Isolate the behavior loss: both dense heads and the shared decoder must
    # receive gradients through the frozen visual encoder and action transformer.
    optimizer.zero_grad(set_to_none=True)
    student_tokens = student_context_tokens(teacher, workspace.model, obs)
    denoiser_loss, _ = workspace._denoiser_losses(batch["action"], student_tokens, teacher_tokens)
    denoiser_loss.backward()
    for parameter in (workspace.model.output.weight, workspace.model.mask_head.weight,
                      workspace.model.backbone.decoders[-1].body[0].weight,
                      workspace.model.wrench_encoder.weight_ih_l0):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
