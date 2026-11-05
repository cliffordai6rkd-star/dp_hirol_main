"""Behavior distillation keeps the shared visual/action networks frozen."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Conv2d(3, 4, 1)

    def encode_image_context(self, images, *, input_grad=False):
        assert input_grad or not torch.is_grad_enabled()
        return self.visual(images["side"][:, 0]).flatten(2).transpose(1, 2)


class RecordingDenoiser(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.readout = nn.Linear(4, action_dim)
        self.calls = []

    def forward(self, action, timestep, context):
        output = (0.1 * action + self.readout(context.mean(1))[:, None]
                  + timestep[:, None, None] * 0.01)
        self.calls.append({"action": action.detach().clone(), "timestep": timestep.clone(),
                           "context_dtype": context.dtype, "input_grad": context.requires_grad,
                           "grad_enabled": torch.is_grad_enabled(), "output": output.detach().clone()})
        return output


class Teacher(nn.Module):
    def __init__(self, prediction_type, action_dim=3):
        super().__init__()
        self.obs_encoder = ImageEncoder()
        self.model = RecordingDenoiser(action_dim)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=10, prediction_type=prediction_type)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit({"action": torch.randn(40, action_dim), "wrench": torch.randn(40, 6)})

    def prepare_context_inputs(self, obs):
        return {"side": obs["side"]}, obs["wrench"], None, None

    def _encode_context(self, obs, apply_curriculum_mask=False):
        tokens = self.obs_encoder.encode_image_context({"side": obs["side"]})
        return tokens + 0.3 + obs["wrench"].mean((1, 2, 3))[:, None, None]


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Linear(6, 3)

    def forward(self, images, wrench, *, return_aux=False):
        image = images["side"] + self.offset(wrench.mean(2))[:, :, :, None, None]
        result = {"side": image}
        return (result, {}) if return_aux else result


def make_workspace(prediction_type="epsilon", action_dim=3, **weights):
    torch.manual_seed(73)
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({"distillation": {"objective": "denoiser_mse",
                                       "validation_seed": 19, **weights},
                                     "logging": {"progress_bar": False}})
    workspace.device = torch.device("cpu")
    workspace.teacher_policy = Teacher(prediction_type, action_dim).eval().requires_grad_(False)
    workspace.model = Renderer()
    workspace.attention_enabled = False
    workspace.mask_loss_weight = 1.0
    workspace.amp_enabled = False
    workspace.shape_info = {}
    workspace.global_step = workspace.epoch = 0
    workspace.max_val_batches = None
    workspace.loss_metrics = {}
    batch = {"obs": {"side": torch.rand(3, 1, 3, 4, 5), "wrench": torch.randn(3, 1, 2, 6)},
             "action": torch.randn(3, 4, action_dim)}
    return workspace, batch


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
@pytest.mark.parametrize("action_dim", [3, 8])
def test_shared_noisy_actions_and_frozen_input_gradients(prediction_type, action_dim):
    workspace, batch = make_workspace(prediction_type, action_dim=action_dim)
    teacher = workspace.teacher_policy
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    renderer_before = {key: value.clone() for key, value in workspace.model.state_dict().items()}
    optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=0.01)
    loss = workspace._batch_loss(batch)
    calls = teacher.model.calls
    assert len(calls) == 2
    torch.testing.assert_close(calls[0]["action"], calls[1]["action"], rtol=0, atol=0)
    torch.testing.assert_close(calls[0]["timestep"], calls[1]["timestep"], rtol=0, atol=0)
    assert calls[0]["grad_enabled"] is False and calls[0]["input_grad"] is False
    assert calls[1]["grad_enabled"] is True and calls[1]["input_grad"] is True
    expected = F.mse_loss(calls[1]["output"], calls[0]["output"])
    torch.testing.assert_close(loss.detach(), expected)
    assert workspace._selection_mse(loss) == pytest.approx(expected.item())
    loss.backward()
    assert workspace.model.offset.weight.grad.abs().sum() > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    optimizer.step()
    assert any(not torch.equal(value, renderer_before[key])
               for key, value in workspace.model.state_dict().items())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_optional_demo_target_uses_scheduler_prediction_type(prediction_type):
    workspace, batch = make_workspace(prediction_type, demo_loss_weight=0.7, token_loss_weight=0.2)
    scheduler = workspace.teacher_policy.noise_scheduler
    original_add_noise = scheduler.add_noise
    captured = {}

    def capture(action, noise, timestep):
        captured.update(action=action, noise=noise, timestep=timestep)
        return original_add_noise(action, noise, timestep)

    scheduler.add_noise = capture
    loss = workspace._batch_loss(batch)
    target = {"epsilon": captured["noise"], "sample": captured["action"]}.get(prediction_type)
    if prediction_type == "v_prediction":
        target = scheduler.get_velocity(captured["action"], captured["noise"], captured["timestep"])
    output = workspace.teacher_policy.model.calls[-1]["output"]
    assert workspace.loss_metrics["demo_mse"] == pytest.approx(F.mse_loss(output, target).item())
    expected = (workspace.loss_metrics["denoiser_mse"] + 0.7 * workspace.loss_metrics["demo_mse"]
                + 0.2 * workspace.loss_metrics["token_mse"])
    assert loss.item() == pytest.approx(expected)
    assert workspace._selection_mse(loss) == workspace.loss_metrics["denoiser_mse"]


def test_cached_half_precision_context_is_reused_without_fusion_forward():
    workspace, batch = make_workspace()
    with torch.no_grad():
        tokens = workspace.teacher_policy._encode_context(batch["obs"])
    batch["teacher_targets"] = {"tokens": tokens.half()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached teacher context was recomputed")

    workspace.teacher_policy._encode_context = forbidden
    workspace._batch_loss(batch).backward()
    assert all(call["context_dtype"] == torch.float32 for call in workspace.teacher_policy.model.calls)
    assert workspace.model.offset.weight.grad.abs().sum() > 0


def test_validation_probes_repeat_without_changing_training_rng():
    workspace, batch = make_workspace(demo_loss_weight=0.7, token_loss_weight=0.2)
    loader = torch.utils.data.DataLoader([batch, batch], batch_size=None, num_workers=0)
    before = torch.random.get_rng_state().clone()
    result = workspace._validate(loader)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    first_calls = list(workspace.teacher_policy.model.calls)
    torch.randn(17)
    before_second = torch.random.get_rng_state().clone()
    second = workspace._validate(loader)
    assert result == second
    torch.testing.assert_close(torch.random.get_rng_state(), before_second, rtol=0, atol=0)
    for first, repeated in zip(first_calls, workspace.teacher_policy.model.calls[len(first_calls):]):
        torch.testing.assert_close(first["action"], repeated["action"], rtol=0, atol=0)
        torch.testing.assert_close(first["timestep"], repeated["timestep"], rtol=0, atol=0)
    assert result == workspace.validation_metrics["denoiser_mse"]
    assert workspace.validation_metrics["demo_mse"] > 0


def test_training_resamples_noise_while_eval_uses_fixed_probes():
    workspace, batch = make_workspace()
    workspace._batch_loss(batch)
    first = workspace.teacher_policy.model.calls[-1]["action"]
    workspace._batch_loss(batch)
    second = workspace.teacher_policy.model.calls[-1]["action"]
    assert not torch.equal(first, second)
    workspace.model.eval()
    with torch.no_grad():
        first_loss = workspace._batch_loss(batch)
        second_loss = workspace._batch_loss(batch)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)


def test_missing_actions_fail_and_legacy_objective_stays_obs_only():
    workspace, batch = make_workspace()
    with pytest.raises(ValueError, match="requires batch"):
        workspace._batch_loss({"obs": batch["obs"]})
    del workspace.cfg["distillation"]
    assert workspace.objective == "token_mse"
    loss = workspace._batch_loss({"obs": batch["obs"]})
    assert loss.item() == workspace.loss_metrics["token_mse"]


@pytest.mark.parametrize("rendering_mode", ["residual", "gaussian_glow"])
def test_real_frozen_visual_encoder_and_action_transformer_pass_input_gradients(rendering_mode):
    from test_render_net_distillation import make_attention_teacher
    from diffusion_policy.model.vision.render_net import RenderNet

    workspace, _ = make_workspace()
    teacher, obs, _ = make_attention_teacher()
    teacher.normalizer.fit({"action": torch.randn(32, 7)})
    workspace.teacher_policy = teacher.eval().requires_grad_(False)
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        patch_size=4, embed_dim=8, image_self_attn_layers=1, cross_attn_layers=1, num_heads=2,
        rendering_mode=rendering_mode,
        high_attention_render={"enabled": rendering_mode == "gaussian_glow"},
        glow={"num_blobs": 4},
    )
    loss = workspace._batch_loss({"obs": obs, "action": torch.randn(2, 4, 7)})
    loss.backward()
    assert workspace.model.output.weight.grad.abs().sum() > 0
    if rendering_mode == "gaussian_glow":
        # Spatial supervision is disabled in this workspace: these gradients
        # must arrive through glow rendering -> frozen visual encoder -> DiT.
        assert not workspace.attention_enabled
        assert workspace.model.mask_head.weight.grad.abs().sum() > 0
        assert workspace.model.backbone.blob_queries.grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())


def test_training_loop_logs_and_selects_denoiser_metric(monkeypatch, tmp_path):
    import json
    from test_render_net_ddp import _config, TinyDataset, TinyTeacher, TinyRenderer, TinyPreprocessor
    from diffusion_policy.workspace import train_distill_render_net_workspace as distill

    class ActionDataset(TinyDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["action"] = torch.full((4, 3), float(self.indices[index, 0]) * 0.03)
            return sample

        def get_validation_dataset(self):
            return ActionDataset(self.dataset_path, range(8, 13))

    teacher = TinyTeacher()
    teacher.model = RecordingDenoiser()
    teacher.noise_scheduler = DDIMScheduler(num_train_timesteps=10)
    teacher.normalizer.fit({"action": torch.randn(32, 3)})
    teacher_config = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_config))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: teacher)
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: ActionDataset(tmp_path / "data"))
    config = _config(tmp_path)
    config.teacher_cache.enabled = False
    config.distillation = {"objective": "denoiser_mse", "validation_seed": 42,
                           "token_loss_weight": 0.2, "demo_loss_weight": 0.3}
    workspace = TrainDistillRenderNetWorkspace(config, str(tmp_path / "run"))
    workspace.run()
    records = [json.loads(line) for line in
               (tmp_path / "run/logs.json.txt").read_text().splitlines()]
    steps = [row for row in records if "train_mse" in row]
    assert len(steps) == 2
    assert all(row["objective"] == "denoiser_mse" and row["train_mse"] == row["train_denoiser_mse"]
               for row in steps)
    assert all(row["train_loss"] > row["train_mse"] for row in steps)
    summary = json.loads((tmp_path / "run/training_summary.json").read_text())
    assert summary["objective"] == "denoiser_mse"
    assert summary["validation_metric"] == "denoiser MSE (excludes auxiliary losses)"
    assert (tmp_path / "run/checkpoints/best.ckpt").is_file()
    assert (tmp_path / "run/checkpoints/latest.ckpt").is_file()
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.parametrize("config", [{"objective": "invalid"}, {"token_loss_weight": -1},
                                     {"demo_loss_weight": float("nan")}])
def test_invalid_distillation_configuration_is_rejected(config):
    workspace, _ = make_workspace()
    workspace.cfg.distillation.update(config)
    with pytest.raises(ValueError, match="distillation"):
        workspace._validate_distillation_config()
