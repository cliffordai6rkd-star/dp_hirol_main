"""Exercise the unified teacher and the full phase/intensity/RGB lifecycle."""

import copy
from pathlib import Path
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from test_configurable_force_fusion import IMAGE_SHAPES, PatchBackbone
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.workspace import train_distill_render_net_workspace as distill


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def teacher_and_obs(history_steps=25):
    encoder = ForceAwareObsEncoder(
        "unused", image_shapes=IMAGE_SHAPES, backbone=PatchBackbone(),
        n_emb=16, n_head=4, max_obs_steps=2, wrench_history_steps=history_steps,
        wrench_dim=7, force_temporal_encoder="lstm", force_encoder_layers=2,
        cross_attn_query="image", cross_attn_layers=1, dropout=0., image_token_masking=True,
    )
    teacher = ForceAwareDiffusionTransformerPolicy(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in IMAGE_SHAPES.items()},
            "tau_ext": {"shape": [history_steps, 7], "type": "low_dim"},
        }, "action": {"shape": [7]}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10),
        horizon=9, n_obs_steps=2, n_action_steps=8, obs_encoder=encoder,
        wrench_key="tau_ext", phase_wrench_key="tau_ext", wrench_history_steps=history_steps,
        precontact_seconds=.5, fusion_gate_mode="phase_binary", preserve_zero_wrench=True,
        input_gate_enabled=True, input_gate_threshold=12.5, input_gate_norm="l1",
        contact_force_dims=list(range(7)), contact_threshold=12.5,
        image_mask_scope="precontact_contact",
        mask_schedule={"schedule_type": "piecewise", "total_steps": 100,
                       "piecewise_fractions": [0., .25, .75, 1.],
                       "piecewise_probabilities": [1., 1., 0., 0.]},
        n_emb=16, n_head=4, n_layer=1, p_drop_attn=0., action_representation="joint_position",
    ).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({"tau_ext": torch.randn(100, 7) + 4, "action": torch.randn(100, 7)}, mode="gaussian")
    teacher.set_normalizer(normalizer)
    obs = {key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()}
    obs.update(tau_ext=torch.zeros(2, 2, history_steps, 7),
               contact_phase=torch.tensor([[1, 0], [2, 1]]),
               contact_alpha=torch.tensor([[0., 0.], [1., .5]]))
    obs["tau_ext"][1, 0, -1, 0] = 13.
    obs["phase_inputs"] = {
        "state": torch.randn(2, 2, history_steps, 21),
        "images": {key: torch.rand(2, 2, 4, *shape) for key, shape in IMAGE_SHAPES.items()},
    }
    return teacher, obs


def test_mask_is_per_observation_and_precontact_accepts_zero_torque():
    teacher, obs = teacher_and_obs()
    captured = []
    handle = teacher.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)
    try:
        teacher._encode_context(obs, apply_curriculum_mask=True)
        clear = teacher._encode_context(obs, apply_curriculum_mask=False)
    finally:
        handle.remove()
    opened = obs["contact_phase"] != 0
    torch.testing.assert_close(captured[0]["image_token_mask"], opened)
    assert not captured[1]["image_token_mask"].any()
    torch.testing.assert_close(captured[0]["fusion_gate"], opened)
    assert not captured[0]["wrench_history"][0].any()
    assert not captured[0]["wrench_history"][1, 1].any()
    for key in IMAGE_SHAPES:
        torch.testing.assert_close(captured[0]["images"][key], obs[key], rtol=0, atol=0)
    visual = teacher.obs_encoder.encode_image_context({key: obs[key] for key in IMAGE_SHAPES})
    clear_frames = clear.reshape(2, 2, -1, 16)
    visual_frames = visual.reshape_as(clear_frames)
    torch.testing.assert_close(clear_frames[~opened], visual_frames[~opened], rtol=0, atol=0)
    assert not torch.equal(clear_frames[0, 0], visual_frames[0, 0])


class MiniDataset:
    image_randomer = None

    def __init__(self, obs):
        self.obs = obs
        self.indices = np.arange(2).reshape(-1, 1)
        self.lerobot_dataset = SimpleNamespace(close=lambda: None)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        def at(value):
            return {key: at(item) for key, item in value.items()} if isinstance(value, dict) else value[index]
        return {"obs": at(self.obs)}

    def get_validation_dataset(self):
        return MiniDataset(self.obs)

    def phase_training_states(self):
        return self.obs["phase_inputs"]["state"].reshape(-1, 21)


def make_workspace(monkeypatch, tmp_path, *, resume=None, history_steps=25):
    teacher, obs = teacher_and_obs(history_steps)
    teacher_cfg = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "resolve_student_teacher", lambda cfg: (str(tmp_path / "teacher.ckpt"), teacher_cfg))
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: teacher)
    monkeypatch.setattr(distill.TrainDistillRenderNetWorkspace, "_instantiate_dataset", lambda *args: MiniDataset(obs))
    monkeypatch.setattr(distill.RenderNetTrainingLogger, "save_plot", lambda self: None)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "diffusion_policy/config/train_distill_render_net_student.yaml")
    del cfg["hydra"]
    cfg.output_dir = str(tmp_path)
    cfg.teacher_checkpoint = str(tmp_path / "teacher.ckpt")
    cfg.teacher_cache.enabled = False
    cfg.training.device = "cpu"
    cfg.training.amp.enabled = cfg.training.compile.enabled = False
    cfg.training.num_epochs = 6
    cfg.training.render_stages.patience = cfg.training.early_stopping.patience = 1
    cfg.training.lr_scheduler.enabled = False
    cfg.training.resume_from = str(resume) if resume else None
    cfg.render_net.phase_predictor.hidden_dim = 8
    cfg.render_net.phase_predictor.image_size = 32
    cfg.render_net.backbone.base_channels = 4
    cfg.render_net.backbone.embed_dim = 16
    cfg.render_net.backbone.image_self_attn_layers = 1
    cfg.render_net.backbone.cross_attn_layers = 1
    cfg.render_net.high_attention_render.threshold = 0.
    cfg.logging.progress_bar = False
    for loader in (cfg.dataloader, cfg.val_dataloader):
        loader.batch_size = 2
        loader.num_workers = 0
        loader.persistent_workers = loader.pin_memory = False
    OmegaConf.resolve(cfg)  # The CLI resolves null history placeholders before loading the teacher.
    return distill.TrainDistillRenderNetWorkspace(cfg, str(tmp_path)), obs


@pytest.mark.parametrize("history_steps", [25, 37, 50])
def test_student_inherits_teacher_history_for_phase_and_rendering(monkeypatch, tmp_path, history_steps):
    workspace, obs = make_workspace(monkeypatch, tmp_path, history_steps=history_steps)
    assert workspace.cfg.tau_ext_history_steps == history_steps
    assert workspace.model.wrench_history_steps == history_steps
    assert workspace.model.phase_predictor.history_steps == history_steps
    assert workspace._dataset_config().phase_predictor.history_steps == history_steps
    workspace.model.phase_predictor.fit_normalizer(obs["phase_inputs"]["state"].reshape(-1, 21))
    phase = workspace.model.phase_predictor(obs["phase_inputs"])
    assert phase["phase_logits"].shape == (2, 2, 3)


def test_full_loop_uses_separate_metrics_best_snapshots_and_freezes_heads(monkeypatch, tmp_path):
    workspace, obs = make_workspace(monkeypatch, tmp_path)
    initial_teacher = copy.deepcopy(workspace.teacher_policy.state_dict())
    stages = []
    actual_loss = workspace._loss

    def loss(*args, **kwargs):
        value = actual_loss(*args, **kwargs)
        if torch.is_grad_enabled():
            stages.append(workspace.training_stage)
            expected = {"phase": "phase_ce", "intensity": "intensity_loss", "rgb": "token_mse"}[workspace.training_stage]
            assert workspace._selection_mse(value) == workspace.loss_metrics[expected]
        return value

    workspace._loss = loss
    actual_validate = workspace._validate

    def plateau(loader):
        actual_validate(loader)  # Validate the real heads; force a plateau to exercise transitions quickly.
        return 1.

    workspace._validate = plateau
    workspace.run()
    assert stages == ["phase", "intensity", "rgb"]
    assert workspace.stop_reason == "early_stopping"
    assert workspace.training_stage == "rgb"
    assert workspace.stage_epoch == 1
    assert all(not p.requires_grad and p.grad is None for p in workspace.model.phase_predictor.parameters())
    assert all(not p.requires_grad and p.grad is None for p in workspace.model.gate_parameters())
    assert any(p.grad is not None for p in workspace.model.rgb_parameters())
    phase_best = torch.load(workspace.get_checkpoint_path("best_phase"), pickle_module=dill)["state_dicts"]["ema_model"]
    intensity_best = torch.load(workspace.get_checkpoint_path("best_intensity"), pickle_module=dill)["state_dicts"]["ema_model"]
    for name, value in workspace.model.state_dict().items():
        if name.startswith("phase_predictor."):
            torch.testing.assert_close(value, phase_best[name], rtol=0, atol=0)
    for name, parameter in workspace.model.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, intensity_best[name], rtol=0, atol=0)
            torch.testing.assert_close(workspace.ema_model.state_dict()[name], parameter, rtol=0, atol=0)
    for name, value in workspace.teacher_policy.state_dict().items():
        torch.testing.assert_close(value, initial_teacher[name], rtol=0, atol=0)

    resumed, _ = make_workspace(monkeypatch, tmp_path / "resumed", resume=workspace.get_checkpoint_path("latest"))
    assert resumed.training_stage == "rgb"
    assert not resumed._maybe_transition_stage()
    assert resumed.stage_epoch == 1
    assert all(not p.requires_grad for p in resumed.model.phase_predictor.parameters())
    assert all(not p.requires_grad for p in resumed.model.gate_parameters())
