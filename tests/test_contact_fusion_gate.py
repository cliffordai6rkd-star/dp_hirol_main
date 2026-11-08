"""Physical future contact and complete teacher-fusion bypass, without DINO data."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_configurable_force_fusion import IMAGE_SHAPES, make_encoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector
from diffusion_policy.model.vision.render_net_attention import (
    teacher_attention_targets, teacher_context_effect_targets,
)
from diffusion_policy.model.vision.wrench_preprocessing import (
    WRENCH_PREPROCESSING_VERSION, WrenchPreprocessor,
)
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.scripts.checkpoint_attention_capture import AttentionCapture
from diffusion_policy.scripts.fusion_visual_attention_trace import FusionVisualAttentionTrace
from diffusion_policy.scripts.image_query_attention_capture import ImageQueryAttentionCapture
from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs, capture_pair
from diffusion_policy.scripts.visualize_wrench_action_sensitivity_episode import capture_fusion_pair, replacement_contexts
from diffusion_policy.workspace import train_distill_render_net_workspace as distill


def make_teacher(**overrides):
    torch.manual_seed(7)
    sa = overrides.pop("pre_cross_self_attn_layers", 1)
    ca = overrides.pop("cross_attn_layers", 2)
    encoder = make_encoder(
        cross_attn_query="image", cross_attn_layers=ca,
        pre_cross_self_attn_layers=sa, force_temporal_encoder="lstm",
    )
    config = dict(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in IMAGE_SHAPES.items()},
            "tau_ext": {"shape": (8, 7), "type": "low_dim"},
            "future_tau_ext": {"shape": (8, 7), "type": "low_dim"},
            "future_wrench": {"shape": (8, 6), "type": "low_dim"},
        }, "action": {"shape": (7,)}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10, prediction_type="epsilon"),
        horizon=9, n_obs_steps=2, n_action_steps=8, num_inference_steps=2,
        obs_encoder=encoder, n_emb=16, n_head=4, n_layer=1, p_drop_attn=0.,
        wrench_key="future_wrench", phase_wrench_key="tau_ext",
        future_phase_wrench_key="future_tau_ext", contact_force_dims=(5, 6),
        contact_norm="l1", contact_threshold=2., free_threshold=1.,
        contact_enter_frames=5, contact_exit_frames=5,
        future_contact_mode="token_threshold", contact_fusion_gate=True,
    )
    config.update(overrides)
    policy = ForceAwareDiffusionTransformerPolicy(**config).eval()
    normalizer = LinearNormalizer()
    # Deliberately nonzero mean: suppressed windows must remain EXACT zeros.
    normalizer.fit({"future_wrench": torch.arange(600).reshape(100, 6).float() + 5,
                    "action": torch.randn(100, 7)})
    policy.set_normalizer(normalizer)
    return policy


def observations(batch=2):
    return {
        **{key: torch.rand(batch, 2, *shape) for key, shape in IMAGE_SHAPES.items()},
        "tau_ext": torch.zeros(batch, 2, 8, 7),
        "future_tau_ext": torch.zeros(batch, 2, 8, 7),
        "future_wrench": torch.arange(batch * 2 * 8 * 6).reshape(batch, 2, 8, 6).float() + 1,
    }


def test_future_threshold_uses_independent_physical_tokens_and_preserves_prefix():
    teacher, obs = make_teacher(), observations(5)
    # XYZ is deliberately enormous everywhere: the configured phase dims are 5,6.
    obs["future_tau_ext"][..., :3] = 1000
    obs["future_tau_ext"][1, :, -1, 5:] = 1.1  # L1=2.2, L2<2; last-point contact.
    obs["future_tau_ext"][2, :, -3:, 5] = 3.
    obs["tau_ext"][3, :, :, 5] = 3.  # current contact, future free.
    obs["future_tau_ext"][4, :, :5, 5] = 3.
    obs["tau_ext"][0, 0, -3:, 5] = 3.  # three history points still do not enter.
    _, wrench, current, gate = teacher.prepare_context_inputs(obs)
    assert gate.tolist() == [[False, False], [True, True], [True, True], [False, False], [True, True]]
    assert current[0].tolist() == [False, False]
    assert current[3].all()
    assert wrench[[0, 3]].count_nonzero() == 0
    assert teacher.normalizer["future_wrench"].normalize(torch.zeros_like(wrench)).count_nonzero() > 0
    expected = obs["future_wrench"][1, :, -1:].expand(-1, 8, -1)
    torch.testing.assert_close(wrench[1], teacher.normalizer["future_wrench"].normalize(expected), rtol=0, atol=0)
    expected = obs["future_wrench"][2].clone()
    expected[:, :5] = expected[:, 5:6]
    torch.testing.assert_close(wrench[2], teacher.normalizer["future_wrench"].normalize(expected), rtol=0, atol=0)
    teacher.future_contact_mode = "legacy_rollout"
    _, legacy_wrench, legacy_current, legacy_gate = teacher.prepare_context_inputs(obs)
    assert not legacy_gate[1:3].any() and legacy_wrench[1:3].count_nonzero() == 0
    assert legacy_gate[3:].all()  # legacy inherits the current latched state.
    torch.testing.assert_close(legacy_current, current)
    teacher.future_contact_mode = "token_threshold"
    zero_wrench, zero_current, zero_gate = WrenchPreprocessor.from_policy(teacher)(make_zero_wrench_obs(teacher, obs))
    assert not zero_gate.any() and not zero_current.any() and zero_wrench.count_nonzero() == 0


def test_token_interface_has_no_state_and_uses_contact_entry_threshold():
    detector = HysteresisContactDetector(1., 2., force_dims=(1, 2), norm="l2", enter_frames=5, exit_frames=5)
    signal = torch.tensor([[[[100., 1.2, 1.6], [0., 1.3, 1.6], [0., 0., 0.]]]])
    assert detector.token_contact(signal).tolist() == [[[False, True, False]]]
    assert not detector(signal).any()
    assert not detector.rollout_future(signal, initial_state=torch.zeros(1, 1, dtype=torch.bool)).any()


@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
@pytest.mark.parametrize("residual", [True, False])
def test_complete_gate_matches_visual_and_legacy_paths_with_mixed_observations(camera_mode, residual):
    encoder = make_encoder(cross_attn_query="image", cross_attn_layers=2,
                           pre_cross_self_attn_layers=2, camera_attention_mode=camera_mode,
                           cross_attn_query_residual=residual).eval()
    obs = observations()
    images = {key: obs[key] for key in IMAGE_SHAPES}
    gate = torch.tensor([[False, True], [True, False]])
    with torch.no_grad():
        visual = encoder.encode_image_context(images)
        fused = encoder(images=images, wrench_history=obs["future_wrench"])
        opened = encoder(images=images, wrench_history=obs["future_wrench"], contact_gate=torch.ones_like(gate))
        closed = encoder(images=images, wrench_history=obs["future_wrench"], contact_gate=torch.zeros_like(gate))
        mixed = encoder(images=images, wrench_history=obs["future_wrench"], contact_gate=gate)
    torch.testing.assert_close(opened, fused, rtol=0, atol=0)
    torch.testing.assert_close(closed, visual, rtol=0, atol=0)
    expected = torch.where(gate[..., None, None], fused.reshape(2, 2, 10, 16), visual.reshape(2, 2, 10, 16))
    torch.testing.assert_close(mixed.reshape_as(expected), expected, rtol=0, atol=0)
    assert (fused - visual).abs().max() > .01


def test_gate_validation_and_unsupported_architectures():
    encoder, obs = make_teacher().obs_encoder, observations()
    inputs = dict(images={key: obs[key] for key in IMAGE_SHAPES}, wrench_history=obs["future_wrench"])
    for bad, error in [(torch.zeros(2, 2), TypeError), ([[False] * 2] * 2, TypeError),
                       (torch.zeros(2, 1, dtype=torch.bool), ValueError),
                       (torch.zeros(2, 2, dtype=torch.bool, device="meta"), ValueError)]:
        with pytest.raises(error, match="contact_gate"):
            encoder(**inputs, contact_gate=bad)
    for query, ca in [("wrench", 1), ("image", 0)]:
        unsupported = make_encoder(cross_attn_query=query, cross_attn_layers=ca)
        with pytest.raises(ValueError, match="image-query"):
            unsupported(**inputs, contact_gate=torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="future_contact_mode"):
        make_teacher(future_contact_mode="typo")
    with pytest.raises(ValueError, match="future contact data"):
        make_teacher(future_phase_wrench_key=None)
    with pytest.raises(KeyError, match="future_tau_ext"):
        make_teacher()._encode_context({k: v for k, v in obs.items() if k != "future_tau_ext"}, False)


@pytest.mark.parametrize("blur", [False, True])
def test_full_context_curriculum_cannot_perturb_closed_observations(blur):
    teacher = make_teacher(image_blur_schedule={
        "total_steps": 1, "start_sigma": 1., "end_sigma": 0.,
        "start_fraction": 0.25, "end_fraction": 0.75,
    } if blur else None)
    obs = observations()
    obs["future_tau_ext"][0, -1, -1, 5] = 3.
    captured = {}
    handle = teacher.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(kwargs), with_kwargs=True)
    try:
        with torch.no_grad():
            context = teacher._encode_context(obs, apply_curriculum_mask=True)
            visual = teacher.obs_encoder.encode_image_context({k: obs[k] for k in IMAGE_SHAPES})
    finally:
        handle.remove()
    gate = captured["contact_gate"]
    assert gate.tolist() == [[False, True], [False, False]]
    for key in IMAGE_SHAPES:
        torch.testing.assert_close(captured["images"][key][~gate], obs[key][~gate], rtol=0, atol=0)
        if blur:
            assert not torch.equal(captured["images"][key][gate], obs[key][gate])
    torch.testing.assert_close(context.reshape(2, 2, 10, 16)[~gate], visual.reshape(2, 2, 10, 16)[~gate], rtol=0, atol=0)


def test_closed_observations_have_no_fusion_gradient_but_open_ones_train():
    encoder = make_teacher().obs_encoder.train()
    obs, gate = observations(), torch.tensor([[False, True], [True, False]])
    wrench = obs["future_wrench"].requires_grad_()
    images = {k: obs[k] for k in IMAGE_SHAPES}
    # Use a nonconstant linear probe, since summing LayerNorm outputs cancels.
    output = encoder(images=images, wrench_history=wrench, contact_gate=gate).reshape(2, 2, 10, 16)
    (output[~gate] * torch.randn_like(output[~gate])).sum().backward()
    assert wrench.grad.count_nonzero() == 0
    fusion_parameters = [*encoder.force_encoder.parameters(), *encoder.cross_attention.parameters(),
                         *encoder.pre_cross_self_attention.parameters(), *encoder.cross_ffn.parameters()]
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in fusion_parameters)
    encoder.zero_grad(set_to_none=True)
    wrench.grad = None
    output = encoder(images=images, wrench_history=wrench, contact_gate=gate).reshape(2, 2, 10, 16)
    (output[gate] * torch.randn_like(output[gate])).sum().backward()
    assert wrench.grad[~gate].count_nonzero() == 0 and wrench.grad[gate].abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in fusion_parameters)


def test_teacher_preprocessor_roundtrip_and_strict_old_weights():
    teacher, obs = make_teacher(), observations()
    obs["future_tau_ext"][0, :, -1, 5] = 3.
    standalone = WrenchPreprocessor.from_policy(teacher)
    config = standalone.get_config()
    assert config["future_contact_mode"] == "token_threshold" and config["contact_fusion_gate"]
    restored = WrenchPreprocessor.from_config(OmegaConf.create(config))
    restored.load_state_dict(standalone.state_dict(), strict=True)
    for a, b, c in zip(teacher.prepare_context_inputs(obs)[1:], standalone(obs), restored(obs)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    old_config = {k: v for k, v in config.items() if k not in {"future_contact_mode", "contact_fusion_gate"}}
    old_config["version"] = 1
    legacy = WrenchPreprocessor.from_config(old_config)
    legacy.load_state_dict(standalone.state_dict(), strict=True)
    assert legacy.future_contact_mode == "legacy_rollout" and not legacy.contact_fusion_gate
    assert not legacy(obs)[2].any()
    old_teacher = make_teacher(future_contact_mode="legacy_rollout", contact_fusion_gate=False)
    teacher.load_state_dict(old_teacher.state_dict(), strict=True)
    assert set(teacher.state_dict()) == set(old_teacher.state_dict())
    assert sum(p.numel() for p in teacher.parameters()) == sum(p.numel() for p in old_teacher.parameters())
    with pytest.raises(ValueError, match="future_contact_mode"):
        WrenchPreprocessor.from_config({**config, "future_contact_mode": "invalid"})


@pytest.mark.parametrize("batched", [False, True])
def test_attention_rollout_closed_branch_is_identity_and_conserves_mass(batched):
    teacher, obs = make_teacher().requires_grad_(False), observations(2 if batched else 1)
    obs["future_tau_ext"][0, 1, -1, 5] = 3.
    with FusionVisualAttentionTrace(teacher, batched=batched) as trace:
        teacher._encode_context(obs, False)
        relevance = torch.rand(3, len(obs["tau_ext"]), 20) if batched else torch.rand(3, 20)
        maps, diagnostics = trace.project(relevance)
    expected = relevance.reshape(*relevance.shape[:-1], 2, 10)
    gate = trace.contact_gate
    for key, sl in [("side", slice(0, 4)), ("wrist", slice(4, 10))]:
        closed = ~gate[None, ..., None].expand_as(maps[key])
        torch.testing.assert_close(maps[key][closed], expected[..., sl][closed], rtol=0, atol=0)
    assert diagnostics["per_observation_wrench_mass"][..., ~gate].count_nonzero() == 0 if not batched else (
        diagnostics["per_observation_wrench_mass"][:, ~gate].count_nonzero() == 0)
    torch.testing.assert_close(diagnostics["input_mass"], diagnostics["visual_mass"] + diagnostics["wrench_mass"])


def test_context_effect_and_attention_targets_use_gated_teacher_and_raw_zero():
    teacher, obs = make_teacher().requires_grad_(False), observations()
    obs["future_tau_ext"][0, -1, -1, 5] = 3.
    gates = []
    handle = teacher.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: gates.append(kwargs["contact_gate"].clone()), with_kwargs=True)
    try:
        tokens, maps = teacher_context_effect_targets(teacher, obs)
        assert gates[0].tolist() == [[False, True], [False, False]] and not gates[1].any()
        with torch.no_grad():
            expected = teacher._encode_context(obs, False)
        torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
        for values in maps.values():
            assert values[~gates[0]].count_nonzero() == 0 and values[gates[0]].abs().sum() > 0
        gates.clear()
        attention_tokens, patches = teacher_attention_targets(teacher, obs, percentiles=(10, 90), return_patch_attention=True)
        torch.testing.assert_close(attention_tokens, expected, rtol=0, atol=0)
        assert all(torch.equal(gate, gates[0]) for gate in gates)
        assert all(torch.isfinite(v).all() for v in patches.values())
    finally:
        handle.remove()


def test_visualization_and_replay_respect_visual_bypass():
    teacher = make_teacher(pre_cross_self_attn_layers=0, cross_attn_layers=1).requires_grad_(False)
    obs = observations(1)
    obs["future_tau_ext"][0, 0, -1, 5] = 3.
    for capture_class in (ImageQueryAttentionCapture, AttentionCapture):
        with capture_class(teacher) as capture:
            teacher._encode_context(obs, False)
            assert all(value.count_nonzero() == 0 for value in capture.attention_by_camera().values())
            assert all(value.count_nonzero() == 0 for value in capture.update_by_camera().values())
    pair = capture_pair(teacher, obs, save_vectors=True)
    assert not pair["baseline_ca_update"].any() and not pair["context_effect"][1].any()
    query, normal, reference = capture_fusion_pair(teacher, obs, "raw_zero")
    assert normal["contact_gate"].tolist() == [True, False] and not reference["contact_gate"].any()
    indices = torch.arange(20)
    replacements = replacement_contexts(teacher.obs_encoder, query, normal, reference, indices)
    for index in indices:
        torch.testing.assert_close(replacements[index, index], reference["context"].reshape(20, 16)[index])
    torch.testing.assert_close(replacements[10:], normal["context"].reshape(1, 20, 16).expand(10, -1, -1))


def test_cache_identity_separates_modes_gates_and_preprocessing_version(tmp_path, monkeypatch):
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"tiny teacher")
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({"teacher_checkpoint": str(checkpoint),
                                     "teacher_cache": {"directory": str(tmp_path / "cache"), "batch_size": 2},
                                     "training": {"amp": {"allow_tf32": False}}})
    workspace.teacher_cfg = OmegaConf.create({"policy": {}})
    workspace.teacher_policy = SimpleNamespace()
    workspace.teacher_state_key = "model"
    workspace.amp_enabled, workspace.amp_dtype = False, torch.bfloat16
    workspace.attention_enabled, workspace.attention_config = False, {}
    dataset = SimpleNamespace(image_randomer=None, dataset_path=str(tmp_path), indices=np.arange(2))
    class Dataset:
        image_randomer = dataset.image_randomer
        dataset_path = dataset.dataset_path
        indices = dataset.indices
        def __len__(self):
            return 2
    dataset = Dataset()
    data_config = OmegaConf.create({"dataset_path": str(tmp_path)})
    monkeypatch.setattr(distill, "dataset_source_signature", lambda path: "same-source")
    legacy_path, legacy = workspace._teacher_cache_spec(dataset, dataset, data_config)
    assert legacy["future_contact_mode"] == "legacy_rollout" and not legacy["contact_fusion_gate"]
    workspace.teacher_policy.future_contact_mode = "token_threshold"
    threshold_path, _ = workspace._teacher_cache_spec(dataset, dataset, data_config)
    workspace.teacher_policy.contact_fusion_gate = True
    gate_path, identity = workspace._teacher_cache_spec(dataset, dataset, data_config)
    assert identity["wrench_preprocessing_version"] == WRENCH_PREPROCESSING_VERSION
    monkeypatch.setattr(distill, "WRENCH_PREPROCESSING_VERSION", WRENCH_PREPROCESSING_VERSION + 1)
    next_path, _ = workspace._teacher_cache_spec(dataset, dataset, data_config)
    assert len({legacy_path, threshold_path, gate_path, next_path}) == 4


def test_experiment_changes_only_contact_behavior_and_log_identity():
    root = Path(__file__).resolve().parents[1] / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        base = compose(config_name="sim_force_aware_dp_train_20hz_lstm2_gripper")
        gated = compose(config_name="sim_force_aware_dp_train_20hz_lstm2_gripper_contact_gate")
    assert gated.policy.future_contact_mode == "token_threshold" and gated.policy.contact_fusion_gate
    assert gated.policy.contact_enter_frames == base.policy.contact_enter_frames == 5
    assert gated.policy.contact_exit_frames == base.policy.contact_exit_frames == 5
    for key in base.policy:
        if key not in {"future_contact_mode", "contact_fusion_gate"}:
            assert OmegaConf.to_container(base.policy, resolve=True)[key] == OmegaConf.to_container(gated.policy, resolve=True)[key]
    assert OmegaConf.to_container(base.task, resolve=True) == OmegaConf.to_container(gated.task, resolve=True)
    assert gated.name != base.name and gated.logging.group != base.logging.group
