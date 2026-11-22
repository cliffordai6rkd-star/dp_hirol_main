"""Every shipped preset resolves to the same causal CIVI method."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from diffusion_policy.common.config_cli import load_config_reference
from diffusion_policy.model.vision.contact_curriculum import MaskProbabilityScheduler
import train_pipeline


CONFIG = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
TEACHER_NAMES = [
    "train_civi_teacher",
    "train_civi_teacher_curriculum_b64_150k",
    "train_civi_teacher_board_blur_legacy_fusion",
    "train_civi_teacher_huanggua_blur_legacy_fusion_b32_150k",
    "train_board_future_tau_ext_token_mask_b32_100k",
    "train_board_future_tau_ext_token_mask_b32_600ep",
]
STUDENT_NAMES = [
    "train_distill_render_net_student",
    "train_distill_render_net_board_legacy",
    "train_distill_render_net_huanggua_legacy",
    "train_distill_board_future_tau_ext_legacy",
]


def load(name):
    return load_config_reference(str(CONFIG / f"{name}.yaml"))


@pytest.mark.parametrize("name", TEACHER_NAMES)
def test_teacher_aliases_change_data_or_budget_without_changing_the_method(name):
    current, canonical = load(name), load("train_civi_teacher")
    for cfg in (current, canonical):
        assert cfg.policy.future_phase_wrench_key is None
        assert set(cfg.shape_meta.obs) == {"side", "wrist", "tau_ext"}
        assert cfg.tau_ext_history_steps == 25
        assert cfg.shape_meta.obs.tau_ext.shape == [25, 7]
        assert cfg.policy.preserve_zero_wrench
        group = cfg.task.dataset.windowed_lowdim_feature_groups.tau_ext
        assert (group.source_key, group.steps, group.start_offset, group.pad_mode) == (
            "observation.tau_ext", 25, -24, "zero")
        assert cfg.task.dataset.wrench_threshold.threshold == 12.5
    policies = [OmegaConf.to_container(cfg.policy, resolve=True) for cfg in (current, canonical)]
    # The mask schedule follows each alias's optimizer/epoch budget.
    for policy in policies:
        policy["mask_schedule"].pop("total_steps")
    assert policies[0] == policies[1]


@pytest.mark.parametrize("name", STUDENT_NAMES)
def test_student_aliases_share_three_stage_heads_and_history(name):
    cfg, canonical = load(name), load("train_distill_render_net_student")
    assert cfg.render_net == canonical.render_net
    assert cfg.training == canonical.training
    assert cfg.render_net.phase_predictor.num_classes == 3
    assert cfg.tau_ext_history_steps is None
    assert cfg.render_net.phase_predictor.history_steps is None
    assert cfg.render_net.phase_predictor.pad_mode == "zero"
    assert cfg.training.render_stages.transition == "validation_plateau"
    assert cfg.training.render_stages.patience == 5
    assert "intensity_epochs" not in cfg.training.render_stages


def test_default_pipeline_uses_core_configs_and_mask_course():
    args = train_pipeline.make_parser().parse_args([])
    assert Path(args.dp_config).name == "train_civi_teacher.yaml"
    assert Path(args.distill_config).name == "train_distill_render_net_student.yaml"
    cfg = load("train_civi_teacher")
    scheduler = MaskProbabilityScheduler(**OmegaConf.to_container(cfg.policy.mask_schedule, resolve=True))
    budget = cfg.training.num_optimizer_steps
    assert [scheduler.probability(step) for step in (0, budget // 4, budget // 2,
                                                    3 * budget // 4, budget)] == [1., 1., .5, 0., 0.]
    assert cfg.policy.image_blur_schedule is None
    assert cfg.policy.image_mask_scope == "precontact_contact"


@pytest.mark.parametrize("steps", [25, 37, 50])
def test_one_history_override_controls_shape_encoder_and_dataset(steps):
    cfg = load_config_reference(str(CONFIG / "train_civi_teacher.yaml"),
                                overrides=[f"tau_ext_history_steps={steps}"])
    assert cfg.shape_meta.obs.tau_ext.shape == [steps, 7]
    assert cfg.policy.wrench_history_steps == steps
    assert cfg.task.wrench_history_steps == cfg.task.phase_wrench_steps == steps
    group = cfg.task.dataset.windowed_lowdim_feature_groups.tau_ext
    assert group.steps == steps
    assert group.start_offset == 1 - steps
