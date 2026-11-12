import json
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import train_civi as civi
from diffusion_policy.common.config_cli import load_config_reference


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    teacher = root / "teacher.yaml"
    student = root / "student.yaml"
    OmegaConf.save(OmegaConf.create({
        "output_dir": str(root / "outputs/teacher"), "dataset_path": "/data/from/teacher/config",
        "task": {"dataset": {"dataset_path": "${dataset_path}"}},
        "training": {"resume": False},
        "checkpoint": {"action_mse": {"enabled": True},
                       "topk": {"monitor_key": "checkpoint_action_mse", "mode": "min", "k": 1}},
    }), teacher)
    OmegaConf.save(OmegaConf.create({
        "output_dir": str(root / "outputs/student"), "training": {"resume_from": None},
    }), student)
    # Real child processes verify teacher -> scores -> student ordering and argv.
    script = '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, ORIGINAL_REPO)
from diffusion_policy.common.config_cli import load_config_reference
from diffusion_policy.common.checkpoint_action_mse import record_checkpoint_action_mse, resolve_student_teacher
config = sys.argv[sys.argv.index('--config') + 1]
overrides = sys.argv[sys.argv.index('--config') + 2:]
cfg = load_config_reference(config, overrides)
output = Path(cfg.output_dir)
is_teacher = Path(__file__).name == 'sim_force_aware_dp_train.py'
if is_teacher:
    if os.environ.get('CIVI_FAIL_TEACHER'):
        sys.exit(7)
    (output/'checkpoints').mkdir()
    for step, mse in ((1, 0.5), (2, 0.1), (3, 0.3)):
        checkpoint = output/'checkpoints'/f'step={step}.ckpt'
        checkpoint.write_text('teacher')
        record_checkpoint_action_mse(output, {'optimizer_step': step, 'checkpoint_action_mse': mse,
                                            'episode_ids': [0, 1]}, checkpoint)
    (output/'checkpoints/latest.ckpt').symlink_to('step=3.ckpt')
else:
    checkpoint, teacher_cfg = resolve_student_teacher(cfg)
    assert checkpoint.is_file()
    (output/'selection.json').write_text(json.dumps({'checkpoint': str(checkpoint),
        'dataset_path': teacher_cfg.task.dataset.dataset_path, 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}))
print('stage completed', flush=True)
'''.replace("ORIGINAL_REPO", repr(str(civi.REPO_ROOT)))
    for name in ("sim_force_aware_dp_train.py", "train_distill_render_net.py"):
        (root / name).write_text(script)
    return root, teacher, student


def _args(repository, *extra):
    root, teacher, student = repository
    return civi.make_parser().parse_args([
        "--teacher-config", str(teacher), "--student-config", str(student), *extra,
    ])


def test_all_trains_teacher_then_student_from_lowest_inference_mse(repository):
    root, _, _ = repository
    plan = civi.run_civi(_args(repository, "--gpus", "0", "--teacher-override", "dataset_path=/new/data"), root)
    assert plan["status"] == "completed"
    assert plan["teacher_score"]["checkpoint_action_mse"] == 0.1
    selected = json.loads((root / "outputs/student/selection.json").read_text())
    assert Path(selected["checkpoint"]).name == "step=2.ckpt"
    assert selected["dataset_path"] == "/new/data" and selected["gpus"] == "0"
    assert (root / "outputs/teacher/pipeline.json").is_file()
    assert all(stage["status"] == "completed" for stage in plan["stages"].values())


def test_student_only_reuses_teacher_config_without_retraining(repository):
    root, _, _ = repository
    civi.run_civi(_args(repository, "--stage", "teacher"), root)
    args = _args(repository, "--stage", "student")
    plan = civi.run_civi(args, root)
    assert set(plan["stages"]) == {"student"}
    assert (root / "outputs/student/selection.json").is_file()


@pytest.mark.parametrize("from_cli", [False, True])
def test_student_uses_explicit_weights_without_mse_scores(repository, from_cli):
    root, _, student = repository
    checkpoint = root / "specified teacher=external.ckpt"
    checkpoint.write_text("teacher")
    student_cfg = OmegaConf.load(student)
    student_cfg.teacher_checkpoint = str(root / "unused.ckpt" if from_cli else checkpoint)
    OmegaConf.save(student_cfg, student)
    extra = ["--stage", "student"]
    if from_cli:
        extra += ["--teacher-checkpoint", str(checkpoint)]

    plan = civi.run_civi(_args(repository, *extra), root)

    assert plan["status"] == "completed"
    assert plan["teacher_selection"] == "explicit" and plan["teacher_score"] is None
    assert plan["teacher_checkpoint"] == str(checkpoint)
    selected = json.loads((root / "outputs/student/selection.json").read_text())
    assert selected["checkpoint"] == str(checkpoint)
    assert selected["dataset_path"] == "/data/from/teacher/config"
    assert not (root / "outputs/teacher").exists()


def test_explicit_weights_dry_run_passes_checkpoint_to_ddp_without_writing(repository, capsys):
    root, _, _ = repository
    checkpoint = root / "specified teacher=external.ckpt"
    checkpoint.touch()
    plan = civi.run_civi(_args(
        repository, "--stage", "student", "--teacher-checkpoint", str(checkpoint),
        "--dry-run", "--gpus", "0,1", "--nproc-per-node", "2",
    ), root)

    assert not (root / "outputs").exists()
    command = civi.stage_command("student", plan["stages"]["student"], plan, root)
    cfg = load_config_reference(plan["stages"]["student"]["config"], command[command.index("--config") + 2:])
    assert cfg.teacher_checkpoint == str(checkpoint)
    assert cfg.training.distributed.enabled
    assert "Student uses specified teacher checkpoint:" in capsys.readouterr().out


def test_missing_explicit_weights_fail_before_creating_outputs(repository):
    root, _, _ = repository
    with pytest.raises(FileNotFoundError, match="Teacher checkpoint does not exist"):
        civi.run_civi(_args(repository, "--stage", "student", "--teacher-checkpoint", str(root / "missing.ckpt")), root)
    assert not (root / "outputs").exists()


def test_teacher_failure_stops_before_student(repository, monkeypatch):
    root, _, _ = repository
    monkeypatch.setenv("CIVI_FAIL_TEACHER", "1")
    with pytest.raises(subprocess.CalledProcessError):
        civi.run_civi(_args(repository), root)
    assert not (root / "outputs/student").exists()
    assert json.loads((root / "outputs/teacher/pipeline.json").read_text())["status"] == "failed"


def test_dry_run_and_ddp_commands_write_nothing(repository):
    root, _, _ = repository
    plan = civi.run_civi(_args(repository, "--dry-run", "--gpus", "0,1", "--nproc-per-node", "2"), root)
    assert not (root / "outputs").exists()
    command = civi.stage_command("student", plan["stages"]["student"], plan, root)
    assert "torch.distributed.run" in command and "--nproc_per_node=2" in command
    assert "++training.distributed.enabled=true" in command
    assert any(arg.startswith("++teacher_config=") for arg in command)


def test_loading_teacher_config_inside_hydra_restores_student_context(repository):
    root, teacher, _ = repository
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        original = GlobalHydra.instance().hydra
        source = load_config_reference(str(teacher))
        assert source.task.dataset.dataset_path == "/data/from/teacher/config"
        assert GlobalHydra.instance().hydra is original
        assert compose(config_name="student").training.resume_from is None


def test_student_flag_alias_and_existing_output(repository):
    root, _, student = repository
    assert civi.make_parser().parse_args(["--student", str(student)]).student_config == str(student)
    output = root / "outputs/teacher"
    output.mkdir(parents=True)
    marker = output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        civi.build_plan(_args(repository), root)
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize("config_name", [
    "train_civi_teacher", "train_civi_teacher_joint", "train_dp_dinobase_libero_drawer",
])
def test_teacher_configs_require_only_causal_measured_tau_ext(config_name):
    import torch

    from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
    from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor

    cfg = load_config_reference(str(civi.REPO_ROOT / f"diffusion_policy/config/{config_name}.yaml"))
    for shape_meta in (cfg.shape_meta, cfg.policy.shape_meta, cfg.task.shape_meta,
                       cfg.task.dataset.shape_meta):
        assert set(shape_meta.obs) == {"side", "wrist", "tau_ext"}
        assert list(shape_meta.obs.tau_ext.shape) == [8, 7]

    policy = cfg.policy
    assert policy.wrench_key == policy.phase_wrench_key == "tau_ext"
    assert policy.future_phase_wrench_key is None
    assert cfg.task.wrench_key == cfg.task.phase_wrench_key == "tau_ext"
    assert cfg.task.future_phase_wrench_key is None
    for config in (policy, cfg.task, cfg.task.dataset):
        assert list(config.contact_force_dims) == list(range(7))
        assert config.contact_norm == "l2"
    assert cfg.task.phase_source_key == cfg.task.dataset.phase_source_key == "observation.tau_ext"
    assert cfg.task.dataset.wrench_filter.source_key == "observation.tau_ext"
    windows = cfg.task.dataset.windowed_lowdim_feature_groups
    assert set(windows) == {"tau_ext"}
    window = windows.tau_ext
    assert window.source_key == "observation.tau_ext"
    assert [window.start_offset + index * window.stride
            for index in range(window.steps)] == list(range(-7, 1))

    # Exercise the saved preprocessing contract without building DINO or DiT.
    # A free sample and a contacting sample need no future observation key.
    preprocessor = WrenchPreprocessor.from_config({
        "n_obs_steps": policy.n_obs_steps,
        "wrench_key": policy.wrench_key,
        "phase_wrench_key": policy.phase_wrench_key,
        "future_phase_wrench_key": policy.future_phase_wrench_key,
        "contact_detector_type": "hysteresis",
        "contact_detector": {
            "free_threshold": policy.free_threshold,
            "contact_threshold": policy.contact_threshold,
            "enter_frames": policy.contact_enter_frames,
            "exit_frames": policy.contact_exit_frames,
            "force_dims": list(policy.contact_force_dims),
            "history_reducer": policy.contact_history_reducer,
            "norm": policy.contact_norm,
        },
        "input_gate": {"threshold": 0.0, "enabled": policy.input_gate_enabled, "norm": "l1"},
    })
    preprocessor.normalizer["tau_ext"] = SingleFieldLinearNormalizer.create_identity()
    assert preprocessor.required_keys == ("tau_ext",)
    tau_ext = torch.zeros(2, policy.n_obs_steps, window.steps, 7)
    # Contact on joint 7 must be recognized without any Cartesian wrench input.
    tau_ext[1, :, :, 6] = 2 * policy.contact_threshold
    normalized, contact, future_contact = preprocessor({"tau_ext": tau_ext})
    torch.testing.assert_close(normalized, tau_ext)
    torch.testing.assert_close(contact, torch.tensor([[False, False], [True, True]]))
    torch.testing.assert_close(preprocessor.contact_alpha({"tau_ext": tau_ext}), contact.float())
    assert future_contact is None


def test_default_current_tau_ext_outputs_do_not_select_historical_teachers():
    teacher = load_config_reference(str(civi.REPO_ROOT / civi.DEFAULT_TEACHER_CONFIG))
    student = load_config_reference(str(civi.REPO_ROOT / civi.DEFAULT_STUDENT_CONFIG))
    teacher_output = Path(teacher.output_dir)
    student_output = Path(student.output_dir)
    assert teacher_output.parent == student_output.parent == Path("data/outputs/civi/current_tau_ext")
    assert teacher_output != student_output
    assert student.teacher_checkpoint is None
