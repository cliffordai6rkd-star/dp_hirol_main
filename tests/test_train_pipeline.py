import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import train_pipeline as pipeline


def test_default_student_uses_checkpoint_data_config_and_allows_explicit_source(tmp_path):
    from diffusion_policy.common.checkpoint_action_mse import resolve_student_teacher
    from diffusion_policy.common.config_cli import load_config_reference

    checkpoint = tmp_path / "current teacher.ckpt"
    checkpoint.touch()
    args = pipeline.make_parser().parse_args([
        "distill", "--teacher-checkpoint", str(checkpoint),
        "--output-dir", str(tmp_path / "pipeline"),
    ])
    plan = pipeline.build_plan(args)
    stage = plan["stages"]["distill"]
    command = pipeline.stage_command("distill", stage, plan)
    cfg = load_config_reference(stage["config"], command[command.index("--config") + 2:])
    selected, source = resolve_student_teacher(cfg)
    assert selected == checkpoint
    assert source is None  # Workspace will inherit the checkpoint's data config.

    stage["overrides"] = ["teacher_config=" + str(pipeline.REPO_ROOT / pipeline.DEFAULT_DP_CONFIG)]
    command = pipeline.stage_command("distill", stage, plan)
    cfg = load_config_reference(stage["config"], command[command.index("--config") + 2:])
    _, source = resolve_student_teacher(cfg)
    assert source.policy.wrench_key == "tau_ext"


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    monkeypatch.chdir(root)
    for config in (pipeline.DEFAULT_DP_CONFIG, pipeline.DEFAULT_DISTILL_CONFIG):
        path = root / config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("training:\n  num_optimizer_steps: 200000\n  num_epochs: 1000\n")
    # Real subprocesses exercise ordering, argv quoting, logs and error propagation
    # without importing torch or allocating GPU memory.
    script = '''
import json, os, sys
from pathlib import Path
def override(key):
    return next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith(key + '='))
output = Path(json.loads(override('hydra.run.dir')))
is_dp = Path(__file__).name == 'sim_force_aware_dp_train.py'
events = output.parent/'stage_order.jsonl'
with events.open('a') as log:
    log.write(json.dumps(['teacher' if is_dp else 'student', 'start']) + '\\n')
if is_dp and os.environ.get('FAKE_DP_FAIL'):
    print('DP failed', flush=True)
    sys.exit(7)
if not is_dp and os.environ.get('FAKE_STUDENT_FAIL'):
    print('student failed', flush=True)
    sys.exit(9)
if is_dp:
    (output/'checkpoints').mkdir()
    (output/'checkpoints/latest.ckpt').write_text('finished DP')
    if os.environ.get('FAKE_DP_BEST'):
        (output/'checkpoints/best.ckpt').write_text('best DP')
else:
    teacher = Path(json.loads(override('teacher_checkpoint')))
    assert teacher.is_file(), 'distillation must wait for DP checkpoint'
    (output/'checkpoints').mkdir()
    (output/'checkpoints/latest.ckpt').write_text('finished RenderDP')
(output/'config.yaml').write_text('stage: teacher\\n' if is_dp else 'stage: student\\n')
(output/'argv.json').write_text(json.dumps({'args':sys.argv[1:], 'gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'dataset_path': os.environ.get('CIVI_DATASET_PATH')}))
print('stage completed', flush=True)
with events.open('a') as log:
    log.write(json.dumps(['teacher' if is_dp else 'student', 'end']) + '\\n')
'''
    for name in ("sim_force_aware_dp_train.py", "train_distill_render_net.py"):
        (root / name).write_text(script)
    return root


@pytest.mark.parametrize("best", [False, True])
def test_all_runs_stages_in_order_and_hands_off_checkpoint(repository, tmp_path, monkeypatch, best):
    if best:
        monkeypatch.setenv("FAKE_DP_BEST", "1")
    output = tmp_path / "output with spaces = run"
    args = pipeline.make_parser().parse_args([
        "all", "--output-dir", str(output), "--gpus", "0",
        "--dp-override", "training.num_optimizer_steps=100000",
        "--distill-override", "training.num_epochs=10",
    ])
    plan = pipeline.run_pipeline(args, repository)
    teacher_name = "best" if best else "latest"
    assert plan["status"] == "completed"
    assert Path(plan["teacher_checkpoint"]) == output / f"teacher/checkpoints/{teacher_name}.ckpt"
    dp_args = json.loads((output / "teacher/argv.json").read_text())
    distill_args = json.loads((output / "student/argv.json").read_text())
    assert "training.num_optimizer_steps=100000" in dp_args["args"]
    assert "training.num_epochs=10" in distill_args["args"]
    assert "training.num_epochs=10" not in dp_args["args"]
    assert dp_args["gpus"] == distill_args["gpus"] == "0"
    manifest = json.loads((output / "pipeline.json").read_text())
    assert all(stage["status"] == "completed" for stage in manifest["stages"].values())
    assert set(manifest["stages"]) == {"dp", "distill"}
    for name in ("teacher", "student"):
        assert "stage completed" in (output / name / "train.log").read_text()
        assert (output / name / "config.yaml").is_file()
        assert (output / name / "checkpoints/latest.ckpt").is_file()
    assert not (output / "dp").exists() and not (output / "distill").exists()
    events = [json.loads(line) for line in (output / "stage_order.jsonl").read_text().splitlines()]
    assert events == [["teacher", "start"], ["teacher", "end"],
                      ["student", "start"], ["student", "end"]]


def test_default_invocation_runs_teacher_then_student(repository, tmp_path):
    output = tmp_path / "default_pipeline"
    args = pipeline.make_parser().parse_args(["--output-dir", str(output), "--gpus", "0"])
    assert args.stage == "all"
    plan = pipeline.run_pipeline(args, repository)
    assert plan["status"] == "completed" and plan["stage"] == "all"
    assert plan["stages"]["dp"]["output_dir"] == str(output / "teacher")
    assert plan["stages"]["distill"]["output_dir"] == str(output / "student")
    assert plan["teacher_checkpoint"] == str(output / "teacher/checkpoints/latest.ckpt")
    student_args = json.loads((output / "student/argv.json").read_text())["args"]
    assert "teacher_checkpoint=" + json.dumps(plan["teacher_checkpoint"]) in student_args
    assert not (output / "dp").exists() and not (output / "distill").exists()


def test_exported_dataset_is_available_to_both_pipeline_children(repository, tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    exported_path = str(tmp_path / "dataset with spaces = selected")
    monkeypatch.setenv("CIVI_DATASET_PATH", exported_path)
    teacher = repository / pipeline.DEFAULT_DP_CONFIG
    OmegaConf.save(OmegaConf.create({
        "dataset_path": "${oc.env:CIVI_DATASET_PATH,null}",
        "task": {"dataset": {"dataset_path": "${dataset_path}"}},
    }), teacher)
    output = tmp_path / "exported_dataset_run"
    args = pipeline.make_parser().parse_args(["--output-dir", str(output)])

    plan = pipeline.run_pipeline(args, repository)

    assert plan["status"] == "completed"
    for stage in ("teacher", "student"):
        record = json.loads((output / stage / "argv.json").read_text())
        assert record["dataset_path"] == exported_path
        assert not any(argument.startswith("dataset_path=") for argument in record["args"])


@pytest.mark.parametrize("exported_path", [None, "", "   "])
def test_missing_exported_dataset_stops_pipeline_before_creating_outputs(
        repository, tmp_path, monkeypatch, exported_path):
    from omegaconf import OmegaConf

    if exported_path is None:
        monkeypatch.delenv("CIVI_DATASET_PATH", raising=False)
    else:
        monkeypatch.setenv("CIVI_DATASET_PATH", exported_path)
    teacher = repository / pipeline.DEFAULT_DP_CONFIG
    OmegaConf.save(OmegaConf.create({
        "dataset_path": "${oc.env:CIVI_DATASET_PATH,null}",
        "task": {"dataset": {"dataset_path": "${dataset_path}"}},
    }), teacher)
    output = tmp_path / "missing_dataset_run"
    args = pipeline.make_parser().parse_args(["--output-dir", str(output)])

    with pytest.raises(ValueError, match="export CIVI_DATASET_PATH"):
        pipeline.run_pipeline(args, repository)

    assert not output.exists()


@pytest.mark.parametrize("dataset,task_name,expected_task", [
    ("board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5", None, "board"),
    ("huanggua1103_25hz_3cam_lerobotv3_clean_ma5_l1_12p5", None, "huanggua1103"),
    ("board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5", "press_button", "press_button"),
])
def test_default_pipeline_names_experiment_for_task_and_keeps_teacher_student_layout(
        repository, monkeypatch, dataset, task_name, expected_task):
    from omegaconf import OmegaConf

    monkeypatch.setenv("CIVI_DATASET_PATH", "data/" + dataset)
    if task_name is None:
        monkeypatch.delenv("CIVI_TASK_NAME", raising=False)
    else:
        monkeypatch.setenv("CIVI_TASK_NAME", task_name)
    OmegaConf.save(OmegaConf.create({
        "dataset_path": "${oc.env:CIVI_DATASET_PATH,null}",
        "task": {"dataset": {"dataset_path": "${dataset_path}"}},
    }), repository / pipeline.DEFAULT_DP_CONFIG)

    plan = pipeline.build_plan(pipeline.make_parser().parse_args([]), repository)

    output = Path(plan["output_dir"])
    assert output.parent.parent == repository / "data/outputs"
    assert output.name.endswith(f"_fdp_dis_{expected_task}")
    assert Path(plan["stages"]["dp"]["output_dir"]) == output / "teacher"
    assert Path(plan["stages"]["distill"]["output_dir"]) == output / "student"
    assert not output.exists()


def test_dp_failure_stops_before_distillation(repository, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DP_FAIL", "1")
    output = tmp_path / "failed"
    args = pipeline.make_parser().parse_args(["all", "--output-dir", str(output)])
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pipeline.run_pipeline(args, repository)
    assert failure.value.returncode == 7
    assert not (output / "student").exists()
    assert (output / "teacher/train.log").is_file()
    manifest = json.loads((output / "pipeline.json").read_text())
    assert manifest["status"] == manifest["stages"]["dp"]["status"] == "failed"


def test_student_failure_preserves_completed_teacher(repository, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_STUDENT_FAIL", "1")
    output = tmp_path / "failed_student"
    args = pipeline.make_parser().parse_args(["--output-dir", str(output)])
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pipeline.run_pipeline(args, repository)
    assert failure.value.returncode == 9
    assert (output / "teacher/checkpoints/latest.ckpt").read_text() == "finished DP"
    assert "student failed" in (output / "student/train.log").read_text()
    manifest = json.loads((output / "pipeline.json").read_text())
    assert manifest["stages"]["dp"]["status"] == "completed"
    assert manifest["status"] == manifest["stages"]["distill"]["status"] == "failed"
    assert not (output / "dp").exists() and not (output / "distill").exists()


def test_dp_mode_writes_teacher_without_starting_student(repository, tmp_path):
    output = tmp_path / "teacher_only"
    args = pipeline.make_parser().parse_args(["dp", "--output-dir", str(output)])
    plan = pipeline.run_pipeline(args, repository)
    assert plan["status"] == "completed" and set(plan["stages"]) == {"dp"}
    assert (output / "teacher/checkpoints/latest.ckpt").is_file()
    assert not (output / "student").exists()
    assert not (output / "dp").exists() and not (output / "distill").exists()


def test_distill_mode_uses_existing_run_without_retraining_dp(repository, tmp_path):
    teacher_run = tmp_path / "teacher"
    (teacher_run / "checkpoints").mkdir(parents=True)
    checkpoint = teacher_run / "checkpoints/latest.ckpt"
    checkpoint.write_text("teacher")
    output = tmp_path / "student"
    args = pipeline.make_parser().parse_args([
        "distill", "--teacher-run", str(teacher_run), "--output-dir", str(output),
    ])
    plan = pipeline.run_pipeline(args, repository)
    assert plan["teacher_checkpoint"] == str(checkpoint)
    assert plan["status"] == "completed" and not (output / "teacher").exists()
    assert (output / "student/checkpoints/latest.ckpt").is_file()
    assert not (output / "dp").exists() and not (output / "distill").exists()


def test_dry_run_creates_nothing_and_builds_ddp_commands(repository, tmp_path, capsys):
    output = tmp_path / "preview"
    args = pipeline.make_parser().parse_args([
        "all", "--dry-run", "--output-dir", str(output), "--gpus", "0,1", "--nproc-per-node", "2",
    ])
    plan = pipeline.run_pipeline(args, repository)
    assert not output.exists() and plan["status"] == "planned"
    command = pipeline.stage_command("dp", plan["stages"]["dp"], plan, repository)
    assert "torch.distributed.run" in command and "--nproc_per_node=2" in command
    assert "++training.distributed.enabled=true" in command
    printed = capsys.readouterr().out
    assert "auto-selected after teacher training" in printed
    assert str(output / "teacher/checkpoints") in printed
    assert str(output / "student") in printed
    assert str(output / "dp/checkpoints") not in printed


def test_all_launches_two_real_torchrun_stages_and_hands_off_checkpoint(repository, tmp_path):
    # Run real launchers and collectives; replace expensive training with a
    # checkpoint marker to exercise the parent/worker boundary and stage order.
    script = '''
import json, os, sys
from pathlib import Path
import torch.distributed as dist
dist.init_process_group('gloo')
rank = dist.get_rank()
def override(key):
    return json.loads(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith(key+'=')))
output = Path(override('hydra.run.dir'))
is_teacher = Path(__file__).name == 'sim_force_aware_dp_train.py'
if not is_teacher:
    checkpoint = Path(override('teacher_checkpoint'))
    assert checkpoint.read_text() == 'teacher complete'
if rank == 0:
    (output/'checkpoints').mkdir()
    (output/'checkpoints/latest.ckpt').write_text('teacher complete' if is_teacher else 'student complete')
(output/f'rank{rank}.json').write_text(json.dumps({'rank': rank, 'world_size': dist.get_world_size()}))
dist.barrier()
dist.destroy_process_group()
'''
    for name in ("sim_force_aware_dp_train.py", "train_distill_render_net.py"):
        (repository / name).write_text(script)
    output = tmp_path / "two_rank_pipeline"
    args = pipeline.make_parser().parse_args([
        "all", "--nproc-per-node", "2", "--gpus", "0,1", "--output-dir", str(output),
    ])
    plan = pipeline.run_pipeline(args, repository)
    assert plan["status"] == "completed"
    for name in ("teacher", "student"):
        for rank in range(2):
            worker = json.loads((output / name / f"rank{rank}.json").read_text())
            assert worker == {"rank": rank, "world_size": 2}


@pytest.mark.parametrize("arguments,message", [
    (["distill"], "requires --teacher"),
    (["all", "--dp-override", "hydra.run.dir=elsewhere"], "--output-dir"),
    (["dp", "--dp-override", "--multirun"], None),
    (["all", "--distill-override", "teacher_checkpoint=old.ckpt"], "--teacher-checkpoint"),
    (["dp", "--nproc-per-node", "2", "--gpus", "0"], "at least"),
])
def test_invalid_arguments_are_rejected_before_launch(repository, arguments, message):
    if message is None:
        with pytest.raises(SystemExit):
            pipeline.make_parser().parse_args(arguments)
    else:
        args = pipeline.make_parser().parse_args(arguments)
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            pipeline.build_plan(args, repository)


def test_explicit_checkpoint_selection_does_not_silently_fallback(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    latest = tmp_path / "checkpoints/latest.ckpt"
    latest.write_text("weights")
    assert pipeline.select_teacher(tmp_path) == latest
    with pytest.raises(FileNotFoundError, match="best"):
        pipeline.select_teacher(tmp_path, "best")


def test_existing_output_is_preserved(repository, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    args = pipeline.make_parser().parse_args(["dp", "--output-dir", str(output)])
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(args, repository)
    assert marker.read_text() == "keep"


@pytest.fixture
def external_teacher(tmp_path):
    """Publish the final checkpoint only after the test releases stdin."""
    processes = []
    script = '''
import sys
from pathlib import Path
import dill
import torch
from omegaconf import OmegaConf
directory = Path(sys.argv[1]) / 'checkpoints'
directory.mkdir(parents=True)
def save(epoch):
    payload = {
        'cfg': OmegaConf.create({'training': {'num_epochs': 2,
            'num_optimizer_steps': None, 'early_stopping': {'enabled': False}}}),
        'pickles': {'epoch': dill.dumps(epoch), 'optimizer_step': dill.dumps(epoch * 10)},
        'state_dicts': {},
    }
    torch.save(payload, directory / 'next.ckpt', pickle_module=dill)
    (directory / 'next.ckpt').replace(directory / 'latest.ckpt')
save(1)
print('ready', flush=True)
sys.stdin.readline()
save(int(sys.argv[2]))
(directory / 'best.ckpt').write_text('final selected teacher')
'''

    def launch(final_epoch=2):
        run = tmp_path / f"external teacher {len(processes)}"
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script, str(run), str(final_epoch)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        processes.append(process)
        assert process.stdout.readline().strip() == "ready"
        return process, run

    yield launch
    for process in processes:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)


@pytest.mark.parametrize("final_epoch", [1, 2])
def test_waiting_pipeline_selects_final_teacher_only_after_completed_exit(
        repository, tmp_path, monkeypatch, external_teacher, final_epoch):
    from diffusion_policy.common import teacher_handoff

    original_sleep = time.sleep
    monkeypatch.setattr(teacher_handoff.time, "sleep", lambda _: original_sleep(0.01))
    process, teacher_run = external_teacher(final_epoch)
    output = tmp_path / "waiting student"
    args = pipeline.make_parser().parse_args([
        "distill", "--teacher-run", str(teacher_run), "--teacher-selection", "best",
        "--wait-for-teacher-pid", str(process.pid), "--output-dir", str(output),
    ])
    plan = pipeline.build_plan(args, repository)
    assert plan["teacher_checkpoint"] is None
    assert plan["teacher_wait"]["pid"] == process.pid
    assert not (teacher_run / "checkpoints/best.ckpt").exists()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pipeline.run_pipeline, args, repository)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                manifest_path = output / "pipeline.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    if (manifest["status"] == "waiting_for_teacher"
                            and (output / "gpu_usage.jsonl").exists()
                            and (output / "gpu_usage.jsonl").stat().st_size > 0):
                        break
                original_sleep(0.01)
            else:
                pytest.fail("Pipeline did not enter waiting_for_teacher")
            assert not (output / "student").exists()
            assert not (output / "stage_order.jsonl").exists()
            samples = [json.loads(line) for line in (output / "gpu_usage.jsonl").read_text().splitlines()]
            assert samples[0]["attempt"] == 0
            assert not future.done()
        finally:
            process.stdin.write("finish\n")
            process.stdin.flush()
            process.wait(timeout=10)

        if final_epoch < 2:
            with pytest.raises(ValueError, match="before completing training"):
                future.result(timeout=10)
            assert not (output / "student").exists()
            manifest = json.loads((output / "pipeline.json").read_text())
            assert manifest["status"] == manifest["stages"]["distill"]["status"] == "failed"
        else:
            result = future.result(timeout=10)
            assert result["status"] == "completed"
            assert result["teacher_checkpoint"] == str(teacher_run / "checkpoints/best.ckpt")
            assert result["teacher_completion"]["epoch"] == 2
            assert result["teacher_completion"]["reason"] == "num_epochs"
            recorded = json.loads((output / "student/argv.json").read_text())["args"]
            assert "teacher_checkpoint=" + json.dumps(result["teacher_checkpoint"]) in recorded


def test_waiting_pipeline_dry_run_does_not_wait_or_create_outputs(
        repository, tmp_path, monkeypatch, external_teacher, capsys):
    from diffusion_policy.common import teacher_handoff

    process, teacher_run = external_teacher()
    output = tmp_path / "waiting preview"
    def unexpected_wait(**kwargs):
        pytest.fail("dry-run must not wait for the teacher")
    monkeypatch.setattr(teacher_handoff, "wait_for_teacher", unexpected_wait)
    args = pipeline.make_parser().parse_args([
        "distill", "--teacher-run", str(teacher_run), "--teacher-selection", "best",
        "--wait-for-teacher-pid", str(process.pid), "--output-dir", str(output), "--dry-run",
    ])
    plan = pipeline.run_pipeline(args, repository)
    assert plan["status"] == "planned" and plan["teacher_checkpoint"] is None
    assert not output.exists()
    assert process.poll() is None
    printed = capsys.readouterr().out
    assert f"Wait for teacher PID {process.pid}" in printed
    assert "auto-selected after teacher training" in printed


@pytest.mark.parametrize("arguments", [
    ["all", "--wait-for-teacher-pid", "123"],
    ["distill", "--teacher-checkpoint", "teacher.ckpt", "--wait-for-teacher-pid", "123"],
])
def test_waiting_requires_distill_and_teacher_run(repository, arguments):
    args = pipeline.make_parser().parse_args(arguments)
    with pytest.raises(ValueError, match="requires distill mode with --teacher-run"):
        pipeline.build_plan(args, repository)


@pytest.fixture
def oom_student(repository):
    (repository / pipeline.DEFAULT_DISTILL_CONFIG).write_text('''
dataloader:
  batch_size: 24
val_dataloader:
  batch_size: 32
teacher_cache:
  batch_size: 8
training:
  compile:
    enabled: true
  distributed:
    batch_size_is_global: true
''')
    (repository / "train_distill_render_net.py").write_text('''
import json, os, sys
from pathlib import Path
def override(key):
    return next(arg.split('=', 1)[1] for arg in reversed(sys.argv) if arg.split('=', 1)[0].lstrip('+') == key)
output = Path(json.loads(override('hydra.run.dir')))
counter = output / 'counter'
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
(output / f'args{attempt}.json').write_text(json.dumps(sys.argv[1:]))
if os.environ.get('FAKE_SAVE_LATEST'):
    (output / 'checkpoints').mkdir(exist_ok=True)
    (output / 'checkpoints/latest.ckpt').write_text('resumable student')
print(f'attempt {attempt}', flush=True)
if attempt == int(os.environ.get('FAKE_OTHER_ATTEMPT', '0')):
    print('RuntimeError: unrelated failure', flush=True)
    sys.exit(6)
if attempt <= int(os.environ.get('FAKE_OOM_ATTEMPTS', '1')):
    print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128 MiB', flush=True)
    sys.exit(7)
print('student completed', flush=True)
''')


@pytest.mark.parametrize("saved_latest", [False, True])
def test_student_cuda_oom_reduces_batches_and_retries_from_latest(
        repository, tmp_path, monkeypatch, oom_student, saved_latest):
    teacher = tmp_path / "teacher.ckpt"
    teacher.touch()
    if saved_latest:
        monkeypatch.setenv("FAKE_SAVE_LATEST", "1")
    output = tmp_path / "oom recovered"
    args = pipeline.make_parser().parse_args([
        "distill", "--teacher-checkpoint", str(teacher), "--output-dir", str(output),
        "--oom-retries", "2",
    ])
    plan = pipeline.run_pipeline(args, repository)
    stage = plan["stages"]["distill"]
    assert plan["status"] == "completed"
    assert [attempt["status"] for attempt in stage["attempts"]] == ["failed", "completed"]
    assert stage["attempts"][0]["cuda_oom"]
    next_args = json.loads((output / "student/args2.json").read_text())
    assert "++dataloader.batch_size=12" in next_args
    assert "++val_dataloader.batch_size=16" in next_args
    assert "++teacher_cache.batch_size=4" in next_args
    assert "++training.compile.enabled=false" in next_args
    if saved_latest:
        assert "++training.resume_from=" + json.dumps(str(output / "student/checkpoints/latest.ckpt")) in next_args
    else:
        assert not any("training.resume_from=" in value for value in next_args)
    first_log = (output / "student/attempt_001.log").read_text()
    second_log = (output / "student/attempt_002.log").read_text()
    assert "CUDA out of memory" in first_log and "student completed" not in first_log
    assert "student completed" in second_log and "CUDA out of memory" not in second_log
    assert len(json.loads((output / "pipeline.json").read_text())["stages"]["distill"]["attempts"]) == 2


@pytest.mark.parametrize("retries,other_attempt,expected_attempts", [(0, 0, 1), (2, 0, 3), (3, 1, 1), (3, 2, 2)])
def test_oom_retries_stop_at_limit_and_do_not_retry_unrelated_failure(
        repository, tmp_path, monkeypatch, oom_student, retries, other_attempt, expected_attempts):
    teacher = tmp_path / "teacher.ckpt"
    teacher.touch()
    monkeypatch.setenv("FAKE_OOM_ATTEMPTS", "10")
    monkeypatch.setenv("FAKE_OTHER_ATTEMPT", str(other_attempt))
    output = tmp_path / "retry limit"
    args = pipeline.make_parser().parse_args([
        "distill", "--teacher-checkpoint", str(teacher), "--output-dir", str(output),
        "--oom-retries", str(retries),
    ])
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args, repository)
    plan = json.loads((output / "pipeline.json").read_text())
    attempts = plan["stages"]["distill"]["attempts"]
    assert plan["status"] == "failed"
    assert len(attempts) == expected_attempts
    assert int((output / "student/counter").read_text()) == expected_attempts
    assert attempts[-1]["cuda_oom"] is (other_attempt == 0)


def test_teacher_cuda_oom_is_not_retried(repository, tmp_path):
    (repository / "sim_force_aware_dp_train.py").write_text(
        "print('RuntimeError: CUDA error: out of memory', flush=True)\nraise SystemExit(7)\n"
    )
    output = tmp_path / "teacher oom"
    args = pipeline.make_parser().parse_args(["all", "--output-dir", str(output), "--oom-retries", "3"])
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args, repository)
    plan = json.loads((output / "pipeline.json").read_text())
    assert len(plan["stages"]["dp"]["attempts"]) == 1
    assert not (output / "student").exists()


@pytest.mark.parametrize("global_batch,expected", [(True, (4, 4, 1)), (False, (5, 7, 1))])
def test_batch_reduction_respects_global_ddp_multiple_and_cache_minimum(
        repository, tmp_path, oom_student, global_batch, expected):
    stage = {
        "config": str(repository / pipeline.DEFAULT_DISTILL_CONFIG),
        "output_dir": str(tmp_path / "student"),
        "overrides": ["dataloader.batch_size=10", "val_dataloader.batch_size=14",
                      "teacher_cache.batch_size=3", f"training.distributed.batch_size_is_global={str(global_batch).lower()}"],
    }
    result = pipeline.reduce_student_batches(stage, {"nproc_per_node": 4})
    keys = ("dataloader.batch_size", "val_dataloader.batch_size", "teacher_cache.batch_size")
    assert tuple(result["overrides"][key] for key in keys) == expected
    again = pipeline.reduce_student_batches(stage, {"nproc_per_node": 4})
    assert again["overrides"]["teacher_cache.batch_size"] == 1
    assert again["overrides"]["dataloader.batch_size"] >= (4 if global_batch else 1)


def test_gpu_monitor_samples_during_silent_stage_and_preserves_attempt_log(tmp_path):
    binary = tmp_path / "nvidia-smi"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' '0, Test GPU, 2048, 24576, 91'\n")
    binary.chmod(0o755)
    output = tmp_path / "monitored stage"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    pipeline.run_stage(
        [sys.executable, "-u", "-c", "import time; time.sleep(0.12); print('completed')"],
        output, env, monitor_interval=0.02,
    )
    samples = [json.loads(line) for line in (output / "gpu_usage.jsonl").read_text().splitlines()]
    assert len(samples) >= 2
    assert all(sample["gpus"][0]["memory_used_mb"] == "2048" for sample in samples)
    assert all(sample["gpus"][0]["utilization_percent"] == "91" for sample in samples)
    assert (output / "attempt_001.log").read_text().strip() == "completed"


def test_gpu_monitor_failure_does_not_interrupt_stage(tmp_path):
    output = tmp_path / "no nvidia-smi"
    pipeline.run_stage([sys.executable, "-c", "print('completed')"], output, dict(os.environ, PATH=""))
    samples = [json.loads(line) for line in (output / "gpu_usage.jsonl").read_text().splitlines()]
    assert "error" in samples[0]
    assert (output / "train.log").read_text().strip() == "completed"
