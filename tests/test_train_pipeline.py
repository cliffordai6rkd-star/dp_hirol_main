import json
import subprocess
from pathlib import Path

import pytest

import train_pipeline as pipeline


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    for config in (pipeline.DEFAULT_DP_CONFIG, pipeline.DEFAULT_DISTILL_CONFIG):
        path = root / config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    # Real subprocesses exercise ordering, argv quoting, logs and error propagation
    # without importing torch or allocating GPU memory.
    script = '''
import json, os, sys
from pathlib import Path
def override(key):
    return next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith(key + '='))
output = Path(json.loads(override('hydra.run.dir')))
is_dp = Path(__file__).name == 'sim_force_aware_dp_train.py'
if is_dp and os.environ.get('FAKE_DP_FAIL'):
    print('DP failed', flush=True)
    sys.exit(7)
if is_dp:
    (output/'checkpoints').mkdir()
    (output/'checkpoints/latest.ckpt').write_text('finished DP')
    if os.environ.get('FAKE_DP_BEST'):
        (output/'checkpoints/best.ckpt').write_text('best DP')
else:
    teacher = Path(json.loads(override('teacher_checkpoint')))
    assert teacher.is_file(), 'distillation must wait for DP checkpoint'
(output/'argv.json').write_text(json.dumps({'args':sys.argv[1:], 'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}))
print('stage completed', flush=True)
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
    assert Path(plan["teacher_checkpoint"]) == output / f"dp/checkpoints/{teacher_name}.ckpt"
    dp_args = json.loads((output / "dp/argv.json").read_text())
    distill_args = json.loads((output / "distill/argv.json").read_text())
    assert "training.num_optimizer_steps=100000" in dp_args["args"]
    assert "training.num_epochs=10" in distill_args["args"]
    assert "training.num_epochs=10" not in dp_args["args"]
    assert dp_args["gpus"] == distill_args["gpus"] == "0"
    manifest = json.loads((output / "pipeline.json").read_text())
    assert all(stage["status"] == "completed" for stage in manifest["stages"].values())
    for name in ("dp", "distill"):
        assert "stage completed" in (output / name / "train.log").read_text()


def test_dp_failure_stops_before_distillation(repository, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DP_FAIL", "1")
    output = tmp_path / "failed"
    args = pipeline.make_parser().parse_args(["all", "--output-dir", str(output)])
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pipeline.run_pipeline(args, repository)
    assert failure.value.returncode == 7
    assert not (output / "distill").exists()
    manifest = json.loads((output / "pipeline.json").read_text())
    assert manifest["status"] == manifest["stages"]["dp"]["status"] == "failed"


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
    assert plan["status"] == "completed" and not (output / "dp").exists()


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
    assert "auto-selected after DP" in capsys.readouterr().out


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
