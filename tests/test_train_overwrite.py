"""Exercise the real training CLI without constructing a policy or dataset."""

from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from diffusion_policy.common.config_cli import parse_overwrite_argv


ROOT = Path(__file__).resolve().parents[1]


def test_missing_flag_preserves_arguments_and_resume():
    args = ["train.py", "--config-name=teacher", "training.resume=true"]
    rewritten, overwrite = parse_overwrite_argv(args)
    assert rewritten == args
    assert overwrite is False


@pytest.mark.parametrize("position", [1, 5])
def test_flag_disables_resume_regardless_of_argument_order(position):
    args = ["train.py", "--config-name=teacher", "training.resume=true",
            "+logging.resume=allow", "++training.resume_from=/old.ckpt"]
    args.insert(position, "--overwrite")
    rewritten, overwrite = parse_overwrite_argv(args)
    assert overwrite is True
    assert rewritten == ["train.py", "--config-name=teacher", "++training.resume=false",
                         "++training.resume_from=null", "++logging.resume=false"]
    assert args[position] == "--overwrite"  # Caller argv is not mutated.


def test_flag_alone_requires_an_explicit_training_config():
    with pytest.raises(SystemExit, match="requires a training config"):
        parse_overwrite_argv(["train.py", "--overwrite"])


@pytest.fixture
def cli_run(tmp_path):
    output = tmp_path / "run with spaces"
    output.mkdir()
    (output / "checkpoints").mkdir()
    (output / "checkpoints/latest.ckpt").write_text("old checkpoint")
    (output / "logs.json.txt").write_text('{"train_loss": 100}\n')
    (output / "train.log").write_text("old run log\n")
    configs = tmp_path / "configs"
    configs.mkdir()
    OmegaConf.save(OmegaConf.create({
        "_target_": "diffusion_policy.workspace.base_workspace.BaseWorkspace",
        "output_dir": str(output),
        "training": {"resume": True, "device": "cpu", "distributed": {"enabled": False}},
        "logging": {"resume": True},
        "hydra": {"run": {"dir": "${output_dir}"}, "job": {"chdir": False}},
    }), configs / "teacher.yaml")
    return output, [sys.executable, str(ROOT / "train.py"),
                    f"--config-dir={configs}", "--config-name=teacher"]


def run_cli(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)


def test_real_cli_resets_artifacts_and_saves_fresh_configuration(cli_run):
    output, command = cli_run
    result = run_cli(command + ["--overwrite"])
    assert "[overwrite] Reset output directory; starting fresh:" in result.stdout
    assert not (output / "checkpoints").exists()
    assert not (output / "logs.json.txt").exists()
    assert (output / "train.log").is_file()
    assert "old run log" not in (output / "train.log").read_text()
    config = OmegaConf.load(output / ".hydra/config.yaml")
    assert config.training.resume is False
    assert config.training.resume_from is None
    assert config.logging.resume is False
    assert "overwrite" not in config.training

    # A later invocation using the saved config must not inherit the
    # destructive operation, even when resuming the same directory.
    marker = output / "keep-on-resume.txt"
    marker.write_text("keep")
    run_cli([sys.executable, str(ROOT / "train.py"),
             f"--config-dir={output / '.hydra'}", "--config-name=config",
             f"hydra.run.dir={output}", "training.resume=true"])
    assert marker.read_text() == "keep"


def test_real_cli_without_flag_preserves_existing_artifacts(cli_run):
    output, command = cli_run
    result = run_cli(command)
    assert "[overwrite]" not in result.stdout
    assert (output / "checkpoints/latest.ckpt").read_text() == "old checkpoint"
    assert (output / "logs.json.txt").read_text() == '{"train_loss": 100}\n'


@pytest.mark.parametrize("preview_args", [
    ["training.device=cpu", "--overwrite", "--cfg", "job", "--resolve"],
    ["--overwrite", "--cfg", "job", "--resolve", "training.device=cpu"],
])
def test_real_cli_config_preview_never_clears_outputs(cli_run, preview_args):
    output, command = cli_run
    result = run_cli(command + preview_args)
    config = OmegaConf.create(result.stdout)
    assert config.training.resume is False
    assert config.logging.resume is False
    assert (output / "checkpoints/latest.ckpt").read_text() == "old checkpoint"
    assert (output / "train.log").read_text() == "old run log\n"
    assert not (output / ".hydra").exists()
