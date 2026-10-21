import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


CONFIGS = [
    "abulation_study/train_dp_image_query_lstm_insert_usb",
    "abulation_study/train_dp_image_selfattn_insert_usb",
    "abulation_study/train_dp_image_wrench_selfattn_insert_usb",
]


def prepare_launcher(tmp_path, fail_config=""):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "scripts/train_usb_abulation_study.sh"
    script.parent.mkdir()
    shutil.copy2(root / "scripts" / script.name, script)
    wrapper = tmp_path / "fake_training_python"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-u', 'train.py']:\n"
        "    config = args[args.index('--config-name') + 1]\n"
        "    def record(event):\n"
        "        with open(os.environ['ABLATION_TEST_EVENTS'], 'a') as f:\n"
        "            f.write(json.dumps([event, config, args]) + '\\n')\n"
        "    record('start')\n"
        "    time.sleep(0.02)\n"
        "    record('end')\n"
        f"    sys.exit(7 if config == {fail_config!r} else 0)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *args])\n"
    )
    wrapper.chmod(0o755)
    env = dict(os.environ, ABLATION_TEST_EVENTS=str(tmp_path / "events.jsonl"))
    args = ["bash", str(script), "--python", str(wrapper), "--log-dir", str(tmp_path / "logs")]
    return args, env


def test_launcher_runs_three_jobs_serially_in_requested_order(tmp_path):
    args, env = prepare_launcher(tmp_path)
    result = subprocess.run(args + ["--", "training.device=cpu"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event, config) for event, config, _ in events] == [
        (event, config) for config in CONFIGS for event in ("start", "end")
    ]
    assert all(args[-1] == "training.device=cpu" for _, _, args in events)
    exits = (tmp_path / "logs/exit_codes.tsv").read_text()
    assert all(config.split("/")[-1] + "\t0" in exits for config in CONFIGS)


@pytest.mark.parametrize("fail_index", [0, 1])
def test_launcher_stops_after_failed_training(tmp_path, fail_index):
    args, env = prepare_launcher(tmp_path, fail_config=CONFIGS[fail_index])
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 7, result.stdout + result.stderr
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [config for event, config, _ in events if event == "start"] == CONFIGS[:fail_index + 1]


def test_launcher_dry_run_does_not_execute_training_or_create_logs(tmp_path):
    args, env = prepare_launcher(tmp_path)
    result = subprocess.run(args + ["--dry-run"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [line.split("--config-name ")[1].strip() for line in result.stdout.splitlines()] == CONFIGS
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "logs").exists()
