import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def prepare_queue(tmp_path, first_exit=0):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/train_usb_baselines_after_current.sh"
    script = scripts / source.name
    shutil.copy2(source, script)
    wrapper = tmp_path / "python_wrapper"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "args=sys.argv[1:]\n"
        "if args[:2] == ['-u','train.py']:\n"
        " config=args[args.index('--config-name')+1]\n"
        " with open(os.environ['QUEUE_TEST_EVENTS'],'a') as f: f.write(json.dumps(config)+'\\n')\n"
        f" sys.exit({first_exit} if 'dinobase' in config else 0)\n"
        f"os.execv({sys.executable!r},[{sys.executable!r},*args])\n"
    )
    wrapper.chmod(0o755)
    env = dict(os.environ, QUEUE_TEST_EVENTS=str(tmp_path / "events.jsonl"))
    args = ["bash", str(script), "--python", str(wrapper), "--poll-seconds", "0.05",
            "--log-dir", str(tmp_path / "logs")]
    return args, env


def test_queue_waits_without_touching_current_job_then_runs_in_order(tmp_path):
    args, env = prepare_queue(tmp_path)
    current = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    queue = None
    try:
        queue = subprocess.Popen(args + ["--wait-pid", str(current.pid)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status = tmp_path / "logs/status.txt"
        deadline = time.monotonic() + 10
        while not status.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert status.exists()
        assert status.read_text().startswith("waiting")
        assert current.poll() is None
        assert not (tmp_path / "events.jsonl").exists()
        current.terminate()
        current.wait(timeout=5)
        output, _ = queue.communicate(timeout=10)
        assert queue.returncode == 0, output
        events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert events == ["baseline_train/train_dp_dinobase_insert_usb",
                          "baseline_train/train_dp_selfattn_insert_usb"]
        assert status.read_text().strip() == "completed"
    finally:
        if current.poll() is None:
            current.terminate()
            current.wait(timeout=5)
        if queue is not None and queue.poll() is None:
            queue.terminate()
            queue.wait(timeout=5)


def test_queue_records_failure_and_still_attempts_second_baseline(tmp_path):
    args, env = prepare_queue(tmp_path, first_exit=7)
    result = subprocess.run(args + ["--wait-pid", "0"], env=env, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 1, result.stdout + result.stderr
    events = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(events) == 2
    exits = (tmp_path / "logs/exit_codes.tsv").read_text()
    assert "train_dp_dinobase_insert_usb\t7" in exits
    assert "train_dp_selfattn_insert_usb\t0" in exits
    assert (tmp_path / "logs/status.txt").read_text().strip() == "completed_with_errors"
