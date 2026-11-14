"""Train an FDP teacher and distill a RenderDP student from one entry point.

    python train_pipeline.py --gpus 0
    python train_pipeline.py dp
    python train_pipeline.py distill --teacher-run data/outputs/<teacher-run>
    python train_pipeline.py --dry-run

The default runs both stages sequentially, writing teacher/ and student/ beneath
one pipeline output directory and handing the teacher checkpoint to the student.
Stage-specific Hydra overrides use repeatable --dp-override/--distill-override.
Launch this entry point once; --nproc-per-node creates torchrun workers.
"""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading

from omegaconf import OmegaConf

from diffusion_policy.common.config_cli import load_config_reference


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DP_CONFIG = "diffusion_policy/config/train_civi_teacher.yaml"
DEFAULT_DISTILL_CONFIG = "diffusion_policy/config/train_distill_render_net_student.yaml"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", nargs="?", default="all", choices=("dp", "distill", "all"),
                        help="dp: FDP teacher; distill: RenderDP student; all: both sequentially (default)")
    parser.add_argument("--dp-config", default=DEFAULT_DP_CONFIG)
    parser.add_argument("--distill-config", default=DEFAULT_DISTILL_CONFIG)
    parser.add_argument("--dp-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--distill-override", action="append", default=[], metavar="KEY=VALUE")
    teacher = parser.add_mutually_exclusive_group()
    teacher.add_argument("--teacher-checkpoint", help="Exact teacher checkpoint for distill mode")
    teacher.add_argument("--teacher-run", help="Teacher run directory containing checkpoints/ (distill mode)")
    parser.add_argument("--wait-for-teacher-pid", type=int,
                        help="Wait for this running teacher before selecting its checkpoint (distill + --teacher-run)")
    parser.add_argument("--teacher-selection", choices=("auto", "best", "latest"), default="auto",
                        help="auto selects best.ckpt if present, otherwise latest.ckpt")
    parser.add_argument("--output-dir", help="New pipeline directory; stages write teacher/ and student/ beneath it")
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Number of local GPU workers per stage")
    parser.add_argument("--oom-retries", type=int, default=0,
                        help="Maximum student retries after CUDA OOM, reducing batches and resuming latest")
    parser.add_argument("--monitor-interval", type=float, default=60,
                        help="Seconds between GPU memory/utilization samples")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating files or starting training")
    return parser


def resolve_config(reference, repo_root):
    path = Path(reference).expanduser()
    for candidate in (path, repo_root / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Config does not exist: {reference}")


def select_teacher(run_dir, selection="auto"):
    directory = Path(run_dir).expanduser().resolve() / "checkpoints"
    names = ("best", "latest") if selection == "auto" else (selection,)
    for name in names:
        checkpoint = directory / f"{name}.ckpt"
        if checkpoint.is_file():
            return checkpoint.resolve()
    raise FileNotFoundError(f"No {selection} teacher checkpoint found in {directory}")


def validate_overrides(overrides):
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if not override or override.startswith("-") or "=" not in override:
            raise ValueError(f"Expected a Hydra KEY=VALUE override, got: {override!r}")
        if key in {"teacher_checkpoint", "hydra", "hydra.run", "hydra.run.dir", "hydra.job.chdir"}:
            raise ValueError(f"Use --teacher-checkpoint/--teacher-run or --output-dir instead of {key}")


def build_plan(args, repo_root=REPO_ROOT):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch train_pipeline.py once with --nproc-per-node; do not wrap it in torchrun")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    if args.oom_retries < 0:
        raise ValueError("--oom-retries must be nonnegative")
    if not 0 < args.monitor_interval < float("inf"):
        raise ValueError("--monitor-interval must be positive and finite")
    if args.gpus is not None:
        devices = args.gpus.split(",")
        if not all(device.strip() for device in devices) or len(devices) < args.nproc_per_node:
            raise ValueError("--gpus must list at least --nproc-per-node devices")
    if args.stage != "distill" and (args.teacher_checkpoint or args.teacher_run):
        raise ValueError("--teacher-checkpoint/--teacher-run are for distill mode; all uses the DP stage's checkpoint")
    if args.stage == "dp" and args.distill_override:
        raise ValueError("--distill-override requires distill or all mode")
    if args.stage == "distill" and args.dp_override:
        raise ValueError("--dp-override requires dp or all mode")
    if args.wait_for_teacher_pid is not None and (args.stage != "distill" or not args.teacher_run):
        raise ValueError("--wait-for-teacher-pid requires distill mode with --teacher-run")
    for overrides in (args.dp_override, args.distill_override):
        validate_overrides(overrides)
    teacher = None
    teacher_wait = None
    if args.stage == "distill":
        if args.teacher_checkpoint:
            teacher = Path(args.teacher_checkpoint).expanduser().resolve()
            if not teacher.is_file():
                raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher}")
        elif args.teacher_run:
            if args.wait_for_teacher_pid is not None:
                from diffusion_policy.common.teacher_handoff import teacher_process_identity

                run_dir = Path(args.teacher_run).expanduser().resolve()
                if not run_dir.is_dir():
                    raise FileNotFoundError(f"Teacher run does not exist: {run_dir}")
                teacher_wait = {
                    "pid": args.wait_for_teacher_pid,
                    "start_time": teacher_process_identity(args.wait_for_teacher_pid),
                    "run_dir": str(run_dir),
                }
            else:
                teacher = select_teacher(args.teacher_run, args.teacher_selection)
        else:
            raise ValueError("distill mode requires --teacher-checkpoint or --teacher-run")
    teacher_config = None
    dataset_path = os.environ.get("CIVI_DATASET_PATH")
    if args.stage in {"dp", "all"}:
        teacher_config = resolve_config(args.dp_config, repo_root)
        teacher_cfg = load_config_reference(str(teacher_config), args.dp_override)
        dataset_cfg = OmegaConf.select(teacher_cfg, "task.dataset")
        if dataset_cfg is not None and "dataset_path" in dataset_cfg:
            dataset_path = dataset_cfg.dataset_path
            if not dataset_path or not str(dataset_path).strip():
                raise ValueError(
                    "Set the training data path first: export CIVI_DATASET_PATH=/path/to/lerobot_dataset "
                    "(or pass --dp-override dataset_path=...)"
                )
    # The data exports identify the task; sampling-rate and processing suffixes
    # describe the dataset version, rather than the experiment's task name.
    dataset_name = Path(str(dataset_path)).name if dataset_path else "task"
    task_name = os.environ.get("CIVI_TASK_NAME") or re.split(
        r"_\d+(?:\.\d+)?hz(?:_|$)", dataset_name, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    now = datetime.datetime.now().astimezone()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        repo_root / "data/outputs" / now.strftime("%Y.%m.%d")
        / (now.strftime("%H.%M.%S_%f_") + f"fdp_dis_{task_name}")
    ).resolve()
    if output.exists():
        raise FileExistsError(f"Pipeline output already exists; choose a new --output-dir: {output}")
    stages = {}
    if args.stage in {"dp", "all"}:
        stages["dp"] = {"script": "sim_force_aware_dp_train.py", "config": str(teacher_config),
                        "overrides": args.dp_override, "output_dir": str(output / "teacher")}
    if args.stage in {"distill", "all"}:
        stages["distill"] = {"script": "train_distill_render_net.py", "config": str(resolve_config(args.distill_config, repo_root)),
                             "overrides": args.distill_override, "output_dir": str(output / "student")}
    return {"status": "planned", "stage": args.stage, "output_dir": str(output), "stages": stages,
            "task_name": task_name, "dataset_path": str(dataset_path) if dataset_path else None,
            "teacher_checkpoint": str(teacher) if teacher else None, "teacher_selection": args.teacher_selection,
            "teacher_wait": teacher_wait,
            "oom_retries": args.oom_retries, "monitor_interval": args.monitor_interval,
            "nproc_per_node": args.nproc_per_node, "gpus": args.gpus, "created_at": now.isoformat()}


def stage_command(name, stage, plan, repo_root=REPO_ROOT):
    script = str(repo_root / stage["script"])
    if plan["nproc_per_node"] > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={plan['nproc_per_node']}", script]
    else:
        command = [sys.executable, "-u", script]
    command += ["--config", stage["config"]]
    if name == "distill":
        # This pipeline hands off a checkpoint whose saved config includes
        # the teacher's actual data windows and overrides.
        command += ["++teacher_config=null"]
    command += stage["overrides"]
    if plan["nproc_per_node"] > 1:
        command += ["++training.distributed.enabled=true", "++training.distributed.timeout_seconds=86400"]
    if name == "distill":
        teacher_run = (
            plan["teacher_wait"]["run_dir"] if plan.get("teacher_wait")
            else plan["stages"].get("dp", {}).get("output_dir", "")
        )
        teacher = plan["teacher_checkpoint"] or (
            f"<auto-selected after teacher training: {Path(teacher_run) / 'checkpoints'}>"
        )
        # JSON quotes are interpreted by Hydra, preserving spaces and '=' in paths.
        command += ["teacher_checkpoint=" + json.dumps(teacher)]
    command += ["hydra.run.dir=" + json.dumps(stage["output_dir"]), "hydra.job.chdir=false"]
    return command


def monitor_gpu_usage(directory, stop, interval, env, attempt):
    """Sample available GPUs until the stage ends; telemetry never stops training."""
    while not stop.is_set():
        record = {"time": datetime.datetime.now().astimezone().isoformat(), "attempt": attempt}
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], env=env, capture_output=True, text=True, timeout=5, check=True)
            record["gpus"] = [dict(zip(
                ("index", "name", "memory_used_mb", "memory_total_mb", "utilization_percent"),
                (value.strip() for value in row),
            )) for row in csv.reader(result.stdout.splitlines())]
        except (OSError, subprocess.SubprocessError) as exc:
            record["error"] = str(exc)
        try:
            with (Path(directory) / "gpu_usage.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
        except OSError:
            return
        stop.wait(interval)


def run_stage(command, directory, env, repo_root=REPO_ROOT, *, allow_existing=False,
              attempt=1, monitor_interval=60):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=allow_existing)
    print("Running: " + shlex.join(command), flush=True)
    with ((directory / "train.log").open("a" if allow_existing else "w") as log,
          (directory / f"attempt_{attempt:03d}.log").open("w") as attempt_log):
        process = subprocess.Popen(command, cwd=repo_root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        stop_monitor = threading.Event()
        monitor = threading.Thread(target=monitor_gpu_usage, args=(
            directory, stop_monitor, monitor_interval, env, attempt,
        ), daemon=True)
        monitor.start()
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                attempt_log.write(line)
                attempt_log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()
            stop_monitor.set()
            monitor.join(timeout=6)
    if code:
        raise subprocess.CalledProcessError(code, command)


def attempt_has_cuda_oom(path):
    pattern = re.compile(r"CUDA(?: error:)? out of memory|torch\.cuda\.OutOfMemoryError", re.IGNORECASE)
    with Path(path).open(errors="replace") as log:
        return any(pattern.search(line) for line in log)


def reduce_student_batches(stage, plan):
    """Build the next attempt's explicit overrides from its current config."""
    cfg = load_config_reference(stage["config"], stage["overrides"])
    world_size = int(plan["nproc_per_node"])
    global_batch = bool(OmegaConf.select(cfg, "training.distributed.batch_size_is_global", default=True))
    multiple = world_size if global_batch else 1
    previous = {}
    updates = {}
    for key in ("dataloader.batch_size", "val_dataloader.batch_size", "teacher_cache.batch_size"):
        old = int(OmegaConf.select(cfg, key, default=1))
        divisor = 1 if key.startswith("teacher_cache") else multiple
        previous[key] = old
        updates[key] = max(divisor, (old // 2 // divisor) * divisor)
    updates["training.compile.enabled"] = False
    latest = Path(stage["output_dir"]) / "checkpoints/latest.ckpt"
    if latest.is_file():
        updates["training.resume_from"] = str(latest.resolve())
    stage["overrides"] = [
        value for value in stage["overrides"]
        if value.split("=", 1)[0].lstrip("+~") not in updates
    ] + ["++" + key + "=" + json.dumps(value) for key, value in updates.items()]
    return {"previous_batches": previous, "overrides": updates}


def save_plan(plan):
    path = Path(plan["output_dir"]) / "pipeline.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n")
    temporary.replace(path)


def run_pipeline(args, repo_root=REPO_ROOT):
    plan = build_plan(args, repo_root)
    print(f"Pipeline output: {plan['output_dir']}", flush=True)
    if args.dry_run:
        if plan["teacher_wait"]:
            print(f"Wait for teacher PID {plan['teacher_wait']['pid']}: {plan['teacher_wait']['run_dir']}")
        for name, stage in plan["stages"].items():
            print(f"[{name}] " + shlex.join(stage_command(name, stage, plan, repo_root)))
        return plan
    Path(plan["output_dir"]).mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if plan["gpus"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = plan["gpus"]
    plan["status"] = "running"
    save_plan(plan)
    try:
        for name, stage in plan["stages"].items():
            if name == "distill" and plan["teacher_wait"]:
                from diffusion_policy.common.teacher_handoff import wait_for_teacher

                plan["status"] = stage["status"] = "waiting_for_teacher"
                save_plan(plan)
                print(f"Waiting for teacher PID {plan['teacher_wait']['pid']}: "
                      f"{plan['teacher_wait']['run_dir']}", flush=True)
                stop_monitor = threading.Event()
                monitor = threading.Thread(target=monitor_gpu_usage, args=(
                    plan["output_dir"], stop_monitor, plan["monitor_interval"], env, 0,
                ), daemon=True)
                monitor.start()
                try:
                    plan["teacher_completion"] = wait_for_teacher(**plan["teacher_wait"])
                finally:
                    stop_monitor.set()
                    monitor.join(timeout=6)
                plan["teacher_checkpoint"] = str(select_teacher(
                    plan["teacher_wait"]["run_dir"], plan["teacher_selection"]))
                print(f"Teacher training completed; selected {plan['teacher_checkpoint']}", flush=True)
            if name == "distill" and plan["stage"] == "all":
                plan["teacher_checkpoint"] = str(select_teacher(
                    plan["stages"]["dp"]["output_dir"], plan["teacher_selection"]))
            retry_limit = plan["oom_retries"] if name == "distill" else 0
            stage["attempts"] = []
            for attempt in range(1, retry_limit + 2):
                stage["command"] = stage_command(name, stage, plan, repo_root)
                plan["status"] = stage["status"] = "running"
                record = {"attempt": attempt, "status": "running", "command": stage["command"],
                          "log": str(Path(stage["output_dir"]) / f"attempt_{attempt:03d}.log"),
                          "started_at": datetime.datetime.now().astimezone().isoformat()}
                stage["attempts"].append(record)
                save_plan(plan)
                try:
                    run_stage(stage["command"], stage["output_dir"], env, repo_root,
                              allow_existing=attempt > 1, attempt=attempt,
                              monitor_interval=plan["monitor_interval"])
                except subprocess.CalledProcessError as exc:
                    record.update(status="failed", returncode=exc.returncode,
                                  cuda_oom=attempt_has_cuda_oom(record["log"]),
                                  ended_at=datetime.datetime.now().astimezone().isoformat())
                    if not record["cuda_oom"] or attempt > retry_limit:
                        raise
                    record["recovery"] = reduce_student_batches(stage, plan)
                    save_plan(plan)
                    print(f"Student attempt {attempt} hit CUDA OOM; retrying with "
                          f"{record['recovery']['overrides']}", flush=True)
                else:
                    record["status"] = "completed"
                    record["ended_at"] = datetime.datetime.now().astimezone().isoformat()
                    break
            stage["status"] = "completed"
            save_plan(plan)
        plan["status"] = "completed"
    except BaseException as exc:
        plan["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        stage["status"] = plan["status"]
        plan["error"] = str(exc)
        raise
    finally:
        save_plan(plan)
    return plan


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode if exc.returncode > 0 else 1, "Training stage failed; see train.log and pipeline.json.\n")
    except KeyboardInterrupt:
        parser.exit(130, "Pipeline interrupted; child training processes stopped.\n")


if __name__ == "__main__":
    main()
