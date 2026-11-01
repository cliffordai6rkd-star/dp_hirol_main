"""One entry point for DP training, RenderNet distillation, or both in sequence.

    python train_pipeline.py dp
    python train_pipeline.py distill --teacher-run data/outputs/<teacher-run>
    python train_pipeline.py all --dry-run

Stage-specific Hydra overrides use repeatable --dp-override/--distill-override.
Launch this entry point once; --nproc-per-node creates torchrun workers.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DP_CONFIG = "diffusion_policy/config/sim_force_aware_dp_train_20hz_lstm2.yaml"
DEFAULT_DISTILL_CONFIG = "diffusion_policy/config/train_distill_render_net.yaml"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("dp", "distill", "all"), help="Run DP, distillation, or DP followed by distillation")
    parser.add_argument("--dp-config", default=DEFAULT_DP_CONFIG)
    parser.add_argument("--distill-config", default=DEFAULT_DISTILL_CONFIG)
    parser.add_argument("--dp-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--distill-override", action="append", default=[], metavar="KEY=VALUE")
    teacher = parser.add_mutually_exclusive_group()
    teacher.add_argument("--teacher-checkpoint", help="Exact teacher checkpoint for distill mode")
    teacher.add_argument("--teacher-run", help="Teacher run directory containing checkpoints/ (distill mode)")
    parser.add_argument("--teacher-selection", choices=("auto", "best", "latest"), default="auto",
                        help="auto selects best.ckpt if present, otherwise latest.ckpt")
    parser.add_argument("--output-dir", help="New pipeline directory; stages write dp/ and distill/ beneath it")
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2,3")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Number of local GPU workers per stage")
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
    for overrides in (args.dp_override, args.distill_override):
        validate_overrides(overrides)
    teacher = None
    if args.stage == "distill":
        if args.teacher_checkpoint:
            teacher = Path(args.teacher_checkpoint).expanduser().resolve()
            if not teacher.is_file():
                raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher}")
        elif args.teacher_run:
            teacher = select_teacher(args.teacher_run, args.teacher_selection)
        else:
            raise ValueError("distill mode requires --teacher-checkpoint or --teacher-run")
    now = datetime.datetime.now().astimezone()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        repo_root / "data/outputs" / now.strftime("%Y.%m.%d") / now.strftime("%H.%M.%S_%f_dp_distill_pipeline")
    ).resolve()
    if output.exists():
        raise FileExistsError(f"Pipeline output already exists; choose a new --output-dir: {output}")
    stages = {}
    if args.stage in {"dp", "all"}:
        stages["dp"] = {"script": "sim_force_aware_dp_train.py", "config": str(resolve_config(args.dp_config, repo_root)),
                        "overrides": args.dp_override, "output_dir": str(output / "dp")}
    if args.stage in {"distill", "all"}:
        stages["distill"] = {"script": "train_distill_render_net.py", "config": str(resolve_config(args.distill_config, repo_root)),
                             "overrides": args.distill_override, "output_dir": str(output / "distill")}
    return {"status": "planned", "stage": args.stage, "output_dir": str(output), "stages": stages,
            "teacher_checkpoint": str(teacher) if teacher else None, "teacher_selection": args.teacher_selection,
            "nproc_per_node": args.nproc_per_node, "gpus": args.gpus, "created_at": now.isoformat()}


def stage_command(name, stage, plan, repo_root=REPO_ROOT):
    script = str(repo_root / stage["script"])
    if plan["nproc_per_node"] > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={plan['nproc_per_node']}", script]
    else:
        command = [sys.executable, "-u", script]
    command += ["--config", stage["config"], *stage["overrides"]]
    if plan["nproc_per_node"] > 1:
        command += ["++training.distributed.enabled=true", "++training.distributed.timeout_seconds=86400"]
    if name == "distill":
        teacher = plan["teacher_checkpoint"] or f"<auto-selected after DP: {Path(plan['output_dir']) / 'dp/checkpoints'}>"
        # JSON quotes are interpreted by Hydra, preserving spaces and '=' in paths.
        command += ["teacher_checkpoint=" + json.dumps(teacher)]
    command += ["hydra.run.dir=" + json.dumps(stage["output_dir"]), "hydra.job.chdir=false"]
    return command


def run_stage(command, directory, env, repo_root=REPO_ROOT):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    print("Running: " + shlex.join(command), flush=True)
    with (directory / "train.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repo_root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
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
    if code:
        raise subprocess.CalledProcessError(code, command)


def save_plan(plan):
    path = Path(plan["output_dir"]) / "pipeline.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n")
    temporary.replace(path)


def run_pipeline(args, repo_root=REPO_ROOT):
    plan = build_plan(args, repo_root)
    print(f"Pipeline output: {plan['output_dir']}", flush=True)
    if args.dry_run:
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
            if name == "distill" and plan["stage"] == "all":
                plan["teacher_checkpoint"] = str(select_teacher(
                    plan["stages"]["dp"]["output_dir"], plan["teacher_selection"]))
            stage["command"] = stage_command(name, stage, plan, repo_root)
            stage["status"] = "running"
            save_plan(plan)
            run_stage(stage["command"], stage["output_dir"], env, repo_root)
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
