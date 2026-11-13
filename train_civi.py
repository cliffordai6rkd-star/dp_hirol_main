"""Train a CiVi teacher, then distill the lowest-MSE or explicitly specified checkpoint.

python train_civi.py --teacher-config diffusion_policy/config/train_civi_teacher.yaml \
    --student-config diffusion_policy/config/train_distill_render_net_student.yaml
python train_civi.py --stage student --teacher-config <teacher.yaml> --student-config <student.yaml>
python train_civi.py --stage student --teacher-checkpoint <teacher.ckpt>
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hydra.errors import HydraException
from omegaconf import OmegaConf

from diffusion_policy.common.checkpoint_action_mse import ACTION_MSE_KEY, select_action_mse_teacher
from diffusion_policy.common.config_cli import configured_output_dir, load_config_reference
from train_pipeline import REPO_ROOT, resolve_config, run_stage, save_plan


DEFAULT_TEACHER_CONFIG = "diffusion_policy/config/train_civi_teacher.yaml"
DEFAULT_STUDENT_CONFIG = "diffusion_policy/config/train_distill_render_net_student.yaml"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-config", default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--student-config", "--student", dest="student_config", default=DEFAULT_STUDENT_CONFIG)
    parser.add_argument("--stage", choices=("all", "teacher", "student"), default="all")
    parser.add_argument("--teacher-checkpoint", help="Exact teacher checkpoint for the student, instead of MSE selection")
    parser.add_argument("--teacher-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--student-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without starting training or writing files")
    return parser


def build_plan(args, repo_root=REPO_ROOT):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch train_civi.py once with --nproc-per-node, without wrapping it in torchrun")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    if args.stage == "teacher" and args.teacher_checkpoint:
        raise ValueError("--teacher-checkpoint requires --stage student or --stage all")
    if args.gpus is not None:
        devices = args.gpus.split(",")
        if not all(device.strip() for device in devices) or len(devices) < args.nproc_per_node:
            raise ValueError("--gpus must list at least --nproc-per-node devices")
    for override in [*args.teacher_override, *args.student_override]:
        key = override.split("=", 1)[0].lstrip("+~")
        if "=" not in override or override.startswith("-"):
            raise ValueError(f"Expected a Hydra KEY=VALUE override: {override!r}")
        if key.startswith("hydra.") or key in {
            "teacher_config", "teacher_checkpoint", "teacher_output_dir", "teacher_config_overrides",
        }:
            raise ValueError(f"{key} is managed by train_civi.py; use the config paths, --teacher-checkpoint and output_dir instead")
    teacher_path = resolve_config(args.teacher_config, repo_root)
    teacher_cfg = load_config_reference(str(teacher_path), args.teacher_override, include_hydra=True)
    dataset_cfg = OmegaConf.select(teacher_cfg, "task.dataset")
    if (dataset_cfg is not None and "dataset_path" in dataset_cfg
            and (not dataset_cfg.dataset_path or not str(dataset_cfg.dataset_path).strip())):
        raise ValueError(
            "Set the training data path first: export CIVI_DATASET_PATH=/path/to/lerobot_dataset "
            "(or pass --teacher-override dataset_path=...)"
        )
    teacher_output = configured_output_dir(teacher_cfg)
    requested_checkpoint = args.teacher_checkpoint
    stages = {}
    if args.stage in {"all", "teacher"}:
        if not OmegaConf.select(teacher_cfg, "checkpoint.action_mse.enabled", default=False):
            raise ValueError("Teacher config must enable checkpoint.action_mse.enabled")
        if (OmegaConf.select(teacher_cfg, "checkpoint.topk.monitor_key") != ACTION_MSE_KEY
                or teacher_cfg.checkpoint.topk.mode != "min" or teacher_cfg.checkpoint.topk.k < 1):
            raise ValueError(f"Teacher checkpoint.topk must retain the minimum {ACTION_MSE_KEY} (k>=1)")
        stages["teacher"] = {
            "script": "sim_force_aware_dp_train.py", "config": str(teacher_path),
            "overrides": args.teacher_override, "output_dir": str(teacher_output),
            "resume": bool(teacher_cfg.training.get("resume", False)),
        }
    if args.stage in {"all", "student"}:
        student_path = resolve_config(args.student_config, repo_root)
        student_cfg = load_config_reference(str(student_path), args.student_override, include_hydra=True)
        requested_checkpoint = requested_checkpoint or student_cfg.get("teacher_checkpoint")
        student_output = configured_output_dir(student_cfg)
        if student_output == teacher_output:
            raise ValueError("Teacher and student output_dir must differ")
        stages["student"] = {
            "script": "train_distill_render_net.py", "config": str(student_path),
            "overrides": args.student_override, "output_dir": str(student_output),
            "resume": bool(student_cfg.training.get("resume_from")),
        }
    for stage in stages.values():
        if Path(stage["output_dir"]).exists() and not stage["resume"]:
            raise FileExistsError(f"Output already exists; change output_dir in the config: {stage['output_dir']}")
    checkpoint = score = None
    if requested_checkpoint:
        checkpoint = Path(str(requested_checkpoint)).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")
    elif args.stage == "student":
        checkpoint, score = select_action_mse_teacher(teacher_output)
    return {
        "status": "planned", "stage": args.stage, "stages": stages,
        "output_dir": next(iter(stages.values()))["output_dir"],
        "teacher_config": str(teacher_path), "teacher_config_overrides": args.teacher_override,
        "teacher_output_dir": str(teacher_output),
        "teacher_checkpoint": str(checkpoint) if checkpoint else None, "teacher_score": score,
        "teacher_selection": "explicit" if requested_checkpoint else "action_mse",
        "gpus": args.gpus, "nproc_per_node": args.nproc_per_node,
    }


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
    if name == "student":
        command += [
            "++teacher_config=" + json.dumps(plan["teacher_config"]),
            "++teacher_config_overrides=" + json.dumps(plan["teacher_config_overrides"]),
            "++teacher_output_dir=" + json.dumps(plan["teacher_output_dir"]),
        ]
        if plan["teacher_selection"] == "explicit":
            command += ["++teacher_checkpoint=" + json.dumps(plan["teacher_checkpoint"])]
    command += ["++output_dir=" + json.dumps(stage["output_dir"]),
                "hydra.run.dir=" + json.dumps(stage["output_dir"]), "hydra.job.chdir=false"]
    return command


def run_civi(args, repo_root=REPO_ROOT):
    plan = build_plan(args, repo_root)
    for name, stage in plan["stages"].items():
        print(f"{name.capitalize()} output: {stage['output_dir']}", flush=True)
    if args.dry_run:
        for name, stage in plan["stages"].items():
            print(f"[{name}] " + shlex.join(stage_command(name, stage, plan, repo_root)))
        if "student" in plan["stages"]:
            if plan["teacher_selection"] == "explicit":
                print(f"Student uses specified teacher checkpoint: {plan['teacher_checkpoint']}")
            else:
                print(f"Student selects minimum {ACTION_MSE_KEY} from {plan['teacher_output_dir']}/checkpoints/action_mse.json")
        return plan
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    plan["status"] = "running"
    Path(plan["output_dir"]).mkdir(parents=True, exist_ok=True)
    save_plan(plan)
    try:
        for name, stage in plan["stages"].items():
            if name == "student":
                if plan["teacher_selection"] == "explicit":
                    print(f"Using specified teacher checkpoint: {plan['teacher_checkpoint']}", flush=True)
                else:
                    checkpoint, score = select_action_mse_teacher(plan["teacher_output_dir"])
                    plan["teacher_checkpoint"], plan["teacher_score"] = str(checkpoint), score
                    print(f"Selected teacher: {checkpoint} (action MSE={score[ACTION_MSE_KEY]:.8f})", flush=True)
            stage["command"] = stage_command(name, stage, plan, repo_root)
            stage["status"] = "running"
            save_plan(plan)
            # Output existence/resume was checked before either stage started.
            run_stage(stage["command"], stage["output_dir"], env, repo_root, allow_existing=True)
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
    os.chdir(REPO_ROOT)
    parser = make_parser()
    args = parser.parse_args()
    try:
        run_civi(args)
    except (ValueError, OSError, HydraException) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode if exc.returncode > 0 else 1, "Training stage failed; see train.log and pipeline.json.\n")
    except KeyboardInterrupt:
        parser.exit(130, "CiVi training interrupted; child training processes stopped.\n")


if __name__ == "__main__":
    main()
