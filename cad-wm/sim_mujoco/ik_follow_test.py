import logging
import math

import yaml

from sim_mujoco.ik_controller import PinocchioIKController
from sim_mujoco.mujocosim_inteface import MujocoSim_interface_fr3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    with open("config/sim_cfg/replay_test.yaml", "r") as f:
        config = yaml.safe_load(f)

    sim = MujocoSim_interface_fr3(config)
    sim.load_model()
    sim.print_model_info()
    sim.save_compiled_mjcf()

    ik = PinocchioIKController(config)

    q_home = [float(value) for value in config["q_reset"]]
    sim.reset_arm_to_q(q_home)

    home_pose = ik.forward_kinematics(q_home)
    base_translation = home_pose.translation.copy()
    base_rotation = home_pose.rotation.copy()

    axis = config.get("ik_follow_test_axis", "z")
    amplitude = float(config.get("ik_follow_test_amplitude", 0.03))
    frequency = float(config.get("ik_follow_test_frequency", 0.15))
    reset_joint_speed = float(config.get("teleop_reset_joint_speed", 0.8))
    reset_position_tolerance = float(config.get("teleop_reset_position_tolerance", 0.01))
    control_dt = float(config.get("control_dt", 1.0 / float(config.get("control_frequency", 100.0))))
    reset_motion = {"active": False, "q_cmd": None}

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError(f"unsupported ik_follow_test_axis: {axis}")
    axis_idx = axis_to_index[axis]

    def target_fn(t, q_current):
        if reset_motion["active"]:
            q_target = [float(value) for value in config["q_reset"]]
            if reset_motion["q_cmd"] is None:
                reset_motion["q_cmd"] = q_current

            max_step = reset_joint_speed * control_dt
            q_cmd = []
            for current, target in zip(reset_motion["q_cmd"], q_target):
                delta = max(-max_step, min(max_step, target - current))
                q_cmd.append(current + delta)

            reset_motion["q_cmd"] = q_cmd
            if max(abs(target - current) for current, target in zip(q_cmd, q_target)) <= reset_position_tolerance:
                reset_motion["active"] = False
                reset_motion["q_cmd"] = None
                log.info("finished smooth reset motion to q_reset")

            return q_cmd

        target_translation = base_translation.copy()
        target_translation[axis_idx] += amplitude * math.sin(2.0 * math.pi * frequency * t)

        q_des, info = ik.solve_pose(
            target_translation,
            rotation=base_rotation,
            q_seed=q_current,
            return_info=True,
        )

        if not info["converged"]:
            log.debug(
                f"IK not fully converged: error={info['error_norm']:.6f}, "
                f"iters={info['iterations']}"
            )

        solution_ok, reject_reasons = ik.check_solution(
            q_des,
            q_reference=q_current,
            info=info,
        )
        if not solution_ok:
            log.warning(f"rejected IK solution: {'; '.join(reject_reasons)}")
            return q_current

        return ik.extract_arm_q(q_des)

    def reset_fn():
        reset_motion["active"] = True
        reset_motion["q_cmd"] = None
        log.info("started smooth reset motion to q_reset")

    log.info(
        f"running IK follow test: axis={axis}, amplitude={amplitude}, frequency={frequency}"
    )
    sim.run_joint_position_control(target_fn, reset_fn=reset_fn)


if __name__ == "__main__":
    main()
