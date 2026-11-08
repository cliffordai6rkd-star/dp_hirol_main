import argparse
import logging
import time

import numpy as np
import pinocchio as pin
import yaml

from sim_mujoco.ik_controller import PinocchioIKController
from sim_mujoco.mujocosim_inteface import MujocoSim_interface_fr3
from sim_mujoco.xbox_controller import LinuxXboxController


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def clamp_vector(value, lower, upper):
    return np.minimum(np.maximum(value, lower), upper)


def rotation_delta_from_roll_pitch(roll_delta, pitch_delta):
    roll = pin.rpy.rpyToMatrix(float(roll_delta), 0.0, 0.0)
    pitch = pin.rpy.rpyToMatrix(0.0, float(pitch_delta), 0.0)
    return roll @ pitch


class XboxTeleoperation:
    def __init__(self, config):
        self.config = config
        self.sim = MujocoSim_interface_fr3(config)
        self.ik = PinocchioIKController(config)
        self.xbox = LinuxXboxController(config)

        self.translation_speed = float(config.get("teleop_translation_speed", 0.15))
        self.rotation_speed = float(config.get("teleop_rotation_speed", 0.8))
        self.left_x_axis = int(config.get("teleop_left_x_index", 1))
        self.forward_axis = int(config.get("teleop_forward_index", 0))
        self.up_axis = int(config.get("teleop_up_index", 2))

        self.workspace_min = np.asarray(
            config.get("teleop_workspace_min", [0.15, -0.45, 0.08]),
            dtype=np.float64,
        )
        self.workspace_max = np.asarray(
            config.get("teleop_workspace_max", [0.75, 0.45, 0.85]),
            dtype=np.float64,
        )

        self.gripper_open_ctrl = float(config.get("teleop_gripper_open_ctrl", -0.11))
        self.gripper_closed_ctrl = float(config.get("teleop_gripper_closed_ctrl", 0.0))
        self.gripper_toggle_threshold = float(config.get("teleop_gripper_toggle_threshold", 0.5))
        self.gripper_initial_closed = bool(config.get("teleop_gripper_initial_closed", False))
        self.gripper_closed = self.gripper_initial_closed
        self.last_gripper_pressed = False
        self.reset_combo_threshold = float(config.get("teleop_reset_combo_threshold", 0.5))
        self.reset_requires_a = bool(config.get("teleop_reset_requires_a", False))
        self.reset_hold_steps = int(config.get("teleop_reset_hold_steps", 20))
        self.reset_joint_speed = float(config.get("teleop_reset_joint_speed", 0.8))
        self.reset_position_tolerance = float(config.get("teleop_reset_position_tolerance", 0.01))
        self.last_reset_combo_pressed = False

    def close(self):
        self.xbox.close()

    def _integrate_target_pose(self, target_pose, state, dt):
        translation = target_pose.translation.copy()

        lateral_delta = state.left_x * self.translation_speed * dt
        forward_delta = state.left_y * self.translation_speed * dt
        vertical_delta = (state.y - state.a) * self.translation_speed * dt

        translation[self.left_x_axis] += lateral_delta
        translation[self.forward_axis] += forward_delta
        translation[self.up_axis] += vertical_delta
        translation = clamp_vector(translation, self.workspace_min, self.workspace_max)

        roll_delta = state.right_x * self.rotation_speed * dt
        pitch_delta = state.right_y * self.rotation_speed * dt
        rotation = target_pose.rotation @ rotation_delta_from_roll_pitch(roll_delta, pitch_delta)

        return pin.SE3(rotation, translation)

    def _update_gripper_toggle(self, state):
        pressed = state.r2 >= self.gripper_toggle_threshold
        if pressed and not self.last_gripper_pressed:
            self.gripper_closed = not self.gripper_closed
            log.info(f"gripper toggled to {'closed' if self.gripper_closed else 'open'}")
        self.last_gripper_pressed = pressed

    def _reset_combo_pressed(self, state):
        modifier_pressed = (
            state.lt >= self.reset_combo_threshold
            or state.reset_modifier > 0.5
        )
        pressed = modifier_pressed and (not self.reset_requires_a or state.a > 0.5)
        triggered = pressed and not self.last_reset_combo_pressed
        self.last_reset_combo_pressed = pressed
        return triggered

    def _gripper_ctrl(self):
        if self.gripper_closed:
            return self.gripper_closed_ctrl
        return self.gripper_open_ctrl

    def _gripper_q(self):
        reset_gripper = self.config.get("q_reset_gripper", [-0.04, 0.04])
        open_q = np.asarray(
            self.config.get("teleop_gripper_open_q", reset_gripper),
            dtype=np.float64,
        )
        closed_q = np.asarray(
            self.config.get("teleop_gripper_closed_q", [0.0, 0.0]),
            dtype=np.float64,
        )
        if self.gripper_closed:
            return closed_q
        return open_q

    def run(self):
        self.sim.load_model()
        self.sim.print_model_info()
        self.sim.save_compiled_mjcf()

        q_home = [float(value) for value in self.config["q_reset"]]
        self.sim.reset_arm_to_q(q_home)

        target_pose = self.ik.forward_kinematics(q_home)
        control_dt = self.sim.control_dt
        sim_dt = float(self.sim.model.opt.timestep)
        steps_per_control = max(1, int(round(control_dt / sim_dt)))
        actual_control_dt = steps_per_control * sim_dt

        q_seed = self.ik.compose_q(
            arm_q=q_home,
            gripper_q=self._gripper_q(),
        )
        q_safe = q_seed.copy()
        reset_requested = {"value": False}
        reset_motion = {"active": False, "q_cmd": None}

        def start_reset_motion():
            nonlocal target_pose, q_seed, q_safe

            self.gripper_closed = self.gripper_initial_closed
            reset_motion["active"] = True
            reset_motion["q_cmd"] = np.asarray(self.sim.get_arm_qpos(), dtype=np.float64)
            self.sim.command_gripper(self._gripper_ctrl(), step=False)
            self.last_gripper_pressed = False
            reset_requested["value"] = False
            log.info("started smooth reset motion to q_reset")

        def step_reset_motion():
            nonlocal target_pose, q_seed, q_safe

            q_target = np.asarray(q_home, dtype=np.float64)
            q_cmd = np.asarray(reset_motion["q_cmd"], dtype=np.float64)
            max_step = self.reset_joint_speed * actual_control_dt
            delta = np.clip(q_target - q_cmd, -max_step, max_step)
            q_cmd = q_cmd + delta
            reset_motion["q_cmd"] = q_cmd

            self.sim.command_gripper(self._gripper_ctrl(), step=False)
            for _ in range(steps_per_control):
                self.sim.command_joint_pos(q_cmd)

            if np.max(np.abs(q_target - q_cmd)) <= self.reset_position_tolerance:
                reset_motion["active"] = False
                q_seed = self.ik.compose_q(
                    arm_q=q_home,
                    gripper_q=self._gripper_q(),
                )
                q_safe = q_seed.copy()
                target_pose = self.ik.forward_kinematics(q_home)
                self.last_reset_combo_pressed = False
                log.info("finished smooth reset motion to q_reset")

            return q_cmd

        def request_reset(keycode):
            if keycode in (79, ord("o"), ord("O")):
                reset_requested["value"] = True
                log.info("reset requested by O key")

        log.info(
            f"running Xbox teleoperation: control_dt={actual_control_dt:.6f}s, "
            f"steps_per_control={steps_per_control}"
        )

        try:
            with mujoco_viewer(self.sim, key_callback=request_reset) as viewer:
                while viewer.is_running():
                    loop_start = time.perf_counter()

                    state = self.xbox.poll()
                    if self._reset_combo_pressed(state):
                        reset_requested["value"] = True
                        log.info("reset requested by Xbox")

                    if reset_requested["value"]:
                        start_reset_motion()

                    if reset_motion["active"]:
                        step_reset_motion()
                        viewer.sync()
                        elapsed = time.perf_counter() - loop_start
                        sleep_time = actual_control_dt - elapsed
                        if sleep_time > 0.0:
                            time.sleep(sleep_time)
                        continue

                    self._update_gripper_toggle(state)

                    candidate_target_pose = self._integrate_target_pose(
                        target_pose,
                        state,
                        actual_control_dt,
                    )
                    q_full, info = self.ik.solve_se3(
                        candidate_target_pose,
                        q_seed=q_seed,
                        return_info=True,
                    )
                    solution_ok, reject_reasons = self.ik.check_solution(
                        q_full,
                        q_reference=q_seed,
                        info=info,
                    )
                    if solution_ok:
                        target_pose = candidate_target_pose
                        q_arm = self.ik.extract_arm_q(q_full)
                        q_safe = q_full
                    else:
                        q_full = q_safe
                        q_arm = self.ik.extract_arm_q(q_safe)
                        log.warning(f"rejected IK solution: {'; '.join(reject_reasons)}")

                    gripper_ctrl = self._gripper_ctrl()
                    gripper_q = self._gripper_q()
                    q_seed = self.ik.compose_q(
                        arm_q=q_arm,
                        gripper_q=gripper_q,
                        base_q=q_full,
                    )
                    self.sim.command_gripper(gripper_ctrl, step=False)

                    for _ in range(steps_per_control):
                        self.sim.command_joint_pos(q_arm)

                    viewer.sync()

                    elapsed = time.perf_counter() - loop_start
                    sleep_time = actual_control_dt - elapsed
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

                    if not info["converged"]:
                        log.debug(
                            f"IK not fully converged: error={info['error_norm']:.6f}, "
                            f"iters={info['iterations']}"
                        )
        finally:
            self.close()


class mujoco_viewer:
    def __init__(self, sim, key_callback=None):
        self.sim = sim
        self.key_callback = key_callback
        self.viewer = None

    def __enter__(self):
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(
            self.sim.model,
            self.sim.data,
            key_callback=self.key_callback,
        )
        return self.viewer

    def __exit__(self, exc_type, exc_value, traceback):
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Xbox teleoperation for MuJoCo FR3.")
    parser.add_argument("--config", default="config/sim_cfg/replay_test.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    teleop = XboxTeleoperation(config)
    teleop.run()


if __name__ == "__main__":
    main()
