import mujoco
import mujoco.viewer
import logging
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()




class MujocoSim_interface_fr3:
    def __init__(self,config):
        model_path = config.get("model_path", "sim_mesh/franka_fr3/fr3_pika_gripper_ati.urdf")
        self.model_path = _project_path(model_path)
        output_path = config.get("output_path", None)
        self.output_path = (
            _project_path(output_path)
            if output_path is not None
            else None
        )
        self.print_info = config.get("print_info", True)
        self.save_mjcf = config.get("save_mjcf", False)
        self.show_sim = config.get("show_sim", True)
        self.pause_between_episodes = config.get("pause_between_episodes", True)
        
        self.model = None
        self.data = None

        self.world_frame_size = config.get("world_frame_size", 0.2)
        self.quick_replay = config.get("quick_replay", True)
        self.sim_frequency = config.get("sim_frequency", None)
        self.sim_dt = config.get("sim_dt", None)
        self.control_frequency = float(config.get("control_frequency", 100.0))
        self.control_dt = float(config.get("control_dt", 1.0 / self.control_frequency))
        if self.control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {self.control_dt}")

        self.arm_joint_names = config.get("arm_joint_names", [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ])

        self.arm_actuator_names = config.get("arm_actuator_names", [
            "fr3_actuator1",
            "fr3_actuator2",
            "fr3_actuator3",
            "fr3_actuator4",
            "fr3_actuator5",
            "fr3_actuator6",
            "fr3_actuator7",
        ])
        self.gripper_actuator_name = config.get("gripper_actuator_name", "pika_gripper_actuator")
        self.gripper_ctrl_range = config.get("gripper_ctrl_range", [-0.11, 0.0])
        self.reset_hold_steps = int(config.get("teleop_reset_hold_steps", 20))

    # Load .urdf or .xml.
    def load_model(self):
        log.info(f"loading model from {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self._configure_sim_timestep()
        self.data = mujoco.MjData(self.model)
        return self.model, self.data

    def _configure_sim_timestep(self):
        if self.sim_dt is not None:
            sim_dt = float(self.sim_dt)
        elif self.sim_frequency is not None:
            sim_frequency = float(self.sim_frequency)
            if sim_frequency <= 0:
                raise ValueError(f"sim_frequency must be positive, got {sim_frequency}")
            sim_dt = 1.0 / sim_frequency
        else:
            sim_dt = float(self.model.opt.timestep)

        if sim_dt <= 0:
            raise ValueError(f"sim timestep must be positive, got {sim_dt}")

        self.model.opt.timestep = sim_dt
        log.info(
            f"sim_dt={sim_dt:.6f}s, sim_frequency={1.0 / sim_dt:.1f}Hz, "
            f"control_dt={self.control_dt:.6f}s, control_frequency={1.0 / self.control_dt:.1f}Hz"
        )

    def ensure_loaded(self):
        if self.model is None or self.data is None:
            self.load_model()

    def name_of(self, obj_type, idx):
        self.ensure_loaded()
        name = mujoco.mj_id2name(self.model, obj_type, idx)
        return name if name is not None else ""

    def print_model_info(self):
        if not self.print_info:
            return
        self.ensure_loaded()

        log.info(f"nq: {self.model.nq}")
        log.info(f"nv: {self.model.nv}")
        log.info(f"nu: {self.model.nu}")
        log.info(f"nbody: {self.model.nbody}")
        log.info(f"njnt: {self.model.njnt}")
        log.info(f"ngeom: {self.model.ngeom}")
        log.info(f"nsite: {self.model.nsite}")

        log.info("Bodies:")
        for i in range(self.model.nbody):
            log.info(f"{i} {self.name_of(mujoco.mjtObj.mjOBJ_BODY, i)}")

        log.info("Joints:")
        for i in range(self.model.njnt):
            log.info(f"{i} {self.name_of(mujoco.mjtObj.mjOBJ_JOINT, i)}")

        log.info("Sites:")
        for i in range(self.model.nsite):
            log.info(f"{i} {self.name_of(mujoco.mjtObj.mjOBJ_SITE, i)}")

    def save_compiled_mjcf(self):
        if not self.save_mjcf:
            return None

        self.ensure_loaded()
        path = self.output_path
        if path is None:
            path = self.model_path.with_suffix(".compiled.xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(path), self.model)
        self._fix_compiled_meshdir(path)
        log.info(f"saved compiled MJCF to {path}")
        return path

    def show_model(self):
        if not self.show_sim:
            return
    
        self.ensure_loaded()
    
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.sync()
            while viewer.is_running():
                viewer.sync()

    def actuator_ids(self, actuator_names):
        self.ensure_loaded()

        ids = []
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            if actuator_id < 0:
                raise ValueError(f"actuator not found: {name}")
            ids.append(actuator_id)
        
        return ids
            
    def command_joint_pos(self, q, actuator_names=None, step=True):
        self.ensure_loaded()

        if actuator_names is None:
            actuator_names = self.arm_actuator_names

        if len(q) != len(actuator_names):
            raise ValueError(f"q length {len(q)} != actuator_names length {len(actuator_names)}")
        
        actuator_ids = self.actuator_ids(actuator_names)

        for value, actuator_id in zip(q, actuator_ids):
            self.data.ctrl[actuator_id] = value
        
        if step:
            mujoco.mj_step(self.model, self.data)

    def command_gripper(self, value, actuator_name=None, step=False):
        self.ensure_loaded()

        if actuator_name is None:
            actuator_name = self.gripper_actuator_name

        actuator_id = self.actuator_ids([actuator_name])[0]
        low, high = self.gripper_ctrl_range
        value = min(max(float(value), float(low)), float(high))
        self.data.ctrl[actuator_id] = value

        if step:
            mujoco.mj_step(self.model, self.data)

    def set_joint_positions(self, q, joint_names=None):
        self.ensure_loaded()
        # log.info(f"joint command::{q}")
        if joint_names is None:
            joint_names = self.arm_joint_names
    
        if len(q) != len(joint_names):
            raise ValueError(f"q length {len(q)} != joint_names length {len(joint_names)}")
    
        for value, joint_name in zip(q, joint_names):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"joint not found: {joint_name}")
    
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = value

        mujoco.mj_forward(self.model, self.data)

    def reset_arm_to_q(self, q, joint_names=None, actuator_names=None):
        self.ensure_loaded()

        if joint_names is None:
            joint_names = self.arm_joint_names
        if actuator_names is None:
            actuator_names = self.arm_actuator_names

        if len(q) != len(joint_names):
            raise ValueError(f"q length {len(q)} != joint_names length {len(joint_names)}")
        if len(q) != len(actuator_names):
            raise ValueError(f"q length {len(q)} != actuator_names length {len(actuator_names)}")

        self.set_joint_positions(q, joint_names=joint_names)

        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"joint not found: {joint_name}")

            dof_addr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_addr] = 0.0

        actuator_ids = self.actuator_ids(actuator_names)
        for value, actuator_id in zip(q, actuator_ids):
            self.data.ctrl[actuator_id] = value

        mujoco.mj_forward(self.model, self.data)

    def get_arm_qpos(self, joint_names=None):
        self.ensure_loaded()

        if joint_names is None:
            joint_names = self.arm_joint_names

        q = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"joint not found: {joint_name}")

            qpos_addr = self.model.jnt_qposadr[joint_id]
            q.append(self.data.qpos[qpos_addr])

        return q

    def run_joint_position_control(self, target_fn, control_dt=None, real_time=True, reset_fn=None):
        self.ensure_loaded()
    
        sim_dt = float(self.model.opt.timestep)
        if control_dt is None:
            control_dt = self.control_dt
        else:
            control_dt = float(control_dt)

        if control_dt <= 0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")

        steps_per_control = max(1, int(round(control_dt / sim_dt)))
        actual_control_dt = steps_per_control * sim_dt

        log.info(
            f"running joint position control: sim_dt={sim_dt:.6f}s, "
            f"target_control_dt={control_dt:.6f}s, actual_control_dt={actual_control_dt:.6f}s, "
            f"steps_per_control={steps_per_control}"
        )
    
        t = 0.0
        reset_requested = {"value": False}

        def key_callback(keycode):
            if keycode in (79, ord("o"), ord("O")):
                reset_requested["value"] = True
                log.info("reset requested by O key")
    
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
        ) as viewer:
            while viewer.is_running():
                if reset_requested["value"]:
                    if reset_fn is not None:
                        reset_fn()
                    reset_requested["value"] = False
                    t = 0.0

                q_current = self.get_arm_qpos()
                q_des = target_fn(t, q_current)
    
                for _ in range(steps_per_control):
                    self.command_joint_pos(q_des)
    
                viewer.sync()
                if real_time:
                    time.sleep(actual_control_dt)
                t += actual_control_dt

    def play_joint_sequences(self, all_q_seqs, dt):
        self.ensure_loaded()

        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"replay dt must be positive, got {dt}")
        if not all_q_seqs:
            raise ValueError("no joint sequences to replay")

        if not self.show_sim:
            for ep_idx, q_seq in enumerate(all_q_seqs):
                log.info(f"playing episode {ep_idx + 1}/{len(all_q_seqs)} without viewer")
                for q in q_seq:
                    self.set_joint_positions(q)
                    if not self.quick_replay:
                        time.sleep(dt)
            return

        go_next = {"value": False}

        def key_callback(keycode):
            if keycode in (257, 335):
                go_next["value"] = True

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
        ) as viewer:
            for ep_idx, q_seq in enumerate(all_q_seqs):
                if not viewer.is_running():
                    break

                log.info(f"playing episode {ep_idx + 1}/{len(all_q_seqs)}")

                # 播放当前 episode
                for q in q_seq:
                    if not viewer.is_running():
                        break

                    self.set_joint_positions(q)
                    viewer.sync()

                    if self.quick_replay is False:
                        time.sleep(dt)

                is_last_episode = ep_idx == len(all_q_seqs) - 1
                if is_last_episode or not self.pause_between_episodes:
                    continue

                log.info("press Enter to continue")
                go_next["value"] = False
                while viewer.is_running() and not go_next["value"]:
                    viewer.sync()
                    time.sleep(0.02)





def main():
    import yaml

    with open("config/sim_cfg/replay_test.yaml", "r") as f:
        config = yaml.safe_load(f)
    viewer = MujocoSim_interface_fr3(config)
    viewer.load_model()
    log.info("MuJoCo compile OK")
    viewer.print_model_info()
    viewer.save_compiled_mjcf()

    q_home = [float(value) for value in config["q_reset"]]
    joint_idx = int(config.get("follow_test_joint_index", 6))
    amplitude = float(config.get("follow_test_amplitude", 0.3))
    frequency = float(config.get("follow_test_frequency", 0.25))
    if joint_idx < 0 or joint_idx >= len(q_home):
        raise ValueError(f"follow_test_joint_index {joint_idx} out of range for q length {len(q_home)}")

    viewer.reset_arm_to_q(q_home)

    def target_fn(t, q_current):
        q_des = q_home.copy()
        q_des[joint_idx] += amplitude * math.sin(2.0 * math.pi * frequency * t)
        return q_des

    viewer.run_joint_position_control(target_fn)


if __name__ == "__main__":
    main()
