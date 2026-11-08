import logging
from pathlib import Path

import numpy as np
import pinocchio as pin


log = logging.getLogger(__name__)


class PinocchioIKController:
    def __init__(self, config):
        urdf_path = config.get("ik_urdf_path", "sim_mesh/franka_fr3/fr3_pika_gripper_ati.urdf")
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.ee_frame_name = config.get("ik_ee_frame_name", "pika_gripper_ee")
        self.lock_joint_names = config.get("ik_lock_joint_names", [])
        self.arm_joint_names = config.get(
            "ik_arm_joint_names",
            [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
        )
        self.gripper_joint_names = config.get(
            "ik_gripper_joint_names",
            ["gripper_left_joint", "gripper_right_joint"],
        )

        self.max_iterations = int(config.get("ik_max_iterations", 80))
        self.tolerance = float(config.get("ik_tolerance", 1e-4))
        self.damping = float(config.get("ik_damping", 1e-3))
        self.step_size = float(config.get("ik_step_size", 0.5))
        self.pos_weight = float(config.get("ik_position_weight", 1.0))
        self.ori_weight = float(config.get("ik_orientation_weight", 0.5))
        self.joint_limit_margin = float(config.get("ik_joint_limit_margin", 1e-4))
        self.quat_order = config.get("teleop_pose_quat_order", "wxyz")
        self.reject_unconverged = bool(config.get("ik_reject_unconverged", False))
        self.max_position_error = float(config.get("ik_max_position_error", 0.03))
        self.max_orientation_error = float(config.get("ik_max_orientation_error", 0.5))
        self.max_joint_delta = float(config.get("ik_max_joint_delta", 0.12))
        self.max_joint_delta_norm = float(config.get("ik_max_joint_delta_norm", 0.35))

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        locked_joint_ids = [
            full_model.getJointId(name)
            for name in self.lock_joint_names
            if full_model.existJointName(name)
        ]
        if locked_joint_ids:
            locked_reference = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, locked_joint_ids, locked_reference)
        else:
            self.model = full_model
        self.data = self.model.createData()

        self.frame_id = self.model.getFrameId(self.ee_frame_name)
        if self.frame_id == len(self.model.frames):
            raise ValueError(f"IK frame not found in URDF: {self.ee_frame_name}")

        self.lower_limits = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper_limits = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.arm_q_indices = self._joint_q_indices(self.arm_joint_names, required=True)
        self.gripper_q_indices = self._joint_q_indices(self.gripper_joint_names, required=False)

        self.q_nominal = np.asarray(pin.neutral(self.model), dtype=np.float64).reshape(-1)
        nominal_arm_q = config.get("ik_nominal_arm_q", config.get("q_reset", None))
        if nominal_arm_q is not None:
            self._set_indexed_q(self.q_nominal, self.arm_q_indices, nominal_arm_q, "ik_nominal_arm_q")

        nominal_gripper_q = config.get(
            "ik_default_gripper_q",
            config.get("q_reset_gripper", [-0.04, 0.04]),
        )
        if self.gripper_q_indices and nominal_gripper_q is not None:
            self._set_indexed_q(
                self.q_nominal,
                self.gripper_q_indices,
                nominal_gripper_q,
                "ik_default_gripper_q",
            )

        self.q_nominal = self._clip_to_limits(self.q_nominal)

        log.info(
            f"loaded IK model from {self.urdf_path}, frame={self.ee_frame_name}, "
            f"nq={self.model.nq}, nv={self.model.nv}, locked_joints={self.lock_joint_names}"
        )

    def _joint_q_indices(self, joint_names, required):
        indices = []
        for joint_name in joint_names:
            if not self.model.existJointName(joint_name):
                if required:
                    raise ValueError(f"joint not found in IK model: {joint_name}")
                continue

            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"expected 1-DoF joint for {joint_name}, got nq={joint.nq}")

            indices.append(joint.idx_q)

        return indices

    def _set_indexed_q(self, q, indices, values, name):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(indices):
            raise ValueError(f"{name} length {values.shape[0]} != expected {len(indices)}")

        for value, idx in zip(values, indices):
            q[idx] = value

    def _clip_to_limits(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        lower = self.lower_limits + self.joint_limit_margin
        upper = self.upper_limits - self.joint_limit_margin
        finite = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
        q[finite] = np.clip(q[finite], lower[finite], upper[finite])
        return q

    def compose_q(self, arm_q=None, gripper_q=None, base_q=None):
        if base_q is None:
            q = self.q_nominal.copy()
        else:
            q = np.asarray(base_q, dtype=np.float64).reshape(-1).copy()
            if q.shape[0] != self.model.nq:
                raise ValueError(f"base_q length {q.shape[0]} != IK model nq {self.model.nq}")

        if arm_q is not None:
            self._set_indexed_q(q, self.arm_q_indices, arm_q, "arm_q")
        if gripper_q is not None and self.gripper_q_indices:
            self._set_indexed_q(q, self.gripper_q_indices, gripper_q, "gripper_q")

        return self._clip_to_limits(q)

    def extract_arm_q(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"q length {q.shape[0]} != IK model nq {self.model.nq}")
        return q[self.arm_q_indices].copy()

    def extract_gripper_q(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"q length {q.shape[0]} != IK model nq {self.model.nq}")
        return q[self.gripper_q_indices].copy()

    def check_solution(self, q_candidate, q_reference=None, info=None):
        q_candidate = np.asarray(q_candidate, dtype=np.float64).reshape(-1)
        reasons = []

        if q_candidate.shape[0] != self.model.nq:
            reasons.append(f"q length {q_candidate.shape[0]} != IK model nq {self.model.nq}")
            return False, reasons

        if not np.all(np.isfinite(q_candidate)):
            reasons.append("q contains NaN or Inf")

        if info is not None:
            if self.reject_unconverged and not info.get("converged", False):
                reasons.append("IK did not converge")

            position_error = float(info.get("position_error_norm", 0.0))
            orientation_error = float(info.get("orientation_error_norm", 0.0))
            if position_error > self.max_position_error:
                reasons.append(
                    f"position error {position_error:.4f} > {self.max_position_error:.4f}"
                )
            if orientation_error > self.max_orientation_error:
                reasons.append(
                    f"orientation error {orientation_error:.4f} > {self.max_orientation_error:.4f}"
                )

        if q_reference is not None:
            q_reference = self._as_q(q_reference)
            arm_delta = self.extract_arm_q(q_candidate) - self.extract_arm_q(q_reference)
            max_abs_delta = float(np.max(np.abs(arm_delta)))
            delta_norm = float(np.linalg.norm(arm_delta))

            if max_abs_delta > self.max_joint_delta:
                reasons.append(
                    f"max joint delta {max_abs_delta:.4f} > {self.max_joint_delta:.4f}"
                )
            if delta_norm > self.max_joint_delta_norm:
                reasons.append(
                    f"joint delta norm {delta_norm:.4f} > {self.max_joint_delta_norm:.4f}"
                )

        return len(reasons) == 0, reasons

    def _as_q(self, q):
        if q is None:
            return self.q_nominal.copy()

        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] == self.model.nq:
            return self._clip_to_limits(q)

        arm_nq = len(self.arm_q_indices)
        gripper_nq = len(self.gripper_q_indices)
        if q.shape[0] == arm_nq:
            return self.compose_q(arm_q=q)
        if gripper_nq and q.shape[0] == arm_nq + gripper_nq:
            return self.compose_q(arm_q=q[:arm_nq], gripper_q=q[arm_nq:])

        raise ValueError(
            f"q_seed length {q.shape[0]} must be arm nq {arm_nq}, "
            f"arm+gripper nq {arm_nq + gripper_nq}, or IK model nq {self.model.nq}"
        )

    def _quat_to_rotation(self, quat, quat_order=None):
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        order = quat_order or self.quat_order

        if order == "wxyz":
            w, x, y, z = quat
        elif order == "xyzw":
            x, y, z, w = quat
        else:
            raise ValueError(f"unsupported quat order: {order}")

        quat_pin = pin.Quaternion(w, x, y, z)
        quat_pin.normalize()
        return quat_pin.toRotationMatrix()

    def pose_to_se3(self, position, quaternion=None, rotation=None, quat_order=None):
        position = np.asarray(position, dtype=np.float64).reshape(3)

        if rotation is not None:
            rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        elif quaternion is not None:
            rotation = self._quat_to_rotation(quaternion, quat_order=quat_order)
        else:
            raise ValueError("either quaternion or rotation must be provided")

        return pin.SE3(rotation, position)

    def forward_kinematics(self, q):
        q = self._as_q(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame_id].copy()

    def solve_pose(
        self,
        position,
        quaternion=None,
        rotation=None,
        q_seed=None,
        quat_order=None,
        return_info=False,
    ):
        target_pose = self.pose_to_se3(
            position,
            quaternion=quaternion,
            rotation=rotation,
            quat_order=quat_order,
        )
        return self.solve_se3(target_pose, q_seed=q_seed, return_info=return_info)

    def solve_matrix(self, target_matrix, q_seed=None, return_info=False):
        target_matrix = np.asarray(target_matrix, dtype=np.float64).reshape(4, 4)
        target_pose = pin.SE3(
            target_matrix[:3, :3],
            target_matrix[:3, 3],
        )
        return self.solve_se3(target_pose, q_seed=q_seed, return_info=return_info)

    def solve_se3(self, target_pose, q_seed=None, return_info=False):
        q = self._as_q(q_seed)
        fixed_gripper_q = self.extract_gripper_q(q) if self.gripper_q_indices else None
        weights = np.array(
            [
                self.pos_weight,
                self.pos_weight,
                self.pos_weight,
                self.ori_weight,
                self.ori_weight,
                self.ori_weight,
            ],
            dtype=np.float64,
        )

        converged = False
        err = np.zeros(6, dtype=np.float64)

        for iteration in range(self.max_iterations):
            pin.computeJointJacobians(self.model, self.data, q)
            pin.framesForwardKinematics(self.model, self.data, q)

            current_pose = self.data.oMf[self.frame_id]
            current_to_target = current_pose.actInv(target_pose)
            err = pin.log(current_to_target).vector
            weighted_err = weights * err

            if np.linalg.norm(weighted_err) < self.tolerance:
                converged = True
                break

            jacobian = pin.getFrameJacobian(
                self.model,
                self.data,
                self.frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            weighted_jacobian = weights[:, None] * jacobian
            lhs = weighted_jacobian @ weighted_jacobian.T
            lhs += (self.damping ** 2) * np.eye(6)
            dq = weighted_jacobian.T @ np.linalg.solve(lhs, weighted_err)

            q = pin.integrate(self.model, q, self.step_size * dq)
            if fixed_gripper_q is not None:
                self._set_indexed_q(q, self.gripper_q_indices, fixed_gripper_q, "fixed_gripper_q")
            q = self._clip_to_limits(q)

        info = {
            "converged": converged,
            "iterations": iteration + 1,
            "error_norm": float(np.linalg.norm(weights * err)),
            "position_error_norm": float(np.linalg.norm(err[:3])),
            "orientation_error_norm": float(np.linalg.norm(err[3:])),
        }

        if return_info:
            return q, info
        return q
