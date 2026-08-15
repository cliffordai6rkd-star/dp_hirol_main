import torch


def normalize_quaternion_xyzw(
    quaternion: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize xyzw quaternions, using identity for degenerate inputs."""
    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"quaternion must have last dimension 4, got {tuple(quaternion.shape)}"
        )
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return torch.where(
        norm > eps,
        quaternion / norm.clamp_min(eps),
        identity,
    )


def quaternion_multiply_xyzw(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Hamilton product of broadcastable xyzw quaternions."""
    if lhs.shape[-1] != 4 or rhs.shape[-1] != 4:
        raise ValueError("both quaternions must have last dimension 4")
    lx, ly, lz, lw = lhs.unbind(dim=-1)
    rx, ry, rz, rw = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def quaternion_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have last dimension 4")
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def _broadcast_pose_reference(
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> torch.Tensor:
    if pose.shape[-1] != 7 or reference_pose.shape[-1] != 7:
        raise ValueError("pose and reference_pose must use xyz + xyzw (last dimension 7)")
    if reference_pose.ndim == pose.ndim - 1:
        reference_pose = reference_pose.unsqueeze(-2)
    try:
        torch.broadcast_shapes(pose.shape, reference_pose.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"pose shape {tuple(pose.shape)} and reference shape "
            f"{tuple(reference_pose.shape)} are not broadcastable"
        ) from exc
    return reference_pose


def absolute_pose_to_relative_pose(
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> torch.Tensor:
    """Convert absolute xyz+xyzw poses to deltas relative to one reference.

    Translation deltas remain in the world/base frame. Rotation is represented
    by q_reference^-1 * q_target. Target quaternion signs are aligned to the
    reference first so equivalent q/-q inputs produce the same short-arc delta.
    """
    reference_pose = _broadcast_pose_reference(pose, reference_pose)
    position_delta = pose[..., :3] - reference_pose[..., :3]

    reference_quaternion = normalize_quaternion_xyzw(reference_pose[..., 3:7])
    target_quaternion = normalize_quaternion_xyzw(pose[..., 3:7])
    same_hemisphere = (reference_quaternion * target_quaternion).sum(
        dim=-1, keepdim=True
    ) >= 0
    target_quaternion = torch.where(
        same_hemisphere,
        target_quaternion,
        -target_quaternion,
    )
    relative_quaternion = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(reference_quaternion),
        target_quaternion,
    )
    relative_quaternion = normalize_quaternion_xyzw(relative_quaternion)
    return torch.cat((position_delta, relative_quaternion), dim=-1)


def relative_pose_to_absolute_pose(
    relative_pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> torch.Tensor:
    """Restore absolute xyz+xyzw poses from world-frame translation deltas."""
    reference_pose = _broadcast_pose_reference(relative_pose, reference_pose)
    position = reference_pose[..., :3] + relative_pose[..., :3]
    reference_quaternion = normalize_quaternion_xyzw(reference_pose[..., 3:7])
    relative_quaternion = normalize_quaternion_xyzw(relative_pose[..., 3:7])
    quaternion = quaternion_multiply_xyzw(
        reference_quaternion,
        relative_quaternion,
    )
    quaternion = normalize_quaternion_xyzw(quaternion)
    return torch.cat((position, quaternion), dim=-1)
