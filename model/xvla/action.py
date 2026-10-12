from __future__ import annotations

import torch
from torch import Tensor


def axis_angle_to_matrix(axis_angle: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert axis-angle vectors to rotation matrices."""
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(eps)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(k)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    mat = eye + sin * k + (1.0 - cos) * (k @ k)
    small = (angle.squeeze(-1) < eps)[..., None, None]
    return torch.where(small, eye, mat)


def axis_angle_to_rotate6d(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to X-VLA's first-two-columns 6D rotation."""
    mat = axis_angle_to_matrix(axis_angle)
    return torch.cat([mat[..., :3, 0], mat[..., :3, 1]], dim=-1)


def libero7_to_xvla_ee6d20(action: Tensor) -> Tensor:
    """Map LIBERO 7D actions `[xyz, axis-angle, gripper]` to X-VLA EE6D 20D actions."""
    if action.shape[-1] != 7:
        raise ValueError(f"Expected LIBERO action dim 7, got {action.shape[-1]}.")
    out = action.new_zeros(*action.shape[:-1], 20)
    out[..., :3] = action[..., :3]
    out[..., 3:9] = axis_angle_to_rotate6d(action[..., 3:6])
    out[..., 9] = action[..., 6]
    return out
