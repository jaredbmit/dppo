"""Shared G1 38-D observation geometry helpers.

The 38-D feature stores gyro as body-local angular velocity but root_vel_xy in
the heading (yaw-only) frame:

    [0:3]   gvec        — gravity direction in the body frame
    [3:6]   gyro        — body-local angular velocity (rad/s)
    [36:38] root_vel_xy — planar velocity in the heading frame (m/s)

Reconstructing world root motion needs a heading yaw rate, not gyro_z directly
(those agree only for an upright frame). For the ZYX decomposition used here:

    heading_yaw_rate = (gyro_y * sin(roll) + gyro_z * cos(roll)) / cos(pitch)

with roll/pitch recovered from gvec. Functions accept numpy arrays or torch
tensors; the last axis is the 3-vector axis, leading dims are free.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # torch optional for pure-numpy call sites
    torch = None

# Guard cos(pitch) near the gimbal singularity (pitch -> ±90°).
_COS_PITCH_EPS = 1e-3


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def roll_pitch_from_gvec(gvec):
    """Recover (roll, pitch) from the body-frame gravity vector obs[0:3]."""
    gx, gy, gz = gvec[..., 0], gvec[..., 1], gvec[..., 2]
    if _is_torch(gvec):
        roll  = torch.atan2(-gy, -gz)
        pitch = torch.atan2(gx, torch.sqrt(gy * gy + gz * gz))
    else:
        roll  = np.arctan2(-gy, -gz)
        pitch = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
    return roll, pitch


def heading_yaw_rate(gvec, gyro):
    """Heading (yaw-only) frame yaw rate from body-local gyro + gravity.

    Args:
        gvec: (..., 3) gravity direction in the body frame.
        gyro: (..., 3) body-local angular velocity (rad/s).
    Returns:
        (...) heading yaw rate, matching the backend of the inputs.
    """
    gyro_y, gyro_z = gyro[..., 1], gyro[..., 2]
    roll, pitch = roll_pitch_from_gvec(gvec)
    if _is_torch(gvec):
        cos_pitch = torch.cos(pitch)
        cos_pitch = torch.sign(cos_pitch) * torch.clamp(torch.abs(cos_pitch), min=_COS_PITCH_EPS)
        return (gyro_y * torch.sin(roll) + gyro_z * torch.cos(roll)) / cos_pitch
    else:
        cos_pitch = np.cos(pitch)
        cos_pitch = np.sign(cos_pitch) * np.clip(np.abs(cos_pitch), _COS_PITCH_EPS, None)
        return (gyro_y * np.sin(roll) + gyro_z * np.cos(roll)) / cos_pitch
