"""
Goal conditioning modules for task-specific BC pretraining.
"""

import numpy as np
import torch
import torch.nn as nn


class XYGoalConditioner(nn.Module):
    """Compute body-frame XY displacement goal on-the-fly from the action sequence.

    During BC pretraining the goal is the integrated endpoint of each horizon
    chunk (hindsight labeling), plus optional Gaussian noise.  At RL time the
    caller sets cond["goal"] directly and this module is bypassed.

    Observation layout (physical, post-denorm):
        [0:3]   gvec
        [3:6]   gyro (gyro_z at index 5)
        [6:35]  jpos
        [35]    root_height
        [36:38] root_vel_xy  (heading frame)
    """

    def __init__(
        self,
        norm_stats_path: str,
        noise_std: float = 0.15,
        freq: float = 50.0,
    ):
        super().__init__()
        n = np.load(norm_stats_path)
        self.register_buffer("mean", torch.from_numpy(n["mean"]).float())
        self.register_buffer("std",  torch.from_numpy(n["std"]).float())
        self.noise_std = noise_std
        self.freq = freq

    def integrate_xy(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate heading-frame velocity to get body-frame XY endpoint.

        Args:
            x: (B, T, 38) normalized action sequence
        Returns:
            (B, 2) displacement in body frame at t=0
        """
        dt = 1.0 / self.freq

        # Denormalize only the channels we need
        gyro_z = x[:, :, 5]    * self.std[5]    + self.mean[5]      # (B, T)
        vel_h  = x[:, :, 36:38] * self.std[36:38] + self.mean[36:38] # (B, T, 2)

        # Yaw at each step (yaw[0] = 0 by convention — goal is body-frame-relative)
        # yaw[:, t] = sum of gyro_z[:, 0..t-1] * dt
        yaw = torch.zeros_like(gyro_z)
        yaw[:, 1:] = torch.cumsum(gyro_z[:, :-1], dim=1) * dt  # (B, T)

        # Rotate heading-frame velocity to world frame, then integrate
        cos_y = torch.cos(yaw[:, :-1])  # (B, T-1)
        sin_y = torch.sin(yaw[:, :-1])
        vx_w = cos_y * vel_h[:, :-1, 0] - sin_y * vel_h[:, :-1, 1]
        vy_w = sin_y * vel_h[:, :-1, 0] + cos_y * vel_h[:, :-1, 1]

        xy_end = torch.stack([vx_w.sum(dim=1), vy_w.sum(dim=1)], dim=-1) * dt  # (B, 2)
        return xy_end

    def forward(self, x: torch.Tensor, cond: dict) -> dict:
        """Add a hindsight XY goal: displacement to the chunk end, anchored at the
        last conditioning frame (or x[0] if no conditioning is provided)."""
        state = cond.get("state")
        if state is not None and state.shape[1] >= 1:
            seq = torch.cat([state[:, -1:], x], dim=1)  # anchor at conditioning frame
        else:
            seq = x  # unconditional / CFG drop / first chunk -> anchor at x[0]
        goal = self.integrate_xy(seq)
        if self.training and self.noise_std > 0:
            goal = goal + torch.randn_like(goal) * self.noise_std
        cond = dict(cond)
        cond["goal"] = goal
        return cond
