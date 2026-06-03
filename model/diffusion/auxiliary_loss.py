"""Auxiliary geometric losses for G1 motion diffusion pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.kinematics import G1Kinematics

_IDX_JOINT_POS = slice(6, 35)


class G1GeometricLoss(nn.Module):
    """End-effector position loss + joint velocity loss for 38-D G1 canonical features.

    Both losses operate on x0 predictions (not epsilon). The EE loss denormalizes
    into physical units before running FK; the joint velocity loss works in
    normalized space (finite-diff of the joint_pos block across the time axis).

    Args:
        xml_path:         Path to G1 MuJoCo scene XML (used to build FK chain).
        norm_stats_path:  Path to norm_stats.npz containing mean (38,) and std (38,).
        lambda_ee_pos:    Weight on end-effector position MSE loss.
        lambda_joint_vel: Weight on joint velocity MSE loss.
    """

    def __init__(
        self,
        xml_path: str,
        norm_stats_path: str,
        lambda_ee_pos: float = 1.0,
        lambda_joint_vel: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_ee_pos    = lambda_ee_pos
        self.lambda_joint_vel = lambda_joint_vel

        stats = np.load(norm_stats_path)
        self.register_buffer("mean", torch.from_numpy(stats["mean"]).float())
        self.register_buffer("std",  torch.from_numpy(stats["std"]).float())

        self.fk = G1Kinematics(xml_path) if lambda_ee_pos > 0.0 else None

    def forward(
        self,
        x0_hat: torch.Tensor,
        x0_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses.

        Args:
            x0_hat:    (B, T, 38) predicted clean sample, normalized.
            x0_target: (B, T, 38) ground-truth clean sample, normalized.

        Returns:
            Dict of weighted loss tensors (already multiplied by their lambdas).
        """
        losses: dict[str, torch.Tensor] = {}

        if self.lambda_ee_pos > 0.0:
            pred = x0_hat    * self.std + self.mean
            tgt  = (x0_target * self.std + self.mean).detach()
            sp   = self.fk(pred[..., _IDX_JOINT_POS])
            st   = self.fk(tgt[...,  _IDX_JOINT_POS])
            ee   = (F.mse_loss(sp["hands"], st["hands"])
                  + F.mse_loss(sp["feet"],  st["feet"])) / 2.0
            losses["ee_pos"] = self.lambda_ee_pos * ee

        if self.lambda_joint_vel > 0.0:
            dj_pred = x0_hat[...,    _IDX_JOINT_POS]
            dj_tgt  = x0_target[..., _IDX_JOINT_POS]
            jvel = F.mse_loss(
                dj_pred[:, 1:] - dj_pred[:, :-1],
                dj_tgt[:, 1:]  - dj_tgt[:, :-1],
            )
            losses["joint_vel"] = self.lambda_joint_vel * jvel

        return losses
