"""Shared motion-quality metrics for diffusion-policy evaluation.

Both eval_generation.py and eval_circle.py subscribe to these. Two metric
families:

  * distribution metrics (frechet_distance, diversity) — generic, frame/chunk
    level, parameterization-free.
  * physics metrics (foot skate, ground penetration) — require world-frame foot
    positions, reconstructed from the canonical 38-D G1 feature layout via FK.

Canonical feature layout (physical units, post-denorm):
    [0:3]   gvec            (gravity in body frame)
    [3:6]   gyro            (body-local angular velocity; gyro_z at index 5)
    [6:35]  joint_pos       (29, MuJoCo qpos[7:] order — FK input)
    [35]    root_height
    [36:38] root_vel_xy     (heading frame)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import sqrtm

from util.g1_obs import heading_yaw_rate

IDX_GVEC   = slice(0, 3)
IDX_GYRO   = slice(3, 6)
IDX_JPOS   = slice(6, 35)
IDX_ROOT_H = 35
IDX_VEL_XY = slice(36, 38)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between two sets of per-frame features (N, D)."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    covmean = sqrtm(ca @ cb)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * covmean))


def diversity(chunks_flat: torch.Tensor) -> float:
    """Mean pairwise L2 among (N, F) chunk embeddings."""
    pair = torch.cdist(chunks_flat, chunks_flat)
    n = pair.shape[0]
    return (pair.sum() / (n * (n - 1))).item()


# ---------------------------------------------------------------------------
# Physics metrics
# ---------------------------------------------------------------------------

def _as_tensor(x, device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device).float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _rot2d(yaw: torch.Tensor) -> torch.Tensor:
    """(...,) yaw -> (..., 2, 2) rotation matrices."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s,  c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _foot_world(jpos, root_h, root_xy, root_yaw, fk) -> torch.Tensor:
    """World-frame foot positions.

    Args (all share leading shape L = (...,T)):
        jpos:     (*L, 29) joint offsets (FK input)
        root_h:   (*L,)    pelvis height above ground
        root_xy:  (*L, 2)  world-frame pelvis xy
        root_yaw: (*L,)    world-frame heading
    Returns:
        (*L, 2, 3) world-frame positions for [left, right] foot.
    """
    feet_local = fk(jpos)["feet"]                          # (*L, 2, 3) pelvis-local
    R = _rot2d(root_yaw)                                   # (*L, 2, 2)
    xy = torch.einsum("...ij,...kj->...ki", R, feet_local[..., :2])
    xy = xy + root_xy[..., None, :]                        # (*L, 2, 2)
    z = feet_local[..., 2] + root_h[..., None]            # (*L, 2)
    return torch.cat([xy, z[..., None]], dim=-1)           # (*L, 2, 3)


def _metrics_from_feet(feet_world, dt, contact_h, ground_z) -> dict:
    """Skate + penetration from (B, T, 2, 3) world-frame foot positions.

    Foot skate: horizontal foot speed weighted by a near-ground contact factor
    w = clamp(2 - 2^(h/contact_h), 0, 1) (≈1 at the ground, →0 by contact_h) —
    a planted foot that slides is penalized; a swing foot is not. Units m/s.

    Penetration: depth of feet below ground_z (m), and the fraction of foot-frames
    that penetrate.
    """
    z = feet_world[..., 2]                                 # (B, T, 2)
    pen = (ground_z - z).clamp(min=0)

    dxy = feet_world[:, 1:, :, :2] - feet_world[:, :-1, :, :2]
    speed = dxy.norm(dim=-1) / dt                          # (B, T-1, 2)
    h = z[:, 1:].clamp(min=0)
    w = (2.0 - torch.pow(2.0, h / contact_h)).clamp(0.0, 1.0)
    skate = speed * w

    return {
        "foot_skate": float(skate.mean()),
        "penetration_mean": float(pen.mean()),
        "penetration_max": float(pen.max()),
        "penetration_frac": float((pen > 1e-4).float().mean()),
        "contact_frac": float((z < contact_h).float().mean()),
    }


def chunk_physics_metrics(
    feats_norm,
    mean,
    std,
    fk,
    freq: float = 50.0,
    contact_h: float = 0.08,
    ground_z: float = 0.0,
) -> dict:
    """Physics metrics for a batch of independent chunks (B, T, 38), normalized.

    Each chunk's world frame is reconstructed internally by integrating
    heading-frame velocity with the heading yaw rate (see util.g1_obs), anchored
    at xy=0, yaw=0. Skate and penetration are velocity-/height-based, hence
    invariant to that anchor.
    """
    device = fk.default_qpos.device
    x = _as_tensor(feats_norm, device)
    mean = _as_tensor(mean, device)
    std = _as_tensor(std, device)
    x = x * std + mean                                     # denormalize (B, T, 38)
    dt = 1.0 / freq

    jpos   = x[..., IDX_JPOS]                              # (B, T, 29)
    root_h = x[..., IDX_ROOT_H]                            # (B, T)
    gvec   = x[..., IDX_GVEC]                              # (B, T, 3)
    gyro   = x[..., IDX_GYRO]                              # (B, T, 3)
    vel_h  = x[..., IDX_VEL_XY]                            # (B, T, 2)
    yaw_rate = heading_yaw_rate(gvec, gyro)                # (B, T)

    # yaw[t] = cumulative yaw_rate up to t-1 (yaw[0] = 0)
    yaw = torch.zeros_like(yaw_rate)
    yaw[:, 1:] = torch.cumsum(yaw_rate[:, :-1], dim=1) * dt
    # rotate heading-frame velocity to world, integrate to xy (xy[0] = 0)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    vx = cos_y * vel_h[..., 0] - sin_y * vel_h[..., 1]
    vy = sin_y * vel_h[..., 0] + cos_y * vel_h[..., 1]
    xy = torch.zeros(*vel_h.shape, device=device)
    xy[:, 1:, 0] = torch.cumsum(vx[:, :-1], dim=1) * dt
    xy[:, 1:, 1] = torch.cumsum(vy[:, :-1], dim=1) * dt

    feet_world = _foot_world(jpos, root_h, xy, yaw, fk)
    return _metrics_from_feet(feet_world, dt, contact_h, ground_z)


def rollout_physics_metrics(
    obs_denorm,
    world_xy,
    world_yaw,
    fk,
    freq: float = 50.0,
    contact_h: float = 0.08,
    ground_z: float = 0.0,
) -> dict:
    """Physics metrics for a single rollout with a known world frame.

    Args:
        obs_denorm: (T, 38) denormalized features.
        world_xy:   (T, 2)  integrated world-frame pelvis xy.
        world_yaw:  (T,)    integrated world-frame heading.
    """
    device = fk.default_qpos.device
    x = _as_tensor(obs_denorm, device)
    jpos   = x[None, :, IDX_JPOS]                          # (1, T, 29)
    root_h = x[None, :, IDX_ROOT_H]                        # (1, T)
    xy  = _as_tensor(world_xy,  device)[None]              # (1, T, 2)
    yaw = _as_tensor(world_yaw, device)[None]              # (1, T)
    feet_world = _foot_world(jpos, root_h, xy, yaw, fk)
    return _metrics_from_feet(feet_world, 1.0 / freq, contact_h, ground_z)
