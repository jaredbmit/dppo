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


# ---------------------------------------------------------------------------
# Smoothness metrics (no reference data needed)
# ---------------------------------------------------------------------------

def _finite_diff(x: np.ndarray, dt: float, order: int) -> np.ndarray:
    """Repeated time-derivative along axis 0 via finite differences."""
    for _ in range(order):
        x = np.diff(x, axis=0) / dt
    return x


def _sparc(speed: np.ndarray, freq: float,
           pad_level: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length of a 1-D movement-speed profile (Balasubramanian 2015).

    A reparameterization-robust smoothness measure: always <= 0, with values
    closer to 0 indicating smoother motion. Returns 0.0 for degenerate input.
    """
    speed = np.asarray(speed, dtype=np.float64)
    if speed.size < 4 or not np.any(speed):
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(speed.size)) + pad_level))
    f = np.arange(0, freq, freq / nfft)
    mf = np.abs(np.fft.fft(speed, nfft))
    mf /= mf.max()
    sel = f <= fc
    f_sel, mf_sel = f[sel], mf[sel]
    above = np.where(mf_sel >= amp_th)[0]
    if above.size:
        f_sel = f_sel[above[0]:above[-1] + 1]
        mf_sel = mf_sel[above[0]:above[-1] + 1]
    if f_sel.size < 2 or (f_sel[-1] - f_sel[0]) == 0:
        return 0.0
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    dm = np.diff(mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


def rollout_smoothness_metrics(obs_denorm, freq: float = 50.0) -> dict:
    """Joint-space smoothness for a single rollout (T, 38) of denormalized feats.

    Smoothness is derivative-based, so the constant joint offset (jpos is angle
    minus default pose) cancels — no FK or world frame needed.

      jerk_rms      RMS of joint-angle jerk    (rad/s^3, lower = smoother)
      acc_rms       RMS of joint-angle accel   (rad/s^2)
      sparc         spectral arc length of the joint-space speed profile
                    (<= 0, closer to 0 = smoother)
      root_jerk_rms RMS jerk of the measured heading-frame root velocity (m/s^3)
    """
    x = np.asarray(obs_denorm, dtype=np.float64)
    dt = 1.0 / freq
    jpos = x[:, IDX_JPOS]                                   # (T, 29)
    vel  = x[:, IDX_VEL_XY]                                 # (T, 2)
    jerk = _finite_diff(jpos, dt, 3)
    acc  = _finite_diff(jpos, dt, 2)
    speed = np.linalg.norm(_finite_diff(jpos, dt, 1), axis=1)
    root_jerk = _finite_diff(vel, dt, 2)
    return {
        "jerk_rms": float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.0,
        "acc_rms": float(np.sqrt(np.mean(acc ** 2))) if acc.size else 0.0,
        "sparc": _sparc(speed, freq),
        "root_jerk_rms": float(np.sqrt(np.mean(root_jerk ** 2))) if root_jerk.size else 0.0,
    }


def rollout_limit_metrics(
    obs_denorm,
    joint_limits,
    default_qpos,
    freq: float = 50.0,
    vel_limit=None,
) -> dict:
    """Joint position-limit violations + peak joint velocity for one rollout.

    Args:
        obs_denorm:   (T, 38) denormalized features (jpos block is offset-from-default).
        joint_limits: (29, 2) absolute [lo, hi] per joint, in qpos[7:] order;
                      unlimited joints use +-inf.
        default_qpos: (29,) default joint angles added back to recover absolute angle.
        vel_limit:    optional (29,) or scalar joint-speed limit (rad/s) for a
                      velocity-violation fraction.

      joint_pos_viol_frac  fraction of (frame x limited-joint) outside limits
      joint_pos_viol_max   largest overshoot beyond a limit (rad)
      joint_vel_max        peak |joint velocity| across the rollout (rad/s)
    """
    x = np.asarray(obs_denorm, dtype=np.float64)
    dt = 1.0 / freq
    lim = np.asarray(joint_limits, dtype=np.float64)        # (29, 2)
    jpos_abs = x[:, IDX_JPOS] + np.asarray(default_qpos, dtype=np.float64)
    over = np.maximum.reduce([lim[:, 0] - jpos_abs, jpos_abs - lim[:, 1],
                              np.zeros_like(jpos_abs)])
    limited = np.isfinite(lim[:, 0]) & np.isfinite(lim[:, 1])
    over_lim = over[:, limited]
    jvel = np.abs(_finite_diff(jpos_abs, dt, 1))            # (T-1, 29)
    out = {
        "joint_pos_viol_frac": float((over_lim > 1e-6).mean()) if over_lim.size else 0.0,
        "joint_pos_viol_max": float(over_lim.max()) if over_lim.size else 0.0,
        "joint_vel_max": float(jvel.max()) if jvel.size else 0.0,
    }
    if vel_limit is not None:
        vel_over = np.maximum(jvel - np.asarray(vel_limit, dtype=np.float64), 0.0)
        out["joint_vel_viol_frac"] = float((vel_over > 1e-6).mean()) if vel_over.size else 0.0
    return out


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
