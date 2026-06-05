"""Empirical goal-bank sampler for goal-conditioned RL finetuning.

The BC prior was trained on *hindsight* goals: the body-frame XY displacement
(meters) the robot actually reaches over an action horizon. That distribution is
strongly anisotropic (a forward walking lobe + a standing cluster, little lateral
or backward mass), so sampling goals isotropically during RL commands many
out-of-distribution directions. This module instead samples goals that match the
measured distribution via a KDE / bootstrap-with-jitter scheme:

    goal = bank[random index] + Gaussian jitter (KDE bandwidth)

`bank` is a set of measured hindsight goals; the jitter covariance is scipy's
Scott-rule Gaussian-KDE kernel covariance. The sampler itself is pure numpy so
the env has no scipy/torch dependency at runtime.

Goal labeling mirrors model.common.goal.XYGoalConditioner.integrate_xy exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from util.g1_obs import heading_yaw_rate

IDX_GVEC = slice(0, 3)
IDX_GYRO = slice(3, 6)
IDX_VEL_XY = slice(36, 38)


def _integrate_xy(seq: np.ndarray, mean: np.ndarray, std: np.ndarray, freq: float) -> np.ndarray:
    """Hindsight XY displacement for a batch of windows.

    Args:
        seq: (N, H+1, 38) normalized [context, chunk] windows.
    Returns:
        (N, 2) body-frame displacement, anchored at the context frame.
    """
    dt = 1.0 / freq
    gvec = seq[..., IDX_GVEC] * std[IDX_GVEC] + mean[IDX_GVEC]
    gyro = seq[..., IDX_GYRO] * std[IDX_GYRO] + mean[IDX_GYRO]
    vel = seq[..., IDX_VEL_XY] * std[IDX_VEL_XY] + mean[IDX_VEL_XY]
    yaw_rate = heading_yaw_rate(gvec, gyro)                  # (N, H+1)

    yaw = np.zeros_like(yaw_rate)
    yaw[:, 1:] = np.cumsum(yaw_rate[:, :-1], axis=1) * dt
    c, s = np.cos(yaw[:, :-1]), np.sin(yaw[:, :-1])
    vx = c * vel[:, :-1, 0] - s * vel[:, :-1, 1]
    vy = s * vel[:, :-1, 0] + c * vel[:, :-1, 1]
    return np.stack([vx.sum(1), vy.sum(1)], axis=-1) * dt    # (N, 2)


def _valid_starts(traj_lengths: np.ndarray, horizon: int) -> np.ndarray:
    """Window start indices that stay within a single trajectory."""
    offs = np.zeros_like(traj_lengths)
    offs[1:] = np.cumsum(traj_lengths)[:-1]
    return np.concatenate([
        off + np.arange(0, L - horizon + 1)
        for off, L in zip(offs, traj_lengths) if L > horizon
    ])


def build_goal_bank(
    dataset_path: str | Path,
    horizon: int,
    freq: float = 50.0,
    n_fit: int = 40000,
    n_bank: int = 8000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute measured hindsight goals and the KDE jitter covariance.

    Returns (bank (n_bank, 2), cov (2, 2)). `cov` is fit on up to n_fit goals for
    accuracy; `bank` is a random subset of size n_bank that the sampler draws from.
    """
    from scipy.stats import gaussian_kde

    data = np.load(dataset_path)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    norm_path = Path(dataset_path).parent / "norm_stats.npz"
    stats = np.load(norm_path)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)

    rng = np.random.default_rng(seed)
    starts = _valid_starts(data["traj_lengths"], horizon)
    pick = rng.choice(len(starts), size=min(n_fit, len(starts)), replace=False)
    s = starts[pick]

    # Build [context, chunk] windows and integrate in batches to bound memory.
    goals = np.empty((len(s), 2), dtype=np.float32)
    win = horizon  # actions[start : start+horizon]
    for b in range(0, len(s), 20000):
        sb = s[b:b + 20000]
        chunk_idx = sb[:, None] + np.arange(win)[None, :]    # (B, H)
        seq = np.concatenate([states[sb][:, None, :], actions[chunk_idx]], axis=1)
        goals[b:b + len(sb)] = _integrate_xy(seq, mean, std, freq)

    cov = gaussian_kde(goals.T).covariance.astype(np.float32)
    bank = goals[rng.choice(len(goals), size=min(n_bank, len(goals)), replace=False)]
    return bank, cov


def load_or_build_goal_bank(
    dataset_path: str | Path,
    horizon: int,
    freq: float = 50.0,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached goal bank for this horizon, building + caching it if absent."""
    cache = Path(dataset_path).parent / f"goal_bank_h{horizon}.npz"
    if cache.exists() and not rebuild:
        d = np.load(cache)
        return d["bank"].astype(np.float32), d["cov"].astype(np.float32)
    bank, cov = build_goal_bank(dataset_path, horizon, freq=freq)
    np.savez(cache, bank=bank, cov=cov)
    return bank, cov


class GoalBankSampler:
    """Draws goals matching the measured distribution: bank point + KDE jitter.

    Uses the global numpy RNG (np.random) so it honors env.seed().
    """

    def __init__(self, bank: np.ndarray, cov: np.ndarray):
        self.bank = np.ascontiguousarray(bank, dtype=np.float32)
        self.chol = np.linalg.cholesky(cov.astype(np.float64)).astype(np.float32)

    def sample(self, n: int) -> np.ndarray:
        """(n, 2) goals."""
        idx = np.random.randint(0, len(self.bank), size=n)
        jitter = np.random.standard_normal((n, 2)).astype(np.float32) @ self.chol.T
        return (self.bank[idx] + jitter).astype(np.float32)
