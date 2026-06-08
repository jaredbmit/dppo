"""Noise-space environment wrapper.

Turns a task env (whose actions are motion chunks) into one whose action space IS
the diffusion prior's initial-noise space. Data flow for one RL step:

    policy -> noise w (B, Ta, Da)
           -> frozen DDIM eta=0 decoder, conditioned on current history
           -> motion chunk (B, Ta, Da)
           -> underlying env.step(chunk)  [plays the chunk, advances AR history]
           -> task reward + next history + (task) goal

One RL timestep == one autoregressive motion chunk (NOT one denoising step, NOT
one frame). The decoder is the FROZEN prior; gradients never flow through it —
the policy is trained purely from the noise-space log-probs and task reward.

The wrapper is task-agnostic: it only needs an underlying vec env that returns a
``{"state": ..., "goal"?: ...}`` obs dict and consumes chunk actions. The goal is
passed straight through to the policy and is NOT given to the (task-agnostic)
decoder.
"""

from __future__ import annotations

import numpy as np
import torch


class NoiseSteeringEnv:
    """Wrap a chunk-action vec env so its action space is the prior's noise.

    Args:
        venv:   underlying vectorized env (e.g. G1KinematicVecEnv). step(actions)
                takes (N, act_steps, obs_dim) chunks and returns
                (obs_dict, reward, terminated, truncated, info); reset() returns
                obs_dict with "state" (N, To, obs_dim) and optional "goal".
        prior:  frozen DiffusionModel with use_ddim=True, eta=0 (deterministic).
        device: torch device for noise / decoding.
    """

    def __init__(self, venv, prior, device: str = "cuda:0"):
        self.venv = venv
        self.prior = prior
        self.device = device
        self.n_envs = venv.n_envs
        self.noise_shape = (prior.horizon_steps, prior.action_dim)

    # ------------------------------------------------------------------ #

    def _to_torch(self, obs: dict) -> dict:
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in obs.items()}

    @torch.no_grad()
    def _decode(self, w: torch.Tensor, obs: dict) -> np.ndarray:
        """noise w (B,Ta,Da) + history -> motion chunk (B,Ta,Da) numpy.

        Conditions ONLY on history ("state"); the prior is task-agnostic so the
        goal is never passed to the decoder.
        """
        cond = {"state": obs["state"]}
        chunk = self.prior.forward(cond, init_noise=w).trajectories  # (B,Ta,Da)
        return chunk.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #

    def reset(self) -> dict:
        return self._to_torch(self.venv.reset())

    @torch.no_grad()
    def step(self, w: torch.Tensor, obs: dict):
        """Decode noise w under the current obs, play the chunk, return outcome.

        Args:
            w:   (N, Ta, Da) initial noise from the policy.
            obs: current obs dict (torch) used to condition the decoder.
        Returns:
            next_obs (torch dict), reward (N,) np, terminated (N,) np,
            truncated (N,) np, info.
        """
        chunk = self._decode(w, obs)
        next_obs, reward, terminated, truncated, info = self.venv.step(chunk)
        return self._to_torch(next_obs), reward, terminated, truncated, info

    def close(self):
        self.venv.close()
