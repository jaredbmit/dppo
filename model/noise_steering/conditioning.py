"""Tap the frozen prior's internal history embedding.

The steering policy/critic condition on (motion history, task goal). For the
*history* we reuse the representation the frozen task-agnostic prior already
computes — its ``obs_proj`` token embedding — rather than training a second
encoder over raw observations:

    obs_proj(state) -> per-history-step token embedding  (B, To, d_model)

The *goal* is deliberately NOT handled here: the frozen prior is task-agnostic
(goal_dim=0, no goal_proj), so the goal is a property of the task/policy, not of
the prior. The steering policy embeds the goal itself (see policy.py) and
concatenates it onto this history feature. This is the whole framework: a generic
state-conditional prior, steered toward a task by a small goal-aware policy.

These projections are frozen and called under no_grad — the policy learns only
the small heads on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenHistoryEncoder(nn.Module):
    """Maps cond dict -> history feature using the frozen prior's obs tokens.

    Args:
        dit: the frozen ``DiffusionDiT`` network (must have ``obs_proj``).
        cond_steps: number of history steps To in cond["state"].
    """

    def __init__(self, dit: nn.Module, cond_steps: int):
        super().__init__()
        # tuple-wrap so the frozen prior is NOT registered as a submodule (keeps it
        # out of the policy/critic param count, optimizer, and state_dict)
        self._dit = (dit,)
        self.cond_steps = cond_steps
        self.d_model = dit.obs_proj.out_features
        self.feature_dim = cond_steps * self.d_model

    @property
    def dit(self) -> nn.Module:
        return self._dit[0]

    @torch.no_grad()
    def forward(self, cond: dict) -> torch.Tensor:
        """cond["state"] -> (B, cond_steps*d_model) history feature (frozen)."""
        state = cond["state"]
        if state.dim() == 2:  # (B, obs_dim) -> (B, 1, obs_dim)
            state = state.unsqueeze(1)
        B = state.shape[0]
        obs_emb = self.dit.obs_proj(state)  # (B, To, d_model)
        return obs_emb.reshape(B, -1)       # (B, To*d_model)
