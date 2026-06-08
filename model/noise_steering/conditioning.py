"""Tap the frozen prior's internal history embedding.

For the history conditioning we reuse what the frozen task-agnostic prior already
computes — its ``obs_proj`` token embedding — instead of training a second encoder:

    obs_proj(state) -> per-step token embedding  (B, To, d_model)

The goal is NOT handled here: the prior is task-agnostic (goal_dim=0), so the goal
is the policy's concern — it embeds the goal and concatenates it on (see policy.py).
Frozen, called under no_grad; the policy learns only the heads on top.
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
