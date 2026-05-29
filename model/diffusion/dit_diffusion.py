"""
DiT (Diffusion Transformer) denoiser for diffusion policies.

Ported from the motion-diffusion DiT in ~/drl/LATENT and adapted to the dppo
network contract: ``forward(x, time, cond) -> (B, Ta, Da)`` with ``cond`` a dict
holding ``"state"`` (and optionally ``"goal"``).

Design (see model/diffusion/diffusion.py for the surrounding DDPM/DDIM wrapper):

  * The noised future chunk x (B, Ta, action_dim) is tokenized along the horizon
    axis -- one token per future step.
  * The recent observation history cond["state"] (B, To, obs_dim) is projected to
    *clean* tokens and prepended to the sequence. Self-attention then lets the
    future tokens condition on the observed pose directly, exploiting the shared
    motion-feature inductive bias (conditioning-by-inpainting). Output tokens for
    the history positions are discarded.
  * The diffusion timestep and the goal vector condition every block via
    AdaLN-zero (Peebles & Xie 2023): both are projected to d_model and summed
    into the modulation signal.

Predicts x0 (not epsilon), matching predict_epsilon=False + the geometric
auxiliary losses.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([args.cos(), args.sin()], dim=-1)  # (B, dim)


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal(t, self.d_model))  # (B, d_model)


# ---------------------------------------------------------------------------
# DiT block (AdaLN-zero)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Single DiT block with AdaLN-zero conditioning."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        # 6 modulation params: scale1, shift1, gate1, scale2, shift2, gate2
        self.adaLN = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  c: (B, d_model) conditioning."""
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)
        # attention sub-block
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        # MLP sub-block
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class DiffusionDiT(nn.Module):
    """DiT denoiser matching the dppo network interface.

    Args:
        action_dim:    per-step action/feature dimension (token width of the
                       noised future chunk).
        horizon_steps: number of future steps Ta (number of noised tokens).
        obs_dim:       per-step observation dimension of cond["state"].
        cond_steps:    number of observation steps To prepended as clean tokens.
        goal_dim:      dimension of cond["goal"] (0 to disable goal conditioning).
        d_model/n_heads/n_layers/ff_mult: transformer width / depth.
        dropout:       attention + MLP dropout.
    """

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        obs_dim: int,
        cond_steps: int = 1,
        goal_dim: int = 0,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.goal_dim = goal_dim

        seq_len = cond_steps + horizon_steps

        # token projections: clean history vs noised future (kept separate so the
        # model can distinguish observed from to-generate tokens)
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.act_proj = nn.Linear(action_dim, d_model)

        # learnable absolute position embedding over [history | future]
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # learnable segment embedding: row 0 = observed, row 1 = to-generate
        self.seg_emb = nn.Parameter(torch.zeros(1, 2, d_model))

        # AdaLN conditioning signal: timestep (+ goal)
        self.time_emb = TimestepEmbedding(d_model)
        if goal_dim > 0:
            self.goal_proj = nn.Linear(goal_dim, d_model)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.seg_emb, std=0.02)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"DiffusionDiT parameters: {n_params:,}")

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs):
        """
        x:    (B, Ta, action_dim)  noised future chunk
        time: (B,)                 diffusion timestep
        cond: dict with
            state: (B, To, obs_dim)
            goal:  (B, goal_dim)   optional
        Returns x0 prediction: (B, Ta, action_dim)
        """
        B, Ta, _ = x.shape

        # --- tokenize: clean history tokens prepended to noised future tokens ---
        state = cond["state"]
        if state.dim() == 2:  # (B, obs_dim) -> (B, 1, obs_dim)
            state = state.unsqueeze(1)
        To = state.shape[1]

        h_tok = self.obs_proj(state)  # (B, To, d_model)
        f_tok = self.act_proj(x)      # (B, Ta, d_model)
        tokens = torch.cat([h_tok, f_tok], dim=1)  # (B, To+Ta, d_model)

        # position + segment embeddings
        tokens = tokens + self.pos_emb[:, : To + Ta]
        seg = torch.cat(
            [
                self.seg_emb[:, 0:1].expand(B, To, -1),
                self.seg_emb[:, 1:2].expand(B, Ta, -1),
            ],
            dim=1,
        )
        tokens = tokens + seg

        # --- AdaLN conditioning: timestep (+ goal) ---
        c = self.time_emb(time.reshape(B))
        if self.goal_dim > 0 and cond.get("goal") is not None:
            c = c + self.goal_proj(cond["goal"].reshape(B, self.goal_dim))

        for block in self.blocks:
            tokens = block(tokens, c)

        out = self.out_proj(self.out_norm(tokens))
        return out[:, To:]  # keep only the future positions -> (B, Ta, action_dim)
