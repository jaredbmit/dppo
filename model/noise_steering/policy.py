"""Tiny Gaussian noise-steering policy + state-value critic.

Both condition on  h = [ frozen history feature ‖ learned goal embedding ]: the
frozen prior's history embedding (conditioning.py) plus a small goal embedding
(the goal lives policy-side because the prior is task-agnostic).

POLICY  pi(w | h): single forward pass h -> diagonal Gaussian over the flattened
(Ta*Da) noise w. No denoising-time input / timestep / iteration — not a DiT, stays
tiny. Init ~N(0, I) (zero-init mean head, log_std 0) so it starts as the base prior.

CRITIC  V(h): standard PPO state-value MLP on the same conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.noise_steering.conditioning import FrozenHistoryEncoder


def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class _GoalConditioning(nn.Module):
    """Shared conditioning: frozen history feature ‖ learned goal embedding."""

    def __init__(self, history_encoder: FrozenHistoryEncoder, goal_dim: int,
                 goal_emb_dim: int = 64):
        super().__init__()
        self.history_encoder = history_encoder
        self.goal_dim = goal_dim
        if goal_dim > 0:
            self.goal_embed = nn.Linear(goal_dim, goal_emb_dim)
            self.feature_dim = history_encoder.feature_dim + goal_emb_dim
        else:  # task with no explicit goal vector -> history only
            self.goal_embed = None
            self.feature_dim = history_encoder.feature_dim

    def features(self, cond: dict) -> torch.Tensor:
        h = self.history_encoder(cond)  # (B, hist_dim), frozen/detached
        if self.goal_embed is not None:
            g = self.goal_embed(cond["goal"].reshape(h.shape[0], self.goal_dim))
            h = torch.cat([h, g], dim=-1)
        return h


class NoisePolicy(nn.Module):
    """Diagonal-Gaussian policy over the flattened initial-noise tensor.

    Args:
        history_encoder: frozen history encoder (tapped from the prior).
        noise_shape:     (Ta, Da) shape of one chunk's initial noise.
        goal_dim:        task goal dimension (0 = no goal vector).
        hidden:          MLP width.
        goal_emb_dim:    width of the learned goal embedding.
        log_std_init:    initial log-std (0.0 -> std 1 -> N(0,I) at init).

    Steering authority (mean drift from 0) is set by PPO's `kl_coef` (the KL term
    penalizes mu^2); there is deliberately no separate mean-gain knob.
    """

    def __init__(
        self,
        history_encoder: FrozenHistoryEncoder,
        noise_shape: tuple[int, int],
        goal_dim: int,
        hidden: int = 256,
        goal_emb_dim: int = 64,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.cond = _GoalConditioning(history_encoder, goal_dim, goal_emb_dim)
        self.noise_shape = noise_shape
        self.noise_dim = int(noise_shape[0] * noise_shape[1])

        self.trunk = _mlp(self.cond.feature_dim, hidden, hidden, n_hidden=1)
        self.mean_head = nn.Linear(hidden, self.noise_dim)
        # state-independent log-std (standard for continuous-control PPO)
        self.log_std = nn.Parameter(torch.full((self.noise_dim,), float(log_std_init)))

        # zero-init -> mean 0 for any input -> with log_std=0, policy == N(0,I) (see above)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def _mean_std(self, cond: dict):
        h = torch.tanh(self.trunk(self.cond.features(cond)))
        mean = self.mean_head(h)  # (B, noise_dim)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def forward(self, cond: dict):
        """Return (dist, mean, std) — distribution over flattened noise."""
        mean, std = self._mean_std(cond)
        dist = torch.distributions.Normal(mean, std)
        return dist, mean, std

    @torch.no_grad()
    def act(self, cond: dict, deterministic: bool = False):
        """Sample noise w. Returns (w_chunk (B,Ta,Da), logprob (B,), flat_w (B,N))."""
        dist, mean, _ = self.forward(cond)
        flat = mean if deterministic else dist.rsample()
        logp = dist.log_prob(flat).sum(-1)  # diagonal Gaussian
        w = flat.reshape(-1, *self.noise_shape)
        return w, logp, flat

    def evaluate(self, cond: dict, flat_w: torch.Tensor):
        """Log-prob + entropy of given flat noise actions under current policy."""
        dist, _, _ = self.forward(cond)
        logp = dist.log_prob(flat_w).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy

    def kl_to_standard_normal(self, cond: dict) -> torch.Tensor:
        """Analytic KL( pi(.|h) || N(0, I) ), per sample (B,).

        Load-bearing: keeps w in the frozen decoder's input distribution and the AR
        rollout on-manifold as steered chunks feed future conditioning.

        KL = 0.5 * sum_i( sigma_i^2 + mu_i^2 - 1 - log sigma_i^2 ).
        """
        mean, std = self._mean_std(cond)
        var = std.pow(2)
        kl = 0.5 * (var + mean.pow(2) - 1.0 - torch.log(var))
        return kl.sum(-1)


class ValueCritic(nn.Module):
    """State-value V(h) on the same conditioning features (small MLP)."""

    def __init__(
        self,
        history_encoder: FrozenHistoryEncoder,
        goal_dim: int,
        hidden: int = 256,
        goal_emb_dim: int = 64,
    ):
        super().__init__()
        # Separate goal embedding from the policy's (critic and actor don't share
        # the small head; the only shared, frozen part is the history encoder).
        self.cond = _GoalConditioning(history_encoder, goal_dim, goal_emb_dim)
        self.net = _mlp(self.cond.feature_dim, hidden, 1, n_hidden=2)

    def forward(self, cond: dict) -> torch.Tensor:
        return self.net(self.cond.features(cond)).squeeze(-1)  # (B,)
