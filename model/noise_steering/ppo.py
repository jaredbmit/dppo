"""Clipped PPO + GAE for the noise-steering policy.

Hand-written CleanRL-style PPO, matching the conventions of DPPO's own
`agent/finetune/train_ppo_diffusion_agent.py` (manual GAE, `RunningRewardScaler`,
`reward_scale_const`, critic-warmup, `terminated`-based bootstrap so time-limit
truncation still bootstraps). DPPO does not use an external RL library, so neither
do we.

Two noise-space specifics on top of vanilla PPO:

  * The "action" is the flattened initial noise w; log-probs / ratios are computed
    in noise space where the policy is a diagonal Gaussian, so the importance
    ratio is well defined.
  * A soft KL penalty pulls the policy toward N(0, I) (analytic, see
    NoisePolicy.kl_to_standard_normal). This keeps w in the frozen decoder's input
    distribution and keeps the AR rollout on-manifold; its coefficient `kl_coef`
    is the main steering-vs-stay-natural knob.

The critic is a state-value V(h) (see policy.py for why not a Q over noise).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from util.reward_scaling import RunningRewardScaler

log = logging.getLogger(__name__)


class NoiseSteeringPPO:
    def __init__(
        self,
        env,                 # NoiseSteeringEnv
        policy,              # NoisePolicy
        critic,              # ValueCritic
        device: str = "cuda:0",
        # rollout
        n_iterations: int = 500,
        n_steps: int = 16,        # chunks collected per env per iteration
        n_critic_warmup_itr: int = 5,
        # ppo
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        kl_coef: float = 0.1,     # KL(pi || N(0,I)) penalty -- load-bearing
        max_grad_norm: float = 0.5,
        target_kl: float | None = 0.05,  # early-stop on policy drift (old||new)
        norm_adv: bool = True,
        # reward scaling (matches DPPO)
        reward_scale_running: bool = True,
        reward_scale_const: float = 1.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
    ):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.device = device
        self.n_envs = env.n_envs
        self.noise_dim = policy.noise_dim

        self.n_iterations = n_iterations
        self.n_steps = n_steps
        self.n_critic_warmup_itr = n_critic_warmup_itr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.reward_scale_const = reward_scale_const

        self.reward_scaler = (
            RunningRewardScaler(self.n_envs, gamma=gamma)
            if reward_scale_running else None
        )

        self.actor_opt = torch.optim.Adam(policy.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        # done flags carried across iterations to seed episode-start markers
        self._prev_done = np.zeros(self.n_envs, dtype=np.float32)

    # ------------------------------------------------------------------ #

    def _slice_obs(self, obs: dict, idx) -> dict:
        return {k: v[idx] for k, v in obs.items()}

    @torch.no_grad()
    def collect(self, obs: dict):
        """Roll out n_steps chunks. Returns a flat batch dict + last obs."""
        N, S = self.n_envs, self.n_steps
        obs_buf = {k: torch.zeros((S, *v.shape), dtype=torch.float32, device=self.device)
                   for k, v in obs.items()}
        acts = torch.zeros((S, N, self.noise_dim), device=self.device)
        logps = torch.zeros((S, N), device=self.device)
        vals = np.zeros((S, N), dtype=np.float32)
        rews = np.zeros((S, N), dtype=np.float32)
        terms = np.zeros((S, N), dtype=np.float32)
        firsts = np.zeros((S, N), dtype=np.float32)

        prev_done = self._prev_done.copy()
        ep_rew = []
        for t in range(S):
            for k in obs_buf:
                obs_buf[k][t] = obs[k]
            firsts[t] = prev_done
            w, logp, flat = self.policy.act(obs, deterministic=False)
            v = self.critic(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(w, obs)

            acts[t] = flat
            logps[t] = logp
            vals[t] = v.cpu().numpy()
            rews[t] = reward
            terms[t] = terminated.astype(np.float32)
            prev_done = np.logical_or(terminated, truncated).astype(np.float32)
            ep_rew.append(float(np.mean(reward)))
            obs = next_obs
        self._prev_done = prev_done

        with torch.no_grad():
            last_val = self.critic(obs).cpu().numpy()

        # optional running reward scaling (matches DPPO: operate on (N, S))
        if self.reward_scaler is not None:
            rews = self.reward_scaler(reward=rews.T, first=firsts.T).T

        # GAE -- bootstrap with `terminated` only (truncation still bootstraps)
        adv = np.zeros((S, N), dtype=np.float32)
        lastgae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(S)):
            nextval = last_val if t == S - 1 else vals[t + 1]
            nonterminal = 1.0 - terms[t]
            delta = (rews[t] * self.reward_scale_const
                     + self.gamma * nextval * nonterminal - vals[t])
            lastgae = delta + self.gamma * self.gae_lambda * nonterminal * lastgae
            adv[t] = lastgae
        returns = adv + vals

        to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        batch = {
            "obs": {k: v.reshape(S * N, *v.shape[2:]) for k, v in obs_buf.items()},
            "acts": acts.reshape(S * N, self.noise_dim),
            "logps": logps.reshape(S * N),
            "advs": to_t(adv).reshape(S * N),
            "returns": to_t(returns).reshape(S * N),
        }
        stats = {"rollout/mean_reward": float(np.mean(ep_rew))}
        return batch, obs, stats

    # ------------------------------------------------------------------ #

    def update(self, batch: dict, update_actor: bool) -> dict:
        B = batch["acts"].shape[0]
        mb_size = max(1, B // self.num_minibatches)
        idxs = np.arange(B)

        advs = batch["advs"]
        if self.norm_adv:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        logs = {}
        for epoch in range(self.update_epochs):
            np.random.shuffle(idxs)
            approx_kls = []
            for start in range(0, B, mb_size):
                mb = idxs[start:start + mb_size]
                mb_obs = self._slice_obs(batch["obs"], mb)
                mb_acts = batch["acts"][mb]
                mb_old_logp = batch["logps"][mb]
                mb_adv = advs[mb]
                mb_ret = batch["returns"][mb]

                # critic
                new_val = self.critic(mb_obs)
                v_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
                self.critic_opt.zero_grad()
                (self.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                # actor (skipped during critic warmup)
                if update_actor:
                    new_logp, entropy = self.policy.evaluate(mb_obs, mb_acts)
                    kl_prior = self.policy.kl_to_standard_normal(mb_obs)  # to N(0,I)

                    logratio = new_logp - mb_old_logp
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kls.append(((ratio - 1) - logratio).mean().item())

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    actor_loss = (pg_loss
                                  - self.ent_coef * entropy.mean()
                                  + self.kl_coef * kl_prior.mean())
                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.actor_opt.step()

                    logs = {
                        "loss/policy": pg_loss.item(),
                        "loss/value": v_loss.item(),
                        "loss/entropy": entropy.mean().item(),
                        "loss/kl_to_prior": kl_prior.mean().item(),
                        "policy/approx_kl": approx_kls[-1],
                    }
                else:
                    logs = {"loss/value": v_loss.item()}
            if (update_actor and self.target_kl is not None
                    and len(approx_kls) and np.mean(approx_kls) > self.target_kl):
                logs["policy/early_stop_epoch"] = epoch
                break
        return logs

    # ------------------------------------------------------------------ #

    def train(self, log_fn=None):
        obs = self.env.reset()
        for it in range(self.n_iterations):
            batch, obs, roll_stats = self.collect(obs)
            update_actor = it >= self.n_critic_warmup_itr
            logs = self.update(batch, update_actor=update_actor)
            logs.update(roll_stats)
            logs["iter"] = it
            log.info(
                f"itr {it}: reward {roll_stats['rollout/mean_reward']:.4f} "
                f"| pg {logs.get('loss/policy', 0):.4f} "
                f"| klN(0,I) {logs.get('loss/kl_to_prior', 0):.3f} "
                f"| approx_kl {logs.get('policy/approx_kl', 0):.4f}"
                f"{'' if update_actor else '  [critic warmup]'}"
            )
            if log_fn is not None:
                log_fn(logs)
        return self.policy, self.critic
