"""Hydra agent for latent-noise steering of a frozen motion prior.

Mirrors the repo's training pattern (cfg `_target_` -> agent(cfg).run(), launched
via script/run.py). The frozen, task-agnostic diffusion prior is instantiated
from cfg.prior (with use_ddim=True / eta=0 for a deterministic decoder) and never
trained; only the small Gaussian policy + value critic are. See
model/noise_steering/README.md for the framework.
"""

from __future__ import annotations

import logging
import os

import hydra
import torch

from env.gym_utils.wrapper.g1_kinematic_lowdim import G1KinematicVecEnv
from model.noise_steering import (
    FrozenHistoryEncoder,
    NoisePolicy,
    ValueCritic,
    NoiseSteeringEnv,
    NoiseSteeringPPO,
)

log = logging.getLogger(__name__)


class TrainNoiseSteeringAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        torch.manual_seed(cfg.seed)

        # --- frozen task-agnostic prior (loads EMA weights via network_path) ---
        self.prior = hydra.utils.instantiate(cfg.prior).to(self.device).eval()
        assert getattr(self.prior, "use_ddim", False), "prior must use DDIM (eta=0)"
        assert self.prior.network.goal_dim == 0, (
            "noise steering expects a task-agnostic prior (network goal_dim=0)"
        )
        for p in self.prior.parameters():
            p.requires_grad_(False)

        # --- underlying task env (chunk actions); one chunk == one RL step ---
        self.venv = G1KinematicVecEnv(
            n_envs=cfg.env.n_envs,
            norm_stats_path=cfg.env.specific.norm_stats_path,
            dataset_path=cfg.env.specific.dataset_path,
            horizon_steps=cfg.horizon_steps,
            max_episode_steps=cfg.env.max_episode_steps,
            goal_scale=cfg.env.specific.get("goal_scale", 1.0),
            goal_sampling=cfg.env.specific.get("goal_sampling", "empirical"),
            goal_radius=cfg.env.specific.get("goal_radius", 2.5),
            n_obs_steps=cfg.cond_steps,
            act_steps=cfg.horizon_steps,
        )
        self.venv.seed(cfg.seed)
        self.env = NoiseSteeringEnv(self.venv, self.prior, device=self.device)

        # --- tiny goal-aware policy + state-value critic on the frozen tap ---
        enc = FrozenHistoryEncoder(self.prior.network, cond_steps=cfg.cond_steps)
        goal_dim = self.venv.goal_dim
        self.policy = NoisePolicy(
            enc,
            noise_shape=self.env.noise_shape,
            goal_dim=goal_dim,
            hidden=cfg.model.policy_hidden,
            goal_emb_dim=cfg.model.goal_emb_dim,
            log_std_init=cfg.model.log_std_init,
        ).to(self.device)
        self.critic = ValueCritic(
            enc, goal_dim=goal_dim,
            hidden=cfg.model.get("critic_hidden", cfg.model.policy_hidden),
            goal_emb_dim=cfg.model.goal_emb_dim,
        ).to(self.device)
        log.info(
            "policy params: %s | critic params: %s | noise dim: %d",
            f"{sum(p.numel() for p in self.policy.parameters()):,}",
            f"{sum(p.numel() for p in self.critic.parameters()):,}",
            self.policy.noise_dim,
        )

        t = cfg.train
        self.ppo = NoiseSteeringPPO(
            self.env, self.policy, self.critic, device=self.device,
            n_iterations=t.n_train_itr, n_steps=t.n_steps,
            n_critic_warmup_itr=t.n_critic_warmup_itr,
            gamma=t.gamma, gae_lambda=t.gae_lambda, clip_coef=t.clip_coef,
            update_epochs=t.update_epochs, num_minibatches=t.num_minibatches,
            vf_coef=t.vf_coef, ent_coef=t.ent_coef, kl_coef=t.kl_coef,
            max_grad_norm=t.max_grad_norm, target_kl=t.target_kl,
            reward_scale_running=t.reward_scale_running,
            reward_scale_const=t.reward_scale_const,
            actor_lr=t.actor_lr, critic_lr=t.critic_lr,
        )

        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_model_freq = t.save_model_freq

        self.use_wandb = cfg.get("wandb") is not None
        if self.use_wandb:
            import wandb
            from omegaconf import OmegaConf
            wandb.init(
                entity=cfg.wandb.entity, project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            self._wandb = wandb

    # ------------------------------------------------------------------ #

    def _on_iter(self, logs: dict) -> None:
        it = logs["iter"]
        if self.use_wandb:
            self._wandb.log(logs, step=it)
        if (it + 1) % self.save_model_freq == 0 or it + 1 == self.cfg.train.n_train_itr:
            self.save(it)

    def save(self, itr: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        torch.save(
            {"policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
             "itr": itr},
            path,
        )
        log.info("saved steering policy -> %s", path)

    def run(self) -> None:
        self.ppo.train(log_fn=self._on_iter)


if __name__ == "__main__":
    raise SystemExit("Launch via: python script/run.py --config-name=steer_xy_goal "
                     "--config-dir=cfg/bones_loco_core/noise_steering")
