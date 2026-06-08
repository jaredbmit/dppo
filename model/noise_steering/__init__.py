"""Latent-noise steering of a frozen, task-agnostic motion diffusion prior.

A small Gaussian policy learns a distribution over the diffusion prior's INITIAL
NOISE (not the model weights). The frozen prior is decoded deterministically
(DDIM eta=0), so the noise fully determines the generated motion chunk; RL (PPO)
then steers the noise toward a downstream task. See README.md for the data flow.
"""

from model.noise_steering.conditioning import FrozenHistoryEncoder
from model.noise_steering.policy import NoisePolicy, ValueCritic
from model.noise_steering.env_wrapper import NoiseSteeringEnv
from model.noise_steering.ppo import NoiseSteeringPPO

__all__ = [
    "FrozenHistoryEncoder",
    "NoisePolicy",
    "ValueCritic",
    "NoiseSteeringEnv",
    "NoiseSteeringPPO",
]
