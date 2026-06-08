"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""

import logging
import torch
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.sampling import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="cuda:0",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        cfg_scale=None,  # >1.0 enables classifier-free guidance at sampling
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        # Auxiliary losses (x0-prediction mode only)
        aux_loss_fn=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # Optional auxiliary loss module (e.g. geometric losses for G1 motion)
        self.aux_loss_fn = aux_loss_fn.to(device) if aux_loss_fn is not None else None

        # Set up models
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"DiffusionModel parameters: {n_params:,}")

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(denoising_steps).to(device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]]
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        """
        DDIM parameters

        In DDIM paper https://arxiv.org/pdf/2010.02502, alpha is alpha_cumprod in DDPM https://arxiv.org/pdf/2102.09672
        """
        if use_ddim:
            if ddim_discretize == "uniform":  # use the HF "leading" style
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = (
                    torch.arange(0, ddim_steps, device=self.device) * step_ratio
                )
            else:
                raise "Unknown discretization method for DDIM."
            self.ddim_alphas = (
                self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            )
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0]).to(torch.float32).to(self.device),
                    self.alphas_cumprod[self.ddim_t[:-1]],
                ]
            )
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (
                ddim_eta
                * (
                    (1 - self.ddim_alphas_prev)
                    / (1 - self.ddim_alphas)
                    * (1 - self.ddim_alphas / self.ddim_alphas_prev)
                )
                ** 0.5
            )

            # Flip all
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(
                self.ddim_sqrt_one_minus_alphas, [0]
            )
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    # ---------- Sampling ----------#

    def p_mean_var(self, x, t, cond, index=None, network_override=None):
        net = network_override if network_override is not None else self.network
        cfg_scale = getattr(self, "cfg_scale", None)
        if cfg_scale is None or cfg_scale == 1.0 or cond.get("goal") is None:
            noise = net(x, t, cond=cond)
        else:
            # CFG: batch [conditional | unconditional(null goal)] in one pass.
            B = x.shape[0]
            cond2 = {
                k: (torch.cat([v, v], 0) if torch.is_tensor(v) else v)
                for k, v in cond.items()
            }
            drop = torch.zeros(2 * B, dtype=torch.bool, device=x.device)
            drop[B:] = True
            cond2["goal_drop"] = drop
            out = net(torch.cat([x, x], 0), torch.cat([t, t], 0), cond=cond2)
            cond_out, uncond_out = out[:B], out[B:]
            noise = uncond_out + cfg_scale * (cond_out - uncond_out)

        # Convert the network output to (x₀, ε).
        if self.use_ddim:
            alpha = extract(self.ddim_alphas, index, x.shape)
            alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
            sqrt_one_minus_alpha = extract(
                self.ddim_sqrt_one_minus_alphas, index, x.shape
            )

        if self.predict_epsilon:
            epsilon = noise
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                x_recon = (x - sqrt_one_minus_alpha * epsilon) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * epsilon
                )
        else:  # network directly predicts x₀
            x_recon = noise
            if self.use_ddim:
                # implied ε from predicted x₀:  ε = (xₜ - √αₜ x₀) / √(1-αₜ)
                epsilon = (x - (alpha**0.5) * x_recon) / sqrt_one_minus_alpha

        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-derive ε so it stays consistent with the clamped x₀
                epsilon = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            epsilon.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε

            eta=0
            """
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * epsilon
            mu = (alpha_prev**0.5) * x_recon + dir_xt
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=True, init_noise=None):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            init_noise: optional (B, Ta, Da) initial noise x_T; with DDIM eta=0
                this makes init_noise -> trajectory deterministic (used by the
                noise-space steering policy). Defaults to a fresh N(0, I) draw.
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Loop
        if init_noise is not None:
            x = init_noise.to(device)
        else:
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = torch.zeros_like(std)
            else:
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None)

    # ---------- Supervised training ----------#

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, *args, t)

    def p_losses(
        self,
        x_start,
        cond: dict,
        t,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction="mean")
        else:
            loss = F.mse_loss(x_recon, x_start, reduction="mean")

        aux = {}
        if not self.predict_epsilon and self.aux_loss_fn is not None:
            aux = self.aux_loss_fn(x_recon, x_start)
            for v in aux.values():
                loss = loss + v
        return loss, aux

    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = x_start.device
            noise = torch.randn_like(x_start, device=device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
