# Latent-Noise Steering

Steer a **frozen, task-agnostic** motion diffusion prior toward a downstream task
by learning a policy over its **initial noise** — never touching the prior's
weights, and never backpropagating through the denoising chain (that would be
DPPO; this is deliberately *not* that).

The motion prior is a generic, state-conditional model `p(future chunk | recent
history)` with **no goal input** (`goal_dim=0`). All task knowledge lives in a
small steering policy on top. The same frozen prior is reusable across tasks;
only the steering policy changes.

## Data flow (one RL step = one autoregressive motion chunk)

```
 recent history ──(frozen obs_proj)──► history feature ┐
 task goal      ──(policy goal embed)─────────────────┼──► policy π(w | h)  ── diagonal Gaussian over flattened noise
                                                       ┘                          │
                                                                                  ▼   w   (Ta × Da initial noise)
                              frozen DDIM eta=0 decoder  (state-conditional prior, goal NOT passed)
                                                                                  │
                                                                                  ▼   motion chunk (Ta × Da)
                                       underlying task env.step(chunk)  ──►  reward, next history, goal
```

Because the decoder is **DDIM with eta=0**, the map `w → motion chunk` is a fixed,
deterministic function of `(w, history)`. The only stochasticity is `w`, which is
exactly the policy's action — so the policy gradient in noise space is clean.

## Why these choices (the non-obvious bits)

- **Init ≈ N(0, I).** The policy's mean head is zero-init and log-std is zero-init,
  so at start `π(w|h) = N(0, I)` for any input — i.e. the steered policy *is* the
  base prior. RL moves it from there.
- **KL-to-N(0, I) penalty (load-bearing).** Pulling `π` toward `N(0, I)` keeps `w`
  in the frozen decoder's input distribution *and* keeps the autoregressive
  rollout on-manifold as steered chunks feed future conditioning. Coefficient
  `kl_coef` is the main steering-vs-stay-natural knob.

## Modules

| file | what |
|------|------|
| `conditioning.py` | `FrozenHistoryEncoder` — taps the prior's frozen `obs_proj` for the history feature (not registered as a submodule, so it stays out of the policy's params/optimizer). |
| `policy.py` | `NoisePolicy` (tiny diagonal-Gaussian over flattened noise; goal embedded policy-side) and `ValueCritic` (state-value MLP). |
| `env_wrapper.py` | `NoiseSteeringEnv` — action space = noise; decodes `w` with the frozen DDIM decoder, plays the chunk in the underlying env. |
| `ppo.py` | `NoiseSteeringPPO` — clipped PPO + GAE, KL-to-N(0,I) penalty, matching DPPO's conventions (running reward scaler, `terminated`-based bootstrap, critic warmup). |

The deterministic decoder is the existing `DiffusionModel.forward(cond,
init_noise=w)` with `use_ddim=True` (eta=0 is hardcoded). The DDIM branch was
extended to support the repo's x0-prediction model (it derives the implied
epsilon for the DDIM direction term).

## Usage

```bash
# 1) train the task-agnostic prior (once; reusable across tasks)
uv run python script/run.py \
    --config-name=pre_diffusion_dit_stateonly \
    --config-dir=cfg/bones_loco_core/pretrain

# 2) train a steering policy on top of the frozen prior
#    (set base_prior_path in the cfg to the state-only checkpoint from step 1)
uv run python script/run.py \
    --config-name=steer_xy_goal \
    --config-dir=cfg/bones_loco_core/noise_steering
```

## Hyperparameters (defaults in `cfg/bones_loco_core/noise_steering/steer_xy_goal.yaml`)

| name | default | meaning |
|------|---------|---------|
| `train.kl_coef` | 0.1 | weight of KL(π ‖ N(0,I)); the single steering-authority knob — ↑ = stay natural, ↓ = steer harder. (Governs how far the mean drifts from 0; the KL term penalizes μ² directly.) |
| `model.log_std_init` | 0.0 | initial log-std (0.0 → std 1 → N(0,I) at init) |
| `model.hidden` / `model.goal_emb_dim` | 256 / 64 | policy/critic MLP width / goal embedding width |
| `env.n_envs` | 256 | parallel envs |
| `train.n_steps` | 16 | chunks collected per env per iteration |
| `train.n_train_itr` | 500 | PPO iterations |
| `ddim_steps` | 8 | denoising steps in the deterministic decoder (≤ `denoising_steps`) |
| `train.gamma` / `train.gae_lambda` | 0.99 / 0.95 | discount (per chunk) / GAE |
| `train.clip_coef` | 0.2 | PPO clip |
| `train.update_epochs` / `train.num_minibatches` | 4 / 4 | PPO update sweep |
| `train.vf_coef` / `train.ent_coef` | 0.5 / 0.0 | value / entropy loss weights |
| `train.reward_scale_running` / `train.reward_scale_const` | True / 1.0 | running reward scaler / constant scale |
| `train.n_critic_warmup_itr` | 5 | critic-only iterations before actor updates |
| `train.target_kl` | 0.05 | early-stop an update sweep on policy drift (old‖new) |
| `train.actor_lr` / `train.critic_lr` | 3e-4 / 1e-3 | learning rates |

Override any of these on the CLI, e.g. `... train.kl_coef=0.2 train.actor_lr=1e-4`.

**Exploration is coupled to `kl_coef`.** With `ent_coef=0`, the only thing keeping
the policy variance up is the KL-to-N(0,I) term — KL(π‖N(0,I)) → ∞ as σ → 0, so it
floors the variance. That's fine, but note: lowering `kl_coef` to "steer harder"
also weakens your only exploration pressure. If you lower it and the policy
collapses (σ shrinks, learning stalls), add a small entropy floor via `ent_coef`
rather than fighting it through `kl_coef` alone.

## Generalizing to other tasks

Only the underlying env changes. Any vec env that consumes chunk actions and
returns `{"state": ..., "goal"?: ...}` works with `NoiseSteeringEnv` as-is; the
prior, conditioning tap, policy, critic, and PPO are task-agnostic. For a task
with no explicit goal vector, set `goal_dim=0` and the policy conditions on
history alone.
