"""Evaluate a noise-steering checkpoint on the episodic goal-reaching task.

Matches the training MDP exactly: isotropic goals sampled at a fixed radius,
remaining-displacement goal passed to the policy each chunk, full chunk executed
before replanning.  Robot state is never reset between goals — a new goal is
sampled from the current state after each episode.

For each policy mode three curves are reported (distance to goal vs chunk index):
  steered  — trained noise-steering policy (deterministic mean)
  random   — w ~ N(0, I) each chunk (prior ignores noise; baseline)
  zero     — w = 0 each chunk (deterministic prior mean; baseline)

Outputs:
  eval_noise_steering/distance_curves.png  — mean ± std distance over chunks
  eval_noise_steering/log.txt              — per-mode aggregate statistics

Usage:
  uv run python script/eval_noise_steering.py \\
      --prior_checkpoint log/bones_loco_core-pretrain/.../checkpoint/state_10.pt \\
      --policy_checkpoint log/bones_loco_core-noise-steering/.../checkpoint/state_N.pt

  uv run python script/eval_noise_steering.py \\
      --prior_checkpoint ... --policy_checkpoint ... \\
      --n_episodes 20 --goal_radius 2.5 --n_chunks 8
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.diffusion.diffusion import DiffusionModel
from model.diffusion.dit_diffusion import DiffusionDiT
from model.noise_steering.conditioning import FrozenHistoryEncoder
from model.noise_steering.policy import NoisePolicy
from agent.dataset.sequence import StitchedSequenceDataset
from sample_diffusion import load_run_cfg, resolve_data_dir, OBS_DIM
from eval_locomotion_utils import (
    FREQ,
    Record,
    CircleTrajectory,
    integrate_step,
    world_to_body,
    seed_obs_deque,
    make_goal,
    per_frame_dists,
    render_to_video,
    plot_episode_tracks,
    plot_ground_tracks,
)


# ---------------------------------------------------------------------------
# Checkpoint loading (unchanged from previous version)
# ---------------------------------------------------------------------------

def _resolve(val, cfg: dict):
    """Replace a raw Hydra interpolation string like '${key}' with cfg[key]."""
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        return cfg[val[2:-1]]
    return val


def load_prior(prior_checkpoint: str, cfg: dict, device: str) -> DiffusionModel:
    p       = cfg["prior"]
    net_cfg = p["network"]
    r       = lambda v: _resolve(v, cfg)  # noqa: E731

    horizon_steps   = r(p["horizon_steps"])
    obs_dim         = r(p["obs_dim"])
    action_dim      = r(p["action_dim"])
    denoising_steps = r(p["denoising_steps"])
    cond_steps      = r(cfg["cond_steps"])

    network = DiffusionDiT(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        cond_steps=cond_steps,
        goal_dim=net_cfg.get("goal_dim", 0),
        d_model=net_cfg.get("d_model", 384),
        n_heads=net_cfg.get("n_heads", 6),
        n_layers=net_cfg.get("n_layers", 6),
        ff_mult=net_cfg.get("ff_mult", 4),
        dropout=0.0,
    )
    prior = DiffusionModel(
        network=network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        predict_epsilon=p.get("predict_epsilon", False),
        denoised_clip_value=p.get("denoised_clip_value"),
        use_ddim=p.get("use_ddim", True),
        ddim_steps=r(p.get("ddim_steps")),
        device=device,
        network_path=prior_checkpoint,
    )
    prior.to(device).eval()
    for param in prior.parameters():
        param.requires_grad_(False)
    return prior


def load_noise_policy(
    policy_checkpoint: str,
    prior: DiffusionModel,
    cfg: dict,
    device: str,
) -> NoisePolicy:
    mc        = cfg.get("model", {})
    cond_steps = cfg.get("cond_steps", 1)
    enc = FrozenHistoryEncoder(prior.network, cond_steps=cond_steps)
    policy = NoisePolicy(
        enc,
        noise_shape=(prior.horizon_steps, prior.action_dim),
        goal_dim=2,
        hidden=mc.get("policy_hidden", mc.get("hidden", 256)),
        goal_emb_dim=mc.get("goal_emb_dim", 64),
        log_std_init=mc.get("log_std_init", 0.0),
    ).to(device)
    ckpt = torch.load(policy_checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    itr = ckpt.get("itr", "?")
    print(f"Loaded steering policy (itr {itr}) from {policy_checkpoint}")
    return policy


# ---------------------------------------------------------------------------
# Episodic evaluation
# ---------------------------------------------------------------------------

def _sample_goal(goal_radius: float) -> np.ndarray:
    # filled disk (matches training env): r = R*sqrt(U) is uniform over area
    theta = np.random.uniform(-np.pi, np.pi)
    r     = goal_radius * np.sqrt(np.random.uniform())
    return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)


@torch.no_grad()
def eval_episodic(
    prior: DiffusionModel,
    policy: NoisePolicy | None,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    seed_obs_norm: torch.Tensor,
    horizon_steps: int,
    cond_steps: int,
    device: str,
    n_episodes: int = 10,
    n_chunks: int = 8,
    goal_radius: float = 2.5,
    mode: str = "steered",
    seed: int = 0,
    collect_records: bool = False,
) -> tuple[list[dict], list[list[Record]]]:
    """Roll out n_episodes consecutive goal-reaching episodes without state reset.

    Each episode: sample an isotropic goal, run n_chunks full chunks, track
    cumulative body-frame displacement.  After each episode a fresh goal is
    sampled from the current (unreset) state.

    mode:
      "steered" — trained policy, deterministic mean
      "random"  — w ~ N(0, I) per chunk
      "zero"    — w = 0 per chunk

    Returns (results, episode_records):
      results: list of per-episode dicts with goal, chunk_dists, final_dist, avg_dist
      episode_records: list of per-episode Record lists (only populated if collect_records=True)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt      = 1.0 / FREQ
    obs_buf = seed_obs_deque(seed_obs_norm, cond_steps)
    results = []
    episode_records: list[list[Record]] = []

    for _ in range(n_episodes):
        goal        = _sample_goal(goal_radius)
        episode_xy  = np.zeros(2, dtype=np.float64)
        episode_yaw = 0.0
        chunk_dists = []
        ep_records: list[Record] = []

        for _ in range(n_chunks):
            remaining = (goal - episode_xy.astype(np.float32))
            remaining_t = torch.tensor(remaining, dtype=torch.float32,
                                       device=device).unsqueeze(0)
            state_t = torch.stack(list(obs_buf)).unsqueeze(0).to(device)
            prior_cond = {"state": state_t}

            if mode == "steered":
                policy_cond = {"state": state_t, "goal": remaining_t}
                w, _, _ = policy.act(policy_cond, deterministic=True)
            elif mode == "random":
                w = torch.randn(1, prior.horizon_steps, prior.action_dim,
                                device=device)
            else:  # zero
                w = torch.zeros(1, prior.horizon_steps, prior.action_dim,
                                device=device)

            chunk_norm = (prior(cond=prior_cond, init_noise=w)
                          .trajectories.squeeze(0).cpu().numpy())

            # Execute all horizon_steps frames (matches training exactly)
            for i in range(horizon_steps):
                obs_denorm = chunk_norm[i] * norm_std + norm_mean
                episode_xy, episode_yaw = integrate_step(
                    obs_denorm, episode_xy, episode_yaw, dt
                )
                obs_buf.append(
                    torch.from_numpy(chunk_norm[i].astype(np.float32)).to(device)
                )
                if collect_records:
                    ep_records.append(
                        (obs_denorm, episode_xy.copy(), float(episode_yaw),
                         goal.copy())
                    )

            chunk_dists.append(float(np.linalg.norm(episode_xy - goal)))

        results.append({
            "goal":        goal.copy(),
            "chunk_dists": chunk_dists,
            "final_dist":  chunk_dists[-1],
            "avg_dist":    float(np.mean(chunk_dists)),
        })
        if collect_records:
            episode_records.append(ep_records)

    return results, episode_records


# ---------------------------------------------------------------------------
# Circle-following evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_circle_steered(
    prior: DiffusionModel,
    policy: NoisePolicy,
    circle: CircleTrajectory,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    seed_obs_norm: torch.Tensor,
    horizon_steps: int,
    cond_steps: int,
    total_steps: int,
    device: str,
    seed: int = 0,
) -> list[Record]:
    """Roll out the steered prior on a circle-following task.

    Replans every horizon_steps frames (full chunk execution, matching training).
    At each replan the goal is the circle position one horizon ahead in body frame.
    The prior never sees the goal.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt        = 1.0 / FREQ
    world_xy  = circle.position(0.0).copy()
    world_yaw = circle.start_yaw()
    obs_buf   = seed_obs_deque(seed_obs_norm, cond_steps)
    records: list[Record] = []

    t_sim = 0
    while t_sim < total_steps:
        goal_body, xy_goal_world = make_goal(
            "circle", circle, world_xy, world_yaw, t_sim, horizon_steps, None
        )
        goal_t  = torch.tensor(goal_body, dtype=torch.float32,
                               device=device).unsqueeze(0)
        state_t = torch.stack(list(obs_buf)).unsqueeze(0).to(device)

        policy_cond = {"state": state_t, "goal": goal_t}
        prior_cond  = {"state": state_t}
        w, _, _ = policy.act(policy_cond, deterministic=True)
        chunk_norm = (prior(cond=prior_cond, init_noise=w)
                      .trajectories.squeeze(0).cpu().numpy())

        steps_to_run = min(horizon_steps, total_steps - t_sim)
        for i in range(steps_to_run):
            obs_denorm = chunk_norm[i] * norm_std + norm_mean
            records.append(
                (obs_denorm, world_xy.copy(), float(world_yaw),
                 xy_goal_world.copy())
            )
            world_xy, world_yaw = integrate_step(obs_denorm, world_xy,
                                                  world_yaw, dt)
            obs_buf.append(
                torch.from_numpy(chunk_norm[i].astype(np.float32)).to(device)
            )
        t_sim += steps_to_run

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict]) -> dict:
    dists    = np.array([r["chunk_dists"] for r in results])   # (E, C)
    finals   = np.array([r["final_dist"]  for r in results])
    avgs     = np.array([r["avg_dist"]    for r in results])
    return {
        "curve_mean": dists.mean(0),   # (C,)
        "curve_std":  dists.std(0),
        "final_mean": float(finals.mean()),
        "final_std":  float(finals.std()),
        "avg_mean":   float(avgs.mean()),
        "avg_std":    float(avgs.std()),
    }


MODES = ["steered", "random", "zero"]
MODE_STYLE = {
    "steered": dict(color="#2196F3", label="Steered (policy)"),
    "random":  dict(color="#F44336", label="Random noise"),
    "zero":    dict(color="#FF9800", label="Zero noise"),
}


def plot_distance_curves(
    agg: dict[str, dict],
    n_chunks: int,
    goal_radius: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    chunks = np.arange(1, n_chunks + 1)
    for mode, stats in agg.items():
        style = MODE_STYLE[mode]
        mean  = stats["curve_mean"]
        std   = stats["curve_std"]
        ax.plot(chunks, mean, color=style["color"], label=style["label"], lw=2)
        ax.fill_between(chunks, mean - std, mean + std,
                        color=style["color"], alpha=0.15)
    ax.axhline(goal_radius, color="k", ls="--", lw=1, label=f"Goal radius ({goal_radius}m)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Distance to goal (m)")
    ax.set_title("Episodic goal-reaching: distance over chunks (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_log(
    agg: dict[str, dict],
    out_path: str,
    provenance: list[str],
    n_episodes: int,
    n_chunks: int,
    goal_radius: float,
) -> None:
    lines = [
        "Noise-steering episodic eval — goal-reaching distance",
        "=" * 60,
        *provenance,
        f"episodes: {n_episodes}  chunks/episode: {n_chunks}  "
        f"goal_radius: {goal_radius}m",
        "",
    ]
    for mode in MODES:
        if mode not in agg:
            continue
        s = agg[mode]
        lines += [
            f"[{mode}]",
            f"  final dist:  {s['final_mean']:.3f} ± {s['final_std']:.3f} m",
            f"  avg dist:    {s['avg_mean']:.3f} ± {s['avg_std']:.3f} m",
            f"  curve mean:  " + "  ".join(f"{v:.2f}" for v in s["curve_mean"]),
            "",
        ]
    txt = "\n".join(lines)
    print(txt)
    Path(out_path).write_text(txt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prior_checkpoint",  type=str, required=True)
    ap.add_argument("--policy_checkpoint", type=str, required=True)
    ap.add_argument("--data_dir",          type=Path, default=None)
    ap.add_argument("--out_dir",           type=str,  default=None)
    ap.add_argument("--n_episodes",        type=int,  default=8,
                    help="goals sampled per seed state")
    ap.add_argument("--n_chunks",          type=int,  default=8,
                    help="chunks executed per episode (matches training)")
    ap.add_argument("--goal_radius",       type=float, default=2.5,
                    help="isotropic goal radius in metres (matches training)")
    ap.add_argument("--n_seeds",           type=int,  default=3,
                    help="number of different seed obs states to average over")
    # Circle eval
    ap.add_argument("--radius",            type=float, default=2.0)
    ap.add_argument("--speed",             type=float, default=1.0)
    ap.add_argument("--duration",          type=float, default=12.0)
    ap.add_argument("--clip_idx",          type=int,  default=0)
    ap.add_argument("--frame_idx",         type=int,  default=0)
    ap.add_argument("--xml_path",          type=str,
                    default=str(Path.home() / "drl/LATENT/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"))
    ap.add_argument("--cam_distance",      type=float, default=5.0)
    ap.add_argument("--cam_elevation",     type=float, default=-55.0)
    ap.add_argument("--cam_azimuth",       type=float, default=90.0)
    ap.add_argument("--device",            type=str,  default="cuda:0")
    ap.add_argument("--seed",              type=int,  default=0)
    args = ap.parse_args()

    cfg = load_run_cfg(args.policy_checkpoint)
    if not cfg:
        raise SystemExit(
            "Could not find .hydra/config.yaml next to the policy checkpoint."
        )

    if args.data_dir is None:
        args.data_dir = resolve_data_dir(cfg)
        if args.data_dir is None:
            raise SystemExit("Could not resolve --data_dir; pass it explicitly.")
        print(f"Resolved --data_dir: {args.data_dir}")

    cond_steps    = cfg.get("cond_steps",    1)
    horizon_steps = cfg.get("horizon_steps", 40)

    norm = np.load(args.data_dir / "norm_stats.npz")
    norm_mean, norm_std = norm["mean"], norm["std"]

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(args.policy_checkpoint).parent.parent / "eval_noise_steering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prior  = load_prior(args.prior_checkpoint, cfg, args.device)
    policy = load_noise_policy(args.policy_checkpoint, prior, cfg, args.device)

    dataset = StitchedSequenceDataset(
        dataset_path=str(args.data_dir / "train.npz"),
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        device=args.device,
    )
    traj_lengths = np.load(args.data_dir / "train.npz")["traj_lengths"]
    frame_start  = int(np.sum(traj_lengths[:args.clip_idx])) + args.frame_idx

    # Draw n_seeds evenly spaced start frames
    total = dataset.states.shape[0]
    seed_frames = [
        dataset.states[int(i)]
        for i in np.linspace(frame_start, total - 1, args.n_seeds, dtype=int)
    ]

    print(f"Config: cond_steps={cond_steps} horizon={horizon_steps}")
    print(f"Eval: {args.n_seeds} seeds × {args.n_episodes} episodes × "
          f"{args.n_chunks} chunks  goal_radius={args.goal_radius}m\n")

    eval_kwargs = dict(
        prior=prior, norm_mean=norm_mean, norm_std=norm_std,
        horizon_steps=horizon_steps, cond_steps=cond_steps, device=args.device,
        n_episodes=args.n_episodes, n_chunks=args.n_chunks,
        goal_radius=args.goal_radius,
    )

    all_agg: dict[str, dict] = {}
    render_records: list[list[Record]] = []   # steered seed-0 episodes for video/plot

    for mode in MODES:
        all_results = []
        for si, seed_obs in enumerate(seed_frames):
            want_records = (mode == "steered" and si == 0)
            results, ep_recs = eval_episodic(
                **eval_kwargs,
                policy=policy if mode == "steered" else None,
                seed_obs_norm=seed_obs,
                mode=mode,
                seed=args.seed + si,
                collect_records=want_records,
            )
            all_results.extend(results)
            if want_records:
                render_records = ep_recs
            finals = [r["final_dist"] for r in results]
            print(f"  [{mode}] seed {si}: "
                  f"final_dist {np.mean(finals):.3f} ± {np.std(finals):.3f} m")
        all_agg[mode] = _aggregate(all_results)
        print()

    plot_distance_curves(
        all_agg, args.n_chunks, args.goal_radius,
        str(out_dir / "distance_curves.png"),
    )

    # Ground tracks for steered seed-0
    if render_records:
        print("Plotting episode ground tracks (steered seed 0)...")
        plot_episode_tracks(
            render_records, args.goal_radius,
            str(out_dir / "episode_tracks.png"),
            title="Steered policy — episodic goal-reaching tracks (seed 0)",
        )
        print("Rendering video (steered seed 0, all episodes)...")
        render_to_video(
            records=[rec for ep in render_records for rec in ep],
            xml_path=args.xml_path,
            out_path=str(out_dir / "steered_ep0.mp4"),
            cam_distance=args.cam_distance,
            cam_elevation=args.cam_elevation,
            cam_azimuth=args.cam_azimuth,
        )

    provenance = [
        f"prior:   {args.prior_checkpoint}",
        f"policy:  {args.policy_checkpoint}",
        f"data:    {args.data_dir}",
    ]
    write_log(all_agg, str(out_dir / "log.txt"), provenance,
              args.n_seeds * args.n_episodes, args.n_chunks, args.goal_radius)

    # ---------------------------------------------------------------------- #
    # Circle-following eval (seed 0 only)
    # ---------------------------------------------------------------------- #
    total_steps  = int(args.duration * FREQ)
    circle = CircleTrajectory(
        center=np.array([0.0, 0.0]), radius=args.radius, speed=args.speed
    )
    print(f"\nCircle eval: radius={args.radius}m  speed={args.speed}m/s  "
          f"duration={args.duration}s  replan=every {horizon_steps} steps")

    circle_seeds_records = []
    for si, seed_obs in enumerate(seed_frames):
        print(f"  seed {si} ...", end=" ", flush=True)
        recs = eval_circle_steered(
            prior=prior, policy=policy, circle=circle,
            norm_mean=norm_mean, norm_std=norm_std,
            seed_obs_norm=seed_obs,
            horizon_steps=horizon_steps, cond_steps=cond_steps,
            total_steps=total_steps,
            device=args.device, seed=args.seed + si,
        )
        circle_seeds_records.append(recs)
        dists = per_frame_dists(recs, circle)
        print(f"mean_dist={dists.mean():.3f}m  final_dist={dists[-1]:.3f}m")

    # Ground tracks
    print("Plotting circle ground tracks...")
    plot_ground_tracks(
        {"circle": circle_seeds_records},
        circle,
        str(out_dir / "circle_tracks.png"),
    )

    # Video of seed 0
    print("Rendering circle video (seed 0)...")
    render_to_video(
        records=circle_seeds_records[0],
        xml_path=args.xml_path,
        out_path=str(out_dir / "circle_steered.mp4"),
        circle=circle,
        cam_distance=args.cam_distance,
        cam_elevation=args.cam_elevation,
        cam_azimuth=args.cam_azimuth,
    )

    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
