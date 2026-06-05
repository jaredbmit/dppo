"""Measure generation quality and memorization for a pretrained diffusion policy.

Metrics
-------
Memorization (nearest-neighbour distances, normalized feature space):
  d_gen  : for each GENERATED chunk, L2 distance to its nearest training window.
  d_self : for held-out REAL windows, L2 to their nearest *other* training window
           -> the natural NN spacing of genuine motion.
  ratio  = median(d_gen) / median(d_self)
    ratio << 1  -> memorization   ratio ~  1  -> generalization

Quality / diversity (per-frame distribution, normalized feature space):
  fid        Fréchet distance between generated and real per-frame distributions
             (lower is better; 0 = identical distributions)
  diversity  mean pairwise L2 among generated chunks (higher = more varied)
  mean_err   mean |μ_gen - μ_real| over features
  std_err    mean |σ_gen - σ_real| over features

Distances are in the dataset's z-scored units so every feature is comparable.

Outputs to --out_dir (default <run>/generation_eval):
  summary.json              all metrics
  generation_analysis.png   NN distance histograms + per-feature stats plot
  pairs.npz                 closest generated chunks + nearest training neighbour
                            (denormalized), render with render_motion.py

Usage:
  uv run python script/eval_generation.py \\
      --checkpoint log/<ds>-pretrain/.../checkpoint/state_12.pt --data_dir data/<ds>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.dataset.sequence import StitchedSequenceDataset
from sample_diffusion import load_model, load_norm_stats, load_run_cfg, resolve_data_dir, OBS_DIM
from util.kinematics import G1Kinematics
from util.motion_metrics import frechet_distance, diversity, chunk_physics_metrics


def _gather_windows(source: torch.Tensor, starts: torch.Tensor, L: int) -> torch.Tensor:
    """(starts,) global start indices -> (len(starts), L, D) windows from source."""
    ar = torch.arange(L, device=source.device)
    idx = starts[:, None] + ar[None, :]          # (K, L)
    return source[idx]                           # (K, L, D)


@torch.no_grad()
def _generate(model, seed_frames: torch.Tensor, cond_steps: int, goal_dim: int,
              batch: int) -> torch.Tensor:
    """Sample one H-chunk per seed frame. Returns (n, H, D) normalized."""
    outs = []
    for i in range(0, seed_frames.shape[0], batch):
        f = seed_frames[i:i + batch]                       # (b, D)
        cond = {"state": f[:, None, :].repeat(1, cond_steps, 1)}  # (b, cond_steps, D)
        if goal_dim > 0:                                   # BC prior: zero goal
            cond["goal"] = torch.zeros(f.shape[0], goal_dim, device=f.device)
        outs.append(model(cond=cond).trajectories)         # (b, H, D)
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _nn_dist(queries: torch.Tensor, pool: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """Min L2 distance from each flattened query row to the pool. (Q,) on cpu."""
    out = torch.empty(queries.shape[0])
    for i in range(0, queries.shape[0], batch):
        d = torch.cdist(queries[i:i + batch], pool)        # (b, P)
        out[i:i + batch] = d.min(dim=1).values.cpu()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=Path, default=None,
                    help="Dataset dir (norm_stats.npz + train.npz). Defaults to the "
                         "dataset the checkpoint was trained with, read from its config.")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--n_gen", type=int, default=512, help="generated chunks")
    ap.add_argument("--pool_size", type=int, default=200_000,
                    help="random training windows used as the NN reference pool")
    ap.add_argument("--probe_size", type=int, default=512,
                    help="held-out real windows for d_self calibration")
    ap.add_argument("--n_render_pairs", type=int, default=4)
    ap.add_argument("--fid_ref_size", type=int, default=2048,
                    help="training windows sampled as the FID / quality reference")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    # arch overrides (default: read from the run's .hydra/config.yaml)
    ap.add_argument("--obs_dim", type=int, default=None)
    ap.add_argument("--goal_dim", type=int, default=None)
    ap.add_argument("--cond_steps", type=int, default=None)
    ap.add_argument("--horizon_steps", type=int, default=None)
    ap.add_argument("--denoising_steps", type=int, default=None)
    ap.add_argument("--cfg_scale", type=float, default=None,
                    help="classifier-free guidance scale; None/1.0 = no CFG")
    ap.add_argument("--xml_path", type=str, default=None,
                    help="G1 scene XML for physics metrics (default: from run config)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    cfg = load_run_cfg(args.checkpoint)
    if args.data_dir is None:
        args.data_dir = resolve_data_dir(cfg)
        if args.data_dir is None:
            raise SystemExit(
                "Could not resolve --data_dir from the checkpoint config; pass it "
                "explicitly (it must match the dataset the model was trained with)."
            )
        print(f"Resolved --data_dir from checkpoint config: {args.data_dir}")
    obs_dim         = args.obs_dim         if args.obs_dim  is not None else cfg.get("obs_dim", OBS_DIM)
    goal_dim        = args.goal_dim        if args.goal_dim is not None else cfg.get("goal_dim", 0)
    cond_steps      = args.cond_steps      or cfg.get("cond_steps", 1)
    horizon_steps   = args.horizon_steps   or cfg.get("horizon_steps", 50)
    denoising_steps = args.denoising_steps or cfg.get("denoising_steps", 20)

    if args.out_dir is None:
        args.out_dir = Path(args.checkpoint).parent.parent / "generation_eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm_stats(args.data_dir)
    action_dim = len(mean)
    L = horizon_steps

    print(f"Loading model (obs_dim={obs_dim}, goal_dim={goal_dim}, cond_steps={cond_steps}, "
          f"horizon={L}, denoising={denoising_steps})...")
    model = load_model(args.checkpoint, cond_steps, L, action_dim, denoising_steps,
                       args.device, obs_dim=obs_dim, goal_dim=goal_dim, cfg=cfg)
    model.cfg_scale = args.cfg_scale

    # FK for physics metrics; xml_path from CLI or the run's aux-loss config.
    xml_path = args.xml_path or (cfg.get("model", {}).get("aux_loss_fn", {}) or {}).get("xml_path")
    fk = G1Kinematics(xml_path).to(args.device) if xml_path else None
    if fk is None:
        print("No xml_path (CLI or config) — skipping physics metrics.")

    print("Loading dataset...")
    ds = StitchedSequenceDataset(
        dataset_path=str(args.data_dir / "train.npz"),
        horizon_steps=L, cond_steps=cond_steps,
        max_n_episodes=10**9, device=args.device,
    )
    all_starts = torch.tensor([s for s, _ in ds.indices], device=args.device)
    n_windows = all_starts.numel()
    print(f"{n_windows:,} valid length-{L} windows")

    # disjoint random subsets for pool / probe
    perm = torch.randperm(n_windows, generator=g, device=args.device)
    pool_size = min(args.pool_size, n_windows - args.probe_size)
    pool_starts  = all_starts[perm[:pool_size]]
    probe_starts = all_starts[perm[pool_size:pool_size + args.probe_size]]

    # action-space windows (the model's output space), flattened, normalized
    pool = _gather_windows(ds.actions, pool_starts, L).reshape(pool_size, L * action_dim)
    probe = _gather_windows(ds.actions, probe_starts, L).reshape(args.probe_size, L * action_dim)
    print(f"pool {tuple(pool.shape)} ({pool.element_size()*pool.nelement()/1e9:.2f} GB)")

    # generate from the model, seeded on random dataset states
    seed_idx = torch.randint(0, ds.states.shape[0], (args.n_gen,), generator=g, device=args.device)
    print(f"Generating {args.n_gen} chunks...")
    gen = _generate(model, ds.states[seed_idx], cond_steps, goal_dim, batch=128)
    gen_flat = gen.reshape(args.n_gen, L * action_dim)

    print("Computing nearest-neighbour distances...")
    d_gen = _nn_dist(gen_flat, pool)
    d_self = _nn_dist(probe, pool)

    # per-element RMS (interpretable: avg z-scored deviation per feature per frame)
    scale = (L * action_dim) ** 0.5
    rms_gen, rms_self = d_gen / scale, d_self / scale
    ratio = float(d_gen.median() / d_self.median())
    near_copy = float((d_gen < 0.25 * d_self.median()).float().mean())

    # --- quality / diversity metrics ---
    print("Computing FID, diversity, mean/std error...")

    # per-frame arrays in normalized feature space
    fid_ref_size = min(args.fid_ref_size, pool_size)
    fid_idx = torch.randperm(pool_size, device=args.device)[:fid_ref_size]
    ref_frames = (pool[fid_idx]
                  .reshape(fid_ref_size, L, action_dim)
                  .reshape(fid_ref_size * L, action_dim)
                  .cpu().numpy().astype(np.float32))
    gen_frames = gen.reshape(args.n_gen * L, action_dim).cpu().numpy().astype(np.float32)

    fid = frechet_distance(gen_frames, ref_frames)
    mean_err = float(np.abs(gen_frames.mean(0) - ref_frames.mean(0)).mean())
    std_err  = float(np.abs(gen_frames.std(0)  - ref_frames.std(0)).mean())
    div = diversity(gen_flat.cpu().float())

    # --- physics metrics (foot skate / penetration), gen vs real for calibration ---
    phys_gen, phys_real = {}, {}
    if fk is not None:
        print("Computing physics metrics (foot skate, penetration)...")
        real_chunks = pool[fid_idx].reshape(fid_ref_size, L, action_dim)
        phys_gen  = chunk_physics_metrics(gen, mean, std, fk)
        phys_real = chunk_physics_metrics(real_chunks, mean, std, fk)

    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": str(args.data_dir),
        "n_windows": int(n_windows),
        "pool_size": int(pool_size),
        "n_gen": int(args.n_gen),
        "horizon": L,
        # memorization
        "d_gen_median": float(d_gen.median()),
        "d_self_median": float(d_self.median()),
        "ratio_dgen_over_dself": ratio,
        "rms_gen_median": float(rms_gen.median()),
        "rms_self_median": float(rms_self.median()),
        "near_copy_fraction": near_copy,
        # quality / diversity
        "fid": fid,
        "diversity": div,
        "mean_err": mean_err,
        "std_err": std_err,
        # physics (compare gen vs real; gen close to real = physically plausible)
        "physics_gen": phys_gen,
        "physics_real": phys_real,
        "interpretation": ("ratio<<1 => memorization; ratio~1 => generalization. "
                           "near_copy_fraction = frac of samples within 25% of the "
                           "median real NN distance (near-exact copies). "
                           "fid: lower is better (Frechet distance vs real frame dist). "
                           "diversity: higher is more varied (mean pairwise L2 of chunks)."),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ MEMORIZATION SUMMARY ================")
    print(f"  d_gen  median (L2): {summary['d_gen_median']:.3f}   "
          f"per-elem RMS: {summary['rms_gen_median']:.4f}")
    print(f"  d_self median (L2): {summary['d_self_median']:.3f}   "
          f"per-elem RMS: {summary['rms_self_median']:.4f}")
    print(f"  ratio d_gen/d_self: {ratio:.3f}   (<<1 = memorized, ~1 = generalized)")
    print(f"  near-copy fraction: {near_copy:.3f}")
    print("=====================================================")
    print("\n================ QUALITY / DIVERSITY =================")
    print(f"  fid:                {fid:.4f}   (lower = closer to real dist)")
    print(f"  diversity:          {div:.4f}   (higher = more varied)")
    print(f"  mean_err:           {mean_err:.4f}   (L1 err of per-feature means)")
    print(f"  std_err:            {std_err:.4f}   (L1 err of per-feature stds)")
    print("=====================================================")
    if phys_gen:
        print("\n================ PHYSICS (gen vs real) ==============")
        print(f"  foot_skate (m/s):   {phys_gen['foot_skate']:.4f}  vs real {phys_real['foot_skate']:.4f}")
        print(f"  penetration_mean:   {phys_gen['penetration_mean']:.4f}  vs real {phys_real['penetration_mean']:.4f}")
        print(f"  penetration_frac:   {phys_gen['penetration_frac']:.4f}  vs real {phys_real['penetration_frac']:.4f}")
        print(f"  contact_frac:       {phys_gen['contact_frac']:.4f}  vs real {phys_real['contact_frac']:.4f}")
        print("  (gen close to real = physically plausible; lower skate/penetration better)")
        print("=====================================================\n")

    # histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    bins = np.linspace(0, float(max(d_gen.max(), d_self.max())), 60)
    ax.hist(d_self.numpy(), bins=bins, alpha=0.6, label="d_self (real→train)", color="tab:green")
    ax.hist(d_gen.numpy(), bins=bins, alpha=0.6, label="d_gen (gen→train)", color="tab:red")
    ax.set_xlabel("nearest-neighbour L2 distance (normalized)")
    ax.set_ylabel("count")
    ax.set_title(f"{args.data_dir.name}: ratio={ratio:.3f}  near-copy={near_copy:.2f}")
    ax.legend()

    ax = axes[1]
    feature_means_gen = gen_frames.mean(0)
    feature_means_ref = ref_frames.mean(0)
    feature_stds_gen  = gen_frames.std(0)
    feature_stds_ref  = ref_frames.std(0)
    feat_idx = np.arange(action_dim)
    ax.plot(feat_idx, feature_means_ref, color="tab:green", label="real mean", alpha=0.7)
    ax.plot(feat_idx, feature_means_gen, color="tab:red",   label="gen mean",  alpha=0.7)
    ax.fill_between(feat_idx,
                    feature_means_ref - feature_stds_ref,
                    feature_means_ref + feature_stds_ref,
                    color="tab:green", alpha=0.15)
    ax.fill_between(feat_idx,
                    feature_means_gen - feature_stds_gen,
                    feature_means_gen + feature_stds_gen,
                    color="tab:red", alpha=0.15)
    ax.set_xlabel("feature index")
    ax.set_ylabel("normalized value")
    ax.set_title(f"per-feature stats: FID={fid:.2f}  div={div:.3f}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.out_dir / "generation_analysis.png", dpi=120)
    plt.close(fig)

    # closest generated chunks + their nearest training neighbour (denormalized)
    k = min(args.n_render_pairs, args.n_gen)
    closest = torch.argsort(d_gen)[:k]
    nbr_starts = []
    for qi in closest:
        d = torch.cdist(gen_flat[qi:qi + 1], pool)
        nbr_starts.append(pool_starts[d.argmin()].item())
    nbr_starts = torch.tensor(nbr_starts, device=args.device)
    gen_pairs = gen[closest].cpu().numpy() * std + mean                      # (k,L,38)
    nbr_pairs = (_gather_windows(ds.actions, nbr_starts, L).cpu().numpy() * std + mean)
    np.savez_compressed(args.out_dir / "pairs.npz",
                        generated=gen_pairs, neighbor=nbr_pairs,
                        d_gen=d_gen[closest].numpy())
    print(f"Wrote summary.json, generation_analysis.png, pairs.npz -> {args.out_dir}")
    print("Render a pair with:")
    print(f"  uv run python script/render_motion.py --samples {args.out_dir/'pairs.npz'} "
          f"--key generated --xml_path <G1_XML> --out gen.mp4")
    print(f"  uv run python script/render_motion.py --samples {args.out_dir/'pairs.npz'} "
          f"--key neighbor  --xml_path <G1_XML> --out nbr.mp4")


if __name__ == "__main__":
    main()
