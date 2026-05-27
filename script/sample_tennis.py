"""Sample from a pre-trained tennis diffusion policy checkpoint.

The policy predicts the full 38-D next observation, so autoregressive rollout
feeds the last predicted frame back as the next conditioning input.
Re-planning happens every horizon_steps frames.

Three output modes (all saved to --out_dir):

  chunks.npz
    N independently sampled next-obs chunks conditioned on random dataset
    frames.  Shape: (n_chunks, T_p, 38) unnormalized.

  teacher_forced.npz
    Long rollout conditioned on ground-truth dataset states every T_p steps.
    Shape: (T, 38) actions, (T, 38) gt_states, unnormalized.

  autoregressive.npz
    Long rollout where the last predicted frame is fed back as the next
    conditioning input.
    Shape: (T, 38) actions, unnormalized.

All outputs are in physical (unnormalized) units.

Usage:
  python script/sample_tennis.py \\
      --checkpoint log/tennis-pretrain/.../checkpoint/state_2000.pt \\
      --cond_steps 1
  python script/sample_tennis.py \\
      --checkpoint ... --cond_steps 4 --clip_idx 2 --rollout_steps 500
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.mlp_diffusion import DiffusionMLP
from agent.dataset.sequence import StitchedSequenceDataset

OBS_DIM = 38
FREQ = 50.0


def load_run_cfg(checkpoint_path: str) -> dict:
    """Load .hydra/config.yaml from the run directory containing this checkpoint."""
    cfg_path = Path(checkpoint_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, cond_steps: int, horizon_steps: int, action_dim: int, denoising_steps: int, device: str, obs_dim: int = OBS_DIM, cfg: dict = {}) -> DiffusionModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ckpt.get("ema", ckpt.get("model"))
    net_cfg = cfg.get("model", {}).get("network", {})
    network = DiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        time_dim=net_cfg.get("time_dim", 16),
        mlp_dims=net_cfg.get("mlp_dims", [512, 512, 512]),
        activation_type=net_cfg.get("activation_type", "ReLU"),
        out_activation_type=net_cfg.get("out_activation_type", "Identity"),
        use_layernorm=net_cfg.get("use_layernorm", False),
        residual_style=net_cfg.get("residual_style", True),
    )
    model = DiffusionModel(
        network=network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        predict_epsilon=False,
        denoised_clip_value=None,
        device=device,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_norm_stats(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    n = np.load(data_dir / "norm_stats.npz")
    return n["mean"], n["std"]


def denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def get_clip_states(dataset: StitchedSequenceDataset, clip_idx: int) -> torch.Tensor:
    """Return all normalized states for one clip, shape (T, 38)."""
    lengths = np.load(dataset.dataset_path, allow_pickle=False)["traj_lengths"]
    start = int(np.sum(lengths[:clip_idx]))
    end = start + int(lengths[clip_idx])
    return dataset.states[start:end]


def make_buffer(frames: deque, cond_steps: int, device: str) -> torch.Tensor:
    """Build (1, cond_steps, 38) conditioning tensor from a rolling frame buffer."""
    assert len(frames) == cond_steps
    return torch.stack(list(frames)).unsqueeze(0).to(device)


def seed_buffer(seed_frame: torch.Tensor, cond_steps: int) -> deque:
    """Initialize a rolling buffer by repeating the seed frame N times."""
    return deque([seed_frame.clone() for _ in range(cond_steps)], maxlen=cond_steps)


@torch.no_grad()
def sample_chunk(model: DiffusionModel, buffer: deque, cond_steps: int, device: str) -> np.ndarray:
    """Run one diffusion forward pass. Returns (T_p, ACT_DIM) numpy, normalized."""
    cond = {"state": make_buffer(buffer, cond_steps, device)}
    out = model(cond=cond)
    return out.trajectories.squeeze(0).cpu().numpy()  # (T_p, ACT_DIM)


# ---------------------------------------------------------------------------
# Sampling modes
# ---------------------------------------------------------------------------

def sample_random_chunks(
    model: DiffusionModel,
    dataset: StitchedSequenceDataset,
    n_chunks: int,
    cond_steps: int,
    device: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    indices = np.random.randint(0, len(dataset.states), size=n_chunks)
    chunks, cond_states = [], []
    for idx in indices:
        buf = seed_buffer(dataset.states[idx], cond_steps)
        chunk = sample_chunk(model, buf, cond_steps, device)
        chunks.append(denorm(chunk, mean, std))
        cond_states.append(denorm(dataset.states[idx].cpu().numpy(), mean, std))
    return {
        "actions": np.stack(chunks),           # (n_chunks, T_p, 38)
        "cond_states": np.stack(cond_states),  # (n_chunks, 38)
    }


def sample_teacher_forced(
    model: DiffusionModel,
    clip_states: torch.Tensor,
    rollout_steps: int,
    cond_steps: int,
    horizon_steps: int,
    device: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    T = min(rollout_steps, len(clip_states) - horizon_steps)
    all_actions, all_gt_states = [], []
    t = 0
    while t + horizon_steps <= T:
        buf = deque(
            [clip_states[max(0, t - i)] for i in reversed(range(cond_steps))],
            maxlen=cond_steps,
        )
        chunk = sample_chunk(model, buf, cond_steps, device)
        all_actions.append(denorm(chunk, mean, std))
        all_gt_states.append(denorm(clip_states[t:t + horizon_steps].cpu().numpy(), mean, std))
        t += horizon_steps
    return {
        "actions": np.concatenate(all_actions, axis=0),
        "gt_states": np.concatenate(all_gt_states, axis=0),
    }


def sample_autoregressive(
    model: DiffusionModel,
    clip_states: torch.Tensor,
    rollout_steps: int,
    cond_steps: int,
    horizon_steps: int,
    device: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    buf = seed_buffer(clip_states[0], cond_steps)
    all_actions = []
    t = 0
    while t < rollout_steps:
        chunk = sample_chunk(model, buf, cond_steps, device)  # (T_p, 38) normalized
        all_actions.append(denorm(chunk, mean, std))
        buf.append(torch.from_numpy(chunk[-1].astype(np.float32)).to(clip_states.device))
        t += horizon_steps
    return {
        "actions": np.concatenate(all_actions, axis=0),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--obs_dim", type=int, default=None,
                    help="Defaults to value from training config (0 for unconditional)")
    ap.add_argument("--cond_steps", type=int, default=None,
                    help="Defaults to value from training config")
    ap.add_argument("--horizon_steps", type=int, default=None,
                    help="Defaults to value from training config")
    ap.add_argument("--denoising_steps", type=int, default=None,
                    help="Defaults to value from training config")
    ap.add_argument("--data_dir", type=Path, default=Path("data/tennis"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--clip_idx", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=64)
    ap.add_argument("--rollout_steps", type=int, default=400)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_run_cfg(args.checkpoint)
    obs_dim         = args.obs_dim         if args.obs_dim is not None else cfg.get("obs_dim",         OBS_DIM)
    cond_steps      = args.cond_steps      or cfg.get("cond_steps",      1)
    horizon_steps   = args.horizon_steps   or cfg.get("horizon_steps",   16)
    denoising_steps = args.denoising_steps or cfg.get("denoising_steps", 100)

    if args.out_dir is None:
        args.out_dir = Path(args.checkpoint).parent.parent / "samples"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm_stats(args.data_dir)
    action_dim = len(mean)

    print(f"Loading model (obs_dim={obs_dim}, cond_steps={cond_steps}, horizon_steps={horizon_steps}, action_dim={action_dim}, denoising_steps={denoising_steps})...")
    model = load_model(args.checkpoint, cond_steps, horizon_steps, action_dim, denoising_steps, args.device, obs_dim=obs_dim, cfg=cfg)

    print("Loading dataset...")
    dataset = StitchedSequenceDataset(
        dataset_path=str(args.data_dir / "train.npz"),
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        device=args.device,
    )

    clip_states = get_clip_states(dataset, args.clip_idx)
    print(f"Clip {args.clip_idx}: {len(clip_states)} frames")

    print(f"\nSampling {args.n_chunks} random chunks...")
    chunks = sample_random_chunks(model, dataset, args.n_chunks, cond_steps, args.device, mean, std)
    np.savez_compressed(args.out_dir / "chunks.npz", **chunks)
    print(f"  actions: {chunks['actions'].shape}")

    print(f"\nSampling teacher-forced rollout ({args.rollout_steps} steps)...")
    tf = sample_teacher_forced(model, clip_states, args.rollout_steps, cond_steps, horizon_steps, args.device, mean, std)
    np.savez_compressed(args.out_dir / "teacher_forced.npz", **tf)
    print(f"  actions: {tf['actions'].shape}  gt_states: {tf['gt_states'].shape}")

    print(f"\nSampling autoregressive rollout ({args.rollout_steps} steps)...")
    ar = sample_autoregressive(model, clip_states, args.rollout_steps, cond_steps, horizon_steps, args.device, mean, std)
    np.savez_compressed(args.out_dir / "autoregressive.npz", **ar)
    print(f"  actions: {ar['actions'].shape}")

    print(f"\nDone. Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
