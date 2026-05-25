"""Sample from a pre-trained tennis diffusion policy checkpoint.

The policy predicts the full 64-D next observation (including gvec and gyro),
so autoregressive rollout is exact. Re-planning happens every T_p steps.
Supports arbitrary --cond_steps: a rolling buffer of the last N frames is
maintained and passed as conditioning. Episodes are padded with the seed frame
at the start, matching the training dataset behaviour.

Three output modes (all saved to --out_dir):

  chunks.npz
    N independently sampled next-obs chunks conditioned on random dataset
    frames.  Shape: (n_chunks, T_p, 64) unnormalized.

  teacher_forced.npz
    Long rollout conditioned on ground-truth dataset states every T_p steps.
    Shape: (T, 64) actions, (T, 64) gt_states, unnormalized.

  autoregressive.npz
    Long rollout where the last predicted frame is fed back as the next
    conditioning input (with a rolling buffer of depth cond_steps).
    Shape: (T, 64) actions, unnormalized.

All outputs are in physical units, i.e. min-max normalization inverted via
normalization.npz.

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

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.mlp_diffusion import DiffusionMLP
from agent.dataset.sequence import StitchedSequenceDataset

HORIZON_STEPS = 16
DENOISING_STEPS = 20
OBS_DIM = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, cond_steps: int, device: str) -> DiffusionModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network = DiffusionMLP(
        action_dim=OBS_DIM,
        horizon_steps=HORIZON_STEPS,
        cond_dim=OBS_DIM * cond_steps,
        time_dim=16,
        mlp_dims=[512, 512, 512],
        activation_type="ReLU",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=True,
    )
    model = DiffusionModel(
        network=network,
        horizon_steps=HORIZON_STEPS,
        obs_dim=OBS_DIM,
        action_dim=OBS_DIM,
        denoising_steps=DENOISING_STEPS,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        device=device,
    )
    state_dict = ckpt.get("ema", ckpt.get("model"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_norm_stats(data_dir: Path):
    n = np.load(data_dir / "normalization.npz")
    return n["obs_min"], n["obs_max"]


def denorm(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return ((x + 1.0) / 2.0) * (x_max - x_min) + x_min


def get_clip_states(dataset: StitchedSequenceDataset, clip_idx: int) -> torch.Tensor:
    """Return all normalized states for one clip, shape (T, 64)."""
    lengths = np.load(dataset.dataset_path, allow_pickle=False)["traj_lengths"]
    start = int(np.sum(lengths[:clip_idx]))
    end = start + int(lengths[clip_idx])
    return dataset.states[start:end]


def make_buffer(frames: deque, cond_steps: int, device: str) -> torch.Tensor:
    """Build (1, cond_steps, 64) conditioning tensor from a rolling frame buffer."""
    assert len(frames) == cond_steps
    return torch.stack(list(frames)).unsqueeze(0).to(device)  # (1, N, 64)


def seed_buffer(seed_frame: torch.Tensor, cond_steps: int) -> deque:
    """Initialize a rolling buffer by repeating the seed frame N times."""
    return deque([seed_frame.clone() for _ in range(cond_steps)], maxlen=cond_steps)


@torch.no_grad()
def sample_chunk(model: DiffusionModel, buffer: deque, cond_steps: int, device: str) -> np.ndarray:
    """Run one diffusion forward pass. Returns (T_p, 64) numpy, normalized."""
    cond = {"state": make_buffer(buffer, cond_steps, device)}
    out = model(cond=cond, deterministic=False)
    return out.trajectories.squeeze(0).cpu().numpy()  # (T_p, 64)


# ---------------------------------------------------------------------------
# Sampling modes
# ---------------------------------------------------------------------------

def sample_random_chunks(
    model: DiffusionModel,
    dataset: StitchedSequenceDataset,
    n_chunks: int,
    cond_steps: int,
    device: str,
    obs_min: np.ndarray,
    obs_max: np.ndarray,
) -> dict:
    indices = np.random.randint(0, len(dataset.states), size=n_chunks)
    chunks, cond_states = [], []
    for idx in indices:
        # Seed buffer with the chosen frame (no true history available for random samples).
        buf = seed_buffer(dataset.states[idx], cond_steps)
        chunk = sample_chunk(model, buf, cond_steps, device)
        chunks.append(denorm(chunk, obs_min, obs_max))
        cond_states.append(denorm(dataset.states[idx].cpu().numpy(), obs_min, obs_max))
    return {
        "actions": np.stack(chunks),          # (n_chunks, T_p, 64)
        "cond_states": np.stack(cond_states), # (n_chunks, 64)
    }


def sample_teacher_forced(
    model: DiffusionModel,
    clip_states: torch.Tensor,
    rollout_steps: int,
    cond_steps: int,
    device: str,
    obs_min: np.ndarray,
    obs_max: np.ndarray,
) -> dict:
    T = min(rollout_steps, len(clip_states) - HORIZON_STEPS)
    all_actions, all_gt_states = [], []
    t = 0
    while t + HORIZON_STEPS <= T:
        # Build buffer from GT frames, padding with frame 0 before episode start.
        buf = deque(
            [clip_states[max(0, t - i)] for i in reversed(range(cond_steps))],
            maxlen=cond_steps,
        )
        chunk = sample_chunk(model, buf, cond_steps, device)
        all_actions.append(denorm(chunk, obs_min, obs_max))
        all_gt_states.append(
            denorm(clip_states[t:t + HORIZON_STEPS].cpu().numpy(), obs_min, obs_max)
        )
        t += HORIZON_STEPS
    return {
        "actions": np.concatenate(all_actions, axis=0),      # (T, 64)
        "gt_states": np.concatenate(all_gt_states, axis=0),  # (T, 64)
    }


def sample_autoregressive(
    model: DiffusionModel,
    clip_states: torch.Tensor,
    rollout_steps: int,
    cond_steps: int,
    device: str,
    obs_min: np.ndarray,
    obs_max: np.ndarray,
) -> dict:
    buf = seed_buffer(clip_states[0], cond_steps)
    all_actions = []
    t = 0
    while t < rollout_steps:
        chunk = sample_chunk(model, buf, cond_steps, device)  # (T_p, 64) normalized
        all_actions.append(denorm(chunk, obs_min, obs_max))
        # Shift last predicted frame into the rolling buffer.
        buf.append(torch.from_numpy(chunk[-1]).float().to(clip_states.device))
        t += HORIZON_STEPS
    return {
        "actions": np.concatenate(all_actions, axis=0),  # (T, 64)
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--cond_steps", type=int, default=1,
                    help="Must match the cond_steps used during training")
    ap.add_argument("--data_dir", type=Path, default=Path("data/tennis"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/tennis/samples"))
    ap.add_argument("--clip_idx", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=64)
    ap.add_argument("--rollout_steps", type=int, default=400)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model (cond_steps={args.cond_steps})...")
    model = load_model(args.checkpoint, args.cond_steps, args.device)

    print("Loading dataset...")
    dataset = StitchedSequenceDataset(
        dataset_path=str(args.data_dir / "train.npz"),
        horizon_steps=HORIZON_STEPS,
        cond_steps=args.cond_steps,
        device=args.device,
    )
    obs_min, obs_max = load_norm_stats(args.data_dir)

    clip_states = get_clip_states(dataset, args.clip_idx)
    print(f"Clip {args.clip_idx}: {len(clip_states)} frames")

    print(f"\nSampling {args.n_chunks} random chunks...")
    chunks = sample_random_chunks(
        model, dataset, args.n_chunks, args.cond_steps, args.device, obs_min, obs_max
    )
    np.savez_compressed(args.out_dir / "chunks.npz", **chunks)
    print(f"  actions: {chunks['actions'].shape}")

    print(f"\nSampling teacher-forced rollout ({args.rollout_steps} steps)...")
    tf = sample_teacher_forced(
        model, clip_states, args.rollout_steps, args.cond_steps, args.device, obs_min, obs_max
    )
    np.savez_compressed(args.out_dir / "teacher_forced.npz", **tf)
    print(f"  actions: {tf['actions'].shape}  gt_states: {tf['gt_states'].shape}")

    print(f"\nSampling autoregressive rollout ({args.rollout_steps} steps)...")
    ar = sample_autoregressive(
        model, clip_states, args.rollout_steps, args.cond_steps, args.device, obs_min, obs_max
    )
    np.savez_compressed(args.out_dir / "autoregressive.npz", **ar)
    print(f"  actions: {ar['actions'].shape}")

    print(f"\nDone. Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
