"""Evaluate stochastic trajectory diversity for one fixed goal-conditioned input.

The script chooses one conditioning tuple (robot state + body-frame XY goal)
from the training dataset distribution, samples several diffusion trajectories
for that exact same conditioning, then writes:

  eval_sampling_diversity/comparison.png   root XY/yaw trajectories for all samples
  eval_sampling_diversity/sample_000.mp4   rendered generated motion sequence
  eval_sampling_diversity/sample_001.mp4   ...
  eval_sampling_diversity/dataset_*.mp4     source dataset chunk(s), when applicable
  eval_sampling_diversity/samples.npz      generated chunks and integrated root traces
  eval_sampling_diversity/log.txt          provenance and diversity metrics

The --checkpoint argument may be either a checkpoint .pt file or a run
directory. When a run directory is provided, the highest-numbered
checkpoint/state_*.pt is used.

Usage:
  uv run python script/eval_sampling_diversity.py \\
      --checkpoint log/lafan-pretrain/.../checkpoint/state_12.pt

  uv run python script/eval_sampling_diversity.py \\
      --checkpoint log/lafan-pretrain/.../2026-06-03_18-07-04_42 \\
      --conditioning_mode matched --n_samples 16
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from agent.dataset.sequence import StitchedSequenceDataset
from eval_circle import FREQ, OBS_DIM, _add_sphere, integrate_step, load_model, obs_to_qpos
from sample_diffusion import load_run_cfg, resolve_data_dir
from util.g1_obs import heading_yaw_rate


FALLBACK_XML_PATH = Path.home() / "drl/LATENT/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"
ConditioningMode = Literal[
    "matched",
    "random_state_random_goal",
    "dataset_state_random_goal",
    "random_state_dataset_goal",
]


@dataclass(frozen=True)
class Window:
    idx: int
    clip_idx: int
    frame_idx: int
    goal: np.ndarray
    goal_dist: float


@dataclass(frozen=True)
class Conditioning:
    mode: str
    state_window: Window
    goal_window: Window
    state_norm: torch.Tensor
    goal: np.ndarray


@dataclass(frozen=True)
class Record:
    obs_denorm: np.ndarray
    world_xy: np.ndarray
    world_yaw: float
    goal_world_xy: np.ndarray


def resolve_checkpoint(path: Path) -> Path:
    """Accept a .pt file or run/checkpoint directory and return one .pt path."""
    if path.is_file():
        return path
    ckpt_dir = path / "checkpoint"
    if path.name == "checkpoint":
        ckpt_dir = path
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"{path} is neither a checkpoint file nor a run directory with checkpoint/"
        )

    candidates = sorted(ckpt_dir.glob("state_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No state_*.pt checkpoints found under {ckpt_dir}")

    def step_num(p: Path) -> int:
        match = re.search(r"state_(\d+)\.pt$", p.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step_num)


def run_dir_from_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoint":
        return checkpoint.parent.parent
    return checkpoint.parent


def resolve_xml_path(cli_xml_path: str | None, data_dir: Path) -> str:
    if cli_xml_path:
        path = Path(cli_xml_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"--xml_path does not exist: {path}")
        return str(path)

    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        xml_path = metadata.get("xml_path")
        if xml_path and Path(xml_path).exists():
            print(f"Resolved --xml_path from dataset metadata: {xml_path}")
            return str(xml_path)

    if FALLBACK_XML_PATH.exists():
        print(f"Using fallback --xml_path: {FALLBACK_XML_PATH}")
        return str(FALLBACK_XML_PATH)

    raise FileNotFoundError(
        "Could not resolve a MuJoCo XML path. Pass --xml_path explicitly, or add "
        "metadata.json with an existing xml_path to the dataset directory."
    )


def clip_starts(traj_lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(traj_lengths[:-1])]).astype(np.int64)


def idx_to_clip_frame(idx: int, starts: np.ndarray, lengths: np.ndarray) -> tuple[int, int]:
    clip_idx = int(np.searchsorted(starts, idx, side="right") - 1)
    frame_idx = int(idx - starts[clip_idx])
    if frame_idx < 0 or frame_idx >= lengths[clip_idx]:
        raise IndexError(f"Global index {idx} is outside trajectory bounds")
    return clip_idx, frame_idx


def clip_frame_to_idx(clip_idx: int, frame_idx: int, starts: np.ndarray, lengths: np.ndarray) -> int:
    if clip_idx < 0 or clip_idx >= len(lengths):
        raise IndexError(f"clip_idx={clip_idx} is outside [0, {len(lengths) - 1}]")
    if frame_idx < 0 or frame_idx >= int(lengths[clip_idx]):
        raise IndexError(
            f"frame_idx={frame_idx} is outside clip {clip_idx} length {int(lengths[clip_idx])}"
        )
    return int(starts[clip_idx] + frame_idx)


def hindsight_goal(
    state_norm: np.ndarray,
    actions_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    freq: float = FREQ,
) -> np.ndarray:
    """Match XYGoalConditioner: integrate [state_t, actions_t:t+H] endpoint."""
    seq = np.concatenate([state_norm[None, :], actions_norm], axis=0) * std + mean
    gvec = seq[:, 0:3]
    gyro = seq[:, 3:6]
    vel_h = seq[:, 36:38]
    yaw_rate = heading_yaw_rate(gvec, gyro)

    yaw = np.zeros(len(seq), dtype=np.float64)
    yaw[1:] = np.cumsum(yaw_rate[:-1], dtype=np.float64) / freq
    cos_y = np.cos(yaw[:-1])
    sin_y = np.sin(yaw[:-1])
    vx_w = cos_y * vel_h[:-1, 0] - sin_y * vel_h[:-1, 1]
    vy_w = sin_y * vel_h[:-1, 0] + cos_y * vel_h[:-1, 1]
    return np.array([vx_w.sum(), vy_w.sum()], dtype=np.float64) / freq



def precompute_windows(
    states: np.ndarray,
    actions: np.ndarray,
    traj_lengths: np.ndarray,
    horizon_steps: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[list[Window], np.ndarray]:
    starts = clip_starts(traj_lengths)
    windows: list[Window] = []
    for clip_idx, (start, length) in enumerate(zip(starts, traj_lengths)):
        max_frame = int(length) - horizon_steps
        for frame_idx in range(max_frame + 1):
            idx = int(start + frame_idx)
            goal = hindsight_goal(states[idx], actions[idx : idx + horizon_steps], mean, std)
            windows.append(
                Window(
                    idx=idx,
                    clip_idx=clip_idx,
                    frame_idx=frame_idx,
                    goal=goal,
                    goal_dist=float(np.linalg.norm(goal)),
                )
            )
    if not windows:
        raise RuntimeError("No valid dataset windows; horizon_steps may be too large")
    return windows, starts


def window_by_clip_frame(
    windows_by_idx: dict[int, Window],
    clip_idx: int,
    frame_idx: int,
    starts: np.ndarray,
    traj_lengths: np.ndarray,
) -> Window:
    idx = clip_frame_to_idx(clip_idx, frame_idx, starts, traj_lengths)
    if idx not in windows_by_idx:
        raise ValueError(
            f"clip {clip_idx} frame {frame_idx} cannot provide a full horizon chunk"
        )
    return windows_by_idx[idx]



def random_window(windows: list[Window], seed: int) -> Window:
    rng = np.random.default_rng(seed)
    return windows[int(rng.integers(len(windows)))]


def choose_window_or_default(
    *,
    label: str,
    default_window: Window,
    windows_by_idx: dict[int, Window],
    starts: np.ndarray,
    traj_lengths: np.ndarray,
    clip_idx: int | None,
    frame_idx: int | None,
) -> Window:
    if clip_idx is None and frame_idx is None:
        return default_window
    if clip_idx is None or frame_idx is None:
        raise ValueError(f"Pass both --{label}_clip_idx and --{label}_frame_idx")
    return window_by_clip_frame(windows_by_idx, clip_idx, frame_idx, starts, traj_lengths)


def choose_conditioning(
    mode: ConditioningMode,
    windows: list[Window],
    windows_by_idx: dict[int, Window],
    starts: np.ndarray,
    traj_lengths: np.ndarray,
    states_t: torch.Tensor,
    args: argparse.Namespace,
) -> Conditioning:
    selected_default = random_window(windows, args.seed)
    random_state_default = random_window(windows, args.seed + 10_007)
    random_goal_seed = args.goal_seed if args.goal_seed is not None else args.seed + 20_011
    random_goal_default = random_window(windows, random_goal_seed)

    selected_state = choose_window_or_default(
        label="state",
        default_window=selected_default,
        windows_by_idx=windows_by_idx,
        starts=starts,
        traj_lengths=traj_lengths,
        clip_idx=args.state_clip_idx,
        frame_idx=args.state_frame_idx,
    )
    selected_goal = choose_window_or_default(
        label="goal",
        default_window=selected_state,
        windows_by_idx=windows_by_idx,
        starts=starts,
        traj_lengths=traj_lengths,
        clip_idx=args.goal_clip_idx,
        frame_idx=args.goal_frame_idx,
    )

    if mode == "matched":
        state_window = selected_state
        goal_window = selected_state
    elif mode == "random_state_random_goal":
        state_window = random_state_default
        goal_window = random_goal_default
    elif mode == "dataset_state_random_goal":
        state_window = selected_state
        goal_window = random_goal_default
    elif mode == "random_state_dataset_goal":
        state_window = random_state_default
        goal_window = selected_goal
    else:
        raise ValueError(f"Unknown conditioning mode: {mode}")

    return Conditioning(
        mode=mode,
        state_window=state_window,
        goal_window=goal_window,
        state_norm=states_t[state_window.idx],
        goal=goal_window.goal.copy(),
    )


def world_goal_from_body(goal_body: np.ndarray, world_xy: np.ndarray, world_yaw: float) -> np.ndarray:
    c, s = np.cos(world_yaw), np.sin(world_yaw)
    return world_xy + np.array(
        [c * goal_body[0] - s * goal_body[1], s * goal_body[0] + c * goal_body[1]],
        dtype=np.float64,
    )


@torch.no_grad()
def sample_one(
    model,
    conditioning: Conditioning,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    cond_steps: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[Record]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_buf = [conditioning.state_norm.clone() for _ in range(cond_steps)]
    state_t = torch.stack(obs_buf).unsqueeze(0).to(device)
    goal_t = torch.tensor(conditioning.goal, dtype=torch.float32, device=device).unsqueeze(0)
    chunk_norm = model(cond={"state": state_t, "goal": goal_t}).trajectories.squeeze(0)
    chunk_norm_np = chunk_norm.cpu().numpy()
    chunk_denorm = chunk_norm_np * norm_std + norm_mean

    world_xy = np.zeros(2, dtype=np.float64)
    world_yaw = 0.0
    goal_world_xy = world_goal_from_body(conditioning.goal, world_xy, world_yaw)
    records: list[Record] = []
    for obs_denorm in chunk_denorm:
        records.append(
            Record(
                obs_denorm=obs_denorm,
                world_xy=world_xy.copy(),
                world_yaw=float(world_yaw),
                goal_world_xy=goal_world_xy.copy(),
            )
        )
        world_xy, world_yaw = integrate_step(obs_denorm, world_xy, world_yaw, 1.0 / FREQ)
    return chunk_norm_np, chunk_denorm, records


def records_from_dataset_window(
    window: Window,
    actions_norm: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, list[Record]]:
    chunk_denorm = actions_norm[window.idx : window.idx + horizon_steps] * norm_std + norm_mean
    world_xy = np.zeros(2, dtype=np.float64)
    world_yaw = 0.0
    goal_world_xy = world_goal_from_body(window.goal, world_xy, world_yaw)
    records: list[Record] = []
    for obs_denorm in chunk_denorm:
        records.append(
            Record(
                obs_denorm=obs_denorm,
                world_xy=world_xy.copy(),
                world_yaw=float(world_yaw),
                goal_world_xy=goal_world_xy.copy(),
            )
        )
        world_xy, world_yaw = integrate_step(obs_denorm, world_xy, world_yaw, 1.0 / FREQ)
    return chunk_denorm, records


def dataset_reference_windows(conditioning: Conditioning) -> list[tuple[str, Window]]:
    if conditioning.mode == "matched":
        return [("dataset_matched", conditioning.state_window)]

    refs = [("dataset_state_window", conditioning.state_window)]
    if conditioning.goal_window.idx != conditioning.state_window.idx:
        refs.append(("dataset_goal_window", conditioning.goal_window))
    return refs


def render_to_video(
    records: list[Record],
    xml_path: str,
    out_path: str,
    fps: int = 50,
    width: int = 640,
    height: int = 480,
    cam_distance: float = 4.0,
    cam_elevation: float = -35.0,
    cam_azimuth: float = 90.0,
) -> None:
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_qpos = (m.key_qpos[kid, 7:] if kid >= 0 else m.qpos0[7:]).copy()
    if default_qpos.shape != (29,):
        raise RuntimeError(f"Expected 29 default joint positions, got {default_qpos.shape}")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth = cam_azimuth

    frames = []
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for rec in records:
            qpos = obs_to_qpos(rec.obs_denorm, rec.world_xy, rec.world_yaw, default_qpos)
            d.qpos[:36] = qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)

            cam.lookat[0] = rec.world_xy[0]
            cam.lookat[1] = rec.world_xy[1]
            cam.lookat[2] = 0.9

            renderer.update_scene(d, camera=cam)
            _add_sphere(
                renderer.scene,
                np.array([rec.goal_world_xy[0], rec.goal_world_xy[1], 0.1]),
                0.12,
                np.array([1.0, 0.85, 0.0, 1.0]),
            )
            frames.append(renderer.render().copy())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    print(f"  saved {len(frames)} frames ({len(frames) / fps:.1f}s) -> {out_path}")


def plot_tracks(
    all_records: list[list[Record]],
    conditioning: Conditioning,
    out_path: str,
    yaw_stride: int,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(all_records)))
    fig, ax = plt.subplots(figsize=(7, 7))

    for i, records in enumerate(all_records):
        xy = np.array([r.world_xy for r in records])
        yaw = np.array([r.world_yaw for r in records])
        ax.plot(xy[:, 0], xy[:, 1], color=colors[i], linewidth=1.2, alpha=0.75)
        stride = max(1, yaw_stride)
        arrow_idx = np.arange(0, len(records), stride)
        ax.quiver(
            xy[arrow_idx, 0],
            xy[arrow_idx, 1],
            np.cos(yaw[arrow_idx]),
            np.sin(yaw[arrow_idx]),
            color=colors[i],
            angles="xy",
            scale_units="xy",
            scale=8.0,
            width=0.004,
            alpha=0.65,
        )

    goal = all_records[0][0].goal_world_xy
    ax.scatter([0.0], [0.0], s=80, color="black", marker="o", label="Start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], s=120, color="#FFC107", edgecolor="black",
               marker="*", label="Goal", zorder=6)

    if conditioning.state_window.idx == conditioning.goal_window.idx:
        ax.plot(
            [0.0, goal[0]],
            [0.0, goal[1]],
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label="Dataset endpoint",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Generated Root XY/Yaw Diversity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved comparison plot -> {out_path}")


def write_log(
    out_path: Path,
    args: argparse.Namespace,
    checkpoint: Path,
    data_dir: Path,
    conditioning: Conditioning,
    all_records: list[list[Record]],
) -> None:
    final_xy = np.array([records[-1].world_xy for records in all_records])
    goal_xy = all_records[0][0].goal_world_xy
    final_dist = np.linalg.norm(final_xy - goal_xy[None, :], axis=1)
    pairwise = []
    for i in range(len(final_xy)):
        for j in range(i + 1, len(final_xy)):
            pairwise.append(float(np.linalg.norm(final_xy[i] - final_xy[j])))
    pairwise_arr = np.array(pairwise, dtype=np.float64) if pairwise else np.zeros(1)

    lines = [
        "Trajectory diversity for fixed conditioning",
        "=" * 52,
        "",
        f"checkpoint: {checkpoint}",
        f"data_dir:   {data_dir}",
        f"out_dir:    {args.out_dir}",
        f"mode:       {conditioning.mode}",
        f"n_samples:  {args.n_samples}",
        f"seed:       {args.seed}",
        f"goal_seed:  {args.goal_seed if args.goal_seed is not None else args.seed + 20_011}",
        "",
        "[conditioning]",
        f"state: clip={conditioning.state_window.clip_idx} "
        f"frame={conditioning.state_window.frame_idx} idx={conditioning.state_window.idx} "
        f"matched_goal_dist={conditioning.state_window.goal_dist:.6f}",
        f"goal:  clip={conditioning.goal_window.clip_idx} "
        f"frame={conditioning.goal_window.frame_idx} idx={conditioning.goal_window.idx} "
        f"xy_body=[{conditioning.goal[0]:.6f}, {conditioning.goal[1]:.6f}] "
        f"dist={conditioning.goal_window.goal_dist:.6f}",
        "",
        "[diversity]",
        f"final_dist_to_goal_mean={final_dist.mean():.6f}",
        f"final_dist_to_goal_std={final_dist.std():.6f}",
        f"final_dist_to_goal_min={final_dist.min():.6f}",
        f"final_dist_to_goal_max={final_dist.max():.6f}",
        f"endpoint_spread_x_std={final_xy[:, 0].std():.6f}",
        f"endpoint_spread_y_std={final_xy[:, 1].std():.6f}",
        f"pairwise_endpoint_dist_mean={pairwise_arr.mean():.6f}",
        f"pairwise_endpoint_dist_max={pairwise_arr.max():.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"  saved log -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Dataset dir (norm_stats.npz + train.npz). Defaults to the "
        "dataset the checkpoint was trained with, read from its config.",
    )
    ap.add_argument("--xml_path", type=str, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument(
        "--conditioning_mode",
        choices=[
            "matched",
            "random_state_random_goal",
            "dataset_state_random_goal",
            "random_state_dataset_goal",
        ],
        default="matched",
        help="Conditioning is selected once per run, then reused for all samples.",
    )
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="Seed for conditioning selection")
    ap.add_argument(
        "--goal_seed",
        type=int,
        default=None,
        help="Override only the random-goal window seed; defaults to --seed + 20011.",
    )
    ap.add_argument("--sample_seed_start", type=int, default=0)
    ap.add_argument("--state_clip_idx", type=int, default=None)
    ap.add_argument("--state_frame_idx", type=int, default=None)
    ap.add_argument("--goal_clip_idx", type=int, default=None)
    ap.add_argument("--goal_frame_idx", type=int, default=None)
    ap.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--yaw_stride", type=int, default=5)
    ap.add_argument("--obs_dim", type=int, default=None)
    ap.add_argument("--goal_dim", type=int, default=None)
    ap.add_argument("--cond_steps", type=int, default=None)
    ap.add_argument("--horizon_steps", type=int, default=None)
    ap.add_argument("--denoising_steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cam_distance", type=float, default=4.0)
    ap.add_argument("--cam_elevation", type=float, default=-35.0)
    ap.add_argument("--cam_azimuth", type=float, default=90.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    cfg = load_run_cfg(str(checkpoint))
    if args.data_dir is None:
        args.data_dir = resolve_data_dir(cfg)
        if args.data_dir is None:
            raise SystemExit(
                "Could not resolve --data_dir from the checkpoint config; pass it explicitly."
            )
        print(f"Resolved --data_dir from checkpoint config: {args.data_dir}")

    obs_dim = args.obs_dim if args.obs_dim is not None else cfg.get("obs_dim", OBS_DIM)
    goal_dim = args.goal_dim if args.goal_dim is not None else cfg.get("goal_dim", 2)
    cond_steps = args.cond_steps or cfg.get("cond_steps", 1)
    horizon_steps = args.horizon_steps or cfg.get("horizon_steps", 50)
    denoising_steps = args.denoising_steps or cfg.get("denoising_steps", 100)
    if goal_dim != 2:
        raise SystemExit(f"This evaluator expects a 2-D XY goal, got goal_dim={goal_dim}")

    run_dir = run_dir_from_checkpoint(checkpoint)
    if args.out_dir is None:
        args.out_dir = run_dir / "eval_sampling_diversity"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = resolve_xml_path(args.xml_path, args.data_dir)

    norm = np.load(args.data_dir / "norm_stats.npz")
    norm_mean, norm_std = norm["mean"], norm["std"]
    action_dim = len(norm_mean)

    raw = np.load(args.data_dir / "train.npz", allow_pickle=False)
    states_np = raw["states"]
    actions_np = raw["actions"]
    traj_lengths = raw["traj_lengths"]

    print(f"Checkpoint: {checkpoint}")
    print(f"Output:     {args.out_dir}")
    print(
        f"Config: obs_dim={obs_dim} goal_dim={goal_dim} cond_steps={cond_steps} "
        f"horizon={horizon_steps} denoising={denoising_steps}"
    )
    print(f"XML:        {xml_path}")
    print(f"Conditioning mode: {args.conditioning_mode} (fixed across {args.n_samples} samples)")

    print("Precomputing dataset window goals...")
    windows, starts = precompute_windows(
        states_np, actions_np, traj_lengths, horizon_steps, norm_mean, norm_std
    )
    windows_by_idx = {w.idx: w for w in windows}
    dataset = StitchedSequenceDataset(
        dataset_path=str(args.data_dir / "train.npz"),
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        device=args.device,
    )
    conditioning = choose_conditioning(
        args.conditioning_mode,
        windows,
        windows_by_idx,
        starts,
        traj_lengths,
        dataset.states,
        args,
    )
    print(
        "Selected conditioning: "
        f"state clip={conditioning.state_window.clip_idx}/frame={conditioning.state_window.frame_idx}, "
        f"goal clip={conditioning.goal_window.clip_idx}/frame={conditioning.goal_window.frame_idx}, "
        f"goal={conditioning.goal.tolist()}"
    )

    dataset_ref_names: list[str] = []
    refs = dataset_reference_windows(conditioning)
    if refs:
        print("Rendering dataset reference clip(s)...")
        for name, window in refs:
            _, dataset_records = records_from_dataset_window(
                window, actions_np, norm_mean, norm_std, horizon_steps
            )
            out_path = args.out_dir / f"{name}.mp4"
            render_to_video(
                dataset_records,
                xml_path,
                str(out_path),
                fps=int(FREQ),
                width=args.width,
                height=args.height,
                cam_distance=args.cam_distance,
                cam_elevation=args.cam_elevation,
                cam_azimuth=args.cam_azimuth,
            )
            dataset_ref_names.append(name)

    print("Loading model...")
    model = load_model(
        str(checkpoint),
        cond_steps,
        horizon_steps,
        action_dim,
        denoising_steps,
        goal_dim,
        args.device,
        obs_dim=obs_dim,
        cfg=cfg,
    )

    chunks_norm, chunks_denorm, xy_traces, yaw_traces, all_records = [], [], [], [], []
    for i in range(args.n_samples):
        seed = args.sample_seed_start + i
        print(f"[sample {i:03d}] seed={seed}")
        chunk_norm, chunk_denorm, records = sample_one(
            model,
            conditioning,
            norm_mean,
            norm_std,
            cond_steps,
            args.device,
            seed,
        )
        chunks_norm.append(chunk_norm)
        chunks_denorm.append(chunk_denorm)
        xy_traces.append(np.array([r.world_xy for r in records]))
        yaw_traces.append(np.array([r.world_yaw for r in records]))
        all_records.append(records)

        if args.render:
            render_to_video(
                records,
                xml_path,
                str(args.out_dir / f"sample_{i:03d}.mp4"),
                fps=int(FREQ),
                width=args.width,
                height=args.height,
                cam_distance=args.cam_distance,
                cam_elevation=args.cam_elevation,
                cam_azimuth=args.cam_azimuth,
            )

    print("Saving plot and arrays...")
    plot_tracks(
        all_records,
        conditioning,
        str(args.out_dir / "comparison.png"),
        yaw_stride=args.yaw_stride,
    )
    np.savez_compressed(
        args.out_dir / "samples.npz",
        chunks_norm=np.stack(chunks_norm),
        chunks_denorm=np.stack(chunks_denorm),
        xy=np.stack(xy_traces),
        yaw=np.stack(yaw_traces),
        goal_body=conditioning.goal,
        state_idx=np.array(conditioning.state_window.idx, dtype=np.int64),
        state_clip_idx=np.array(conditioning.state_window.clip_idx, dtype=np.int64),
        state_frame_idx=np.array(conditioning.state_window.frame_idx, dtype=np.int64),
        goal_idx=np.array(conditioning.goal_window.idx, dtype=np.int64),
        goal_clip_idx=np.array(conditioning.goal_window.clip_idx, dtype=np.int64),
        goal_frame_idx=np.array(conditioning.goal_window.frame_idx, dtype=np.int64),
        dataset_reference_names=np.array(dataset_ref_names, dtype=str),
    )
    print(f"  saved arrays -> {args.out_dir / 'samples.npz'}")

    write_log(
        args.out_dir / "log.txt",
        args,
        checkpoint,
        args.data_dir,
        conditioning,
        all_records,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
