"""Evaluate a goal-conditioned diffusion policy on circle trajectory following.

Runs N seeds per goal mode (circle / zero / random) and produces:
  eval_circle/circle.mp4      — receding-horizon circle subgoal (seed 0)
  eval_circle/zero.mp4        — goal always [0, 0]  (seed 0)
  eval_circle/random.mp4      — goal ~ N(0, radius) each replan step (seed 0)
  eval_circle/comparison.png  — ground tracks for all seeds overlaid
  eval_circle/log.txt         — per-mode tracking error, physics (foot skate /
                                penetration), smoothness (jerk / SPARC), joint-limit
                                violations, and FID / diversity vs the training data

Re-planning period T_p is independent of the action horizon T_a.

Usage:
  uv run python script/eval_circle.py \\
      --checkpoint log/tennis-pretrain/.../checkpoint/state_2000.pt

  uv run python script/eval_circle.py \\
      --checkpoint ... --n_seeds 5 --radius 3.0 --speed 1.2 --duration 15.0
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from collections import deque
from pathlib import Path
from typing import Literal

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from model.diffusion.diffusion import DiffusionModel
from agent.dataset.sequence import StitchedSequenceDataset
from sample_diffusion import (
    load_model,
    load_run_cfg,
    resolve_data_dir,
    compute_hindsight_goal,
    OBS_DIM,
)
from util.g1_obs import heading_yaw_rate, roll_pitch_from_gvec
from util.kinematics import G1Kinematics
from util.motion_metrics import (
    rollout_physics_metrics,
    rollout_smoothness_metrics,
    rollout_limit_metrics,
    frechet_distance,
    diversity,
)

FREQ     = 50.0
GoalMode = Literal["circle", "zero", "random"]

MODE_STYLE = {
    "circle": dict(color="#2196F3", label="Circle goal"),
    "zero":   dict(color="#FF9800", label="Zero goal"),
    "random": dict(color="#F44336", label="Random goal"),
}


# Model loading (load_model, load_run_cfg) lives in sample_diffusion.py so eval
# scripts share one checkpoint-loading code path.


# ---------------------------------------------------------------------------
# Circle reference trajectory
# ---------------------------------------------------------------------------

class CircleTrajectory:
    """Counter-clockwise circle in world XY at constant tangential speed."""

    def __init__(self, center: np.ndarray, radius: float, speed: float, start_theta: float = 0.0):
        self.center      = np.asarray(center, dtype=np.float64)
        self.radius      = radius
        self.omega       = speed / radius
        self.start_theta = start_theta

    def position(self, t_seconds: float) -> np.ndarray:
        theta = self.start_theta + self.omega * t_seconds
        return self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])

    def dist_to(self, xy: np.ndarray) -> float:
        """Signed-magnitude distance from xy to the circle ring."""
        return abs(float(np.linalg.norm(xy - self.center)) - self.radius)

    def start_yaw(self) -> float:
        return self.start_theta + np.pi / 2.0

    def reference_points(self, n: int = 360) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.center + self.radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


# ---------------------------------------------------------------------------
# State integration helpers
# ---------------------------------------------------------------------------

def world_to_body(dx_world: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * dx_world[0] + s * dx_world[1],
                     -s * dx_world[0] + c * dx_world[1]])


def body_to_world(dx_body: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * dx_body[0] - s * dx_body[1],
                     s * dx_body[0] + c * dx_body[1]])


def integrate_step(
    obs_denorm: np.ndarray,
    world_xy: np.ndarray,
    world_yaw: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    yaw_rate = heading_yaw_rate(obs_denorm[0:3], obs_denorm[3:6])
    vel_h  = obs_denorm[36:38]
    c, s   = np.cos(world_yaw), np.sin(world_yaw)
    world_xy  = world_xy + np.array([c * vel_h[0] - s * vel_h[1],
                                      s * vel_h[0] + c * vel_h[1]]) * dt
    world_yaw = world_yaw + yaw_rate * dt
    return world_xy, world_yaw


def obs_to_qpos(
    obs_denorm: np.ndarray,
    world_xy: np.ndarray,
    world_yaw: float,
    default_qpos: np.ndarray,
) -> np.ndarray:
    gvec = obs_denorm[0:3].astype(np.float64)
    jpos = obs_denorm[6:35]
    roll, pitch = roll_pitch_from_gvec(gvec)
    R_root  = Rotation.from_euler("ZYX", [world_yaw, pitch, roll])
    xyzw    = R_root.as_quat()
    quat_mj = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    qpos = np.zeros(36)
    qpos[0:2]  = world_xy
    qpos[2]    = float(obs_denorm[35])
    qpos[3:7]  = quat_mj
    qpos[7:36] = jpos + default_qpos
    return qpos


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

Record = tuple[np.ndarray, np.ndarray, float, np.ndarray]


@torch.no_grad()
def training_goal_norms(
    dataset: StitchedSequenceDataset,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    horizon_steps: int,
    n_samples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Sample hindsight goal norms over training windows (same labeling as training)."""
    rng = np.random.default_rng(seed)
    mean_t = torch.from_numpy(norm_mean).float().to(dataset.states.device)
    std_t  = torch.from_numpy(norm_std).float().to(dataset.states.device)
    starts = [s for (s, _) in dataset.indices]
    pick = rng.choice(len(starts), size=min(n_samples, len(starts)), replace=False)
    norms = []
    for i in pick:
        start = starts[i]
        context = dataset.states[start]                              # anchor frame
        chunk   = dataset.actions[start:start + horizon_steps]       # (H, D)
        goal = compute_hindsight_goal(chunk, context, mean_t, std_t)  # (1, 2)
        norms.append(float(torch.linalg.norm(goal)))
    return np.array(norms)


@torch.no_grad()
def eval_circle(
    model: DiffusionModel,
    circle: CircleTrajectory,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    seed_obs_norm: torch.Tensor,
    horizon_steps: int,
    cond_steps: int,
    replan_steps: int,
    total_steps: int,
    device: str,
    goal_mode: GoalMode = "circle",
    seed: int = 0,
    goal_clip: float | None = None,
) -> list[Record]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt        = 1.0 / FREQ
    world_xy  = circle.position(0.0).copy()
    world_yaw = circle.start_yaw()
    obs_buf   = deque([seed_obs_norm.clone() for _ in range(cond_steps)], maxlen=cond_steps)
    records: list[Record] = []

    t_sim = 0
    while t_sim < total_steps:
        if goal_mode == "circle":
            t_goal_s      = (t_sim + horizon_steps) / FREQ
            xy_goal_world = circle.position(t_goal_s)
            goal_body     = world_to_body(xy_goal_world - world_xy, world_yaw)
        elif goal_mode == "zero":
            xy_goal_world = world_xy.copy()
            goal_body     = np.zeros(2)
        else:  # random
            goal_body = np.random.randn(2) * circle.radius
            c, s = np.cos(world_yaw), np.sin(world_yaw)
            xy_goal_world = world_xy + np.array([c * goal_body[0] - s * goal_body[1],
                                                  s * goal_body[0] + c * goal_body[1]])

        # Clamp the commanded goal to the training goal-norm range.
        if goal_clip is not None:
            n = float(np.linalg.norm(goal_body))
            if n > goal_clip:
                goal_body = goal_body * (goal_clip / n)
                xy_goal_world = world_xy + body_to_world(goal_body, world_yaw)

        goal_t  = torch.tensor(goal_body, dtype=torch.float32, device=device).unsqueeze(0)
        state_t = torch.stack(list(obs_buf)).unsqueeze(0).to(device)
        cond    = {"state": state_t, "goal": goal_t}

        chunk_norm = model(cond=cond).trajectories.squeeze(0).cpu().numpy()

        steps_to_run = min(replan_steps, total_steps - t_sim)
        for i in range(steps_to_run):
            obs_denorm = chunk_norm[i] * norm_std + norm_mean
            records.append((obs_denorm, world_xy.copy(), world_yaw, xy_goal_world.copy()))
            world_xy, world_yaw = integrate_step(obs_denorm, world_xy, world_yaw, dt)
            obs_buf.append(torch.from_numpy(chunk_norm[i].astype(np.float32)).to(device))

        t_sim += steps_to_run

    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_frame_dists(records: list[Record], circle: CircleTrajectory) -> np.ndarray:
    return np.array([
        np.linalg.norm(r[1] - circle.position(i / FREQ))
        for i, r in enumerate(records)
    ])


def seed_metric(records: list[Record], circle: CircleTrajectory) -> dict[str, float]:
    d = per_frame_dists(records, circle)
    return {"mean": float(d.mean()), "std": float(d.std()), "max": float(d.max())}


def seed_physics(records: list[Record], fk) -> dict[str, float]:
    """Foot-skate / penetration for one rollout, using its integrated world frame."""
    obs = np.stack([r[0] for r in records])        # (T, 38) denormalized
    xy  = np.stack([r[1] for r in records])        # (T, 2)
    yaw = np.array([r[2] for r in records])        # (T,)
    return rollout_physics_metrics(obs, xy, yaw, fk)


def seed_quality(records: list[Record], fk) -> dict[str, float]:
    """Smoothness + joint-limit violations for one rollout (frame-local, no world frame)."""
    obs = np.stack([r[0] for r in records])        # (T, 38) denormalized
    smooth = rollout_smoothness_metrics(obs)
    limits = rollout_limit_metrics(
        obs,
        fk.joint_limits.cpu().numpy(),
        fk.default_qpos.cpu().numpy(),
    )
    return {**smooth, **limits}


def mode_distribution(
    seeds_records: list[list[Record]],
    ref_frames_norm: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> dict[str, float]:
    """FID (vs the training distribution) and cross-seed diversity for one goal mode.

    Computed in normalized feature space to match eval_generation.py.
    """
    # Per-frame FID against the reference: pool all seeds' frames.
    gen = np.concatenate([np.stack([r[0] for r in recs]) for recs in seeds_records])
    gen_norm = ((gen - norm_mean) / norm_std).astype(np.float32)
    fid = frechet_distance(gen_norm, ref_frames_norm)

    # Diversity: mean pairwise L2 among per-seed flattened (normalized) trajectories.
    T = min(len(recs) for recs in seeds_records)
    flat = np.stack([
        ((np.stack([r[0] for r in recs[:T]]) - norm_mean) / norm_std).reshape(-1)
        for recs in seeds_records
    ]).astype(np.float32)
    div = diversity(torch.from_numpy(flat)) if len(seeds_records) > 1 else 0.0
    return {"fid": float(fid), "diversity": float(div)}


def write_log(
    all_records: dict[GoalMode, list[list[Record]]],
    circle: CircleTrajectory,
    out_path: str,
    provenance: list[str] | None = None,
    fk=None,
    ref_frames_norm: np.ndarray | None = None,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
) -> None:
    lines = ["Circle trajectory following — tracking error (‖robot − circle(t)‖)",
             "=" * 60, ""]
    if provenance:
        lines += provenance + [""]

    for mode, seeds_records in all_records.items():
        seed_means = []
        lines.append(f"[{mode}]")
        for s, records in enumerate(seeds_records):
            m = seed_metric(records, circle)
            lines.append(f"  seed {s}:  mean={m['mean']:.4f}m  std={m['std']:.4f}m  max={m['max']:.4f}m")
            seed_means.append(m["mean"])
        agg_mean = float(np.mean(seed_means))
        agg_std  = float(np.std(seed_means))
        lines.append(f"  aggregate ({len(seeds_records)} seeds):  "
                     f"mean={agg_mean:.4f} ± {agg_std:.4f} m")
        if fk is not None:
            phys = [seed_physics(r, fk) for r in seeds_records]
            agg = {k: float(np.mean([p[k] for p in phys])) for k in phys[0]}
            lines.append(f"  physics:  foot_skate={agg['foot_skate']:.4f} m/s  "
                         f"penetration_mean={agg['penetration_mean']:.4f} m  "
                         f"penetration_frac={agg['penetration_frac']:.4f}")
            qual = [seed_quality(r, fk) for r in seeds_records]
            q = {k: float(np.mean([p[k] for p in qual])) for k in qual[0]}
            lines.append(f"  smooth:   jerk_rms={q['jerk_rms']:.2f} rad/s^3  "
                         f"acc_rms={q['acc_rms']:.2f} rad/s^2  "
                         f"sparc={q['sparc']:.3f}  root_jerk_rms={q['root_jerk_rms']:.3f} m/s^3")
            lines.append(f"  limits:   pos_viol_frac={q['joint_pos_viol_frac']:.4f}  "
                         f"pos_viol_max={q['joint_pos_viol_max']:.4f} rad  "
                         f"joint_vel_max={q['joint_vel_max']:.2f} rad/s")
        if ref_frames_norm is not None and norm_mean is not None and norm_std is not None:
            dist = mode_distribution(seeds_records, ref_frames_norm, norm_mean, norm_std)
            lines.append(f"  distrib:  fid={dist['fid']:.4f} (vs train, lower=closer)  "
                         f"diversity={dist['diversity']:.4f} (cross-seed)")
        lines.append("")

    text = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    print(text)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_MAT_IDENTITY = np.eye(3).flatten().astype(np.float64)
_N_CIRCLE_VIZ = 72


def _add_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos.astype(np.float64),
        _MAT_IDENTITY,
        rgba.astype(np.float32),
    )
    scene.ngeom += 1


def render_to_video(
    records: list[Record],
    circle: CircleTrajectory,
    xml_path: str,
    out_path: str,
    fps: int = 50,
    width: int = 640,
    height: int = 480,
    cam_distance: float = 5.0,
    cam_elevation: float = -55.0,
    cam_azimuth: float = 90.0,
) -> None:
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_qpos = (m.key_qpos[kid, 7:] if kid >= 0 else m.qpos0[7:]).copy()
    if default_qpos.shape != (29,):
        raise RuntimeError(f"Expected 29 default joint positions, got {default_qpos.shape}")

    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth   = cam_azimuth

    angles     = np.linspace(0, 2 * np.pi, _N_CIRCLE_VIZ, endpoint=False)
    circle_pts = circle.center + circle.radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )

    frames = []
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for obs_denorm, world_xy, world_yaw, goal_world_xy in records:
            qpos = obs_to_qpos(obs_denorm, world_xy, world_yaw, default_qpos)
            d.qpos[:36] = qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)

            cam.lookat[0] = world_xy[0]
            cam.lookat[1] = world_xy[1]
            cam.lookat[2] = 0.9

            renderer.update_scene(d, camera=cam)
            scene = renderer.scene

            for pt in circle_pts:
                _add_sphere(scene, np.array([pt[0], pt[1], 0.05]), 0.04,
                            np.array([1.0, 1.0, 1.0, 0.6]))
            _add_sphere(scene, np.array([goal_world_xy[0], goal_world_xy[1], 0.1]), 0.12,
                        np.array([1.0, 0.85, 0.0, 1.0]))

            frames.append(renderer.render().copy())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.v3.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    print(f"  saved {len(frames)} frames ({len(frames)/fps:.1f}s) → {out_path}")


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_ground_tracks(
    all_records: dict[GoalMode, list[list[Record]]],
    circle: CircleTrajectory,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    # Reference circle
    ref = circle.reference_points(360)
    ax.plot(ref[:, 0], ref[:, 1], color="black", linestyle="--", linewidth=1.5,
            label="Reference", zorder=4)

    for mode, seeds_records in all_records.items():
        style = MODE_STYLE[mode]
        for i, records in enumerate(seeds_records):
            xy = np.array([r[1] for r in records])
            ax.plot(xy[:, 0], xy[:, 1],
                    color=style["color"], linewidth=1.0, alpha=0.35,
                    label=style["label"] if i == 0 else None,
                    zorder=3)

    # Start marker
    start = circle.position(0.0)
    ax.scatter(*start, s=80, color="black", zorder=5, marker="o", label="Start")

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ground Track Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved comparison plot → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint",      type=str,   required=True)
    ap.add_argument("--data_dir",        type=Path,  default=None,
                    help="Dataset dir (norm_stats.npz + train.npz). Defaults to the "
                         "dataset the checkpoint was trained with, read from its config.")
    ap.add_argument("--xml_path",        type=str,
                    default=str(Path.home() / "drl/LATENT/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"))
    ap.add_argument("--out_dir",         type=str,   default=None)
    ap.add_argument("--n_seeds",         type=int,   default=5)
    # Circle
    ap.add_argument("--radius",          type=float, default=2.0)
    ap.add_argument("--speed",           type=float, default=1.0)
    ap.add_argument("--duration",        type=float, default=10.0)
    # Planning
    ap.add_argument("--replan_steps",    type=int,   default=None)
    # Goal clamping (keeps commanded goal within the training goal-norm range)
    ap.add_argument("--goal_clip",       type=float, default=None,
                    help="max ||goal|| in meters; overrides --goal_clip_pct")
    ap.add_argument("--goal_clip_pct",   type=float, default=None,
                    help="clamp goal norm to this percentile of the training goal "
                         "distribution (e.g. 95). Ignored if --goal_clip is set.")
    # Model overrides
    ap.add_argument("--obs_dim",         type=int,   default=None)
    ap.add_argument("--goal_dim",        type=int,   default=None)
    ap.add_argument("--cond_steps",      type=int,   default=None)
    ap.add_argument("--horizon_steps",   type=int,   default=None)
    ap.add_argument("--denoising_steps", type=int,   default=None)
    ap.add_argument("--cfg_scale",       type=float, default=None,
                    help="classifier-free guidance scale; None/1.0 = no CFG")
    # Seed obs
    ap.add_argument("--clip_idx",        type=int,   default=0)
    ap.add_argument("--frame_idx",       type=int,   default=0)
    # Camera
    ap.add_argument("--cam_distance",    type=float, default=5.0)
    ap.add_argument("--cam_elevation",   type=float, default=-55.0)
    ap.add_argument("--cam_azimuth",     type=float, default=90.0)
    # Misc
    ap.add_argument("--device",          type=str,   default="cuda:0")
    args = ap.parse_args()

    cfg             = load_run_cfg(args.checkpoint)
    if args.data_dir is None:
        args.data_dir = resolve_data_dir(cfg)
        if args.data_dir is None:
            raise SystemExit(
                "Could not resolve --data_dir from the checkpoint config; pass it "
                "explicitly (it must match the dataset the model was trained with)."
            )
        print(f"Resolved --data_dir from checkpoint config: {args.data_dir}")
    obs_dim         = args.obs_dim         if args.obs_dim         is not None else cfg.get("obs_dim",         OBS_DIM)
    goal_dim        = args.goal_dim        if args.goal_dim        is not None else cfg.get("goal_dim",        2)
    cond_steps      = args.cond_steps      or cfg.get("cond_steps",      1)
    horizon_steps   = args.horizon_steps   or cfg.get("horizon_steps",   50)
    denoising_steps = args.denoising_steps or cfg.get("denoising_steps", 100)
    replan_steps    = args.replan_steps    or (horizon_steps // 2)
    total_steps     = int(args.duration * FREQ)

    norm = np.load(args.data_dir / "norm_stats.npz")
    norm_mean, norm_std = norm["mean"], norm["std"]
    action_dim = len(norm_mean)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent.parent / "eval_circle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: obs_dim={obs_dim} goal_dim={goal_dim} cond_steps={cond_steps} "
          f"horizon={horizon_steps} replan={replan_steps} denoising={denoising_steps}")
    print(f"Circle: radius={args.radius}m  speed={args.speed}m/s  duration={args.duration}s")
    print(f"Seeds: {args.n_seeds} per mode  |  Output: {out_dir}\n")

    model = load_model(
        args.checkpoint, cond_steps, horizon_steps, action_dim, denoising_steps,
        args.device, obs_dim=obs_dim, goal_dim=goal_dim, cfg=cfg,
    )
    model.cfg_scale = args.cfg_scale

    fk = G1Kinematics(args.xml_path).to(args.device)

    dataset = StitchedSequenceDataset(
        dataset_path=str(args.data_dir / "train.npz"),
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        device=args.device,
    )
    traj_lengths = np.load(args.data_dir / "train.npz")["traj_lengths"]
    frame_start  = int(np.sum(traj_lengths[:args.clip_idx])) + args.frame_idx
    seed_obs     = dataset.states[frame_start]

    circle = CircleTrajectory(center=np.array([0.0, 0.0]), radius=args.radius, speed=args.speed)

    # Report the training goal-norm distribution and resolve the clamp threshold.
    g = training_goal_norms(dataset, norm_mean, norm_std, horizon_steps)
    pcts = {p: float(np.percentile(g, p)) for p in (50, 90, 95, 99)}
    print(f"Training goal norms (m): mean={g.mean():.3f} "
          f"p50={pcts[50]:.3f} p90={pcts[90]:.3f} p95={pcts[95]:.3f} "
          f"p99={pcts[99]:.3f} max={g.max():.3f}")
    goal_clip = args.goal_clip
    if goal_clip is None and args.goal_clip_pct is not None:
        goal_clip = float(np.percentile(g, args.goal_clip_pct))
    if goal_clip is not None:
        print(f"Clamping commanded goal norm to {goal_clip:.3f} m")

    eval_kwargs = dict(
        model=model, circle=circle, norm_mean=norm_mean, norm_std=norm_std,
        seed_obs_norm=seed_obs, horizon_steps=horizon_steps, cond_steps=cond_steps,
        replan_steps=replan_steps, total_steps=total_steps, device=args.device,
        goal_clip=goal_clip,
    )
    render_kwargs = dict(
        circle=circle, xml_path=args.xml_path, fps=int(FREQ),
        cam_distance=args.cam_distance, cam_elevation=args.cam_elevation,
        cam_azimuth=args.cam_azimuth,
    )

    all_records: dict[GoalMode, list[list[Record]]] = {}
    for mode in ("circle", "zero", "random"):
        print(f"[{mode}]")
        seeds_records = []
        for s in range(args.n_seeds):
            print(f"  seed {s} ...", end=" ", flush=True)
            records = eval_circle(**eval_kwargs, goal_mode=mode, seed=s)
            seeds_records.append(records)
            m = seed_metric(records, circle)
            print(f"mean_dist={m['mean']:.3f}m")
        all_records[mode] = seeds_records

        # Render video for seed 0 only
        print(f"  rendering seed-0 video...")
        render_to_video(records=seeds_records[0], out_path=str(out_dir / f"{mode}.mp4"), **render_kwargs)
        print()

    print("Plotting ground tracks...")
    plot_ground_tracks(all_records, circle, out_path=str(out_dir / "comparison.png"))

    print("\nAggregating metrics...")
    provenance = [
        f"checkpoint: {args.checkpoint}",
        f"data_dir:   {args.data_dir}  (norm_stats + seed pose)",
        f"circle:     radius={args.radius}m speed={args.speed}m/s duration={args.duration}s",
        f"planning:   horizon={horizon_steps} replan={replan_steps} denoising={denoising_steps} "
        f"cond_steps={cond_steps} goal_dim={goal_dim}  seed_obs=clip{args.clip_idx}/frame{args.frame_idx}",
        f"goal_clip:  {goal_clip}",
    ]
    # Reference frame distribution (normalized) for FID, sampled from training data.
    n_ref = min(20000, dataset.states.shape[0])
    ref_idx = np.random.default_rng(0).choice(dataset.states.shape[0], size=n_ref, replace=False)
    ref_frames_norm = dataset.states[ref_idx].cpu().numpy().astype(np.float32)

    write_log(all_records, circle, out_path=str(out_dir / "log.txt"), provenance=provenance,
              fk=fk, ref_frames_norm=ref_frames_norm, norm_mean=norm_mean, norm_std=norm_std)

    print("\nDone.")


if __name__ == "__main__":
    main()
