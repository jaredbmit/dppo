"""Evaluate a goal-conditioned diffusion policy on circle trajectory following.

Runs N seeds per goal mode (circle / zero / random) and produces:
  eval_circle/circle.mp4      — receding-horizon circle subgoal (seed 0)
  eval_circle/zero.mp4        — goal always [0, 0]  (seed 0)
  eval_circle/random.mp4      — goal ~ N(0, radius) each replan step (seed 0)
  eval_circle/comparison.png  — ground tracks for all seeds overlaid
  eval_circle/log.txt         — per-seed and aggregate distance-to-circle metrics

Re-planning period T_p is independent of the action horizon T_a.

Usage:
  uv run python script/eval_circle.py \\
      --checkpoint log/tennis-pretrain/.../checkpoint/state_2000.pt

  uv run python script/eval_circle.py \\
      --checkpoint ... --n_seeds 5 --radius 3.0 --speed 1.2 --duration 15.0
"""

from __future__ import annotations

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
import yaml
from scipy.spatial.transform import Rotation

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.mlp_diffusion import DiffusionMLP
from agent.dataset.sequence import StitchedSequenceDataset

OBS_DIM  = 38
FREQ     = 50.0
GoalMode = Literal["circle", "zero", "random"]

MODE_STYLE = {
    "circle": dict(color="#2196F3", label="Circle goal"),
    "zero":   dict(color="#FF9800", label="Zero goal"),
    "random": dict(color="#F44336", label="Random goal"),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_run_cfg(checkpoint_path: str) -> dict:
    cfg_path = Path(checkpoint_path).parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_model(
    checkpoint_path: str,
    cond_steps: int,
    horizon_steps: int,
    action_dim: int,
    denoising_steps: int,
    goal_dim: int,
    device: str,
    obs_dim: int = OBS_DIM,
    cfg: dict = {},
) -> DiffusionModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ckpt.get("ema", ckpt.get("model"))
    net_cfg = cfg.get("model", {}).get("network", {})
    network = DiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        goal_dim=goal_dim,
        time_dim=net_cfg.get("time_dim", 64),
        mlp_dims=net_cfg.get("mlp_dims", [512] * 13),
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


def integrate_step(
    obs_denorm: np.ndarray,
    world_xy: np.ndarray,
    world_yaw: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    gyro_z = obs_denorm[5]
    vel_h  = obs_denorm[36:38]
    c, s   = np.cos(world_yaw), np.sin(world_yaw)
    world_xy  = world_xy + np.array([c * vel_h[0] - s * vel_h[1],
                                      s * vel_h[0] + c * vel_h[1]]) * dt
    world_yaw = world_yaw + gyro_z * dt
    return world_xy, world_yaw


def obs_to_qpos(
    obs_denorm: np.ndarray,
    world_xy: np.ndarray,
    world_yaw: float,
    default_qpos: np.ndarray,
) -> np.ndarray:
    gvec = obs_denorm[0:3].astype(np.float64)
    jpos = obs_denorm[6:35]
    gx, gy, gz = gvec
    roll    = np.arctan2(-gy, -gz)
    pitch   = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
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


def write_log(
    all_records: dict[GoalMode, list[list[Record]]],
    circle: CircleTrajectory,
    out_path: str,
) -> None:
    lines = ["Circle trajectory following — tracking error (‖robot − circle(t)‖)",
             "=" * 60, ""]

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
    if kid < 0:
        raise RuntimeError("No 'home' keyframe in XML")
    default_qpos = m.key_qpos[kid, 7:].copy()

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
    ap.add_argument("--data_dir",        type=Path,  default=Path("data/tennis"))
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
    # Model overrides
    ap.add_argument("--obs_dim",         type=int,   default=None)
    ap.add_argument("--goal_dim",        type=int,   default=None)
    ap.add_argument("--cond_steps",      type=int,   default=None)
    ap.add_argument("--horizon_steps",   type=int,   default=None)
    ap.add_argument("--denoising_steps", type=int,   default=None)
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

    cfg             = _load_run_cfg(args.checkpoint)
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
        goal_dim, args.device, obs_dim=obs_dim, cfg=cfg,
    )

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

    eval_kwargs = dict(
        model=model, circle=circle, norm_mean=norm_mean, norm_std=norm_std,
        seed_obs_norm=seed_obs, horizon_steps=horizon_steps, cond_steps=cond_steps,
        replan_steps=replan_steps, total_steps=total_steps, device=args.device,
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
    write_log(all_records, circle, out_path=str(out_dir / "log.txt"))

    print("\nDone.")


if __name__ == "__main__":
    main()
