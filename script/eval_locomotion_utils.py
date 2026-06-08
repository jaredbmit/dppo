"""Shared utilities for locomotion eval scripts (eval_circle, eval_noise_steering).

Covers: circle geometry, world-frame integration, MuJoCo rendering, ground-track
and episode-track plots, and per-frame distance helpers.  Metrics aggregation,
log writing, and dataset-level sampling are script-specific and live in each
eval script.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from util.g1_obs import heading_yaw_rate, roll_pitch_from_gvec

FREQ = 50.0

# (obs_denorm (38,), world_xy (2,), world_yaw float, goal_world_xy (2,))
Record = tuple[np.ndarray, np.ndarray, float, np.ndarray]

MODE_STYLE = {
    "circle": dict(color="#2196F3", label="Circle goal"),
    "zero":   dict(color="#FF9800", label="Zero goal"),
    "random": dict(color="#F44336", label="Random goal"),
}


# ---------------------------------------------------------------------------
# Circle reference trajectory
# ---------------------------------------------------------------------------

class CircleTrajectory:
    """Counter-clockwise circle in world XY at constant tangential speed."""

    def __init__(self, center: np.ndarray, radius: float, speed: float,
                 start_theta: float = 0.0):
        self.center      = np.asarray(center, dtype=np.float64)
        self.radius      = radius
        self.omega       = speed / radius
        self.start_theta = start_theta

    def position(self, t_seconds: float) -> np.ndarray:
        theta = self.start_theta + self.omega * t_seconds
        return self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])

    def dist_to(self, xy: np.ndarray) -> float:
        return abs(float(np.linalg.norm(xy - self.center)) - self.radius)

    def start_yaw(self) -> float:
        return self.start_theta + np.pi / 2.0

    def reference_points(self, n: int = 360) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.center + self.radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


# ---------------------------------------------------------------------------
# World-frame helpers
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
    vel_h    = obs_denorm[36:38]
    c, s     = np.cos(world_yaw), np.sin(world_yaw)
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
# Goal utilities
# ---------------------------------------------------------------------------

def apply_goal_clip(
    goal_body: np.ndarray,
    world_xy: np.ndarray,
    world_yaw: float,
    goal_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp goal_body to goal_clip radius; return (goal_body, xy_goal_world)."""
    xy_goal_world = world_xy + body_to_world(goal_body, world_yaw)
    if goal_clip is not None:
        n = float(np.linalg.norm(goal_body))
        if n > goal_clip:
            goal_body     = goal_body * (goal_clip / n)
            xy_goal_world = world_xy + body_to_world(goal_body, world_yaw)
    return goal_body, xy_goal_world


def make_goal(
    goal_mode: str,
    circle: CircleTrajectory,
    world_xy: np.ndarray,
    world_yaw: float,
    t_sim: int,
    horizon_steps: int,
    goal_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (goal_body, xy_goal_world) for the current replanning step."""
    if goal_mode == "circle":
        t_goal_s      = (t_sim + horizon_steps) / FREQ
        xy_goal_world = circle.position(t_goal_s)
        goal_body     = world_to_body(xy_goal_world - world_xy, world_yaw)
    elif goal_mode == "zero":
        goal_body     = np.zeros(2)
        xy_goal_world = world_xy.copy()
    else:  # random
        goal_body     = np.random.randn(2) * circle.radius
        xy_goal_world = world_xy + body_to_world(goal_body, world_yaw)

    goal_body, xy_goal_world = apply_goal_clip(goal_body, world_xy, world_yaw, goal_clip)
    return goal_body, xy_goal_world


def seed_obs_deque(seed_obs_norm: torch.Tensor, cond_steps: int) -> deque:
    return deque([seed_obs_norm.clone() for _ in range(cond_steps)], maxlen=cond_steps)


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def per_frame_dists(records: list[Record], circle: CircleTrajectory) -> np.ndarray:
    return np.array([
        np.linalg.norm(r[1] - circle.position(i / FREQ))
        for i, r in enumerate(records)
    ])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_MAT_IDENTITY  = np.eye(3).flatten().astype(np.float64)
_N_CIRCLE_VIZ  = 72


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
    xml_path: str,
    out_path: str,
    circle: CircleTrajectory | None = None,
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

    circle_pts = None
    if circle is not None:
        angles     = np.linspace(0, 2 * np.pi, _N_CIRCLE_VIZ, endpoint=False)
        circle_pts = circle.center + circle.radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

    frames = []
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for obs_denorm, world_xy, world_yaw, goal_world_xy in records:
            qpos = obs_to_qpos(obs_denorm, world_xy, world_yaw, default_qpos)
            d.qpos[:36] = qpos
            d.qvel[:]   = 0.0
            mujoco.mj_forward(m, d)

            cam.lookat[0] = world_xy[0]
            cam.lookat[1] = world_xy[1]
            cam.lookat[2] = 0.9

            renderer.update_scene(d, camera=cam)
            scene = renderer.scene

            if circle_pts is not None:
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
# Ground track plots
# ---------------------------------------------------------------------------

def plot_ground_tracks(
    all_records: dict[str, list[list[Record]]],
    circle: CircleTrajectory,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ref = circle.reference_points(360)
    ax.plot(ref[:, 0], ref[:, 1], color="black", linestyle="--", linewidth=1.5,
            label="Reference", zorder=4)

    for mode, seeds_records in all_records.items():
        style = MODE_STYLE.get(mode, dict(color="#9E9E9E", label=mode))
        for i, records in enumerate(seeds_records):
            xy = np.array([r[1] for r in records])
            ax.plot(xy[:, 0], xy[:, 1],
                    color=style["color"], linewidth=1.0, alpha=0.35,
                    label=style["label"] if i == 0 else None,
                    zorder=3)

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


def plot_episode_tracks(
    episode_records: list[list[Record]],
    goal_radius: float,
    out_path: str,
    title: str = "Episodic goal-reaching ground tracks",
) -> None:
    """Plot robot XY paths and goal markers for a set of episodes."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.9, min(len(episode_records), 10)))

    for i, records in enumerate(episode_records):
        c    = colors[i % len(colors)]
        xy   = np.array([r[1] for r in records])
        goal = records[0][3]
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1.0, alpha=0.7)
        ax.scatter(*xy[0],  marker="o", s=40, color=c, zorder=4)
        ax.scatter(*goal,   marker="*", s=120, color=c, zorder=5,
                   edgecolors="k", linewidths=0.5)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(goal_radius * np.cos(theta), goal_radius * np.sin(theta),
            color="black", ls="--", lw=1, label=f"Goal radius ({goal_radius}m)")
    ax.scatter(0, 0, marker="x", s=80, color="black", zorder=6, label="Start")

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved episode tracks → {out_path}")
