"""Render sampled G1 actions from sample_diffusion.py as a video.

Observation layout expected in the .npz (physical units, post-denorm):
  [0:3]   gvec         — gravity direction in pelvis frame
  [3:6]   gyro         — angular velocity in pelvis frame (rad/s)
  [6:35]  jpos         — joint angles minus default_qpos (rad)
  [35]    root_height  — base z position (m)
  [36:38] root_vel_xy  — planar velocity in heading frame (m/s)

Yaw is integrated from gyro_z; XY position is integrated from root_vel_xy
rotated into the world frame.

Usage:
  uv run python script/render_motion.py \\
      --samples log/.../samples/teacher_forced.npz \\
      --xml_path ~/drl/LATENT/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml \\
      --out /tmp/render.mp4
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _load_default_qpos(m: mujoco.MjModel) -> np.ndarray:
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError("No 'home' keyframe in XML")
    return m.key_qpos[kid, 7:].copy()  # (29,)


def _integrate_trajectory(actions: np.ndarray, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-integrate yaw and XY position from gyro_z and root_vel_xy.

    Returns:
        yaw: (T,) integrated yaw angles (rad)
        xy:  (T, 2) world-frame XY positions (m)
    """
    T = len(actions)
    dt = 1.0 / freq

    gyro_z   = actions[:, 5]         # (T,)
    vel_h    = actions[:, 36:38]     # (T, 2) heading-frame velocity

    yaw = np.zeros(T)
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + gyro_z[t - 1] * dt

    xy = np.zeros((T, 2))
    cos, sin = np.cos(yaw), np.sin(yaw)
    for t in range(1, T):
        vx_w = cos[t - 1] * vel_h[t - 1, 0] - sin[t - 1] * vel_h[t - 1, 1]
        vy_w = sin[t - 1] * vel_h[t - 1, 0] + cos[t - 1] * vel_h[t - 1, 1]
        xy[t, 0] = xy[t - 1, 0] + vx_w * dt
        xy[t, 1] = xy[t - 1, 1] + vy_w * dt

    return yaw, xy


def render_samples(
    npz_path: str,
    xml_path: str,
    out_path: str,
    key: str = "actions",
    fps: int = 50,
    width: int = 640,
    height: int = 480,
    max_frames: int | None = None,
) -> None:
    data = np.load(npz_path)
    actions = data[key]
    if actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    if max_frames is not None:
        actions = actions[:max_frames]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    default_qpos = _load_default_qpos(m)

    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    renderer = mujoco.Renderer(m, height=height, width=width)

    yaw, xy = _integrate_trajectory(actions, fps)

    frames = []
    for t, obs in enumerate(actions):
        gvec = obs[0:3].astype(np.float64)
        jpos = obs[6:35]

        gx, gy, gz = gvec
        roll  = np.arctan2(-gy, -gz)
        pitch = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
        R_root = Rotation.from_euler("ZYX", [yaw[t], pitch, roll])
        xyzw = R_root.as_quat()
        quat_mj = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        d.qpos[:] = 0.0
        d.qpos[0:2] = xy[t]
        d.qpos[2]   = float(obs[35])
        d.qpos[3:7] = quat_mj
        d.qpos[7:36] = jpos + default_qpos
        d.qvel[:] = 0.0

        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam_id if cam_id >= 0 else -1)
        frames.append(renderer.render().copy())

    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    print(f"Saved {len(frames)} frames ({len(frames)/fps:.1f}s) → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", required=True, help="Path to .npz from sample_diffusion.py")
    ap.add_argument("--xml_path", default=str(Path.home() / "drl/LATENT/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"), help="Path to G1 MuJoCo scene XML")
    ap.add_argument("--out", default=None)
    ap.add_argument("--key", default="actions", help="Key to render from npz (actions / gt_states)")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    samples_path = Path(args.samples)
    out_path = args.out or str(samples_path.parent / (samples_path.stem + ".mp4"))

    render_samples(
        npz_path=args.samples,
        xml_path=args.xml_path,
        out_path=out_path,
        key=args.key,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
