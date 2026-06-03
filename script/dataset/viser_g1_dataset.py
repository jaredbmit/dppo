"""Interactive Viser viewer for preprocessed 38-D G1 datasets.

The preprocessed 38-D representation does not store full root pose directly.
This viewer reconstructs a MuJoCo-style qpos sequence for visual inspection:
  * root XY and yaw are integrated from root_vel_xy and gyro_z
  * root roll/pitch are inferred from the gravity vector
  * joint positions are default_qpos + the normalized joint-offset block

Use this as a qualitative sanity check for any dataset using the shared G1
38-D motion feature layout, such as Lafan or tennis: joint ordering, default-pose
offset, root height, and gross motion should look plausible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

IDX_GYRO_Z = 5
IDX_HEIGHT = 35
IDX_JOINT_POS = slice(6, 35)
IDX_VEL_XY = slice(36, 38)
N_JOINTS = 29

DEFAULT_DATA_DIR = Path("data/lafan/robot_only")
DEFAULT_XML_PATH = Path(
    "/home/maxi/src/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
)
DEFAULT_URDF_PATH = Path(
    "/home/maxi/src/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"
)


def _load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_default_qpos(xml_path: Path, keyframe_name: str = "home") -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id >= 0:
        qpos = model.key_qpos[key_id, 7:].copy()
    else:
        qpos = model.qpos0[7:].copy()
    if qpos.shape != (N_JOINTS,):
        raise ValueError(f"Expected {N_JOINTS} default joints from {xml_path}, got {qpos.shape}")
    return qpos.astype(np.float64)


def _trajectory_bounds(traj_lengths: np.ndarray, clip_index: int) -> tuple[int, int]:
    if clip_index < 0 or clip_index >= len(traj_lengths):
        raise IndexError(f"clip_index {clip_index} outside [0, {len(traj_lengths) - 1}]")
    start = int(np.sum(traj_lengths[:clip_index]))
    end = start + int(traj_lengths[clip_index])
    return start, end


def _denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def _integrate_xy_yaw(obs: np.ndarray, freq: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(obs)
    dt = 1.0 / freq
    yaw = np.zeros(n, dtype=np.float64)
    xy = np.zeros((n, 2), dtype=np.float64)
    gyro_z = obs[:, IDX_GYRO_Z]
    vel_h = obs[:, IDX_VEL_XY]
    for t in range(1, n):
        yaw[t] = yaw[t - 1] + gyro_z[t - 1] * dt
        c = np.cos(yaw[t - 1])
        s = np.sin(yaw[t - 1])
        vx_w = c * vel_h[t - 1, 0] - s * vel_h[t - 1, 1]
        vy_w = s * vel_h[t - 1, 0] + c * vel_h[t - 1, 1]
        xy[t, 0] = xy[t - 1, 0] + vx_w * dt
        xy[t, 1] = xy[t - 1, 1] + vy_w * dt
    return xy, yaw


def _root_quat_from_gvec_yaw(gvec: np.ndarray, yaw: float) -> np.ndarray:
    gx, gy, gz = gvec.astype(np.float64)
    roll = np.arctan2(-gy, -gz)
    pitch = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
    xyzw = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def load_clip_observations(data_dir: Path, clip_index: int, max_frames: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.load(data_dir / "train.npz")
    stats = np.load(data_dir / "norm_stats.npz")
    meta = _load_metadata(data_dir)
    start, end = _trajectory_bounds(train["traj_lengths"], clip_index)
    obs_norm = np.concatenate([train["states"][start:end], train["actions"][end - 1 : end]], axis=0)
    if max_frames is not None:
        obs_norm = obs_norm[:max_frames]
    obs = _denormalize(obs_norm, stats["mean"], stats["std"])
    return obs, meta


def reconstruct_qpos(
    obs: np.ndarray,
    *,
    default_qpos: np.ndarray,
    freq: float,
) -> np.ndarray:
    xy, yaw = _integrate_xy_yaw(obs, freq)
    qpos = np.zeros((len(obs), 7 + N_JOINTS), dtype=np.float64)
    qpos[:, 0:2] = xy
    qpos[:, 2] = obs[:, IDX_HEIGHT]
    qpos[:, 7:] = obs[:, IDX_JOINT_POS] + default_qpos
    for i in range(len(obs)):
        qpos[i, 3:7] = _root_quat_from_gvec_yaw(obs[i, 0:3], yaw[i])
    return qpos


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q if n == 0.0 else q / n


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + u * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1
    ) / sin_theta


def _interp_qpos(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    out = (1.0 - u) * q0 + u * q1
    out[3:7] = _slerp(q0[3:7], q1[3:7], u)
    return out


def _auto_grid_from_qpos(
    qpos: np.ndarray,
    *,
    grid_width: float | None,
    grid_height: float | None,
    margin: float,
    min_size: float = 1.0,
) -> tuple[float, float, tuple[float, float, float]]:
    xy = qpos[:, 0:2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = (lo + hi) / 2.0
    span = np.maximum(hi - lo, 0.0)
    width = float(grid_width) if grid_width is not None else float(max(min_size, span[0] + 2.0 * margin))
    height = float(grid_height) if grid_height is not None else float(max(min_size, span[1] + 2.0 * margin))
    return width, height, (float(center[0]), float(center[1]), 0.0)


def run_viewer(
    qpos: np.ndarray,
    *,
    urdf_path: Path,
    joint_names: list[str],
    fps: float,
    port: int | None,
    grid_width: float,
    grid_height: float,
    grid_position: tuple[float, float, float],
    show_meshes: bool,
) -> None:
    try:
        import viser  # type: ignore[import-not-found]
        import yourdfpy  # type: ignore[import-untyped]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Viser playback requires optional dependencies that are not installed in this venv. "
            "Install them with something like: uv pip install viser yourdfpy trimesh"
        ) from exc

    server_kwargs = {} if port is None else {"port": port}
    server = viser.ViserServer(**server_kwargs)
    server.scene.add_grid("/grid", width=grid_width, height=grid_height, position=grid_position)
    robot_root = server.scene.add_frame("/robot", show_axes=False)

    # Load from the URDF directory so relative mesh paths resolve reliably.
    cwd = Path.cwd()
    try:
        import os

        os.chdir(urdf_path.parent)
        urdf = yourdfpy.URDF.load(urdf_path.name, load_meshes=True, build_scene_graph=True)
    finally:
        os.chdir(cwd)

    robot = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/robot")
    robot.show_visual = show_meshes
    urdf_joint_order = list(robot.get_actuated_joint_limits().keys())
    name_to_qpos_col = {name: 7 + i for i, name in enumerate(joint_names)}
    try:
        urdf_cols = np.array([name_to_qpos_col[name] for name in urdf_joint_order], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(
            f"URDF joint {exc.args[0]!r} was not found in metadata joint_names. "
            "Use metadata from process_lafan_dataset.py so joint-name mapping is available."
        ) from exc

    frame_state = {
        "index": 0.0,
        "playing": True,
        "last": time.perf_counter(),
        "suppress_slider_update": False,
        "programmatic_slider_value": 0,
    }

    def apply_q(q: np.ndarray) -> None:
        robot_root.position = q[0:3]
        robot_root.wxyz = _quat_normalize(q[3:7])
        robot.update_cfg(q[urdf_cols])

    apply_q(qpos[0])

    with server.gui.add_folder("Playback"):
        playing_cb = server.gui.add_checkbox("Playing", initial_value=True)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(qpos) - 1, step=1, initial_value=0)
        fps_input = server.gui.add_number("FPS", initial_value=float(fps), min=1.0, max=240.0, step=1.0)
        interp_input = server.gui.add_number("Visual FPS multiplier", initial_value=2.0, min=1.0, max=8.0, step=1.0)
    with server.gui.add_folder("Display"):
        show_meshes = server.gui.add_checkbox("Show meshes", initial_value=show_meshes)

    @playing_cb.on_update
    def _(_) -> None:
        frame_state["playing"] = bool(playing_cb.value)
        frame_state["last"] = time.perf_counter()

    @frame_slider.on_update
    def _(_) -> None:
        slider_value = int(frame_slider.value)
        if frame_state["suppress_slider_update"]:
            return
        if frame_state["playing"] and slider_value == frame_state["programmatic_slider_value"]:
            return
        frame_state["index"] = float(slider_value)
        frame_state["playing"] = False
        playing_cb.value = False
        apply_q(qpos[slider_value])

    @show_meshes.on_update
    def _(_) -> None:
        robot.show_visual = bool(show_meshes.value)

    print(f"[viser_g1_dataset] Loaded {len(qpos)} frames from preprocessed G1 data")
    print(f"[viser_g1_dataset] URDF actuated joints: {len(urdf_joint_order)}")
    print("Open the Viser URL printed above. Press Ctrl+C to stop.")

    while True:
        now = time.perf_counter()
        dt = now - float(frame_state["last"])
        frame_state["last"] = now
        if frame_state["playing"]:
            visual_fps = float(fps_input.value) * float(interp_input.value)
            frame_state["index"] = (float(frame_state["index"]) + dt * float(fps_input.value)) % max(1, len(qpos) - 1)
            f = float(frame_state["index"])
            i0 = int(np.floor(f))
            i1 = min(i0 + 1, len(qpos) - 1)
            apply_q(_interp_qpos(qpos[i0], qpos[i1], f - i0))
            frame_state["programmatic_slider_value"] = i0
            frame_state["suppress_slider_update"] = True
            frame_slider.value = i0
            frame_state["suppress_slider_update"] = False
            time.sleep(max(0.001, 1.0 / max(1.0, visual_fps)))
        else:
            time.sleep(0.03)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--clip_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=1500)
    parser.add_argument("--xml_path", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--grid_width", type=float, default=None, help="Override grid width in meters. Defaults to motion span plus margin.")
    parser.add_argument("--grid_height", type=float, default=None, help="Override grid height in meters. Defaults to motion span plus margin.")
    parser.add_argument("--grid_margin", type=float, default=0.5, help="Meters of padding around the reconstructed XY motion span.")
    parser.add_argument("--show_meshes", action="store_true", default=True)
    parser.add_argument("--no_show_meshes", action="store_false", dest="show_meshes")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only reconstruct qpos and print diagnostics; do not import or start Viser.",
    )
    parser.add_argument(
        "--export_qpos_npz",
        type=Path,
        default=None,
        help="Optional path to save reconstructed qpos/fps for other viewers.",
    )
    args = parser.parse_args()

    obs, meta = load_clip_observations(args.data_dir, args.clip_index, args.max_frames)
    freq = float(args.fps if args.fps is not None else meta.get("freq_hz", 50.0))
    default_qpos = _load_default_qpos(args.xml_path)
    qpos = reconstruct_qpos(obs, default_qpos=default_qpos, freq=freq)
    joint_names = [str(x) for x in meta.get("joint_names", [])]
    if len(joint_names) != N_JOINTS:
        raise ValueError(
            f"Expected {N_JOINTS} joint names in {args.data_dir / 'metadata.json'}, got {len(joint_names)}"
        )

    grid_width, grid_height, grid_position = _auto_grid_from_qpos(
        qpos,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        margin=args.grid_margin,
    )

    if args.export_qpos_npz is not None:
        args.export_qpos_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export_qpos_npz, qpos=qpos, fps=np.array([freq], dtype=np.float64), joint_names=np.array(joint_names))
        print(f"Saved reconstructed qpos to {args.export_qpos_npz}")

    print(
        f"qpos shape={qpos.shape} fps={freq:g} "
        f"height_range=[{qpos[:, 2].min():.3f}, {qpos[:, 2].max():.3f}] "
        f"xy_end=({qpos[-1, 0]:.3f}, {qpos[-1, 1]:.3f}) "
        f"grid=({grid_width:.3f}m x {grid_height:.3f}m @ "
        f"{grid_position[0]:.3f}, {grid_position[1]:.3f})"
    )
    if args.dry_run:
        return

    run_viewer(
        qpos,
        urdf_path=args.urdf_path,
        joint_names=joint_names,
        fps=freq,
        port=args.port,
        grid_width=grid_width,
        grid_height=grid_height,
        grid_position=grid_position,
        show_meshes=args.show_meshes,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
