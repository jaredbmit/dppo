"""Process Lafan/Holosoma retargeted robot-only .npz files for DPPO BC.

The input clips are expected to contain retargeted G1 robot trajectories with
``joint_pos`` shaped like MuJoCo qpos: root position, root quaternion, and 29
robot joints. The output follows DPPO's stitched pretraining format:

  train.npz       states (N, 38), actions (N, 38), traj_lengths (C,)
  norm_stats.npz  mean (38,), std (38,)
  metadata.json   feature names, filters, selected clips, provenance

Feature layout (D=38):
  [0:3]   gvec_pelvis  - gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  - angular velocity in pelvis frame (rad/s)
  [6:35]  joint_pos    - joint angles minus default qpos[7:] (rad)
  [35]    root_height  - base z position (m)
  [36:38] root_vel_xy  - planar velocity in heading frame (m/s)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

RAW_ROOT = Path(
    "/home/maxi/src/holosoma/src/holosoma_retargeting/holosoma_retargeting/converted_res/robot_only"
)
XML_PATH = Path(
    "/home/maxi/src/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
)

OBS_DIM = 38
ACT_DIM = 38
N_JOINTS = 29

FEATURE_NAMES: list[str] = (
    [f"gvec_{a}" for a in "xyz"]
    + [f"gyro_{a}" for a in "xyz"]
    + [f"jpos_{i}" for i in range(N_JOINTS)]
    + ["root_height"]
    + ["root_vel_x_heading", "root_vel_y_heading"]
)
assert len(FEATURE_NAMES) == OBS_DIM


def _load_default_qpos(xml_path: Path, keyframe_name: str = "home") -> tuple[np.ndarray, str]:
    """Load the robot default joint pose from a keyframe or MuJoCo qpos0."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if keyframe_name:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id >= 0:
            qpos = model.key_qpos[key_id, 7:].copy()
            source = f"keyframe:{keyframe_name}"
        else:
            qpos = model.qpos0[7:].copy()
            source = "qpos0"
    else:
        qpos = model.qpos0[7:].copy()
        source = "qpos0"
    if qpos.shape != (N_JOINTS,):
        raise ValueError(f"Expected {N_JOINTS} default joints from {xml_path}, got {qpos.shape}")
    return qpos.astype(np.float64), source


def _mj_to_scipy_quat(q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion order (w, x, y, z) to scipy order (x, y, z, w)."""
    return np.concatenate([q[:, 1:], q[:, :1]], axis=1)


def _angular_velocity_local(rotation: Rotation, freq: float) -> np.ndarray:
    """Finite-difference local angular velocity, shape (T, 3)."""
    n = len(rotation)
    omega = np.zeros((n, 3), dtype=np.float32)
    if n <= 1:
        return omega
    delta = rotation[:-1].inv() * rotation[1:]
    omega[:-1] = delta.as_rotvec() * freq
    omega[-1] = omega[-2]
    return omega


def _heading_yaw(rotation: Rotation) -> np.ndarray:
    """Yaw of the root heading frame from the body x-axis in world coordinates."""
    fwd = rotation.apply(np.array([1.0, 0.0, 0.0]))
    return np.arctan2(fwd[:, 1], fwd[:, 0])


def _clip_fps(path: Path) -> float:
    data = np.load(path, allow_pickle=False)
    if "fps" not in data.files:
        raise KeyError(f"{path} is missing required key 'fps'")
    return float(np.asarray(data["fps"]).reshape(-1)[0])


def _selected_files(raw_root: Path, include: list[str], exclude: list[str]) -> list[Path]:
    files = sorted(p for p in raw_root.glob("*.npz") if p.is_file())
    selected: list[Path] = []
    for path in files:
        name = path.name
        if not any(fnmatch.fnmatchcase(name, pat) for pat in include):
            continue
        if any(fnmatch.fnmatchcase(name, pat) for pat in exclude):
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            f"No .npz clips under {raw_root} matched include={include} exclude={exclude}"
        )
    return selected


def _validate_clip(data: np.lib.npyio.NpzFile, path: Path) -> None:
    required = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
        "fps",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required key(s): {missing}")
    qpos = data["joint_pos"]
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{path} joint_pos must have shape (T, 36), got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError(f"{path} must contain at least 2 frames, got {qpos.shape[0]}")
    if data["joint_names"].shape[0] != N_JOINTS:
        raise ValueError(f"{path} joint_names must contain {N_JOINTS} names")


def extract_features(path: Path, default_qpos: np.ndarray, freq: float) -> np.ndarray:
    """Extract the 38-D Lafan/G1 observation features from one retargeted clip."""
    data = np.load(path, allow_pickle=False)
    _validate_clip(data, path)
    qpos = np.asarray(data["joint_pos"], dtype=np.float64)
    n_frames = qpos.shape[0]

    root_rot = Rotation.from_quat(_mj_to_scipy_quat(qpos[:, 3:7]))
    joint_ang = qpos[:, 7:]

    gvec = root_rot.inv().apply(
        np.broadcast_to([0.0, 0.0, -1.0], (n_frames, 3)).copy()
    ).astype(np.float32)
    gyro = _angular_velocity_local(root_rot, freq).astype(np.float32)
    joint_pos = (joint_ang - default_qpos).astype(np.float32)
    root_height = qpos[:, 2:3].astype(np.float32)

    root_xy = qpos[:, 0:2]
    vel_xy = np.empty_like(root_xy)
    vel_xy[:-1] = (root_xy[1:] - root_xy[:-1]) * freq
    vel_xy[-1] = vel_xy[-2]
    yaw = _heading_yaw(root_rot)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    vx_heading = cos_yaw * vel_xy[:, 0] + sin_yaw * vel_xy[:, 1]
    vy_heading = -sin_yaw * vel_xy[:, 0] + cos_yaw * vel_xy[:, 1]
    root_vel_heading = np.stack([vx_heading, vy_heading], axis=1).astype(np.float32)

    features = np.concatenate([gvec, gyro, joint_pos, root_height, root_vel_heading], axis=1)
    if features.shape != (n_frames, OBS_DIM):
        raise ValueError(f"Feature shape mismatch for {path}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features found in {path}")
    return features


def zscore_normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw_root", type=Path, default=RAW_ROOT)
    parser.add_argument("--xml_path", type=Path, default=XML_PATH)
    parser.add_argument("--out_dir", type=Path, default=Path("data/lafan/robot_only"))
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Shell-style basename glob to include. Repeatable. Default: *.npz",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Shell-style basename glob to exclude. Repeatable. Default: none",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=None,
        help="Override clip fps. By default, require selected clips to share their stored fps.",
    )
    parser.add_argument(
        "--keyframe_name",
        type=str,
        default="home",
        help="Preferred MuJoCo keyframe for default joints; falls back to qpos0 if absent.",
    )
    args = parser.parse_args()

    include = args.include or ["*.npz"]
    exclude = args.exclude or []
    raw_files = _selected_files(args.raw_root, include, exclude)

    actual_fps = [_clip_fps(path) for path in raw_files]
    fps_values = sorted({float(x) for x in actual_fps})
    if args.freq is None:
        if len(fps_values) != 1:
            raise ValueError(f"Selected clips have mixed fps values: {fps_values}; pass --freq to override")
        freq = fps_values[0]
    else:
        freq = args.freq

    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_qpos, default_qpos_source = _load_default_qpos(args.xml_path, args.keyframe_name)

    print(f"Selected {len(raw_files)} clip(s) from {args.raw_root}")
    print(f"Using freq={freq:g} Hz; stored fps values={fps_values}")
    print(f"Default qpos source: {default_qpos_source}")

    all_features: list[np.ndarray] = []
    joint_names_ref: list[str] | None = None
    body_names_ref: list[str] | None = None
    for path in raw_files:
        data = np.load(path, allow_pickle=False)
        joint_names = [str(x) for x in data["joint_names"]]
        body_names = [str(x) for x in data["body_names"]]
        if joint_names_ref is None:
            joint_names_ref = joint_names
            body_names_ref = body_names
        elif joint_names != joint_names_ref:
            raise ValueError(f"Joint-name order differs in {path}")

        features = extract_features(path, default_qpos, freq)
        all_features.append(features)
        print(f"  {path.name:<45} T={features.shape[0]:6d}")

    all_frames = np.concatenate(all_features, axis=0)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std = np.clip(all_frames.std(axis=0), 1e-6, None).astype(np.float32)

    states_list = [zscore_normalize(features[:-1], mean, std) for features in all_features]
    actions_list = [zscore_normalize(features[1:], mean, std) for features in all_features]
    traj_lengths = np.array([features.shape[0] - 1 for features in all_features], dtype=np.int64)

    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite normalized states/actions")

    np.savez_compressed(
        args.out_dir / "train.npz",
        states=states,
        actions=actions,
        traj_lengths=traj_lengths,
    )
    np.savez_compressed(args.out_dir / "norm_stats.npz", mean=mean, std=std)

    metadata = {
        "dataset": "lafan",
        "source": "holosoma_retargeting robot_only",
        "raw_root": str(args.raw_root),
        "xml_path": str(args.xml_path),
        "default_qpos_source": default_qpos_source,
        "include_patterns": include,
        "exclude_patterns": exclude,
        "selected_files": [str(path) for path in raw_files],
        "selected_basenames": [path.name for path in raw_files],
        "stored_fps_values": fps_values,
        "freq_hz": float(freq),
        "feature_names": FEATURE_NAMES,
        "obs_dim": OBS_DIM,
        "action_dim": ACT_DIM,
        "action_cols": "obs[t+1] - full 38-D canonical features",
        "normalization": "z-score (zero mean, unit std) per dimension",
        "n_clips": len(all_features),
        "traj_lengths": traj_lengths.tolist(),
        "total_frames_raw": int(all_frames.shape[0]),
        "total_transitions": int(states.shape[0]),
        "joint_names": joint_names_ref,
        "body_names": body_names_ref,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        f"Saved {args.out_dir / 'train.npz'} clips={len(all_features)} "
        f"transitions={states.shape[0]:,} states={states.shape} actions={actions.shape}"
    )
    print(f"Saved {args.out_dir / 'norm_stats.npz'}")
    print(f"Saved {args.out_dir / 'metadata.json'}")

    print("\n--- Feature ranges before normalization ---")
    for name, start, end in [
        ("gvec", 0, 3),
        ("gyro", 3, 6),
        ("joint_pos", 6, 35),
        ("root_height", 35, 36),
        ("root_vel_xy", 36, 38),
    ]:
        block = all_frames[:, start:end]
        print(
            f"  {name:<12} mean={block.mean():+8.4f} std={block.std():8.4f} "
            f"[{block.min():+8.4f}, {block.max():+8.4f}]"
        )


if __name__ == "__main__":
    main()
