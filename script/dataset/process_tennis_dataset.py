"""Process Tennis motion-capture .npz files into DPPO pretraining format.

Extracts 38-D canonical features from raw qpos, z-score normalises per
dimension, and saves (states, actions, traj_lengths) where actions[t] = obs[t+1].

Feature layout (D=38):
  [0:3]   gvec_pelvis  — gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  — body-local angular velocity in pelvis frame (rad/s)
  [6:35]  joint_pos    — joint angles minus default_qpos[7:] (rad)
  [35]    root_height  — base z position (m)
  [36:38] root_vel_xy  — planar velocity in the heading (yaw-only) frame (m/s)

Note gyro is body-local but root_vel_xy is heading-frame, so reconstructing world
root motion needs a heading yaw rate, not gyro_z directly (see util/g1_obs.py).

Outputs written to --out_dir:
  train.npz       states (N,38), actions (N,38), traj_lengths (C,)
  norm_stats.npz  mean (38,), std (38,)
  metadata.json   feature names, freq, provenance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

RAW_ROOT = Path("/home/jared/drl/LATENT/storage/data/mocap/Tennis")
XML_PATH = Path("/home/jared/drl/LATENT/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml")

OBS_DIM = 38
ACT_DIM = 38

FEATURE_NAMES: list[str] = (
    [f"gvec_{a}" for a in "xyz"]
    + [f"gyro_{a}" for a in "xyz"]
    + [f"jpos_{i}" for i in range(29)]
    + ["root_height"]
    + ["root_vel_x_heading", "root_vel_y_heading"]
)
assert len(FEATURE_NAMES) == OBS_DIM


def _load_default_qpos(xml_path: Path) -> np.ndarray:
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError("No keyframe named 'home' in G1 XML")
    return m.key_qpos[kid, 7:].copy()  # (29,)


def _mj_to_scipy_quat(q: np.ndarray) -> np.ndarray:
    """(T, 4) MuJoCo (w,x,y,z) → scipy (x,y,z,w)."""
    return np.concatenate([q[:, 1:], q[:, :1]], axis=1)


def _angular_velocity_local(R: Rotation, freq: float) -> np.ndarray:
    """Finite-diff angular velocity in local body frame, (T, 3); last row duplicated."""
    dR = R[:-1].inv() * R[1:]
    omega = np.empty((len(R), 3), dtype=np.float32)
    omega[:-1] = dR.as_rotvec() * freq
    omega[-1] = omega[-2]
    return omega


def _heading_yaw(R: Rotation) -> np.ndarray:
    """Yaw of the heading (yaw-only) frame from body x-axis in world. (T,)."""
    fwd = R.apply(np.array([1.0, 0.0, 0.0]))   # (T, 3)
    return np.arctan2(fwd[:, 1], fwd[:, 0])


def extract_features(path: Path, default_qpos: np.ndarray, freq: float) -> np.ndarray:
    """Return (T, 38) float32 observation array for one motion clip."""
    d = np.load(path, allow_pickle=True)
    qpos = np.asarray(d["qpos"], dtype=np.float64)  # (T, 36)
    T = qpos.shape[0]

    R_root = Rotation.from_quat(_mj_to_scipy_quat(qpos[:, 3:7]))
    joint_ang = qpos[:, 7:]  # (T, 29)

    gvec = R_root.inv().apply(
        np.broadcast_to([0.0, 0.0, -1.0], (T, 3)).copy()
    ).astype(np.float32)

    angvel_root = _angular_velocity_local(R_root, freq)
    gyro = angvel_root.astype(np.float32)

    joint_pos = (joint_ang - default_qpos).astype(np.float32)

    root_height = qpos[:, 2:3].astype(np.float32)

    root_xy = qpos[:, 0:2]
    vel_xy = np.empty_like(root_xy)
    vel_xy[:-1] = (root_xy[1:] - root_xy[:-1]) * freq
    vel_xy[-1] = vel_xy[-2]
    yaw = _heading_yaw(R_root)
    cos, sin = np.cos(yaw), np.sin(yaw)
    vx_h =  cos * vel_xy[:, 0] + sin * vel_xy[:, 1]
    vy_h = -sin * vel_xy[:, 0] + cos * vel_xy[:, 1]
    root_vel_h = np.stack([vx_h, vy_h], axis=1).astype(np.float32)

    feats = np.concatenate([gvec, gyro, joint_pos, root_height, root_vel_h], axis=1)
    assert feats.shape == (T, OBS_DIM), f"Shape mismatch: {feats.shape}"
    return feats


def zscore_normalize(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw_root", type=Path, default=RAW_ROOT,
                    help="Directory containing raw MoCap .npz files")
    ap.add_argument("--xml_path", type=Path, default=XML_PATH,
                    help="Path to G1 MuJoCo XML (for default_qpos keyframe)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/tennis"))
    ap.add_argument("--freq", type=float, default=50.0,
                    help="Motion capture frequency in Hz")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    default_qpos = _load_default_qpos(args.xml_path)
    print(f"default_qpos: shape={default_qpos.shape}  mean={default_qpos.mean():.4f}")

    raw_files = sorted(
        p for p in args.raw_root.rglob("*.npz") if "__MACOSX" not in p.parts
    )
    if not raw_files:
        raise FileNotFoundError(f"No .npz files found under {args.raw_root}")

    print(f"\nExtracting features from {len(raw_files)} clip(s)...")
    all_feats: list[np.ndarray] = []
    for path in raw_files:
        feats = extract_features(path, default_qpos, args.freq)
        all_feats.append(feats)
        print(f"  {path.name:<45}  T={feats.shape[0]:5d}")

    all_frames = np.concatenate(all_feats, axis=0)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std  = np.clip(all_frames.std(axis=0), 1e-6, None).astype(np.float32)

    # 1-frame shift: states=obs[t], actions=obs[t+1] (both 38-D)
    states_list  = [zscore_normalize(f[:-1], mean, std) for f in all_feats]
    actions_list = [zscore_normalize(f[1:],  mean, std) for f in all_feats]
    traj_lengths = np.array([f.shape[0] - 1 for f in all_feats], dtype=np.int64)

    states  = np.concatenate(states_list,  axis=0)  # (N, 38)
    actions = np.concatenate(actions_list, axis=0)  # (N, 38)

    np.savez_compressed(
        args.out_dir / "train.npz",
        states=states,
        actions=actions,
        traj_lengths=traj_lengths,
    )
    print(
        f"\nSaved train.npz  —  clips={len(all_feats)}  frames={states.shape[0]:,}  "
        f"states={states.shape}  actions={actions.shape}"
    )

    np.savez_compressed(
        args.out_dir / "norm_stats.npz",
        mean=mean,
        std=std,
    )
    print("Saved norm_stats.npz")

    print("\n--- Feature ranges (pre-normalization) ---")
    for name, a, b in [
        ("gvec",        0,  3),
        ("gyro",        3,  6),
        ("joint_pos",   6, 35),
        ("root_height", 35, 36),
        ("root_vel_xy", 36, 38),
    ]:
        blk = all_frames[:, a:b]
        print(
            f"  {name:<12}  mean={blk.mean():+7.3f}  std={blk.std():6.3f}  "
            f"[{blk.min():+7.3f}, {blk.max():+7.3f}]"
        )

    meta = {
        "dataset": "tennis",
        "feature_names": FEATURE_NAMES,
        "obs_dim": OBS_DIM,
        "action_dim": ACT_DIM,
        "action_cols": "obs[t+1] — full 38-D canonical features",
        "freq_hz": args.freq,
        "normalization": "z-score (zero mean, unit std) per dimension",
        "n_clips": len(all_feats),
        "total_frames": int(all_frames.shape[0]),
        "sources": [str(p) for p in raw_files],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved metadata.json\n\nDone.")


if __name__ == "__main__":
    main()
