"""Process Tennis motion-capture .npz files into DPPO pretraining format.

Both states and actions are the full 64-D observation vector so the policy is
self-contained for autoregressive rollout:
  states[t]  = obs[t]      — current observation, used as conditioning
  actions[t] = obs[t+1]    — next observation, the prediction target

Per clip of length T this produces T-1 (state, action) pairs. The 1-frame
shift means the policy learns to predict the next full observation given the
current one, including gvec and gyro (root orientation / angular velocity).

Observation layout (D=64):
  [0:3]   gvec_pelvis  — gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  — angular velocity in pelvis frame (rad/s)
  [6:35]  joint_pos    — joint angles minus default_qpos[7:] (rad)
  [35:64] joint_vel    — joint velocities finite-diff (rad/s)

Both are min-max normalized to [-1, 1] per dimension using the same stats.
Raw min/max are saved in normalization.npz for use during RL fine-tuning.

Outputs written to --out_dir:
  train.npz          states (N,64), actions (N,64), traj_lengths (C,)
  normalization.npz  obs_min, obs_max, action_min, action_max (identical)
  metadata.json      feature names, freq, provenance

Usage:
  python script/dataset/process_tennis_dataset.py \\
      --raw_root storage/data/mocap/Tennis \\
      --xml_path /path/to/g1.xml \\
      --out_dir  data/tennis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

OBS_DIM = 64
ACT_DIM = 64  # full next-frame observation

FEATURE_NAMES: list[str] = (
    [f"gvec_{a}" for a in "xyz"]
    + [f"gyro_{a}" for a in "xyz"]
    + [f"jpos_{i}" for i in range(29)]
    + [f"jvel_{i}" for i in range(29)]
)


def _load_default_qpos(xml_path: str) -> np.ndarray:
    m = mujoco.MjModel.from_xml_path(xml_path)
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


def extract_features(path: Path, default_qpos: np.ndarray, freq: float) -> np.ndarray:
    """Return (T, 64) float32 observation array for one motion clip."""
    d = np.load(path, allow_pickle=True)
    qpos = np.asarray(d["qpos"], dtype=np.float64)  # (T, 36)
    T = qpos.shape[0]

    R_root = Rotation.from_quat(_mj_to_scipy_quat(qpos[:, 3:7]))
    joint_ang = qpos[:, 7:]  # (T, 29)

    gvec = R_root.inv().apply(
        np.broadcast_to([0.0, 0.0, -1.0], (T, 3)).copy()
    ).astype(np.float32)

    gyro = _angular_velocity_local(R_root, freq)  # (T, 3) rad/s in pelvis frame

    joint_pos = (joint_ang - default_qpos).astype(np.float32)

    joint_ang_u = np.unwrap(joint_ang, axis=0)
    jvel_raw = np.empty_like(joint_ang)
    jvel_raw[:-1] = (joint_ang_u[1:] - joint_ang_u[:-1]) * freq
    jvel_raw[-1] = jvel_raw[-2]
    joint_vel = jvel_raw.astype(np.float32)

    feats = np.concatenate([gvec, gyro, joint_pos, joint_vel], axis=1)
    assert feats.shape == (T, OBS_DIM), f"Shape mismatch: {feats.shape}"
    return feats


def minmax_normalize(
    x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray
) -> np.ndarray:
    return (2.0 * (x - x_min) / np.maximum(x_max - x_min, 1e-6) - 1.0).astype(
        np.float32
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw_root", type=Path, required=True,
                    help="Directory containing raw MoCap .npz files")
    ap.add_argument("--xml_path", type=str, required=True,
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

    # 1-frame shift per clip: states=obs[t], actions=obs[t+1], length=T-1
    states_list  = [f[:-1] for f in all_feats]
    actions_list = [f[1:]  for f in all_feats]
    traj_lengths = np.array([f.shape[0] for f in states_list], dtype=np.int64)

    states_concat  = np.concatenate(states_list,  axis=0)  # (N, 64)
    actions_concat = np.concatenate(actions_list, axis=0)  # (N, 64)

    # Same normalization stats for both since they share the same space.
    all_frames = np.concatenate(all_feats, axis=0)
    obs_min = all_frames.min(axis=0).astype(np.float32)  # (64,)
    obs_max = all_frames.max(axis=0).astype(np.float32)  # (64,)

    states  = minmax_normalize(states_concat,  obs_min, obs_max)
    actions = minmax_normalize(actions_concat, obs_min, obs_max)

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
        args.out_dir / "normalization.npz",
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=obs_min,  # identical — same observation space
        action_max=obs_max,
    )
    print("Saved normalization.npz")

    print("\n--- Feature ranges (pre-normalization) ---")
    for name, a, b in [
        ("gvec",      0,  3),
        ("gyro",      3,  6),
        ("joint_pos", 6, 35),
        ("joint_vel", 35, 64),
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
        "action_cols": "obs[t+1] — full next-frame observation (64-D)",
        "freq_hz": args.freq,
        "normalization": "minmax to [-1, 1] per dimension",
        "n_clips": len(all_feats),
        "total_frames": int(all_frames.shape[0]),
        "sources": [str(p) for p in raw_files],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved metadata.json\n\nDone.")


if __name__ == "__main__":
    main()
