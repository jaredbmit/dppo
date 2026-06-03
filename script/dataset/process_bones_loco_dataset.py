"""Process the LATENT g1_loco_core feature store into DPPO pretraining format.

Unlike process_tennis_dataset.py (which extracts features from raw qpos), the
BONES locomotion clips are ALREADY canonical 38-D features on disk — each clip
is an .npz with key "features" of shape (T, 38) in the LATENT feature store
(storage/data/features/<dataset>/feats/*.npz). So this script only needs to
read, z-score normalise, 1-frame-shift into (states, actions), and stitch.

Feature layout (D=38), identical to tennis:
  [0:3]   gvec_pelvis  — gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  — angular velocity in pelvis frame (rad/s)
  [6:35]  joint_pos    — joint angles minus default_qpos[7:] (rad)
  [35]    root_height  — base z position (m)
  [36:38] root_vel_xy  — planar velocity in heading frame (m/s)

Outputs written to --out_dir:
  train.npz       states (N,38), actions (N,38), traj_lengths (C,)
  norm_stats.npz  mean (38,), std (38,)
  metadata.json   feature names, freq, provenance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES_DIR = Path(
    "/home/jared/drl/LATENT/storage/data/features/g1_loco_core/feats"
)

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


def zscore_normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features_dir", type=Path, default=FEATURES_DIR,
                    help="LATENT feature store dir of per-clip (T,38) .npz files")
    ap.add_argument("--meta", type=Path, default=None,
                    help="metadata.json for the feature store (to read freq); "
                         "defaults to <features_dir>/../metadata.json")
    ap.add_argument("--out_dir", type=Path, default=Path("data/bones_loco_core"))
    ap.add_argument("--freq", type=float, default=None,
                    help="Motion frequency in Hz (overrides metadata; default 50)")
    ap.add_argument("--max_clips", type=int, default=0,
                    help="Cap the number of clips for a quick test (0 = all)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve freq: CLI > metadata.json > 50.0
    meta_path = args.meta or (args.features_dir.parent / "metadata.json")
    src_freq = None
    if meta_path.exists():
        src_freq = float(json.loads(meta_path.read_text()).get("freq", 0)) or None
    freq = args.freq or src_freq or 50.0

    files = sorted(args.features_dir.glob("*.npz"))
    if args.max_clips:
        files = files[: args.max_clips]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {args.features_dir}")

    print(f"Reading {len(files):,} clip(s) from {args.features_dir}  (freq={freq:g} Hz)")
    all_feats: list[np.ndarray] = []
    log_every = max(1, len(files) // 20)
    for i, path in enumerate(files, 1):
        feats = np.load(path)["features"].astype(np.float32)  # (T, 38)
        if feats.ndim != 2 or feats.shape[1] != OBS_DIM:
            raise ValueError(f"{path.name}: expected (T,{OBS_DIM}), got {feats.shape}")
        if feats.shape[0] < 2:
            continue  # need >=2 frames for a single (state, action) pair
        all_feats.append(feats)
        if i % log_every == 0 or i == len(files):
            print(f"  [{i:>6,}/{len(files):,}]  {path.stem[:48]:<48}  T={feats.shape[0]:5d}")

    all_frames = np.concatenate(all_feats, axis=0)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std = np.clip(all_frames.std(axis=0), 1e-6, None).astype(np.float32)

    # 1-frame shift: states=obs[t], actions=obs[t+1] (both 38-D)
    states_list = [zscore_normalize(f[:-1], mean, std) for f in all_feats]
    actions_list = [zscore_normalize(f[1:], mean, std) for f in all_feats]
    traj_lengths = np.array([f.shape[0] - 1 for f in all_feats], dtype=np.int64)

    states = np.concatenate(states_list, axis=0)   # (N, 38)
    actions = np.concatenate(actions_list, axis=0)  # (N, 38)

    np.savez_compressed(
        args.out_dir / "train.npz",
        states=states,
        actions=actions,
        traj_lengths=traj_lengths,
    )
    print(
        f"\nSaved train.npz  —  clips={len(all_feats):,}  frames={states.shape[0]:,}  "
        f"states={states.shape}  actions={actions.shape}"
    )

    np.savez_compressed(args.out_dir / "norm_stats.npz", mean=mean, std=std)
    print("Saved norm_stats.npz")

    print("\n--- Feature ranges (pre-normalization) ---")
    for name, a, b in [
        ("gvec", 0, 3),
        ("gyro", 3, 6),
        ("joint_pos", 6, 35),
        ("root_height", 35, 36),
        ("root_vel_xy", 36, 38),
    ]:
        blk = all_frames[:, a:b]
        print(
            f"  {name:<12}  mean={blk.mean():+7.3f}  std={blk.std():6.3f}  "
            f"[{blk.min():+7.3f}, {blk.max():+7.3f}]"
        )

    meta = {
        "dataset": "bones_loco_core",
        "feature_names": FEATURE_NAMES,
        "obs_dim": OBS_DIM,
        "action_dim": ACT_DIM,
        "action_cols": "obs[t+1] — full 38-D canonical features",
        "freq_hz": freq,
        "normalization": "z-score (zero mean, unit std) per dimension",
        "n_clips": len(all_feats),
        "total_frames": int(all_frames.shape[0]),
        "provenance": f"LATENT feature store: {args.features_dir}",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved metadata.json\n\nDone.")


if __name__ == "__main__":
    main()
