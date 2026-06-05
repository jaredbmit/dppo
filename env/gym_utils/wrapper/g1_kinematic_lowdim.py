"""
Kinematic gym environment for the G1 XY goal-reaching task.

This environment is purely kinematic — there is no physics simulation,
no controller, and no MuJoCo. The diffusion policy is a motion generator
that predicts the next 38-D observation directly. Each env step accepts
that prediction as the new state.

Observation layout (normalized, 38-D):
  [0:3]   gvec  — gravity vector in pelvis frame
  [3:6]   gyro  — angular velocity in pelvis frame (gyro_z at idx 5)
  [6:35]  jpos  — 29 joint positions
  [35]    root_height
  [36:38] root_vel_xy — horizontal root velocity in pelvis frame

Action: predicted next 38-D normalized observation (kinematic state transition).

Goal: 2-D body-frame XY displacement to accumulate over horizon_steps steps.
  Resampled every horizon_steps steps; returned in cond["goal"].

Reward: negative Euclidean distance between accumulated body-frame XY
  displacement and the goal, computed each step.

Episode terminates when predicted root height (denormalized obs[35]) falls
below min_height, or max_episode_steps is reached.

Start states are sampled uniformly from the provided motion-capture dataset.
"""

from __future__ import annotations

import numpy as np
import gym
from gym import spaces

# Observation layout indices
IDX_GYRO_Z = 5
IDX_HEIGHT = 35
IDX_VEL_XY = slice(36, 38)

FREQ = 50.0  # dataset / policy control frequency (Hz)


class G1KinematicEnv(gym.Env):
    """
    Kinematic G1 XY goal-reaching environment.

    The policy predicts the next observation; this env accepts that prediction
    as the next state. Displacement is integrated from the velocity fields
    embedded in the observation. No physics or controllers are used.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        norm_stats_path: str,
        dataset_path: str,
        horizon_steps: int = 50,
        max_episode_steps: int = 1000,
        min_height: float = 0.3,
        goal_range: float = 2.0,
    ):
        super().__init__()

        self.horizon_steps = horizon_steps
        self.max_episode_steps = max_episode_steps
        self.min_height = min_height
        self.goal_range = goal_range

        stats = np.load(norm_stats_path)
        self._mean = stats["mean"].astype(np.float32)   # (38,)
        self._std  = stats["std"].astype(np.float32)    # (38,)

        data = np.load(dataset_path)
        self._states = data["states"].astype(np.float32)  # (N, 38) normalized

        obs_dim  = self._mean.shape[0]   # 38
        goal_dim = 2

        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(obs_dim,),  dtype=np.float32),
            "goal":  spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._current_obs    = np.zeros(obs_dim,  dtype=np.float32)
        self._goal           = np.zeros(goal_dim, dtype=np.float32)
        self._chunk_xy       = np.zeros(2,        dtype=np.float32)
        self._chunk_rel_yaw  = 0.0   # yaw accumulated from chunk start (rad)
        self._step_count     = 0
        self._goal_step      = 0

    # ---------------------------------------------------------------------- #
    # Gym interface
    # ---------------------------------------------------------------------- #

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, **kwargs):
        options = kwargs.get("options", {})
        seed = options.get("seed", None)
        if seed is not None:
            self.seed(seed)

        idx = np.random.randint(0, len(self._states))
        self._current_obs = self._states[idx].copy()

        self._step_count    = 0
        self._goal_step     = 0
        self._chunk_xy[:]   = 0.0
        self._chunk_rel_yaw = 0.0
        self._goal          = self._sample_goal()

        return {"state": self._current_obs.copy(), "goal": self._goal.copy()}

    def step(self, action: np.ndarray):
        """
        action: (38,) normalized predicted next observation.

        The kinematic transition: accept the prediction as the next state.
        Displacement is integrated from the velocity in the current obs, rotated
        into the chunk-start body frame using accumulated relative yaw.
        """
        dt = 1.0 / FREQ

        # Denormalize velocity and yaw rate from the CURRENT obs
        gyro_z = (self._current_obs[IDX_GYRO_Z] * self._std[IDX_GYRO_Z]
                  + self._mean[IDX_GYRO_Z])
        vel_h  = (self._current_obs[IDX_VEL_XY] * self._std[IDX_VEL_XY]
                  + self._mean[IDX_VEL_XY])

        # Rotate body-frame velocity into chunk-start body frame
        c = np.cos(self._chunk_rel_yaw)
        s = np.sin(self._chunk_rel_yaw)
        dxy = np.array([c * vel_h[0] - s * vel_h[1],
                        s * vel_h[0] + c * vel_h[1]], dtype=np.float32) * dt

        self._chunk_xy      += dxy
        self._chunk_rel_yaw += gyro_z * dt

        # Advance state
        self._current_obs = action.astype(np.float32)
        self._step_count  += 1
        self._goal_step   += 1

        # Reward: negative distance from accumulated displacement to goal
        dist   = float(np.linalg.norm(self._chunk_xy - self._goal))
        reward = -dist

        # Termination: height from next predicted state
        height = (self._current_obs[IDX_HEIGHT] * self._std[IDX_HEIGHT]
                  + self._mean[IDX_HEIGHT])
        done = bool(float(height) < self.min_height
                    or self._step_count >= self.max_episode_steps)

        # Resample goal each horizon_steps
        if self._goal_step >= self.horizon_steps:
            self._goal          = self._sample_goal()
            self._chunk_xy[:]   = 0.0
            self._chunk_rel_yaw = 0.0
            self._goal_step     = 0

        info = {"dist_to_goal": dist, "height": float(height)}
        return {"state": self._current_obs.copy(), "goal": self._goal.copy()}, reward, done, info

    def render(self, mode="human"):
        pass

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _sample_goal(self) -> np.ndarray:
        """Sample a random body-frame XY displacement reachable in horizon_steps."""
        r     = np.random.uniform(0, self.goal_range)
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)


class G1KinematicVecEnv:
    """
    Fully vectorized kinematic environment for N parallel G1 goal-reaching tasks.

    All N envs run as batched numpy operations — no subprocesses, no IPC.
    Obs-history stacking is handled internally, replacing MultiStep.

    Interface expected by the DPPO train loop:
      reset() -> dict{"state": (N, n_obs_steps, obs_dim),
                      "goal":  (N, n_obs_steps, goal_dim)}
      step(actions) -> (obs_dict, reward, done, info)
        actions: (N, act_steps, obs_dim)   predicted next obs for each inner step
        reward:  (N,)  summed over act_steps
        done:    (N,)  bool — episode ended during this chunk
    """

    def __init__(
        self,
        n_envs: int,
        norm_stats_path: str,
        dataset_path: str,
        horizon_steps: int = 50,
        max_episode_steps: int = 1000,
        min_height: float = 0.3,
        goal_range: float = 2.0,
        n_obs_steps: int = 1,
        act_steps: int = 50,
    ):
        self.n_envs           = n_envs
        self.horizon_steps    = horizon_steps
        self.max_episode_steps = max_episode_steps
        self.min_height       = min_height
        self.goal_range       = goal_range
        self.n_obs_steps      = n_obs_steps
        self.act_steps        = act_steps

        stats = np.load(norm_stats_path)
        self._mean = stats["mean"].astype(np.float32)   # (obs_dim,)
        self._std  = stats["std"].astype(np.float32)    # (obs_dim,)

        self._dataset = np.load(dataset_path)["states"].astype(np.float32)  # (D, obs_dim)

        self.obs_dim  = self._mean.shape[0]   # 38
        self.goal_dim = 2

        # Per-env state arrays — all leading dim is N
        self._obs           = np.zeros((n_envs, self.obs_dim),  dtype=np.float32)
        self._goal          = np.zeros((n_envs, self.goal_dim), dtype=np.float32)
        self._chunk_xy      = np.zeros((n_envs, 2),             dtype=np.float32)
        self._chunk_rel_yaw = np.zeros(n_envs,                  dtype=np.float32)
        self._step_count    = np.zeros(n_envs,                  dtype=np.int32)
        self._goal_step     = np.zeros(n_envs,                  dtype=np.int32)

        # Obs history: (N, n_obs_steps, dim) — rolling buffer for cond stacking.
        self._obs_hist  = np.zeros((n_envs, n_obs_steps, self.obs_dim),  dtype=np.float32)

    # ------------------------------------------------------------------ #

    def seed(self, seeds) -> None:
        seed = seeds[0] if hasattr(seeds, "__len__") else seeds
        np.random.seed(seed)

    def reset(self) -> dict:
        self._reset_envs(np.ones(self.n_envs, dtype=bool))
        return {"state": self._obs_hist.copy(), "goal": self._goal.copy()}

    def step(self, actions: np.ndarray):
        """
        actions: (N, act_steps, obs_dim)

        Runs act_steps inner transitions for all N envs in one vectorized pass.
        Done envs are reset in-place; returned obs is from the fresh episode.
        """
        dt         = 1.0 / FREQ
        reward     = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)   # fell over
        truncated  = np.zeros(self.n_envs, dtype=bool)   # hit time limit

        for t in range(self.act_steps):
            # Denormalize gyro_z and vel_xy from current obs
            gyro_z = (self._obs[:, IDX_GYRO_Z] * self._std[IDX_GYRO_Z]
                      + self._mean[IDX_GYRO_Z])                          # (N,)
            vel_h  = (self._obs[:, IDX_VEL_XY] * self._std[IDX_VEL_XY]
                      + self._mean[IDX_VEL_XY])                          # (N, 2)

            # Rotate body-frame velocity into chunk-start body frame
            c = np.cos(self._chunk_rel_yaw)
            s = np.sin(self._chunk_rel_yaw)
            self._chunk_xy[:, 0] += (c * vel_h[:, 0] - s * vel_h[:, 1]) * dt
            self._chunk_xy[:, 1] += (s * vel_h[:, 0] + c * vel_h[:, 1]) * dt
            self._chunk_rel_yaw  += gyro_z * dt

            # Kinematic transition: predicted obs becomes new state
            np.copyto(self._obs, actions[:, t])
            self._step_count += 1
            self._goal_step  += 1

            # Reward: negative distance at the final inner step only
            if t == self.act_steps - 1:
                reward = -np.linalg.norm(self._chunk_xy - self._goal, axis=1)

            # Termination
            height = (self._obs[:, IDX_HEIGHT] * self._std[IDX_HEIGHT]
                      + self._mean[IDX_HEIGHT])
            terminated |= height < self.min_height
            truncated  |= self._step_count >= self.max_episode_steps

            # Per-env goal resampling at horizon boundary
            goal_mask = self._goal_step >= self.horizon_steps
            if goal_mask.any():
                self._goal[goal_mask]           = self._sample_goals(int(goal_mask.sum()))
                self._chunk_xy[goal_mask]       = 0.0
                self._chunk_rel_yaw[goal_mask]  = 0.0
                self._goal_step[goal_mask]      = 0

            # Roll obs history and append latest
            if self.n_obs_steps > 1:
                self._obs_hist[:,  :-1] = self._obs_hist[:,  1:]
            self._obs_hist[:,  -1] = self._obs

        # Reset finished envs; returned obs is from the fresh episode
        done = terminated | truncated
        if done.any():
            self._reset_envs(done)

        return (
            {"state": self._obs_hist.copy(), "goal": self._goal.copy()},
            reward,
            terminated,
            truncated,
            {},
        )

    def reset_arg(self, options_list=None) -> dict:
        """Reset all envs. options_list is ignored (seeds handled via seed())."""
        return self.reset()

    def reset_one_arg(self, env_ind: int, options=None) -> dict:
        """Reset a single env by index; returns obs without the leading N dim."""
        mask = np.zeros(self.n_envs, dtype=bool)
        mask[env_ind] = True
        self._reset_envs(mask)
        return {
            "state": self._obs_hist[env_ind].copy(),
            "goal":  self._goal[env_ind].copy(),
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #

    def _reset_envs(self, mask: np.ndarray) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        idx = np.random.randint(0, len(self._dataset), size=n)
        self._obs[mask]           = self._dataset[idx]
        self._step_count[mask]    = 0
        self._goal_step[mask]     = 0
        self._chunk_xy[mask]      = 0.0
        self._chunk_rel_yaw[mask] = 0.0
        self._goal[mask]          = self._sample_goals(n)
        # Fill history with the reset obs repeated across all n_obs_steps
        self._obs_hist[mask]  = self._obs[mask,  None, :]   # broadcasts over n_obs_steps

    def _sample_goals(self, n: int) -> np.ndarray:
        r     = np.random.uniform(0.0, self.goal_range, size=n).astype(np.float32)
        theta = np.random.uniform(-np.pi, np.pi,        size=n).astype(np.float32)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
