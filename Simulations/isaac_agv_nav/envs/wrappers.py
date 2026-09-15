"""Arm 2: symmetric data augmentation wrapper.

At each `reset`, sample a random D4 element g and present the *transformed*
view of the episode to the policy: observations are g(true_obs), and actions
the policy selects (in g-space) are mapped back to true env-space actions
before being executed. This forces a plain (non-equivariant) network to see
every training episode under a random symmetry transform, so it must learn a
policy that is *consistent* across the D4 orbit — the cheap alternative to
baking equivariance into the architecture (arm 3).

Note on the action transform: our differential-drive action [linear_v,
angular_v] is already expressed in the robot's body frame, so a map rotation
does not change it — only a map mirror flips the sign of angular_v. Since
that flip is its own inverse, `transform_action` is used both to go into
g-space (for the policy's action distribution, not needed here since we only
transform the observation) and back out of it.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from envs.symmetry import D4, ALL_D4, transform_grid, transform_point, transform_action


class SymmetricAugmentationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, group: list[D4] | None = None, seed: int | None = None):
        super().__init__(env)
        self.group = group or ALL_D4
        self._rng = np.random.default_rng(seed)
        self._g: D4 = D4.IDENTITY

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._g = self._rng.choice(self.group)
        return self._transform_obs(obs), info

    def step(self, action: np.ndarray):
        true_action = transform_action(np.asarray(action, dtype=np.float32), self._g)
        obs, reward, terminated, truncated, info = self.env.step(true_action)
        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs: dict) -> dict:
        grid_size = obs["occupancy"].shape[0]
        occupancy = transform_grid(obs["occupancy"], self._g)
        goal_relative = transform_point(obs["goal_relative"] + grid_size / 2, self._g, grid_size) - grid_size / 2
        return {"occupancy": occupancy, "goal_relative": goal_relative.astype(np.float32)}
