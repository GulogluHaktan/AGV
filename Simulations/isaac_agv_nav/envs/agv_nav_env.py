"""Lightweight, simulator-agnostic AGV navigation environment.

This is a fast, pure-numpy differential-drive navigation task used to
prototype and unit-test the RL/symmetry pipeline (all 3 arms + the G3
generalization benchmark) *before* paying the cost of the full Isaac Sim
warehouse scene. It implements the same occupancy-grid observation, action
space, and D4-symmetric map generation as the eventual Isaac Sim backend, so
policies developed here transfer directly once `isaac_agv_nav_env.py`
(Isaac-Sim-backed, TODO) replaces the kinematics with the real simulator for
the sim-to-real study.

Not a substitute for Isaac Sim in the final sim-to-real experiments — no
sensor noise model, no wheel/floor friction, no rendering. It exists purely
to make arms 1-2 (and the D4 augmentation wrapper) runnable today, without
waiting on the Isaac Sim install.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.map_generator import symmetric_corridor, asymmetric_corridor, sample_free_cell


class AGVNavEnv(gym.Env):
    """Observation: (grid_size, grid_size) local occupancy patch + goal-relative
    polar coords, flattened. Action: [linear_v, angular_v] in [-1, 1] (differential drive).
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = 32, map_type: str = "symmetric",
                 max_steps: int = 250, goal_radius: float = 1.5, seed: int | None = None):
        super().__init__()
        assert map_type in ("symmetric", "asymmetric")
        self.grid_size = grid_size
        self.map_type = map_type
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(0, 1, shape=(grid_size, grid_size), dtype=np.uint8),
            "goal_relative": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.grid: np.ndarray | None = None
        self.pose = np.zeros(3, dtype=np.float32)  # x, y, theta
        self.goal = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_dist = 0.0

    def _new_map(self):
        gen = symmetric_corridor if self.map_type == "symmetric" else asymmetric_corridor
        self.grid = gen(size=self.grid_size, rng=self._rng)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._new_map()
        start = sample_free_cell(self.grid, self._rng)
        self.goal = sample_free_cell(self.grid, self._rng).astype(np.float32)
        self.pose = np.array([start[0], start[1], self._rng.uniform(-np.pi, np.pi)], dtype=np.float32)
        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self.pose[:2] - self.goal))
        return self._obs(), {}

    def step(self, action: np.ndarray):
        linear_v, angular_v = np.clip(action, -1.0, 1.0)
        dt = 0.5  # max ~0.5 grid units/step — covers a ~45-unit diagonal in <100 steps at full speed
        theta = self.pose[2] + angular_v * dt
        dx = linear_v * np.cos(theta) * dt
        dy = linear_v * np.sin(theta) * dt
        new_xy = self.pose[:2] + np.array([dx, dy])

        collided = self._collides(new_xy)
        if not collided:
            self.pose = np.array([new_xy[0], new_xy[1], theta], dtype=np.float32)
        else:
            self.pose[2] = theta

        self._step_count += 1
        dist_to_goal = float(np.linalg.norm(self.pose[:2] - self.goal))
        reached = dist_to_goal < self.goal_radius
        truncated = self._step_count >= self.max_steps

        # potential-based shaping: reward getting closer, penalize getting farther
        progress = self._prev_dist - dist_to_goal
        self._prev_dist = dist_to_goal
        reward = -0.01 + 1.0 * progress
        if collided:
            reward -= 0.5
        if reached:
            reward += 20.0

        return self._obs(), reward, reached, truncated, {"collided": collided, "dist_to_goal": dist_to_goal}

    def _collides(self, xy: np.ndarray) -> bool:
        gx, gy = int(round(xy[0])), int(round(xy[1]))
        if gx < 0 or gy < 0 or gx >= self.grid_size or gy >= self.grid_size:
            return True
        return bool(self.grid[gy, gx])

    def _obs(self) -> dict:
        # normalized to roughly [-1, 1] so its scale doesn't get drowned out
        # by the 1024-dim binary occupancy vector in the concatenated MLP input
        goal_relative = ((self.goal - self.pose[:2]) / self.grid_size).astype(np.float32)
        return {"occupancy": self.grid.copy(), "goal_relative": goal_relative}
