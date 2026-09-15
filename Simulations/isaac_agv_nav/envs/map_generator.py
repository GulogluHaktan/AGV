"""Procedural symmetric / asymmetric corridor-warehouse layouts.

Kept simulator-agnostic (pure numpy occupancy grids) so the same map
generator backs both the lightweight standalone env (`agv_nav_env.py`, used
for fast algorithm development) and the eventual Isaac Sim warehouse scene
(obstacles placed at the same grid cells). This is what lets G3 (generalization
to unseen mirrored/rotated maps) be evaluated identically in both.
"""

from __future__ import annotations

import numpy as np


def symmetric_corridor(size: int = 32, seed: int | None = None,
                        n_obstacle_pairs: int = 6, rng: np.random.Generator | None = None) -> np.ndarray:
    """A left-right mirror-symmetric corridor: every obstacle at (x, y) has a
    twin at (size-1-x, y)."""
    rng = rng or np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1  # walls
    half = size // 2
    placed = 0
    while placed < n_obstacle_pairs:
        x = rng.integers(2, half - 1)
        y = rng.integers(2, size - 2)
        if grid[y, x] or grid[y, size - 1 - x]:
            continue
        grid[y, x] = 1
        grid[y, size - 1 - x] = 1
        placed += 1
    return grid


def asymmetric_corridor(size: int = 32, seed: int | None = None,
                         n_obstacles: int = 10, rng: np.random.Generator | None = None) -> np.ndarray:
    """Same wall footprint, but obstacles placed independently (no mirror twin) —
    used as the *broken-symmetry* condition for RQ3 (sim-to-real symmetry-breaking
    decomposition) and as an extra generalization stress test."""
    rng = rng or np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    placed = 0
    while placed < n_obstacles:
        x = rng.integers(2, size - 2)
        y = rng.integers(2, size - 2)
        if grid[y, x]:
            continue
        grid[y, x] = 1
        placed += 1
    return grid


def sample_free_cell(grid: np.ndarray, rng: np.random.Generator, margin: int = 2) -> np.ndarray:
    size = grid.shape[0]
    while True:
        xy = rng.integers(margin, size - margin, size=2)
        if grid[xy[1], xy[0]] == 0:
            return xy
