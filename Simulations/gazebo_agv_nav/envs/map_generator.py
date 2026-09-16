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


def make_map_pool(size: int = 16, n_maps: int = 24, seed: int = 0,
                   n_obstacle_pairs: int = 6) -> list[np.ndarray]:
    """A pool of canonical-orientation training layouts.

    `symmetric_corridor` places every obstacle together with its left-right
    twin, so each layout is already mirror-symmetric about the vertical axis.
    That fixes the "canonical orientation" the paper trains on; the held-out
    generalization sets come from rotating/flipping these (see `d4_orbit`).

    Every map has the same obstacle count so the physical obstacle pool in the
    Gazebo world is fully used on every episode and difficulty stays uniform.
    """
    rng = np.random.default_rng(seed)
    return [symmetric_corridor(size=size, n_obstacle_pairs=n_obstacle_pairs, rng=rng)
            for _ in range(n_maps)]


def d4_orbit(grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Distinct D4 images of `grid`, excluding the grid itself.

    Note the degeneracy this exists to handle: layouts from
    `symmetric_corridor` are invariant under the left-right mirror, so that
    element of D4 maps a training map to *itself* and is useless as a
    generalization test. Rotations and the top-bottom mirror are the elements
    that actually produce unseen layouts, and this returns only those.
    """
    from envs.symmetry import ALL_D4, D4, transform_grid  # local: avoid cycle
    seen = [grid]
    out: list[tuple[str, np.ndarray]] = []
    for g in ALL_D4:
        if g is D4.IDENTITY:
            continue
        t = transform_grid(grid, g)
        if any(np.array_equal(t, s) for s in seen):
            continue
        seen.append(t)
        out.append((g.value, t))
    return out


def sample_free_cell(grid: np.ndarray, rng: np.random.Generator, margin: int = 2) -> np.ndarray:
    size = grid.shape[0]
    while True:
        xy = rng.integers(margin, size - margin, size=2)
        if grid[xy[1], xy[0]] == 0:
            return xy


def sample_goal_near(grid: np.ndarray, start: np.ndarray, max_dist: float,
                      rng: np.random.Generator, margin: int = 2, tries: int = 100) -> np.ndarray:
    """Curriculum-friendly goal sampler: free cell within `max_dist` of `start`.
    Samples offsets directly inside the max_dist box around `start` (instead of
    across the whole grid) so the distance constraint is actually hit often --
    the old whole-grid sampling could exhaust `tries` near edges/corners and
    fall back to a FULLY unconstrained free cell, silently breaking the
    curriculum's distance cap. If no in-radius cell turns up, falls back to
    the closest free cell seen among the tries (never to an unbounded one)."""
    size = grid.shape[0]
    lo, hi = margin, size - margin
    best, best_d = None, None
    for _ in range(tries):
        offset = rng.uniform(-max_dist, max_dist, size=2)
        xy = np.clip(np.round(start + offset), lo, hi - 1).astype(int)
        if grid[xy[1], xy[0]] != 0:
            continue
        d = float(np.linalg.norm(xy - start))
        if d <= max_dist:
            return xy
        if best is None or d < best_d:
            best, best_d = xy, d
    if best is not None:
        return best
    return sample_free_cell(grid, rng, margin)
