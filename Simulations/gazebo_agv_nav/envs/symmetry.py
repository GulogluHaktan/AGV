"""Symmetry-group transforms for the AGV occupancy-grid navigation task.

Used in two places in the experiment design (Paper Idea 5):
  1. Arm 2 (symmetric data augmentation) — randomly transform (obs, action,
     next_obs) triples under the corridor's mirror/rotation group during
     training, as a cheap alternative to a structurally-equivariant network.
  2. Evaluation (G3 benchmark) — generate held-out mirrored/rotated map
     variants to measure zero-shot generalization, independent of which
     arm produced the policy.

The corridor/warehouse layout's symmetry group is the dihedral group on
axis-aligned reflections + 90-degree rotations, D4 (8 elements), which is
also the group `escnn`'s C4/D4 steerable layers are built around — so the
group used for augmentation here must match the group used to build the
equivariant policy network in arm 3 for a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class D4(Enum):
    """The 8 elements of the dihedral group D4: 4 rotations x {identity, mirror}."""

    IDENTITY = "identity"
    ROT90 = "rot90"
    ROT180 = "rot180"
    ROT270 = "rot270"
    MIRROR_X = "mirror_x"          # flip left-right
    MIRROR_X_ROT90 = "mirror_x_rot90"
    MIRROR_X_ROT180 = "mirror_x_rot180"
    MIRROR_X_ROT270 = "mirror_x_rot270"


ALL_D4 = list(D4)


def transform_grid(grid: np.ndarray, g: D4) -> np.ndarray:
    """Apply a D4 element to a 2D occupancy grid (H, W)."""
    out = grid
    if g in (D4.MIRROR_X, D4.MIRROR_X_ROT90, D4.MIRROR_X_ROT180, D4.MIRROR_X_ROT270):
        out = np.fliplr(out)
    k = {
        D4.IDENTITY: 0,
        D4.MIRROR_X: 0,
        D4.ROT90: 1,
        D4.MIRROR_X_ROT90: 1,
        D4.ROT180: 2,
        D4.MIRROR_X_ROT180: 2,
        D4.ROT270: 3,
        D4.MIRROR_X_ROT270: 3,
    }[g]
    return np.rot90(out, k=k)


def transform_point(xy: np.ndarray, g: D4, grid_size: int) -> np.ndarray:
    """Apply a D4 element to a 2D point in grid coordinates (for goal / pose)."""
    x, y = xy
    if g in (D4.MIRROR_X, D4.MIRROR_X_ROT90, D4.MIRROR_X_ROT180, D4.MIRROR_X_ROT270):
        x = grid_size - 1 - x
    k = {
        D4.IDENTITY: 0, D4.MIRROR_X: 0,
        D4.ROT90: 1, D4.MIRROR_X_ROT90: 1,
        D4.ROT180: 2, D4.MIRROR_X_ROT180: 2,
        D4.ROT270: 3, D4.MIRROR_X_ROT270: 3,
    }[g]
    for _ in range(k):
        x, y = y, grid_size - 1 - x
    return np.array([x, y])


MIRRORED = (D4.MIRROR_X, D4.MIRROR_X_ROT90, D4.MIRROR_X_ROT180, D4.MIRROR_X_ROT270)

_ROT_K = {
    D4.IDENTITY: 0, D4.MIRROR_X: 0,
    D4.ROT90: 1, D4.MIRROR_X_ROT90: 1,
    D4.ROT180: 2, D4.MIRROR_X_ROT180: 2,
    D4.ROT270: 3, D4.MIRROR_X_ROT270: 3,
}


def is_mirrored(g: D4) -> bool:
    """Whether g reverses orientation (carries the mirror factor)."""
    return g in MIRRORED


def transform_direction(v: np.ndarray, g: D4) -> np.ndarray:
    """Apply a D4 element to a direction vector (heading, velocity, ...).

    This is the linear part of `transform_point`: the same map with the
    grid-size translation dropped, so mirroring negates x and each rotation
    sends (dx, dy) to (dy, -dx). Directions need their own entry point because
    routing them through `transform_point` would add the grid offset and yield
    a position rather than a direction."""
    dx, dy = float(v[0]), float(v[1])
    if is_mirrored(g):
        dx = -dx
    for _ in range(_ROT_K[g]):
        dx, dy = dy, -dx
    return np.array([dx, dy], dtype=np.float32)


def transform_goal_body(goal_body: np.ndarray, g: D4) -> np.ndarray:
    """Apply a D4 element to a body-frame goal offset (forward, left).

    Body-frame quantities do not behave like world-frame ones: under a world
    *rotation* the robot turns with the world, so the goal's bearing relative
    to the robot is unchanged and this is the identity. Only a *mirror*
    matters, and it flips the left/right component -- the same reason
    `transform_action` only has to negate angular velocity.

    Derivation: goal_body = R(-yaw)(goal - pose). For a mirror F,
    heading' = F heading gives yaw' = pi - yaw, and
    R(-yaw')F = diag(1, -1) R(-yaw)."""
    out = np.asarray(goal_body, dtype=np.float32).copy()
    if is_mirrored(g):
        out[1] = -out[1]
    return out


def transform_action(action: np.ndarray, g: D4) -> np.ndarray:
    """Apply a D4 element to a differential-drive action [linear_v, angular_v].

    Mirroring flips the sign of angular velocity; rotation of the *map* does
    not change a body-frame differential-drive action, so only the mirror
    component matters here.
    """
    linear_v, angular_v = action
    return np.array([linear_v, -angular_v if is_mirrored(g) else angular_v])


@dataclass
class SymmetricAugmentation:
    """Sample a random D4 element per call, for training-time augmentation (arm 2)."""

    group: list[D4] = None
    rng: np.random.Generator = None

    def __post_init__(self):
        if self.group is None:
            self.group = ALL_D4
        if self.rng is None:
            self.rng = np.random.default_rng()

    def sample(self) -> D4:
        return self.rng.choice(self.group)
