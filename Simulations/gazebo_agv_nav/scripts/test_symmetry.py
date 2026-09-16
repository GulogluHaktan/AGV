"""Check that the D4 action on an observation is the real thing.

The augmentation arm (arm 2) and the equivariant network (arm 3) both rest on
one claim: transforming an observation with a D4 element gives exactly the
observation you would have measured had the whole world been transformed. If
that is off by a sign, arm 2 trains on inconsistent labels and arm 3's
equivariance is meaningless -- and neither failure shows up as a crash, only as
a quietly worse result. So verify it directly rather than trusting the algebra.

For random world states and every g in D4 this compares two routes:

  measured: transform the world (grid, robot pose, yaw, goal), then build the
            observation from the transformed state
  claimed:  build the observation from the original state, then apply the
            wrapper's transform to it

No simulator needed -- it only exercises the observation algebra.

    python3 scripts/test_symmetry.py
"""
import os, sys, math
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from envs.map_generator import symmetric_corridor, asymmetric_corridor
from envs.symmetry import (ALL_D4, transform_grid, transform_point,
                           transform_direction, transform_goal_body,
                           transform_action, is_mirrored)

TOL = 1e-4


def build_obs(grid, pose, yaw, goal):
    """The observation as gazebo_agv_env._obs builds it."""
    grid_size = grid.shape[0]
    d = (np.asarray(goal, dtype=np.float64) - np.asarray(pose, dtype=np.float64)) / grid_size
    c, s = math.cos(yaw), math.sin(yaw)
    return {
        "occupancy": grid,
        "goal_body": np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]),
        "heading": np.array([c, s]),
    }


def transform_obs(obs, g):
    """What SymmetricAugmentationWrapper._transform_obs claims."""
    return {
        "occupancy": transform_grid(obs["occupancy"], g),
        "goal_body": transform_goal_body(obs["goal_body"], g),
        "heading": transform_direction(obs["heading"], g),
    }


def main():
    rng = np.random.default_rng(0)
    failures = []
    checks = 0
    for trial in range(200):
        size = 16
        grid = (symmetric_corridor(size=size, rng=rng) if trial % 2
                else asymmetric_corridor(size=size, rng=rng))
        pose = rng.uniform(1.0, size - 2.0, size=2)
        goal = rng.uniform(1.0, size - 2.0, size=2)
        yaw = rng.uniform(-np.pi, np.pi)
        obs = build_obs(grid, pose, yaw, goal)

        for g in ALL_D4:
            # route 1: transform the world, then observe it
            grid_t = transform_grid(grid, g)
            pose_t = transform_point(pose, g, size)
            goal_t = transform_point(goal, g, size)
            head_t = transform_direction(obs["heading"], g)
            yaw_t = math.atan2(head_t[1], head_t[0])
            measured = build_obs(grid_t, pose_t, yaw_t, goal_t)

            # route 2: observe, then transform the observation
            claimed = transform_obs(obs, g)

            for key in ("occupancy", "goal_body", "heading"):
                err = float(np.max(np.abs(
                    np.asarray(measured[key], dtype=np.float64)
                    - np.asarray(claimed[key], dtype=np.float64))))
                checks += 1
                if err > TOL:
                    failures.append((trial, g.value, key, err))

    # the action transform must be an involution, since the wrapper uses it to
    # map the policy's action back into env space without tracking an inverse
    for g in ALL_D4:
        a = rng.uniform(-1, 1, size=2)
        back = transform_action(transform_action(a, g), g)
        checks += 1
        if float(np.max(np.abs(back - a))) > TOL:
            failures.append(("involution", g.value, "action", float(np.max(np.abs(back - a)))))

    print(f"{checks} checks over 200 random world states x {len(ALL_D4)} group elements")
    if failures:
        print(f"\n{len(failures)} FAILURES (first 10):")
        for f in failures[:10]:
            print(f"  trial={f[0]} g={f[1]} field={f[2]} max_err={f[3]:.3e}")
        by_g = {}
        for f in failures:
            by_g.setdefault((f[1], f[2]), 0)
            by_g[(f[1], f[2])] += 1
        print("\nfailures by (group element, field):")
        for (gv, key), n in sorted(by_g.items(), key=lambda kv: -kv[1]):
            print(f"  {gv:20s} {key:10s} {n}")
        return 1
    print("PASS: the observation transform matches transforming the world, "
          "for every D4 element; action transform is an involution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
