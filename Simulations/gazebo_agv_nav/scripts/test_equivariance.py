"""Check that the arm-3 extractor transforms the way it claims to.

Arm 3's premise is structural, so it is worth measuring rather than asserting:
if it is false the comparison against arms 1 and 2 is meaningless, and nothing
crashes to tell you. The premise has two halves, and this tests both against a
world that is actually transformed:

  rotations   the feature vector is UNCHANGED. This is what the G3 benchmark
              needs -- `d4_orbit` drops the degenerate mirror for the symmetric
              layouts, so the held-out set is rotations of training maps.
  reflections the EVEN half is unchanged and the ODD half flips sign. Full
              invariance would be wrong: under a mirror the correct action is
              (linear, -angular), so mirror-invariant features could not
              express the flipped behaviour. The features transform like the
              action, which is the point.

The observation is transformed with the same group action the augmentation
wrapper uses (verified independently by scripts/test_symmetry.py), so a pass
here means encoder and wrapper agree.

Two traps this test fell into earlier, both fixed:
  * measuring on `symmetric_corridor` maps, which are mirror-invariant, so the
    mirror elements scored a free 0.0000 that said nothing;
  * assuming a residual error was escnn's discretization. It was not -- escnn
    is exact here, and the error came from the encoder flattening its spatial
    feature map into a Linear layer (HANDOFF §15).

    python3 scripts/test_equivariance.py
"""
import os, sys, argparse, math
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from envs.curriculum import GRID_SIZE, N_OBSTACLE_PAIRS
from envs.map_generator import asymmetric_corridor
from envs.symmetry import (ALL_D4, transform_grid, transform_direction,
                           transform_goal_body, is_mirrored)
from envs.equivariant_extractor import D4EquivariantExtractor

# escnn is exact to machine precision on this grid, so a correct extractor
# scores ~0. The budget stays tight on purpose: the structural break this test
# exists to catch showed up at ~0.25.
ERROR_BUDGET = 0.02


def make_obs(grid, yaw, goal_body):
    c, s = math.cos(yaw), math.sin(yaw)
    return {
        "occupancy": torch.from_numpy(np.ascontiguousarray(grid)).float()[None],
        "heading": torch.tensor([[c, s]], dtype=torch.float32),
        "goal_body": torch.tensor([goal_body], dtype=torch.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_size", type=int, default=GRID_SIZE)
    ap.add_argument("--n_cases", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    obs_space = spaces.Dict({
        "occupancy": spaces.Box(0, 1, shape=(args.grid_size, args.grid_size), dtype=np.uint8),
        "goal_body": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        "heading": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    })
    ex = D4EquivariantExtractor(obs_space)
    ex.eval()  # freeze BatchNorm: in train mode it renormalizes per batch
    n_even = ex.n_even

    rng = np.random.default_rng(args.seed)
    errors = {g.value: [] for g in ALL_D4}
    with torch.no_grad():
        for i in range(args.n_cases):
            # asymmetric layouts so every group element is non-trivial
            grid = asymmetric_corridor(size=args.grid_size,
                                       n_obstacles=2 * N_OBSTACLE_PAIRS,
                                       rng=np.random.default_rng(args.seed + i))
            yaw = float(rng.uniform(-np.pi, np.pi))
            goal_body = rng.uniform(-0.5, 0.5, size=2).tolist()

            base = ex.raw_features(make_obs(grid, yaw, goal_body))
            base_norm = float(torch.linalg.norm(base))

            for g in ALL_D4:
                # transform the WORLD: grid, heading, and the body-frame goal
                heading_t = transform_direction(
                    np.array([math.cos(yaw), math.sin(yaw)]), g)
                yaw_t = math.atan2(heading_t[1], heading_t[0])
                gb_t = transform_goal_body(np.asarray(goal_body), g)
                got = ex.raw_features(
                    make_obs(transform_grid(grid, g), yaw_t, gb_t.tolist()))

                # expected: even half unchanged, odd half negated under mirror
                want = base.clone()
                if is_mirrored(g):
                    want[:, n_even:] *= -1.0
                rel = float(torch.linalg.norm(got - want)) / max(base_norm, 1e-9)
                errors[g.value].append(rel)

    print(f"arm-3 extractor under D4, grid {args.grid_size}x{args.grid_size}, "
          f"{args.n_cases} cases")
    print("rotations: features unchanged | reflections: odd half negated\n")
    print(f"{'group element':22s} {'kind':11s} {'mean rel err':>12s} {'max':>8s}")
    worst = 0.0
    for g in ALL_D4:
        errs = errors[g.value]
        m, mx = float(np.mean(errs)), float(np.max(errs))
        worst = max(worst, m)
        kind = "reflection" if is_mirrored(g) else "rotation"
        print(f"{g.value:22s} {kind:11s} {m:12.5f} {mx:8.5f}")

    print(f"\nworst mean relative error: {worst:.5f} (budget {ERROR_BUDGET})")
    if worst > ERROR_BUDGET:
        print("FAIL: the extractor does not transform as claimed -- arm 3's "
              "premise is broken and the arm-1/2/3 comparison would be "
              "meaningless")
        return 1
    print("PASS: rotation-invariant and mirror-odd to numerical precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
