"""Measure how D4-invariant the arm-3 feature extractor actually is.

Arm 3's claim is structural: the escnn encoder pools to D4-invariant features,
so the network cannot tell a map from its D4 transform apart. That claim is
worth measuring rather than asserting: it is the arm's whole reason for
existing, so if it is false the comparison against arms 1 and 2 is
meaningless -- and nothing crashes to tell you.

Reports relative error per group element:

    ||f(g.grid) - f(grid)|| / ||f(grid)||

Two things the first version of this test got wrong, both worth keeping in
mind when reading it:

  * It measured on `symmetric_corridor` maps, which are invariant under the
    left-right mirror. That element therefore maps each map to ITSELF and
    scored a perfect 0.0000 for free, which says nothing about the encoder.
    Asymmetric maps are used here so all eight elements are non-trivial.
  * A residual error was assumed to be escnn's discretization. It is not:
    escnn is exact to machine precision here, and the error came from the
    encoder flattening its spatial feature map into a Linear layer. See
    HANDOFF §15.

BatchNorm is put in eval mode first; in train mode it renormalizes by batch
statistics, which would swamp the effect being measured.

    python3 scripts/test_equivariance.py
"""
import os, sys, argparse
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from envs.curriculum import GRID_SIZE, N_OBSTACLE_PAIRS
from envs.map_generator import asymmetric_corridor
from envs.symmetry import ALL_D4, transform_grid
from envs.equivariant_extractor import D4EquivariantGridEncoder

# escnn itself is exact to machine precision on this grid, so a correct
# encoder scores ~0. The budget is small on purpose: the structural break this
# test exists to catch (a spatially-varying head applied to equivariant
# features) shows up at ~0.2.
ERROR_BUDGET = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_size", type=int, default=GRID_SIZE)
    ap.add_argument("--n_grids", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    encoder = D4EquivariantGridEncoder(args.grid_size, out_dim=64)
    encoder.eval()  # freeze BatchNorm; see module docstring

    # asymmetric layouts: symmetric ones are mirror-invariant, which would
    # hand the mirror elements a free 0.0000 (see module docstring)
    pool = [asymmetric_corridor(size=args.grid_size,
                                n_obstacles=2 * N_OBSTACLE_PAIRS,
                                rng=np.random.default_rng(args.seed + i))
            for i in range(args.n_grids)]

    errors = {g.value: [] for g in ALL_D4}
    with torch.no_grad():
        for grid in pool:
            x = torch.from_numpy(grid).float()[None, None]
            base = encoder(x)
            base_norm = float(torch.linalg.norm(base))
            for g in ALL_D4:
                xt = torch.from_numpy(
                    np.ascontiguousarray(transform_grid(grid, g))
                ).float()[None, None]
                out = encoder(xt)
                rel = float(torch.linalg.norm(out - base)) / max(base_norm, 1e-9)
                errors[g.value].append(rel)

    print(f"D4 invariance of the arm-3 encoder, grid {args.grid_size}x"
          f"{args.grid_size}, {len(pool)} maps\n")
    print(f"{'group element':22s} {'mean rel err':>12s} {'max':>8s}")
    worst = 0.0
    for gv, errs in errors.items():
        m, mx = float(np.mean(errs)), float(np.max(errs))
        worst = max(worst, m)
        print(f"{gv:22s} {m:12.4f} {mx:8.4f}")

    print(f"\nworst mean relative error: {worst:.4f} (budget {ERROR_BUDGET})")
    if worst > ERROR_BUDGET:
        print("FAIL: encoder is not acting D4-invariantly -- arm 3's premise is "
              "broken, and the arm-1/2/3 comparison would be meaningless")
        return 1
    print("PASS: encoder is D4-invariant to numerical precision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
