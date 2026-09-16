"""G3 generalization evaluation: does the policy hold up on layouts it never
trained on, and specifically on D4 images of the ones it did?

This is the measurement behind RQ34 ("to what extent does a symmetry-
regularized policy generalize better to unseen mirrored/rotated warehouse
layouts"), and it is run identically for all three arms so the comparison is
about the arm and not the protocol.

Three conditions:

  train      the training pool itself -- an in-distribution reference, not a
             generalization result. Every other number is read relative to it.
  d4_unseen  D4 images of the training layouts. Note `d4_orbit` drops
             degenerate images: `symmetric_corridor` layouts are invariant
             under the left-right mirror, so that element returns the training
             map itself and would inflate this score with maps the policy has
             in fact seen. What is left is the rotations and the top-bottom
             mirror.
  fresh      layouts from the same generator with a disjoint seed range: new
             obstacle arrangements in the canonical orientation. Separates
             "generalizes across orientation" from "generalizes across layout",
             which a single held-out set would confound.

Both deterministic and stochastic success are reported, because a gap between
them is diagnostic on its own (see HANDOFF §12: the two used to disagree
wildly, which is how the unlearnable-task bug surfaced).

Usage (sim must be up, e.g. `scripts/sim_up.sh 42 evalpart` for an instance
isolated from a running training job):
    python3 scripts/eval_g3.py --ckpt sac_baseline_v4_s5.zip
"""
import os, sys, argparse, json
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from stable_baselines3 import SAC, PPO
from envs.curriculum import (STAGES, GRID_SIZE, N_MAPS, N_OBSTACLE_PAIRS,
                             N_OBSTACLE_SLOTS)
from envs.map_generator import make_map_pool, d4_orbit
from envs.gazebo_agv_env import GazeboAGVEnv

# offset so `fresh` layouts cannot collide with the training seed stream
FRESH_SEED_OFFSET = 10_000


def success_rate(model, pool, grid_size, seed, stage, n_episodes, deterministic):
    env = GazeboAGVEnv(grid=pool, grid_size=grid_size, seed=seed,
                       min_goal_dist=stage["min_goal_dist"],
                       max_goal_dist=stage["max_goal_dist"],
                       max_steps=stage["max_steps"],
                       n_obstacle_slots=N_OBSTACLE_SLOTS)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
        successes += int(terminated)
    env.close()
    return successes / n_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--grid_size", type=int, default=GRID_SIZE)
    ap.add_argument("--seed", type=int, default=3,
                    help="must match the training run's seed, so the train "
                         "condition really is the pool the policy trained on")
    ap.add_argument("--n_maps", type=int, default=N_MAPS,
                    help="must match the training run's --n_maps")
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--stage", default="s5",
                    help="which curriculum stage's difficulty to evaluate at")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    stage = next(s for s in STAGES if s["name"] == args.stage)
    train_pool = make_map_pool(size=args.grid_size, n_maps=args.n_maps,
                               seed=args.seed, n_obstacle_pairs=N_OBSTACLE_PAIRS)
    d4_pool = [t for g in train_pool for _, t in d4_orbit(g)]
    fresh_pool = make_map_pool(size=args.grid_size, n_maps=args.n_maps,
                               seed=args.seed + FRESH_SEED_OFFSET,
                               n_obstacle_pairs=N_OBSTACLE_PAIRS)

    conditions = [("train", train_pool), ("d4_unseen", d4_pool),
                  ("fresh", fresh_pool)]
    model = (SAC if args.algo == "sac" else PPO).load(args.ckpt)

    print(f"G3 evaluation of {args.ckpt}")
    print(f"{args.n_episodes} episodes/condition at stage {stage['name']} "
          f"(dist {stage['min_goal_dist']}-{stage['max_goal_dist']}, "
          f"max_steps={stage['max_steps']})\n")
    results = {}
    for name, pool in conditions:
        row = {}
        for det in (True, False):
            # separate seed per condition so the start/goal stream is not
            # shared, but fixed so re-running is reproducible
            row["deterministic" if det else "stochastic"] = success_rate(
                model, pool, args.grid_size, args.seed + len(name),
                stage, args.n_episodes, det)
        row["n_maps"] = len(pool)
        results[name] = row
        print(f"{name:10s} n_maps={len(pool):3d}  "
              f"deterministic={row['deterministic']:.1%}  "
              f"stochastic={row['stochastic']:.1%}", flush=True)

    base = results["train"]["deterministic"]
    print("\nretention vs the training pool (deterministic):")
    for name in ("d4_unseen", "fresh"):
        got = results[name]["deterministic"]
        rel = got / base if base > 0 else float("nan")
        print(f"  {name:10s} {got:.1%} of task, {rel:.0%} of in-distribution")
    results["retention"] = {
        n: (results[n]["deterministic"] / base if base > 0 else None)
        for n in ("d4_unseen", "fresh")}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
