"""Gate test: can a hand-written controller solve this env at all?

Run this after ANY change to the observation, action, reward, or step timing,
and BEFORE spending GPU hours on a training run. A scripted turn-then-drive
controller should reach ~90%+ on a navigation task this simple; if it cannot,
the environment is broken and no amount of RL tuning will help.

This exists because two overnight runs (~16 GPU-hours) were spent training on
an env whose observation omitted the robot's yaw while the action was
body-frame, making the task formally unsolvable (HANDOFF §12). This check takes
a couple of minutes and would have caught it immediately.

The controller only uses `goal_body`, i.e. exactly the information the fix added
-- so a pass here also proves that information is present and correctly framed.

Usage (sim must already be up, e.g. `scripts/sim_up.sh`):
    python3 scripts/gate_test.py                 # all 5 curriculum difficulties
    python3 scripts/gate_test.py --stage s3      # just one
"""
import os, sys, argparse, math
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from envs.curriculum import (STAGES, GRID_SIZE, N_MAPS, N_OBSTACLE_PAIRS,
                             N_OBSTACLE_SLOTS)
from envs.map_generator import make_map_pool
from envs.gazebo_agv_env import GazeboAGVEnv


# A raw success rate cannot tell "the env is broken" from "this controller is
# too dumb for this map", and those call for opposite responses. So failures are
# classified and the verdict keys off the cause:
#
#   wedged       stopped in contact with an obstacle. The controller is a pure
#                go-to-goal law with NO avoidance, so this is its known blind
#                spot, not an env fault -- an RL policy sees the occupancy grid
#                and can route around. Counted as "env fine".
#   no_progress  barely closed any distance while NOT touching anything. This is
#                the signature of a broken env: it is exactly what the missing-
#                yaw bug looked like (policy hedging, robot milling about in
#                open space). Any meaningful rate here fails the gate.
#   budget       made real progress and was still moving when the step budget
#                ran out. Means max_steps is too tight for the band.
HANDLED_MIN = 0.80      # success + wedged
NO_PROGRESS_MAX = 0.10
BUDGET_MAX = 0.15
# fraction of the closable distance below which an episode counts as no-progress
PROGRESS_FLOOR = 0.25
# contact/stall thresholds used to recognize wedging
WEDGE_OBSTACLE_DIST = 1.1
STALL_DISPLACEMENT = 0.5
STALL_WINDOW = 25


def scripted_action(obs, max_ang, control_dt, drive_cone=math.radians(60)):
    """Turn toward the goal, then drive. Reads only goal_body = (forward, left)
    in the robot frame, so `bearing` is directly the angle to steer off.

    The angular command is scaled by what one step can actually deliver
    (max_ang * control_dt) instead of an arbitrary gain, so it asks for exactly
    the rotation still needed rather than saturating and overshooting."""
    fwd, left = obs["goal_body"]
    bearing = math.atan2(left, fwd)
    ang = np.clip(bearing / (max_ang * control_dt), -1.0, 1.0)
    # drive while roughly pointed at the goal, easing off as bearing grows;
    # outside the cone, rotate in place
    lin = max(0.0, math.cos(bearing)) if abs(bearing) < drive_cone else 0.0
    return np.array([lin, ang], dtype=np.float32)


def run_stage(stage, pool, grid_size, seed, n_episodes):
    env = GazeboAGVEnv(grid=pool, grid_size=grid_size, seed=seed,
                       min_goal_dist=stage["min_goal_dist"],
                       max_goal_dist=stage["max_goal_dist"],
                       max_steps=stage["max_steps"],
                       n_obstacle_slots=N_OBSTACLE_SLOTS)
    counts = {"success": 0, "wedged": 0, "no_progress": 0, "budget": 0}
    lens, final_dists, start_dists = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        start_dist = float(np.linalg.norm(env.goal - env._pose))
        start_dists.append(start_dist)
        terminated = truncated = False
        steps, info, trail = 0, {}, []
        while not (terminated or truncated):
            action = scripted_action(obs, env.max_ang, env.control_dt)
            obs, _, terminated, truncated, info = env.step(action)
            trail.append(env._pose.copy())
            steps += 1
        lens.append(steps)
        final_dists.append(info.get("dist_to_goal", float("nan")))

        if terminated:
            counts["success"] += 1
            continue
        tail = np.asarray(trail[-STALL_WINDOW:])
        displacement = (float(np.linalg.norm(tail[-1] - tail[0]))
                        if len(tail) > 1 else 0.0)
        obstacle_dist = info.get("obstacle_dist", float("inf"))
        closable = max(start_dist - env.goal_radius, 1e-6)
        progress = (start_dist - info.get("dist_to_goal", start_dist)) / closable
        if displacement < STALL_DISPLACEMENT and obstacle_dist < WEDGE_OBSTACLE_DIST:
            counts["wedged"] += 1
        elif progress < PROGRESS_FLOOR:
            counts["no_progress"] += 1
        else:
            counts["budget"] += 1
    env.close()
    rates = {k: v / n_episodes for k, v in counts.items()}
    return dict(rates=rates, mean_len=float(np.mean(lens)),
                mean_final_dist=float(np.mean(final_dists)),
                mean_start_dist=float(np.mean(start_dists)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="s1..s5 (default: all)")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--grid_size", type=int, default=GRID_SIZE)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_maps", type=int, default=N_MAPS,
                    help="size of the map pool the env samples per episode")
    args = ap.parse_args()

    pool = make_map_pool(size=args.grid_size, n_maps=args.n_maps,
                     seed=args.seed, n_obstacle_pairs=N_OBSTACLE_PAIRS)
    stages = [s for s in STAGES if args.stage in (None, s["name"])]
    if not stages:
        sys.exit(f"unknown stage {args.stage!r}")

    print(f"scripted-controller gate test, {args.n_episodes} episodes/stage, "
          f"{len(pool)} maps in pool", flush=True)
    print(f"gate: success+wedged >= {HANDLED_MIN:.0%}, "
          f"no_progress <= {NO_PROGRESS_MAX:.0%}, budget <= {BUDGET_MAX:.0%}\n",
          flush=True)
    results = {}
    for stage in stages:
        r = run_stage(stage, pool, args.grid_size, args.seed, args.n_episodes)
        results[stage["name"]] = r
        q = r["rates"]
        print(f"{stage['name']}: success={q['success']:.0%} wedged={q['wedged']:.0%} "
              f"no_progress={q['no_progress']:.0%} budget={q['budget']:.0%} | "
              f"steps={r['mean_len']:.0f}/{stage['max_steps']} "
              f"start={r['mean_start_dist']:.1f}m end={r['mean_final_dist']:.2f}m",
              flush=True)

    def mean_of(key):
        return float(np.mean([r["rates"][key] for r in results.values()]))

    success, wedged = mean_of("success"), mean_of("wedged")
    no_progress, budget = mean_of("no_progress"), mean_of("budget")
    problems = []
    if success + wedged < HANDLED_MIN:
        problems.append(f"success+wedged {success + wedged:.0%} < {HANDLED_MIN:.0%}")
    if no_progress > NO_PROGRESS_MAX:
        problems.append(f"no_progress {no_progress:.0%} > {NO_PROGRESS_MAX:.0%} "
                        "(robot not making headway in open space -- suspect the "
                        "observation/action contract)")
    if budget > BUDGET_MAX:
        problems.append(f"budget {budget:.0%} > {BUDGET_MAX:.0%} "
                        "(max_steps too tight for these bands)")

    print(f"\noverall success={success:.0%} wedged={wedged:.0%} "
          f"no_progress={no_progress:.0%} budget={budget:.0%}")
    if problems:
        print("\nGATE FAILED, do not train:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nGATE PASSED: env is solvable; remaining failures are the scripted "
          "controller's missing obstacle avoidance, which an RL policy can learn "
          "(it sees the occupancy grid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
