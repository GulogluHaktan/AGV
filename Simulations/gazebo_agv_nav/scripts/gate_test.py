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
from envs.map_generator import symmetric_corridor
from envs.gazebo_agv_env import GazeboAGVEnv

# same ladder as scripts/curriculum_train.py STAGES
STAGES = [
    dict(name="s1", max_goal_dist=4,    max_steps=80),
    dict(name="s2", max_goal_dist=8,    max_steps=140),
    dict(name="s3", max_goal_dist=12,   max_steps=200),
    dict(name="s4", max_goal_dist=16,   max_steps=260),
    dict(name="s5", max_goal_dist=None, max_steps=320),
]

# pass mark for the scripted controller. Not 100%: the controller is a pure
# go-to-goal law with no obstacle avoidance, so it legitimately gets wedged on
# the shelf racks in some layouts. Below this, suspect the env, not the agent.
PASS_THRESHOLD = 0.80


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


def run_stage(stage, grid, grid_size, seed, n_episodes):
    env = GazeboAGVEnv(grid=grid, grid_size=grid_size, seed=seed,
                       max_goal_dist=stage["max_goal_dist"],
                       max_steps=stage["max_steps"])
    successes, lens, final_dists, start_dists = 0, [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        start_dists.append(float(np.linalg.norm(env.goal - env._pose)))
        terminated = truncated = False
        steps = 0
        info = {}
        while not (terminated or truncated):
            action = scripted_action(obs, env.max_ang, env.control_dt)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
        successes += int(terminated)
        lens.append(steps)
        final_dists.append(info.get("dist_to_goal", float("nan")))
    env.close()
    return dict(sr=successes / n_episodes, mean_len=float(np.mean(lens)),
                mean_final_dist=float(np.mean(final_dists)),
                mean_start_dist=float(np.mean(start_dists)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="s1..s5 (default: all)")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    grid = symmetric_corridor(size=args.grid_size,
                              rng=np.random.default_rng(args.seed))
    stages = [s for s in STAGES if args.stage in (None, s["name"])]
    if not stages:
        sys.exit(f"unknown stage {args.stage!r}")

    print(f"scripted-controller gate test, {args.n_episodes} episodes/stage, "
          f"pass mark {PASS_THRESHOLD:.0%}\n", flush=True)
    results = {}
    for stage in stages:
        r = run_stage(stage, grid, args.grid_size, args.seed, args.n_episodes)
        results[stage["name"]] = r
        verdict = "PASS" if r["sr"] >= PASS_THRESHOLD else "FAIL"
        print(f"[{verdict}] {stage['name']}: success={r['sr']:.0%} "
              f"mean_steps={r['mean_len']:.0f}/{stage['max_steps']} "
              f"start_dist={r['mean_start_dist']:.1f}m "
              f"end_dist={r['mean_final_dist']:.2f}m", flush=True)

    overall = float(np.mean([r["sr"] for r in results.values()]))
    passed = overall >= PASS_THRESHOLD
    print(f"\noverall {overall:.0%} -> "
          f"{'ENV OK, safe to train' if passed else 'ENV STILL BROKEN, do not train'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
