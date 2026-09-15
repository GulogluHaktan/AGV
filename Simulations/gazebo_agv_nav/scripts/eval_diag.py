"""Diagnose the train-time vs deterministic-eval success gap (HANDOFF §6.9):
evaluate one checkpoint both with deterministic=True and deterministic=False
on the same stage difficulty. If stochastic ~= train-time rolling success but
deterministic collapses, the policy mean is miscalibrated relative to its own
exploration noise; if both collapse, the rolling train metric itself is
misleading. Run inside an isolated sim instance (GZ_PARTITION + ROS_DOMAIN_ID)
so it can run concurrently with an ongoing training run.
"""
import os, sys, argparse
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from stable_baselines3 import SAC, PPO
from envs.map_generator import symmetric_corridor
from envs.gazebo_agv_env import GazeboAGVEnv


def run(model, env, n_episodes, deterministic):
    successes, lens = 0, []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        successes += int(terminated)
        lens.append(steps)
    return successes / n_episodes, float(np.mean(lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max_goal_dist", type=float, default=4)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--n_episodes", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    grid = symmetric_corridor(size=args.grid_size, rng=rng)
    env = GazeboAGVEnv(grid=grid, grid_size=args.grid_size, seed=args.seed,
                       max_goal_dist=args.max_goal_dist, max_steps=args.max_steps)
    model = (SAC if args.algo == "sac" else PPO).load(args.ckpt)

    for det in (True, False):
        sr, mean_len = run(model, env, args.n_episodes, det)
        print(f"deterministic={det}: success_rate={sr:.2%} "
              f"mean_ep_len={mean_len:.1f} (n={args.n_episodes})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
