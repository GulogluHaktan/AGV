"""Diagnose the train-time vs deterministic-eval success gap (HANDOFF §6.9):
evaluate one checkpoint both with deterministic=True and deterministic=False
on the same stage difficulty. If stochastic ~= train-time rolling success but
deterministic collapses, the policy mean is miscalibrated relative to its own
exploration noise; if both collapse, the rolling train metric itself is
misleading. Run inside an isolated sim instance (GZ_PARTITION + ROS_DOMAIN_ID)
so it can run concurrently with an ongoing training run.
"""
import os, sys, argparse, time
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from stable_baselines3 import SAC, PPO
from envs.map_generator import symmetric_corridor
from envs.gazebo_agv_env import GazeboAGVEnv


def run(model, env, n_episodes, deterministic, step_sleep=0.0):
    """step_sleep emulates the per-step wall-clock overhead that gradient
    updates add during training. The sim runs in real time (async, RTF~5), so
    wall time between actions directly controls how much sim time — and hence
    robot displacement — one env step buys. 0 = pure fast eval."""
    successes, lens, sim_dts = 0, [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        steps = 0
        t_prev = getattr(env, "last_odom_stamp", None)
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            if step_sleep > 0:
                time.sleep(step_sleep)
            obs, reward, terminated, truncated, info = env.step(action)
            t_now = getattr(env, "last_odom_stamp", None)
            if t_prev is not None and t_now is not None and t_now > t_prev:
                sim_dts.append(t_now - t_prev)
            t_prev = t_now
            steps += 1
        successes += int(terminated)
        lens.append(steps)
    mean_dt = float(np.mean(sim_dts)) if sim_dts else float("nan")
    return successes / n_episodes, float(np.mean(lens)), mean_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max_goal_dist", type=float, default=4)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--step_sleep", type=float, default=0.0,
                    help="artificial per-step wall delay (s) emulating training-time "
                         "gradient overhead")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    grid = symmetric_corridor(size=args.grid_size, rng=rng)
    env = GazeboAGVEnv(grid=grid, grid_size=args.grid_size, seed=args.seed,
                       max_goal_dist=args.max_goal_dist, max_steps=args.max_steps)
    model = (SAC if args.algo == "sac" else PPO).load(args.ckpt)

    for det in (True, False):
        sr, mean_len, mean_dt = run(model, env, args.n_episodes, det, args.step_sleep)
        print(f"deterministic={det} step_sleep={args.step_sleep}: "
              f"success_rate={sr:.2%} mean_ep_len={mean_len:.1f} "
              f"sim_dt_per_step={mean_dt:.4f}s (n={args.n_episodes})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
