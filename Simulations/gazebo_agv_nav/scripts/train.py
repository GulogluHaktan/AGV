import sys, argparse
sys.path.insert(0, "/workspace")
import numpy as np
from stable_baselines3 import PPO
from envs.map_generator import symmetric_corridor
from envs.gazebo_agv_env import GazeboAGVEnv
from envs.wrappers import SymmetricAugmentationWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "symmetric_augmentation"], default="baseline")
    ap.add_argument("--total_timesteps", type=int, default=5000)
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="/workspace/model.zip")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--max_goal_dist", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=120)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    grid = symmetric_corridor(size=args.grid_size, rng=rng)

    env = GazeboAGVEnv(grid=grid, grid_size=args.grid_size, seed=args.seed,
                        max_goal_dist=args.max_goal_dist, max_steps=args.max_steps)
    if args.arm == "symmetric_augmentation":
        env = SymmetricAugmentationWrapper(env, seed=args.seed)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"Resumed from {args.resume_from}")
    else:
        model = PPO("MultiInputPolicy", env, seed=args.seed, verbose=1, n_steps=256, batch_size=64,
                    ent_coef=0.01)
    model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=not bool(args.resume_from))
    model.save(args.out)
    print(f"Saved {args.out}")
    env.close()


if __name__ == "__main__":
    main()
