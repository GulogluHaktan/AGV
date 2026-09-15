import sys, argparse
sys.path.insert(0, "/workspace")
import numpy as np
from stable_baselines3 import PPO
from envs.map_generator import symmetric_corridor
from envs.gazebo_agv_env import GazeboAGVEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/workspace/baseline_v5.zip")
ap.add_argument("--n_episodes", type=int, default=15)
ap.add_argument("--grid_size", type=int, default=16)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--max_goal_dist", type=float, default=None)
ap.add_argument("--max_steps", type=int, default=120)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
grid = symmetric_corridor(size=args.grid_size, rng=rng)
env = GazeboAGVEnv(grid=grid, grid_size=args.grid_size, seed=args.seed,
                    max_goal_dist=args.max_goal_dist, max_steps=args.max_steps)
model = PPO.load(args.model)

successes = 0
for ep in range(args.n_episodes):
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    successes += int(terminated)
    print(f"episode {ep}: {'REACHED' if terminated else 'timeout'} dist={info['dist_to_goal']:.2f}")

print(f"\nSuccess rate: {successes}/{args.n_episodes} = {successes/args.n_episodes:.2%}")
env.close()
