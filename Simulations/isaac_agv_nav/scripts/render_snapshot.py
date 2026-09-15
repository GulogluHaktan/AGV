"""Quick visualization: occupancy grid map + start/goal + (optional) a
rollout trajectory from a trained model. Not part of the training loop —
just a debug/inspection tool since the lightweight env has no built-in
renderer.

Usage:
  pixi run -e rl-only python scripts/render_snapshot.py \
      map_type=symmetric out=snapshot.png [model_path=outputs/.../baseline_symmetric.zip]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from envs.agv_nav_env import AGVNavEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_type", default="symmetric", choices=["symmetric", "asymmetric"])
    ap.add_argument("--grid_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--out", default="snapshot.png")
    args = ap.parse_args()

    env = AGVNavEnv(grid_size=args.grid_size, map_type=args.map_type, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    trajectory = [env.pose[:2].copy()]
    outcome = "not run"
    if args.model_path:
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.pose[:2].copy())
        outcome = "reached goal" if terminated else "timed out"

    trajectory = np.array(trajectory)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(env.grid, cmap="Greys", origin="lower", vmin=0, vmax=1)
    ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=12, label="start")
    ax.plot(env.goal[0], env.goal[1], "r*", markersize=18, label="goal")
    if len(trajectory) > 1:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=1.5, label="trajectory")
    ax.set_title(f"AGVNavEnv — {args.map_type} map, seed={args.seed}\n({outcome})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.5, args.grid_size - 0.5)
    ax.set_ylim(-0.5, args.grid_size - 0.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
