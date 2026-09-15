"""G3 benchmark: zero-shot success rate of a trained model across all 8 D4
map transforms (identity + 3 rotations + 4 mirror-rotations), on held-out
maps never seen during training. This is the "mirrored/rotated map
generalization heatmap" the paper design calls for (Figure 3).

Usage:
  pixi run -e rl-only python scripts/evaluate.py model_path=outputs/.../baseline_symmetric.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO

from envs.agv_nav_env import AGVNavEnv
from envs.symmetry import ALL_D4, transform_grid, transform_point


def evaluate_on_transform(model: PPO, cfg: DictConfig, g, n_episodes: int) -> float:
    """Evaluate `model` on `n_episodes` fresh held-out maps, each presented
    under D4 transform `g`, and return the success rate."""
    successes = 0
    for ep in range(n_episodes):
        env = AGVNavEnv(grid_size=cfg.grid_size, map_type=cfg.map.map_type,
                         max_steps=cfg.max_steps, seed=cfg.seed + 100_000 + ep)
        obs, _ = env.reset()
        grid_size = obs["occupancy"].shape[0]
        terminated = truncated = False
        while not (terminated or truncated):
            t_obs = {
                "occupancy": transform_grid(obs["occupancy"], g),
                "goal_relative": (transform_point(obs["goal_relative"] + grid_size / 2, g, grid_size)
                                   - grid_size / 2).astype(np.float32),
            }
            action, _ = model.predict(t_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        if terminated:  # goal reached before truncation
            successes += 1
    return successes / n_episodes


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("Pass model_path=... (path to a saved PPO .zip)")

    model = PPO.load(model_path)
    n_eval = cfg.n_eval_maps

    print(f"{'transform':<20} success_rate")
    for g in ALL_D4:
        rate = evaluate_on_transform(model, cfg, g, n_eval)
        print(f"{g.value:<20} {rate:.3f}")


if __name__ == "__main__":
    main()
