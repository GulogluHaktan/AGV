"""Train one arm (baseline / symmetric_augmentation / equivariant_policy) on
the lightweight AGVNavEnv. Isaac-Sim-backed training will reuse this same
Hydra config structure once envs/isaac_agv_nav_env.py exists — swap the env
constructor, keep everything else.

Usage:
  pixi run -e rl-only python scripts/train.py arm=baseline
  pixi run -e rl-only python scripts/train.py arm=symmetric_augmentation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO

from envs.agv_nav_env import AGVNavEnv
from envs.wrappers import SymmetricAugmentationWrapper


def build_env(cfg: DictConfig):
    env = AGVNavEnv(
        grid_size=cfg.grid_size,
        map_type=cfg.map.map_type,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
    )
    if cfg.arm.use_augmentation:
        env = SymmetricAugmentationWrapper(env, seed=cfg.seed)
    return env


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.arm.use_equivariant:
        raise NotImplementedError(
            "Arm 3 (equivariant_policy) needs escnn — run scripts/install_escnn.sh "
            "(requires gcc-fortran) and implement envs/equivariant_policy.py first."
        )

    env = build_env(cfg)
    model = PPO(cfg.arm.policy, env, seed=cfg.seed, verbose=1,
                tensorboard_log=cfg.output_dir)
    model.learn(total_timesteps=cfg.total_timesteps)

    out_path = Path(cfg.output_dir) / f"{cfg.arm.name}_{cfg.map.map_type}.zip"
    model.save(str(out_path))
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
