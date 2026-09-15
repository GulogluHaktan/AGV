# isaac_agv_nav

Simulation + RL code for Paper Idea 5: *"Symmetry-Exploiting Deep Reinforcement
Learning for Sample-Efficient AGV Path Planning and Sim-to-Real Transfer"*
(target venue: *Symmetry*, MDPI, Computer section).

See `Resources/Literatur-Taramasi/` for the literature review backing this
design, and `Resources/Mission/Symmetry_Makale_Fikirleri.docx` (Makale Fikri 5)
for the original proposal.

## Environment setup (pixi)

This project uses [pixi](https://pixi.sh) instead of native ROS2/Gazebo or a
manual conda env — on Arch/CachyOS this avoids both native ROS2 packaging
pain and Docker+Wayland GUI-forwarding friction, and mirrors the working
setup already used in the sibling `~/Projects/omnilrs` project (Isaac Sim 5.0
via pixi).

```bash
pixi install -e rl-only   # gymnasium, stable-baselines3, torch, wandb — no Isaac Sim, fast
pixi install              # default env, adds Isaac Sim 5.0 (large download, GPU-heavy)
```

### Known install snags (and fixes already applied in `pixi.toml`)

- `isaacsim` pins `numpy==1.26.0` — kept unpinned in `core` so it doesn't
  conflict; pinned only inside the `isaac` feature.
- `escnn` (used by the equivariant-policy arm) depends on `lie_learn`, whose
  legacy `setup.py` build fails under pixi/uv's isolated PEP517 builds and
  additionally needs a Fortran compiler (`py3nj` → `gfortran`) not present by
  default on Arch. Install order:
  ```bash
  sudo pacman -S gcc-fortran        # one-time, needs sudo password
  pixi run -e rl-only bash scripts/install_escnn.sh
  ```
  Until this is done, arms 1 (baseline) and 2 (symmetric data augmentation)
  work fine; arm 3 (equivariant policy network) is blocked.

## Experiment design (three arms, per the paper's methodology)

1. **Baseline** — standard PPO/SAC (Stable-Baselines3), symmetry-agnostic CNN/MLP policy.
2. **Symmetric data augmentation** — same baseline architecture, but training
   observation/action pairs are randomly transformed under the corridor's D4
   symmetry group each step (`envs/symmetry.py`).
3. **Equivariant policy network** — `escnn`-based E(2)-steerable convolutional
   policy, natively equivariant under the same D4 group (requires the escnn
   install above).

Evaluated on:
- **Sample efficiency** — episodes to reach target success rate (all 3 arms, same maps).
- **Generalization (G3)** — zero-shot success rate on held-out *mirrored/rotated*
  map layouts never seen during training.
- **Sim-to-real transfer** — physical AGV success rate + error decomposition
  by symmetry-breaking source (asymmetric sensor noise, uneven floor friction),
  once the Isaac Sim policy is validated in simulation.

## Layout

```
envs/         gym-style environment + symmetry-group transforms
configs/      hydra configs per arm and per map layout
scripts/      train.py / evaluate.py entrypoints, install_escnn.sh
assets/maps/  procedurally generated symmetric/asymmetric corridor layouts
```
