# Environment (native, Docker-free)

Created 2026-09-16 on the machine that took over the project. There is no
Docker and no passwordless sudo here, so everything runs out of a micromamba
environment instead of the `agv-gazebo` image described in HANDOFF §3-4.

## Recreate

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
~/bin/micromamba create -y -n agv -c conda-forge -c robostack-jazzy \
    ros-jazzy-ros-base ros-jazzy-ros-gz python=3.12 pip
~/bin/micromamba run -n agv pip install -r requirements-pip.txt
```

TurtleBot3 models are vendored in `models/` (from ROBOTIS-GIT/turtlebot3_simulations,
jazzy branch), so no turtlebot3 apt package is needed.

## Pin numpy < 2, and install escnn BEFORE training anything

`escnn` depends on `lie-learn`, which requires numpy < 2. Installing escnn into
an environment that already had numpy 2.x silently **downgraded** it (2.5.3 ->
1.26.4), and that made every checkpoint saved under numpy 2.x unreadable:
SB3 unpickles through cloudpickle, and numpy 1.26 has no `numpy._core`, so
loading fails with `ModuleNotFoundError: No module named 'numpy._core.numeric'`.

This cost a restart of `sac_baseline_v5` partway through. Arm 3 needs escnn, so
the whole experiment has to run under the escnn-compatible stack -- all three
arms on one numeric stack, or the comparison is not defensible. Install escnn
up front and keep `requirements-pip.txt` as the pin.

## Verified working set

See `requirements-pip.txt`. Checked after the downgrade: torch CUDA on the
RTX 4060, escnn field types, SB3, gymnasium, and the scripted-controller gate
test all pass.

## Running on a different machine

Everything needed is in the repo; nothing is tied to the machine that wrote it.

1. Create the environment with the block above (micromamba + `requirements-pip.txt`).
   **Install escnn before training anything** -- see the numpy note above.
2. `git clone` gives you `models/` (vendored TurtleBot3) and
   `worlds/agv_nav.sdf`. The world's model URI is `model://turtlebot3_burger`,
   resolved through `GZ_SIM_RESOURCE_PATH`, which `sim_up.sh` and
   `run_native.sh` both point at `models/`. It previously embedded an absolute
   path, so the committed world only worked on the machine that generated it.
   To regenerate (e.g. a different grid size):

   ```bash
   PYTHONPATH=$PWD python3 worlds/gen_world.py \
       --grid_size 24 --seed 3 --n_obstacle_pairs 14 --out worlds/agv_nav.sdf
   ```

   `--grid_size` and `--n_obstacle_pairs` must match `envs/curriculum.py`
   (`GRID_SIZE`, `N_OBSTACLE_PAIRS`); the env teleports `obs_0..obs_{2N-1}` by
   name to realize each episode's map and errors out if the world has too few.

3. Run the checks before spending GPU time. They take minutes and each one has
   already caught a bug that would otherwise have cost a night:

   ```bash
   ./scripts/sim_up.sh 17
   AGV_WORKSPACE=$PWD python3 scripts/gate_test.py        # env solvable?
   AGV_WORKSPACE=$PWD python3 scripts/test_symmetry.py    # group action correct?
   AGV_WORKSPACE=$PWD python3 scripts/test_equivariance.py # arm 3 transforms right?
   ./scripts/sim_down.sh 17
   ```

4. Train one arm:

   ```bash
   ./scripts/run_native.sh baseline sac sac_baseline_v7
   ./scripts/run_native.sh symmetric_augmentation sac sac_aug_v1
   ./scripts/run_native.sh equivariant sac sac_equi_v1
   ```

   To run two arms at once, give the second a different ROS domain so the two
   sims cannot see each other's topics:

   ```bash
   AGV_ROS_DOMAIN_ID=23 ./scripts/run_native.sh symmetric_augmentation sac sac_aug_v1
   ```

   Stop a specific instance with `./scripts/sim_down.sh <domain>`, or all with
   `--all`. Do **not** reach for `pkill -f "gz sim"`: it matches the calling
   shell's own command line and cannot tell two instances apart -- doing that
   killed a training sim 170k steps into a run (HANDOFF §16a).

5. Afterwards: `scripts/extract_results.py <log>` for the learning curves, and
   `scripts/eval_g3.py --ckpt <stage5.zip>` for the generalization table.

Throughput on the machine this was developed on (RTX 4060 Laptop, i5-12450HX,
12 threads): ~18-21 fps during training, so the 830k-step curriculum takes
about 12 hours per arm. The simulator is the bottleneck, not the GPU, so a
machine with faster cores helps more than a bigger card.
