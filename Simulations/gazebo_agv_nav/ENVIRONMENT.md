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
