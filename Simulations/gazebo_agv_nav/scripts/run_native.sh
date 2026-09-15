#!/usr/bin/env bash
# Native (Docker-free) training run via micromamba/RoboStack (ROS2 Jazzy +
# Gazebo Harmonic from conda-forge). Mirrors the docker command in
# Resources/HANDOFF.md section 4.
#
# Usage: run_native.sh <arm> <algo> <out_prefix>
#   e.g. run_native.sh baseline sac sac_baseline_v2
# no -u: robostack's conda activate.d scripts reference unset vars (CONDA_BUILD)
set -eo pipefail

ARM="${1:-baseline}"
ALGO="${2:-sac}"
PREFIX="${3:-sac_baseline_v2}"

WS="$(cd "$(dirname "$0")/.." && pwd)"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/bin/micromamba" shell hook --shell bash)"
micromamba activate agv

export AGV_WORKSPACE="$WS"
export GZ_IP=127.0.0.1        # same discovery fix as in the Docker setup
export ROS_DOMAIN_ID=17       # isolate from any other ROS graph on the host
export GZ_SIM_RESOURCE_PATH="$WS/models"   # turtlebot3_common meshes
cd "$WS"

# gz sim: headless server only (-s), same flags as the docker launch file
gz sim -r -s -v3 "$WS/worlds/agv_nav.sdf" > /tmp/claude-1000/gz.log 2>&1 &
GZ_PID=$!
sleep 5

ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:="$WS/launch/bridge.yaml" > /tmp/claude-1000/bridge.log 2>&1 &
BRIDGE_PID=$!
sleep 5

cleanup() { kill "$GZ_PID" "$BRIDGE_PID" 2>/dev/null || true; }
trap cleanup EXIT

python3 scripts/curriculum_train.py --arm "$ARM" --algo "$ALGO" \
  --out_prefix "$WS/$PREFIX" 2>&1 | tee "$WS/${PREFIX}.log"
echo DONE > "$WS/${PREFIX}.done"
