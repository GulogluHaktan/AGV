#!/usr/bin/env bash
# Start a headless Gazebo instance + ros_gz_bridge for this project and leave
# them running in the background. Used by the standalone eval/gate-test scripts
# (run_native.sh starts its own pair for training).
#
# Usage: sim_up.sh [ros_domain_id] [gz_partition]
#   Defaults to domain 17 / no partition (the training instance's settings).
#   Pass e.g. `sim_up.sh 42 evalpart` for a second, isolated instance that can
#   run alongside a training job.
#
# NOTE: do not kill these with `pkill -f "gz sim"` from a shell whose own
# command line contains that string -- pkill -f matches the caller too. Use
# sim_down.sh.
# no -u: robostack's conda activate.d scripts reference unset vars
set -eo pipefail

DOMAIN="${1:-17}"
PARTITION="${2:-}"

WS="$(cd "$(dirname "$0")/.." && pwd)"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/bin/micromamba" shell hook --shell bash)"
micromamba activate agv

export GZ_IP=127.0.0.1
export ROS_DOMAIN_ID="$DOMAIN"
export GZ_SIM_RESOURCE_PATH="$WS/models"
[ -n "$PARTITION" ] && export GZ_PARTITION="$PARTITION"

LOGDIR="${TMPDIR:-/tmp}"
cd "$WS"

gz sim -r -s -v1 "$WS/worlds/agv_nav.sdf" > "$LOGDIR/gz_${DOMAIN}.log" 2>&1 &
echo "gz pid $!"

# wait for the world to actually load before starting the bridge, so the
# bridge's lazy topic discovery sees the model's publishers
for _ in $(seq 1 60); do
  grep -q "Serving world controls" "$LOGDIR/gz_${DOMAIN}.log" && break
  grep -q "Failed to load" "$LOGDIR/gz_${DOMAIN}.log" && { echo "WORLD LOAD FAILED"; exit 1; }
  sleep 1
done

ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:="$WS/launch/bridge.yaml" > "$LOGDIR/bridge_${DOMAIN}.log" 2>&1 &
echo "bridge pid $!"
sleep 5
echo "sim up (domain $DOMAIN${PARTITION:+, partition $PARTITION})"
