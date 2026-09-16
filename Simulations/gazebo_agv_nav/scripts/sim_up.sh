#!/usr/bin/env bash
# Start a headless Gazebo instance + ros_gz_bridge for this project and leave
# them running in the background.
#
# Usage: sim_up.sh [ros_domain_id] [gz_partition]
#   Defaults to domain 17 / no partition (what run_native.sh uses for training).
#   Pass e.g. `sim_up.sh 42 evalpart` for a second, isolated instance that can
#   run alongside a training job.
#
# The PIDs are recorded in a per-domain file so sim_down.sh can stop exactly
# this instance. That matters: killing by process-name pattern cannot tell two
# concurrent instances apart, and doing so once took down a training sim that
# was mid-run while only the eval instance was meant to stop.
# no -u: robostack's conda activate.d scripts reference unset vars
set -eo pipefail

DOMAIN="${1:-17}"
PARTITION="${2:-}"

WS="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_activate.sh
. "$(dirname "$0")/_activate.sh"

export GZ_IP=127.0.0.1
export ROS_DOMAIN_ID="$DOMAIN"
export GZ_SIM_RESOURCE_PATH="$WS/models"
[ -n "$PARTITION" ] && export GZ_PARTITION="$PARTITION"

LOGDIR="${TMPDIR:-/tmp}"
PIDFILE="$LOGDIR/agv_sim_${DOMAIN}.pids"
cd "$WS"

if [ -f "$PIDFILE" ]; then
  for pid in $(cat "$PIDFILE"); do
    if kill -0 "$pid" 2>/dev/null; then
      echo "instance for domain $DOMAIN already running (pid $pid); "\
"stop it with sim_down.sh $DOMAIN" >&2
      exit 1
    fi
  done
  rm -f "$PIDFILE"
fi

# setsid: each launch gets its own process group, so sim_down.sh can signal
# the whole group. `ros2 run` in particular forks the real parameter_bridge as
# a CHILD, and killing only the recorded parent leaves that child orphaned --
# eight of them accumulated across a session before this was noticed.
setsid gz sim -r -s -v1 "$WS/worlds/agv_nav.sdf" > "$LOGDIR/gz_${DOMAIN}.log" 2>&1 &
GZ_PID=$!
echo "$GZ_PID" > "$PIDFILE"
echo "gz pid $GZ_PID"

# wait for the world to load before starting the bridge, so the bridge's lazy
# topic discovery sees the model's publishers
for _ in $(seq 1 60); do
  grep -q "Serving world controls" "$LOGDIR/gz_${DOMAIN}.log" && break
  grep -q "Failed to load" "$LOGDIR/gz_${DOMAIN}.log" && { echo "WORLD LOAD FAILED"; exit 1; }
  sleep 1
done

setsid ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:="$WS/launch/bridge.yaml" > "$LOGDIR/bridge_${DOMAIN}.log" 2>&1 &
BRIDGE_PID=$!
echo "$BRIDGE_PID" >> "$PIDFILE"
echo "bridge pid $BRIDGE_PID"
sleep 5
echo "sim up (domain $DOMAIN${PARTITION:+, partition $PARTITION}), pids in $PIDFILE"
