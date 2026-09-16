#!/usr/bin/env bash
# Native (Docker-free) training run via micromamba/RoboStack (ROS2 Jazzy +
# Gazebo Harmonic from conda-forge). Mirrors the docker command in
# Resources/HANDOFF.md section 4.
#
# Usage: run_native.sh <arm> <algo> <out_prefix> [extra curriculum_train args...]
#   e.g. run_native.sh baseline sac sac_baseline_v5
#        run_native.sh baseline sac sac_baseline_v5 --resume_from sac_baseline_v5_s1.zip --start_stage 1
#
# The sim is started through sim_up.sh rather than inline, so its PIDs land in
# the per-domain PID file and sim_down.sh can stop exactly this instance. That
# also keeps one copy of the launch logic instead of two that can drift.
# no -u: robostack's conda activate.d scripts reference unset vars
set -eo pipefail

ARM="${1:-baseline}"
ALGO="${2:-sac}"
PREFIX="${3:-sac_baseline}"
shift 3 2>/dev/null || true

DOMAIN="${AGV_ROS_DOMAIN_ID:-17}"
WS="$(cd "$(dirname "$0")/.." && pwd)"

"$WS/scripts/sim_down.sh" "$DOMAIN" >/dev/null 2>&1 || true
"$WS/scripts/sim_up.sh" "$DOMAIN"
cleanup() { "$WS/scripts/sim_down.sh" "$DOMAIN" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# shellcheck source=scripts/_activate.sh
. "$(dirname "$0")/_activate.sh"

export AGV_WORKSPACE="$WS"
export GZ_IP=127.0.0.1                     # GZ Transport discovery fix
export ROS_DOMAIN_ID="$DOMAIN"
export GZ_SIM_RESOURCE_PATH="$WS/models"   # turtlebot3_common meshes
cd "$WS"

python3 scripts/curriculum_train.py --arm "$ARM" --algo "$ALGO" \
  --out_prefix "$WS/$PREFIX" "$@" 2>&1 | tee "$WS/${PREFIX}.log"
echo DONE > "$WS/${PREFIX}.done"
