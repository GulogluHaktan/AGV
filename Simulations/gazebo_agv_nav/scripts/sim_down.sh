#!/usr/bin/env bash
# Stop the Gazebo instance started by sim_up.sh for one ROS domain.
#
# Usage: sim_down.sh [ros_domain_id]      # default 17
#        sim_down.sh --all                # every instance this project started
#
# Scoped by the PID file sim_up.sh writes, NOT by process-name pattern. Two
# instances coexist by design (a training sim plus an isolated eval sim on
# another domain/partition), and a pattern kill cannot tell them apart: doing
# that once killed a training sim 170k steps into a run while only the eval
# instance was meant to go down. Every step afterwards timed out with no
# physics, which is silent data corruption rather than an obvious failure.
set -eo pipefail

# env.local.sh TMPDIR'i tasiyabilir (kurulum her seyi tek diskte tutmak icin
# oraya yaziyor). sim_up.sh bunu _activate.sh uzerinden aliyor; bu betik conda
# ortamina ihtiyac duymadigi icin dosyayi dogrudan okuyor -- yoksa iki betik
# farkli dizinlere bakar ve pid dosyasi bulunamaz.
_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$_WS/env.local.sh" ] && . "$_WS/env.local.sh"

LOGDIR="${TMPDIR:-/tmp}"

stop_domain() {
  local domain="$1"
  local pidfile="$LOGDIR/agv_sim_${domain}.pids"
  if [ ! -f "$pidfile" ]; then
    echo "no pid file for domain $domain ($pidfile); nothing stopped"
    return 0
  fi
  local pids
  pids=$(cat "$pidfile")
  # signal the whole PROCESS GROUP (negative pid), not just the recorded
  # leader: `ros2 run` forks the real parameter_bridge as a child, so killing
  # the leader alone leaks that child. sim_up.sh starts each launch under
  # setsid precisely so the group id equals the recorded pid.
  for pid in $pids; do
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in $pids; do
    kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  rm -f "$pidfile"
  echo "sim down (domain $domain)"
}

if [ "${1:-}" = "--all" ]; then
  shopt -s nullglob
  found=0
  for pidfile in "$LOGDIR"/agv_sim_*.pids; do
    domain="${pidfile##*/agv_sim_}"
    stop_domain "${domain%.pids}"
    found=1
  done
  [ "$found" = 0 ] && echo "no running instances recorded"
  exit 0
fi

stop_domain "${1:-17}"
