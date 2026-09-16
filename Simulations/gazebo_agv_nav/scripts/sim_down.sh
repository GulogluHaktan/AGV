#!/usr/bin/env bash
# Stop the Gazebo server and ros_gz_bridge started by sim_up.sh.
#
# Exists because `pkill -f "gz sim"` typed directly into a shell also matches
# that shell's own command line (pkill -f sees the whole cmdline), which kills
# the caller. Matching from inside this script is safe: the patterns live in the
# script file, not in the invoking command line.
set -eo pipefail

for pat in "gz sim -r -s" "ros_gz_bridge" "parameter_bridge"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    kill "$pid" 2>/dev/null || true
  done
done
sleep 2
# escalate on anything that ignored SIGTERM
for pat in "gz sim -r -s" "parameter_bridge"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    kill -9 "$pid" 2>/dev/null || true
  done
done
echo "sim down"
