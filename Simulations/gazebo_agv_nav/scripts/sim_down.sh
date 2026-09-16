#!/usr/bin/env bash
# Stop the Gazebo server and ros_gz_bridge started by sim_up.sh.
#
# Exists because `pkill -f "gz sim"` also matches the command line of whatever
# shell issued it (pkill -f sees whole cmdlines), so typing it inline kills the
# caller. Matching from inside a script is not enough on its own either: a
# caller whose own command line happens to mention "gz sim ..." still matches.
# So this only kills processes that are NOT shells -- the real gz/bridge
# processes -- and never touches itself or its ancestors.
set -eo pipefail

is_shell() {
  case "$(cat "/proc/$1/comm" 2>/dev/null)" in
    bash|sh|dash|zsh|ksh) return 0 ;;
    *) return 1 ;;
  esac
}

# collect this process and all of its ancestors, so we can never kill our own
# invocation chain
ancestors=" "
p=$$
while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
  ancestors="$ancestors$p "
  p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || echo "")
done

kill_matching() {
  local sig="$1" pat="$2" pid
  for pid in $(pgrep -f "$pat" 2>/dev/null || true); do
    case "$ancestors" in *" $pid "*) continue ;; esac
    is_shell "$pid" && continue
    kill "$sig" "$pid" 2>/dev/null || true
  done
}

for pat in "gz sim" "parameter_bridge"; do
  kill_matching -TERM "$pat"
done
sleep 2
for pat in "gz sim" "parameter_bridge"; do
  kill_matching -KILL "$pat"
done
echo "sim down"
