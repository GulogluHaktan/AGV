# Sourced by sim_up.sh and run_native.sh -- not meant to be executed directly.
#
# Activates the project's micromamba environment, wherever it happens to live.
# The location used to be hardcoded to $HOME/micromamba in both scripts, which
# broke as soon as the environment was installed on a separate disk; install.sh
# writes env.local.sh next to this directory's parent to point here instead.
# Kept as one shared file so the two callers cannot drift apart.
#
# Honoured variables (env.local.sh normally sets the first two):
#   MAMBA_ROOT_PREFIX  root of the micromamba installation
#   MICROMAMBA_BIN     path to the micromamba binary
#   AGV_ENV_NAME       environment name (default: agv)

_AGV_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$_AGV_WS/env.local.sh" ] && . "$_AGV_WS/env.local.sh"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/bin/micromamba}"

if [ ! -x "$MICROMAMBA_BIN" ]; then
  echo "micromamba not found at $MICROMAMBA_BIN." >&2
  echo "Run install.sh, or set MICROMAMBA_BIN/MAMBA_ROOT_PREFIX." >&2
  exit 1
fi

eval "$("$MICROMAMBA_BIN" shell hook --shell bash)"
micromamba activate "${AGV_ENV_NAME:-agv}"
