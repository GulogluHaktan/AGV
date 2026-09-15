#!/usr/bin/env bash
# escnn (E(2)-steerable equivariant network library, used by the "equivariant
# policy" arm) depends on lie_learn, whose legacy setup.py needs numpy already
# importable at build time and fails under isolated PEP517 resolvers (pixi/uv
# default to build isolation). Run this once, after `pixi install`, from
# inside the pixi env so numpy is already on the interpreter's path.
#
# Usage: pixi run -e rl-only bash scripts/install_escnn.sh
#        pixi run bash scripts/install_escnn.sh   (default env, incl. isaac)

set -euo pipefail

python -c "import numpy" || { echo "numpy not importable in this env — run inside a pixi shell/task first"; exit 1; }

python -m pip install --no-build-isolation lie_learn
python -m pip install escnn
python -c "import escnn; print('escnn OK, version', escnn.__version__)"
