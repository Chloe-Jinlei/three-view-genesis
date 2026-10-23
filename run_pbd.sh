#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sim_python="${GENESIS_PYTHON:-/home/chloe/Desktop/genesis-local/.venv/bin/python}"
exec "$sim_python" run_sim.py --contact-extension --detail-video "$@"
