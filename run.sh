#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
GENESIS_INTERPRETER="${GENESIS_PYTHON:-/home/chloe/Desktop/genesis-local/.venv/bin/python}"
exec "$GENESIS_INTERPRETER" run_all.py "$@"
