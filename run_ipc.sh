#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Only shared CUDA runtime libraries are reused from this existing environment.
# No application code, original demonstrations, or simulation data is loaded.
sim_python="${GENESIS_PYTHON:-/home/chloe/Desktop/genesis-local/.venv/bin/python}"
cuda12_root="${CUDA12_LIBRARY_ROOT:-/home/chloe/miniconda3/envs/adaptigraph/lib/python3.9/site-packages/nvidia}"
ipc_library_path=""
for component in cublas cusparse cusolver cuda_runtime nvjitlink; do
  if [[ -d "$cuda12_root/$component/lib" ]]; then
    ipc_library_path="$ipc_library_path$cuda12_root/$component/lib:"
  fi
done
export LD_LIBRARY_PATH="${ipc_library_path}${LD_LIBRARY_PATH:-}"
exec "$sim_python" run_ipc.py "$@"
