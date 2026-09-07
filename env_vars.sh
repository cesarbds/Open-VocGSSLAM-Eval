#!/usr/bin/env bash

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the Conda environment before sourcing env_vars.sh" >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -x "${CUDACXX}" ]]; then
    echo "nvcc was not found at ${CUDACXX}" >&2
    echo "Recreate/update the environment from environment.yml." >&2
    return 1 2>/dev/null || exit 1
fi

echo "Using isolated CUDA toolkit: ${CUDA_HOME}"
"${CUDACXX}" --version | tail -n 1
