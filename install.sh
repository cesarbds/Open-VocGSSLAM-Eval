#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel ninja
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
# Both CUDA packages share names with several incompatible public forks.
# Explicitly replace any binary already installed in this environment.
"${PYTHON_BIN}" -m pip uninstall -y simple-knn diff-gaussian-rasterization || true
"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps --no-build-isolation \
    "${REPO_ROOT}/submodules/simple-knn"
"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps --no-build-isolation \
    "${REPO_ROOT}/submodules/diff-gaussian-rasterization"

# Runtime dependencies used by ICP tracking and semantic keyframe extraction.
"${PYTHON_BIN}" -m pip install --no-build-isolation \
    "${REPO_ROOT}/submodules/fast_gicp"
"${PYTHON_BIN}" -m pip install --no-deps \
    "${REPO_ROOT}/submodules/segment-anything-langsplat"
"${PYTHON_BIN}" -m pip install --no-deps \
    "${REPO_ROOT}/third_party/MobileSAM"

echo "Installation complete. Run: ${PYTHON_BIN} scripts/smoke_test.py"
