#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel ninja
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install "${REPO_ROOT}/submodules/simple-knn"
"${PYTHON_BIN}" -m pip install "${REPO_ROOT}/submodules/diff-gaussian-rasterization"

echo "Installation complete. Run: ${PYTHON_BIN} scripts/smoke_test.py"
