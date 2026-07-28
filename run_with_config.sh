#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:?Set CONFIG to a YAML/JSON config path}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

# Run the config directly so wrapper defaults do not override YAML/JSON values.
exec "${PYTHON_BIN}" main.py --config "${CONFIG}" "$@"
