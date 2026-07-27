#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:?Set CONFIG to a YAML/JSON config path}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

# Run the config directly so wrapper defaults do not override YAML/JSON values.
exec python main.py --config "${CONFIG}" "$@"
