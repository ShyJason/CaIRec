#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH_FILE="${ROOT_DIR}/variants/adapter_decoupled_latent/adapter_decoupled_latent.patch"

cd "${ROOT_DIR}"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "[adapter-variant] patch not found: ${PATCH_FILE}" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[adapter-variant] working tree has existing changes." >&2
  echo "[adapter-variant] create a clean branch or commit/stash changes before applying." >&2
  exit 1
fi

git apply --check "${PATCH_FILE}"
git apply "${PATCH_FILE}"

echo "[adapter-variant] applied decoupled latent adapter patch."
