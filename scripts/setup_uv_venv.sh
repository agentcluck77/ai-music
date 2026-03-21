#!/usr/bin/env bash
set -euo pipefail

# Creates a uv-managed venv in .venv and installs deps.

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first." >&2
  exit 1
fi

# Avoid permission issues in shared/global uv cache directories.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"

uv venv .venv
uv pip install -r requirements.txt

echo "Venv ready in .venv"
