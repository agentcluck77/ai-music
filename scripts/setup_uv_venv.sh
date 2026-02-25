#!/usr/bin/env bash
set -euo pipefail

# Creates a uv-managed venv in .venv and installs deps.

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first." >&2
  exit 1
fi

uv venv .venv
uv pip install -r requirements.txt

echo "Venv ready in .venv"
