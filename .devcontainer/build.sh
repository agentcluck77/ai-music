#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[bootstrap] $*"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BREWFILE="${BREWFILE:-$REPO_ROOT/.devcontainer/Brewfile}"

if command -v apt-get >/dev/null 2>&1; then
  log "Installing system packages via apt-get"
  sudo apt-get update -y
  sudo apt-get install -y \
    ca-certificates \
    curl \
    git-lfs \
    jq \
    openslide-tools \
    rsync
  sudo git lfs install --system || true
elif command -v dnf >/dev/null 2>&1; then
  log "Installing system packages via dnf"
  sudo dnf install -y \
    ca-certificates \
    curl \
    git-lfs \
    jq \
    openslide-tools \
    rsync
  sudo git lfs install --system || true
elif command -v brew >/dev/null 2>&1; then
  log "Installing system packages via brew"
  if [ -d /home/linuxbrew/.linuxbrew ]; then
    sudo chown -R "$(id -u)":"$(id -g)" /home/linuxbrew/.linuxbrew || true
  fi
  brew install git-lfs rsync jq openslide || true
  git lfs install --system || true
else
  log "No supported package manager found for system deps"
fi

if command -v brew >/dev/null 2>&1; then
  if [ -d /home/linuxbrew/.linuxbrew ]; then
    sudo chown -R "$(id -u)":"$(id -g)" /home/linuxbrew/.linuxbrew || true
  fi
  if [ -f "$BREWFILE" ]; then
    log "Running brew bundle install"
    brew bundle install --file "$BREWFILE" || true
  else
    log "Brewfile not found at $BREWFILE"
  fi
fi

MINIFORGE_HOME="$HOME/miniforge3"
if [ ! -d "$MINIFORGE_HOME" ]; then
  log "Installing Miniforge"
  curl -L -o /tmp/Miniforge3.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/Miniforge3.sh -b -p "$MINIFORGE_HOME"
  rm -f /tmp/Miniforge3.sh
else
  log "Miniforge already present"
fi

CONDA_BIN="$MINIFORGE_HOME/bin/conda"
if [ ! -x "$CONDA_BIN" ]; then
  log "Conda binary not found at $CONDA_BIN"
  exit 1
fi

touch "$HOME/.bashrc"
if ! grep -q 'conda" shell.bash hook' "$HOME/.bashrc"; then
  log "Adding conda init hook to .bashrc"
  echo 'eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"' >> "$HOME/.bashrc"
else
  log "Conda init hook already present"
fi

"$CONDA_BIN" env config vars set -n base PYTHONPATH=/workspaces/content
