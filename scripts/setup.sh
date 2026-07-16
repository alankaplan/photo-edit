#!/usr/bin/env bash
# Bootstrap the environment for the ORF Photo Editor.
# Installs Python dependencies and the system GL libraries PySide6 needs to
# run (even headless, under the offscreen platform).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[setup] installing Python dependencies…"
python3 -m pip install --quiet -r requirements.txt pytest

# PySide6 needs these shared libraries present to import, even headless.
if command -v apt-get >/dev/null 2>&1; then
  if ! python3 -c "import PySide6.QtWidgets" >/dev/null 2>&1; then
    echo "[setup] installing Qt system libraries…"
    apt-get update -q || true
    apt-get install -y -q libegl1 libgl1 libxkbcommon0 libdbus-1-3 || true
  fi
fi

echo "[setup] done."
