#!/usr/bin/env bash
# Packs the deliverables into ROLL-NUM_SECTION.zip in the project root
# (only needed if a ZIP is asked for; the main submission is the GitHub repo + PDF).
# Excludes the dataset, the virtual environment and caches.
# Usage: bash scripts/make_submission.sh 23i-0022_A
set -e
NAME=${1:?"usage: bash scripts/make_submission.sh ROLL-NUM_SECTION   e.g. 23i-0022_A"}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rm -f "$NAME.zip"
# -y stores symlinks as links (so a symlinked data folder is never followed)
zip -r -y "$NAME.zip" . -x "data/*" "data" ".venv/*" "*__pycache__*" "*.zip" \
    ".ipynb_checkpoints/*" ".git/*" "uv.lock"
echo "Created $NAME.zip ($(du -h "$NAME.zip" | cut -f1))"
