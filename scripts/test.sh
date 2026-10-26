#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECHOBRIDGE_PYTHON:-$HOME/.venvs/echobridge/bin/python}"
cd "$ROOT"
PYTHONPATH=src:. "$PYTHON" -m pytest -q "$@"
