#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECHOBRIDGE_PYTHON:-$HOME/.venvs/echobridge/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python venv not found: $PYTHON" >&2
  echo "Create it with: uv venv ~/.venvs/echobridge --python 3.12 && uv pip install --python ~/.venvs/echobridge/bin/python -e '$ROOT[dev]'" >&2
  exit 1
fi

mkdir -p "$ROOT/var"
TOKEN_FILE="$ROOT/var/dev-token"
if [[ ! -s "$TOKEN_FILE" ]]; then
  "$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))' > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE" 2>/dev/null || true
fi

export ECHOBRIDGE_DEV=1
export ECHOBRIDGE_TOKEN="$(cat "$TOKEN_FILE")"
export ECHOBRIDGE_BIND="${ECHOBRIDGE_BIND:-0.0.0.0}"
export ECHOBRIDGE_PORT="${ECHOBRIDGE_PORT:-18787}"
export ECHOBRIDGE_ENABLE_PROBE="${ECHOBRIDGE_ENABLE_PROBE:-1}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "EchoBridge dev server"
echo "  bind:  ${ECHOBRIDGE_BIND}:${ECHOBRIDGE_PORT}"
echo "  token: $TOKEN_FILE (not printed)"
echo "  probe: /probe (use scripts/device_probe.sh)"
exec "$PYTHON" -m echobridge
