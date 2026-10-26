#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECHOBRIDGE_ASR_PYTHON:-$HOME/.venvs/echobridge-asr/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ASR venv not found: $PYTHON" >&2
  exit 1
fi

export ECHOBRIDGE_ASR_MODEL="${ECHOBRIDGE_ASR_MODEL:-$HOME/.cache/echobridge/models/Qwen3-ASR-0.6B}"
export ECHOBRIDGE_ASR_HOST="${ECHOBRIDGE_ASR_HOST:-127.0.0.1}"
export ECHOBRIDGE_ASR_PORT="${ECHOBRIDGE_ASR_PORT:-18801}"
export ECHOBRIDGE_ASR_LANGUAGE="${ECHOBRIDGE_ASR_LANGUAGE:-Chinese}"
export ECHOBRIDGE_ASR_MAX_NEW_TOKENS="${ECHOBRIDGE_ASR_MAX_NEW_TOKENS:-192}"
export HF_HOME="${HF_HOME:-$HOME/.cache/echobridge/hf}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "EchoBridge ASR worker"
echo "  model: ${ECHOBRIDGE_ASR_MODEL}"
echo "  bind:  ${ECHOBRIDGE_ASR_HOST}:${ECHOBRIDGE_ASR_PORT}"
exec "$PYTHON" -m echobridge.asr_worker
