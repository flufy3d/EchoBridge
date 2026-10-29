#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_OMNI="${ECHOBRIDGE_VLLM_OMNI_BIN:-$HOME/.venvs/echobridge-tts-vllm/bin/vllm-omni}"
SRC="${ECHOBRIDGE_VLLM_OMNI_SRC:-$HOME/.cache/echobridge/src/vllm-omni}"
MODEL="${ECHOBRIDGE_TTS_MODEL:-$HOME/.cache/echobridge/models/Qwen3-TTS-12Hz-0.6B-Base}"
DEPLOY="${ECHOBRIDGE_TTS_DEPLOY_CONFIG:-$ROOT/deploy/qwen3_tts_echobridge.yaml}"
HOST="${ECHOBRIDGE_TTS_HOST:-127.0.0.1}"
PORT="${ECHOBRIDGE_TTS_PORT:-18802}"
GPU_UTIL="${ECHOBRIDGE_TTS_GPU_UTIL:-0.30}"

if [[ ! -x "$VLLM_OMNI" ]]; then
  echo "vLLM-Omni not found: $VLLM_OMNI" >&2
  exit 1
fi
if [[ ! -d "$SRC" ]]; then
  echo "vLLM-Omni source not found: $SRC" >&2
  exit 1
fi
if [[ ! -f "$DEPLOY" ]]; then
  echo "deploy config not found: $DEPLOY" >&2
  exit 1
fi

echo "EchoBridge vLLM-Omni TTS server"
echo "  model:  ${MODEL}"
echo "  deploy: ${DEPLOY}"
echo "  bind:   ${HOST}:${PORT}"
echo "  gpu:    ${GPU_UTIL}"

cd "$SRC"
exec "$VLLM_OMNI" serve "$MODEL" \
  --deploy-config "$DEPLOY" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --omni
