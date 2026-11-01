#!/usr/bin/env bash
# Launch the vLLM-Omni Qwen3-TTS server with two EchoBridge safety layers:
#   1. the WSL2 pin-memory patch (idempotent; skipped with
#      ECHOBRIDGE_TTS_PATCH_VLLM_OMNI=0);
#   2. a small supervisor that exits the unit when /health stays 503 after a
#      successful start.  vLLM-Omni's API server can survive a dead stage-0
#      engine and keep answering 500; systemd only restarts on process exit.
#      Set ECHOBRIDGE_TTS_ENGINE_SUPERVISOR=0 to restore the old exec path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_OMNI="${ECHOBRIDGE_VLLM_OMNI_BIN:-$HOME/.venvs/echobridge-tts-vllm/bin/vllm-omni}"
SRC="${ECHOBRIDGE_VLLM_OMNI_SRC:-$HOME/.cache/echobridge/src/vllm-omni}"
MODEL="${ECHOBRIDGE_TTS_MODEL:-$HOME/.cache/echobridge/models/Qwen3-TTS-12Hz-0.6B-Base}"
DEPLOY="${ECHOBRIDGE_TTS_DEPLOY_CONFIG:-$ROOT/deploy/qwen3_tts_echobridge.yaml}"
HOST="${ECHOBRIDGE_TTS_HOST:-127.0.0.1}"
PORT="${ECHOBRIDGE_TTS_PORT:-18802}"
GPU_UTIL="${ECHOBRIDGE_TTS_GPU_UTIL:-0.30}"
PATCH="${ECHOBRIDGE_TTS_PATCH_VLLM_OMNI:-1}"
SUPERVISOR="${ECHOBRIDGE_TTS_ENGINE_SUPERVISOR:-1}"
PROBE_S="${ECHOBRIDGE_TTS_ENGINE_PROBE_S:-5}"
FAILURES="${ECHOBRIDGE_TTS_ENGINE_FAILURES:-2}"

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
if [[ ! "$PROBE_S" =~ ^[0-9]+$ ]] || ((PROBE_S < 1)); then
  echo "ECHOBRIDGE_TTS_ENGINE_PROBE_S must be a positive integer" >&2
  exit 1
fi
if [[ ! "$FAILURES" =~ ^[0-9]+$ ]] || ((FAILURES < 1)); then
  echo "ECHOBRIDGE_TTS_ENGINE_FAILURES must be a positive integer" >&2
  exit 1
fi

if [[ "$PATCH" != "0" ]]; then
  PATCH_PY="${ECHOBRIDGE_TTS_VLLM_PYTHON:-$(dirname "$VLLM_OMNI")/python}"
  if [[ -x "$PATCH_PY" ]]; then
    if ! PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PATCH_PY" -m echobridge.vllm_omni_patch "$SRC"; then
      echo "warning: vLLM-Omni WSL pin_memory patch not applied" >&2
      if [[ "${ECHOBRIDGE_TTS_STRICT_PATCH:-0}" == "1" ]]; then
        exit 1
      fi
    fi
  else
    echo "warning: patch python not found: $PATCH_PY" >&2
  fi
fi

echo "EchoBridge vLLM-Omni TTS server"
echo "  model:  ${MODEL}"
echo "  deploy: ${DEPLOY}"
echo "  bind:   ${HOST}:${PORT}"
echo "  gpu:    ${GPU_UTIL}"
echo "  engine supervisor: ${SUPERVISOR} (probe ${PROBE_S}s x ${FAILURES})"

cd "$SRC"
SERVER=(
  "$VLLM_OMNI" serve "$MODEL"
  --deploy-config "$DEPLOY"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_UTIL"
  --trust-remote-code
  --omni
)

if [[ "$SUPERVISOR" == "0" ]]; then
  exec "${SERVER[@]}"
fi

HEALTH_URL="${ECHOBRIDGE_TTS_URL:-http://${HOST}:${PORT}}"
HEALTH_URL="${HEALTH_URL%/}/health"
child=""

forward_term() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  exit 143
}
trap forward_term TERM INT

"${SERVER[@]}" &
child=$!

# Do not count startup 503s: the model takes ~2 minutes to load and
# worker-ready.sh is already watching that phase.  Only a worker that was
# healthy and then lost its engine is force-restarted.
healthy=0
failures=0
while kill -0 "$child" 2>/dev/null; do
  sleep "$PROBE_S"
  if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    healthy=1
    failures=0
    continue
  fi
  if ((healthy == 0)); then
    continue
  fi
  failures=$((failures + 1))
  echo "tts supervisor: /health failed (${failures}/${FAILURES})" >&2
  if ((failures >= FAILURES)); then
    echo "tts supervisor: engine is dead; stopping the API server so systemd restarts it" >&2
    kill -TERM "$child" 2>/dev/null || true
    sleep 2
    kill -KILL "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    exit 1
  fi
done

wait "$child"
