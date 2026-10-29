#!/usr/bin/env bash
# Wait until configured local model workers answer before starting the gateway.
# systemd runs this as ExecStartPre so the gateway's warmup has a real target.
set -euo pipefail

ASR_URL="${ECHOBRIDGE_ASR_URL:-http://127.0.0.1:18801}"
TTS_URL="${ECHOBRIDGE_TTS_URL:-http://127.0.0.1:18802}"
ASR_BACKEND="${ECHOBRIDGE_ASR_BACKEND:-mock}"
TTS_BACKEND="${ECHOBRIDGE_TTS_BACKEND:-mock}"
TIMEOUT="${ECHOBRIDGE_STARTUP_TIMEOUT_S:-300}"
INTERVAL="${ECHOBRIDGE_STARTUP_INTERVAL_S:-2}"

deadline=$((SECONDS + TIMEOUT))

check() {
  local name="$1" url="$2"
  if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
    echo "wait-for-workers: $name ready ($url)"
    return 0
  fi
  return 1
}

wait_for() {
  local name="$1" url="$2"
  while ((SECONDS < deadline)); do
    if check "$name" "$url"; then
      return 0
    fi
    sleep "$INTERVAL"
  done
  echo "wait-for-workers: $name not ready after ${TIMEOUT}s ($url)" >&2
  return 1
}

if [[ "$ASR_BACKEND" == "qwen3" ]]; then
  wait_for asr "${ASR_URL%/}/healthz"
elif [[ "$ASR_BACKEND" == "vllm" ]]; then
  wait_for asr "${ASR_URL%/}/health"
fi
if [[ "$TTS_BACKEND" == "vllm" ]]; then
  wait_for tts "${TTS_URL%/}/health"
elif [[ "$TTS_BACKEND" == "qwen3" ]]; then
  wait_for tts "${TTS_URL%/}/healthz"
fi

echo "wait-for-workers: all configured workers are ready"
