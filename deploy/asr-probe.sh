#!/usr/bin/env bash
# Probe the vLLM ASR worker with a short silent WAV.
#
# This is intentionally a real transcription request rather than a bare
# /health check: it exercises the CUDA graph and the HTTP path that the first
# user utterance uses.  Two consecutive failures restart the worker.  The
# probe never logs audio, base64 or tokens; only status/latency/count.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${ECHOBRIDGE_PROBE_MODE:-user}"
ASR_URL="${ECHOBRIDGE_ASR_URL:-http://127.0.0.1:18801}"
ASR_URL="${ASR_URL%/}"
MODEL="${ECHOBRIDGE_ASR_VLLM_MODEL:-qwen3-asr}"
MAX_TOKENS="${ECHOBRIDGE_ASR_MAX_NEW_TOKENS:-192}"
GATEWAY_URL="${ECHOBRIDGE_GATEWAY_URL:-http://127.0.0.1:${ECHOBRIDGE_PORT:-18787}}"
THRESHOLD="${ECHOBRIDGE_ASR_PROBE_FAILURES:-2}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ "$MODE" == "system" ]]; then
  SYSTEMCTL=(systemctl)
  STATE_DIR="${ECHOBRIDGE_ASR_PROBE_STATE_DIR:-/run/echobridge-asr-probe}"
else
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  SYSTEMCTL=(systemctl --user)
  STATE_DIR="${ECHOBRIDGE_ASR_PROBE_STATE_DIR:-${XDG_RUNTIME_DIR}/echobridge-asr-probe}"
fi

mkdir -p "$STATE_DIR" 2>/dev/null || STATE_DIR="/tmp/echobridge-asr-probe-$(id -u)"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/failures"
WAV="$STATE_DIR/probe.wav"

unit_state="$("${SYSTEMCTL[@]}" is-active echobridge-asr-vllm 2>/dev/null || true)"
if [[ "$unit_state" == "activating" || "$unit_state" == "reloading" ]]; then
  echo "asr probe: unit is ${unit_state}; skip"
  exit 0
fi

# Do not compete with a live user turn for the GPU.  A missing gateway is not
# a reason to skip: the ASR worker still needs to be healthy for the next turn.
healthz="$(curl -fsS --max-time 2 "${GATEWAY_URL%/}/healthz" 2>/dev/null || true)"
if [[ -n "$healthz" && "$healthz" != *'"active":null'* ]]; then
  echo "asr probe: session active; skip"
  exit 0
fi

if [[ ! -s "$WAV" ]]; then
  if ! "$PYTHON_BIN" - "$WAV" <<'PY'
import sys
import wave

with wave.open(sys.argv[1], "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(16000)
    handle.writeframes(b"\x00\x00" * 8000)
PY
  then
    echo "asr probe: could not create probe WAV" >&2
    exit 1
  fi
fi

started="$(date +%s%N)"
status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
  -F "file=@${WAV};type=audio/wav" \
  -F "model=${MODEL}" \
  -F "response_format=json" \
  -F "temperature=0" \
  -F "max_completion_tokens=${MAX_TOKENS}" \
  "${ASR_URL}/v1/audio/transcriptions" 2>/dev/null || true)"
[[ "$status" =~ ^[0-9]{3}$ ]] || status=000
elapsed_ms=$(( ($(date +%s%N) - started) / 1000000 ))

if [[ "$status" == "200" ]]; then
  rm -f "$STATE_FILE"
  echo "asr probe: ok status=200 latency_ms=${elapsed_ms}"
  exit 0
fi

count=0
[[ -f "$STATE_FILE" ]] && count="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"
[[ "$count" =~ ^[0-9]+$ ]] || count=0
count=$((count + 1))
echo "$count" >"$STATE_FILE"
echo "asr probe: failed status=${status} latency_ms=${elapsed_ms} (${count}/${THRESHOLD})" >&2

if ((count >= THRESHOLD)); then
  echo "asr probe: restarting echobridge-asr-vllm" >&2
  "${SYSTEMCTL[@]}" restart echobridge-asr-vllm || true
  rm -f "$STATE_FILE"
fi
exit 0
