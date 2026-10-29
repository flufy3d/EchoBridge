#!/usr/bin/env bash
# ExecStartPost for the model worker units.
#
# Keeps the unit "activating" until the server answers /health, so After=
# ordering means "after the worker is ready" rather than "after its process
# was forked".  That is what keeps the ASR model load out of the TTS worker's
# memory-profiling window at boot.
#
# For the vLLM ASR server it also sends two warm-up transcriptions: without
# them the first utterance after a restart takes 0.5-0.9 s instead of ~0.1 s,
# and the gateway only warms its backends when the gateway itself starts.
set -euo pipefail

kind="${1:?usage: worker-ready.sh tts|asr}"
TIMEOUT="${ECHOBRIDGE_WORKER_READY_TIMEOUT_S:-420}"

case "$kind" in
  tts) url="${ECHOBRIDGE_TTS_URL:-http://127.0.0.1:18802}" ;;
  asr) url="${ECHOBRIDGE_ASR_URL:-http://127.0.0.1:18801}" ;;
  *)
    echo "worker-ready: unknown worker: $kind" >&2
    exit 2
    ;;
esac
url="${url%/}"

deadline=$((SECONDS + TIMEOUT))
until curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "worker-ready: $kind not healthy after ${TIMEOUT}s ($url)" >&2
    exit 1
  fi
  if [[ -n "${MAINPID:-}" ]] && ! kill -0 "$MAINPID" 2>/dev/null; then
    echo "worker-ready: $kind exited before becoming healthy" >&2
    exit 1
  fi
  sleep 2
done
echo "worker-ready: $kind healthy ($url)"

if [[ "$kind" == "asr" ]]; then
  wav="$(mktemp --suffix=.wav)"
  trap 'rm -f "$wav"' EXIT
  python3 - "$wav" <<'PY'
import sys
import wave

with wave.open(sys.argv[1], "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(16000)
    handle.writeframes(b"\x00\x00" * 16000)
PY
  for _ in 1 2; do
    curl -fsS --max-time 30 -o /dev/null "$url/v1/audio/transcriptions" \
      -F "file=@$wav;type=audio/wav" \
      -F "model=${ECHOBRIDGE_ASR_VLLM_MODEL:-qwen3-asr}" \
      -F "response_format=json" \
      -F "temperature=0" ||
      echo "worker-ready: asr warm-up request failed" >&2
  done
  echo "worker-ready: asr warmed up"
fi
