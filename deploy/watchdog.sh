#!/usr/bin/env bash
# Lightweight health watchdog for a single-host EchoBridge deployment.
#
# It runs from a systemd timer every 30 s.  A unit that fails three checks in a
# row is restarted; systemd's own Restart=always handles hard crashes.
set -uo pipefail

MODE="${ECHOBRIDGE_WATCHDOG_MODE:-user}"
GATEWAY_URL="${ECHOBRIDGE_GATEWAY_URL:-http://127.0.0.1:${ECHOBRIDGE_PORT:-18787}}"
ASR_URL="${ECHOBRIDGE_ASR_URL:-http://127.0.0.1:18801}"
TTS_URL="${ECHOBRIDGE_TTS_URL:-http://127.0.0.1:18802}"
ASR_BACKEND="${ECHOBRIDGE_ASR_BACKEND:-mock}"
TTS_BACKEND="${ECHOBRIDGE_TTS_BACKEND:-mock}"
THRESHOLD="${ECHOBRIDGE_WATCHDOG_FAILURES:-3}"
GRACE_S="${ECHOBRIDGE_WATCHDOG_GRACE_S:-300}"

if [[ "$MODE" == "system" ]]; then
  SYSTEMCTL=(systemctl)
  STATE_DIR="/run/echobridge-watchdog"
else
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  SYSTEMCTL=(systemctl --user)
  STATE_DIR="${XDG_RUNTIME_DIR}/echobridge-watchdog"
fi

if ! mkdir -p "$STATE_DIR" 2>/dev/null; then
  STATE_DIR="/tmp/echobridge-watchdog-$(id -u)"
  mkdir -p "$STATE_DIR"
fi

log() {
  echo "echobridge-watchdog: $*" >&2
}

unit_state() {
  "${SYSTEMCTL[@]}" is-active "$1" 2>/dev/null || true
}

unit_age_s() {
  local unit="$1" start now
  start="$("${SYSTEMCTL[@]}" show "$unit" -p ActiveEnterTimestampMonotonic --value 2>/dev/null || true)"
  if [[ ! "$start" =~ ^[0-9]+$ ]] || ((start == 0)); then
    echo 999999
    return
  fi
  now="$(awk '{printf "%.0f", $1 * 1000000}' /proc/uptime)"
  echo $(((now - start) / 1000000))
}

count_file() {
  echo "$STATE_DIR/$1.failures"
}

record_failure() {
  local unit="$1" file count
  file="$(count_file "$unit")"
  count=0
  [[ -f "$file" ]] && count="$(cat "$file" 2>/dev/null || echo 0)"
  count=$((count + 1))
  echo "$count" >"$file"
  echo "$count"
}

clear_failure() {
  rm -f "$(count_file "$1")"
}

restart_unit() {
  local unit="$1"
  log "restarting $unit after repeated health failures"
  if ! "${SYSTEMCTL[@]}" restart "$unit"; then
    log "restart failed for $unit"
  fi
  clear_failure "$unit"
}

check_unit() {
  local unit="$1" healthy="$2" label="$3" state age
  state="$(unit_state "$unit")"
  if [[ "$state" == "activating" || "$state" == "reloading" ]]; then
    return 0
  fi
  if [[ "$state" == "active" ]]; then
    age="$(unit_age_s "$unit")"
    if ((age < GRACE_S)); then
      # Type=simple units look active immediately, while vLLM/ASR may still be
      # loading for minutes.  Do not restart a cold model.
      return 0
    fi
  fi
  if [[ "$state" != "active" ]]; then
    log "$label unit is $state"
    healthy=1
  fi
  if ((healthy == 0)); then
    clear_failure "$unit"
    return 0
  fi
  local count
  count="$(record_failure "$unit")"
  log "$label health check failed ($count/$THRESHOLD)"
  if ((count >= THRESHOLD)); then
    restart_unit "$unit"
  fi
  return 0
}

# Gateway liveness.  Readiness is intentionally not used here: a cold model
# worker should not cause the gateway itself to be restarted in a loop.
gateway_ok=0
curl -fsS --max-time 3 "${GATEWAY_URL%/}/healthz" >/dev/null 2>&1 || gateway_ok=1
check_unit echobridge-gateway "$gateway_ok" gateway

if [[ "$ASR_BACKEND" == "qwen3" ]]; then
  asr_ok=0
  body="$(curl -fsS --max-time 3 "${ASR_URL%/}/healthz" 2>/dev/null || true)"
  grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' <<<"$body" || asr_ok=1
  check_unit echobridge-asr "$asr_ok" asr
fi

if [[ "$TTS_BACKEND" == "vllm" ]]; then
  tts_ok=0
  curl -fsS --max-time 3 "${TTS_URL%/}/health" >/dev/null 2>&1 || tts_ok=1
  check_unit echobridge-tts "$tts_ok" tts
elif [[ "$TTS_BACKEND" == "qwen3" ]]; then
  tts_ok=0
  body="$(curl -fsS --max-time 3 "${TTS_URL%/}/healthz" 2>/dev/null || true)"
  grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' <<<"$body" || tts_ok=1
  check_unit echobridge-tts "$tts_ok" tts
fi
