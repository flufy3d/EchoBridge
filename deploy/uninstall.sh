#!/usr/bin/env bash
# Remove EchoBridge systemd units.  Configuration and models are kept unless
# --purge-env is given, so a rollback does not need a token re-entry.
set -euo pipefail

MODE=""
PURGE_ENV=0
while (($#)); do
  case "$1" in
    --user) MODE=user ;;
    --system) MODE=system ;;
    --purge-env) PURGE_ENV=1 ;;
    -h|--help)
      echo "Usage: deploy/uninstall.sh [--user|--system] [--purge-env]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ -z "$MODE" ]]; then
  if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then MODE=user; else MODE=system; fi
fi
if [[ "$MODE" == "system" && "$(id -u)" -ne 0 ]]; then
  echo "run with sudo: sudo bash deploy/uninstall.sh --system" >&2
  exit 1
fi

if [[ "$MODE" == "system" ]]; then
  UNIT_DIR=/etc/systemd/system
  ENV_DIR=/etc/echobridge
  SYSTEMCTL=(systemctl)
else
  UNIT_DIR="$HOME/.config/systemd/user"
  ENV_DIR="$HOME/.config/echobridge"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  SYSTEMCTL=(systemctl --user)
fi

"${SYSTEMCTL[@]}" disable --now echobridge-watchdog.timer 2>/dev/null || true
"${SYSTEMCTL[@]}" disable --now echobridge-gateway.service 2>/dev/null || true
"${SYSTEMCTL[@]}" disable --now echobridge-asr.service 2>/dev/null || true
"${SYSTEMCTL[@]}" disable --now echobridge-asr-vllm.service 2>/dev/null || true
"${SYSTEMCTL[@]}" disable --now echobridge-tts.service 2>/dev/null || true
rm -f \
  "$UNIT_DIR/echobridge-gateway.service" \
  "$UNIT_DIR/echobridge-asr.service" \
  "$UNIT_DIR/echobridge-asr-vllm.service" \
  "$UNIT_DIR/echobridge-tts.service" \
  "$UNIT_DIR/echobridge-watchdog.service" \
  "$UNIT_DIR/echobridge-watchdog.timer"
"${SYSTEMCTL[@]}" daemon-reload

if ((PURGE_ENV)); then
  rm -rf "$ENV_DIR"
  echo "removed $ENV_DIR"
else
  echo "kept $ENV_DIR (use --purge-env to remove)"
fi

echo "EchoBridge units removed. Device fallback to Doubao is unchanged."
