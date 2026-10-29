#!/usr/bin/env bash
# Print the deployment status an operator needs in one screen.
set -uo pipefail

MODE="${ECHOBRIDGE_STATUS_MODE:-user}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${ECHOBRIDGE_PORT:-18787}"
URL="http://127.0.0.1:${PORT}"

while (($#)); do
  case "$1" in
    --user) MODE=user ;;
    --system) MODE=system ;;
    -h|--help)
      echo "Usage: deploy/status.sh [--user|--system]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$MODE" == "system" ]]; then
  SYSTEMCTL=(systemctl)
  JOURNAL=(journalctl)
else
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  SYSTEMCTL=(systemctl --user)
  JOURNAL=(journalctl --user)
fi

echo "== units =="
"${SYSTEMCTL[@]}" --no-pager --lines=0 status \
  echobridge-gateway.service echobridge-asr.service echobridge-asr-vllm.service \
  echobridge-tts.service echobridge-watchdog.timer 2>&1 | sed -n '1,100p'

echo
echo "== health =="
curl -fsS --max-time 3 "${URL}/healthz" 2>/dev/null || echo "gateway: unreachable"
echo
curl -fsS --max-time 3 "${URL}/readyz" 2>/dev/null || echo "readiness: not ready"
echo
echo "== metrics =="
curl -fsS --max-time 3 "${URL}/metrics" 2>/dev/null || true

echo
echo "== voice models =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

if command -v tailscale >/dev/null 2>&1; then
  echo
  echo "== tailscale serve =="
  tailscale serve status 2>&1 | sed -n '1,20p'
fi

echo
echo "state: ${ROOT}/var (local, ignored)"
echo "logs:  ${JOURNAL[*]} -u echobridge-gateway -n 50"
