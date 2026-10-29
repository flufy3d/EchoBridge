#!/usr/bin/env bash
# Install EchoBridge as a systemd service.
#
#   WSL2 (no sudo):   bash deploy/install.sh --user
#   native Ubuntu:    sudo bash deploy/install.sh --system
#
# The script never prints tokens or API keys.  It reuses var/dev-token when
# present so the X10A's saved EchoBridge token does not change during migration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=""
START=1
STOP_LEGACY=0
ENV_SRC=""
PORT="${ECHOBRIDGE_PORT:-18787}"
PYTHON="${ECHOBRIDGE_PYTHON:-$HOME/.venvs/echobridge/bin/python}"
PYTHON_EXPLICIT=0
if [[ -n "${ECHOBRIDGE_PYTHON:-}" ]]; then PYTHON_EXPLICIT=1; fi

usage() {
  cat <<'EOF'
Usage: deploy/install.sh [--user|--system] [options]

  --user              install per-user units (WSL2, no sudo)
  --system            install system units (native Ubuntu, requires root)
  --no-start          install and enable units but do not start them
  --stop-legacy       stop processes currently listening on 18787/18801/18802
  --env-file PATH     copy an existing env file instead of creating one
  --python PATH       gateway venv python (default ~/.venvs/echobridge/bin/python)
  --port PORT         gateway port (default 18787)

Defaults to --user inside WSL and --system elsewhere.  Run --system with sudo.
EOF
}

while (($#)); do
  case "$1" in
    --user) MODE=user ;;
    --system) MODE=system ;;
    --no-start) START=0 ;;
    --stop-legacy) STOP_LEGACY=1 ;;
    --env-file) ENV_SRC="${2:?--env-file needs a path}"; shift ;;
    --python) PYTHON="${2:?--python needs a path}"; PYTHON_EXPLICIT=1; shift ;;
    --port) PORT="${2:?--port needs a number}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -z "$MODE" ]]; then
  if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then MODE=user; else MODE=system; fi
fi
if [[ "$MODE" == "system" && "$(id -u)" -ne 0 ]]; then
  echo "system install requires root; run: sudo bash deploy/install.sh --system" >&2
  exit 1
fi

if [[ "$MODE" == "system" ]]; then
  SERVICE_USER="${SUDO_USER:-root}"
  SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
  if ((PYTHON_EXPLICIT == 0)); then
    PYTHON="$SERVICE_HOME/.venvs/echobridge/bin/python"
  fi
  UNIT_DIR=/etc/systemd/system
  ENV_DIR=/etc/echobridge
  WANTED_BY=multi-user.target
  MODE_NAME=system
  SYSTEMCTL=(systemctl)
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
else
  SERVICE_USER="$(id -un)"
  SERVICE_HOME="$HOME"
  UNIT_DIR="$HOME/.config/systemd/user"
  ENV_DIR="$HOME/.config/echobridge"
  WANTED_BY=default.target
  MODE_NAME=user
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  SYSTEMCTL=(systemctl --user)
fi
ENV_FILE="$ENV_DIR/echobridge.env"

if [[ ! -x "$PYTHON" ]]; then
  echo "gateway venv python not found: $PYTHON" >&2
  echo "set --python or ECHOBRIDGE_PYTHON, or create the venv" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR" "$ENV_DIR"
chmod 700 "$ENV_DIR"

render() {
  local src="$1" dst="$2"
  sed \
    -e "s|@ROOT@|$ROOT|g" \
    -e "s|@HOME@|$SERVICE_HOME|g" \
    -e "s|@USER@|$SERVICE_USER|g" \
    -e "s|@ENV_FILE@|$ENV_FILE|g" \
    -e "s|@PYTHON@|$PYTHON|g" \
    -e "s|@WANTED_BY@|$WANTED_BY|g" \
    -e "s|@WORKER_UNITS@|$WORKER_UNITS|g" \
    -e "s|@MODE@|$MODE_NAME|g" \
    "$src" >"$dst"
  if [[ "$MODE" == "user" ]]; then
    sed -i '/^User=/d' "$dst"
  fi
}

if [[ -n "$ENV_SRC" ]]; then
  install -m 600 "$ENV_SRC" "$ENV_FILE"
fi

if [[ ! -s "$ENV_FILE" ]]; then
  install -m 600 "$ROOT/config/echobridge.example.env" "$ENV_FILE"
  TOKEN=""
  if [[ -s "$ROOT/var/dev-token" ]]; then
    TOKEN="$(tr -d '[:space:]' <"$ROOT/var/dev-token")"
  else
    TOKEN="$("$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))')"
  fi
  sed -i \
    -e "s|^ECHOBRIDGE_TOKEN=.*|ECHOBRIDGE_TOKEN=$TOKEN|" \
    -e "s|^ECHOBRIDGE_BIND=.*|ECHOBRIDGE_BIND=127.0.0.1|" \
    -e "s|^ECHOBRIDGE_PORT=.*|ECHOBRIDGE_PORT=$PORT|" \
    -e "s|^ECHOBRIDGE_DEV=.*|ECHOBRIDGE_DEV=0|" \
    -e "s|^ECHOBRIDGE_ENABLE_PROBE=.*|ECHOBRIDGE_ENABLE_PROBE=0|" \
    "$ENV_FILE"
  # Import the local development LLM key without echoing it.
  if [[ -f "$ROOT/var/llm.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ROOT/var/llm.env"
    set +a
    if [[ -n "${ECHOBRIDGE_LLM_API_KEY:-}" ]]; then
      sed -i "s|^ECHOBRIDGE_LLM_API_KEY=.*|ECHOBRIDGE_LLM_API_KEY=${ECHOBRIDGE_LLM_API_KEY}|" "$ENV_FILE"
    fi
  fi
  if ! grep -q '^ECHOBRIDGE_TTS_GPU_UTIL=' "$ENV_FILE"; then
    echo 'ECHOBRIDGE_TTS_GPU_UTIL=0.40' >>"$ENV_FILE"
  fi
  chmod 600 "$ENV_FILE"
fi

if [[ "$MODE" == "system" ]]; then
  chown "$SERVICE_USER":"$SERVICE_USER" "$ENV_FILE"
fi

# In-process backends live inside the gateway; do not start (or even order
# after) their legacy worker units.  Units stay rendered for explicit rollback
# to ECHOBRIDGE_ASR_BACKEND=qwen3 / ECHOBRIDGE_TTS_BACKEND=vllm.
ASR_BACKEND="$(grep -E '^ECHOBRIDGE_ASR_BACKEND=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
ASR_BACKEND="${ASR_BACKEND:-inprocess}"
TTS_BACKEND="$(grep -E '^ECHOBRIDGE_TTS_BACKEND=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
TTS_BACKEND="${TTS_BACKEND:-vllm}"
WORKER_UNITS=""
ASR_UNIT_ENABLE=0
ASR_VLLM_UNIT_ENABLE=0
TTS_UNIT_ENABLE=0
if [[ "$ASR_BACKEND" == "qwen3" ]]; then
  WORKER_UNITS="$WORKER_UNITS echobridge-asr.service"
  ASR_UNIT_ENABLE=1
elif [[ "$ASR_BACKEND" == "vllm" ]]; then
  WORKER_UNITS="$WORKER_UNITS echobridge-asr-vllm.service"
  ASR_VLLM_UNIT_ENABLE=1
fi
if [[ "$TTS_BACKEND" == "vllm" || "$TTS_BACKEND" == "qwen3" ]]; then
  WORKER_UNITS="$WORKER_UNITS echobridge-tts.service"
  TTS_UNIT_ENABLE=1
fi
WORKER_UNITS="${WORKER_UNITS# }"

render "$ROOT/deploy/systemd/echobridge-gateway.service.in" "$UNIT_DIR/echobridge-gateway.service"
render "$ROOT/deploy/systemd/echobridge-asr.service.in" "$UNIT_DIR/echobridge-asr.service"
render "$ROOT/deploy/systemd/echobridge-asr-vllm.service.in" "$UNIT_DIR/echobridge-asr-vllm.service"
render "$ROOT/deploy/systemd/echobridge-tts.service.in" "$UNIT_DIR/echobridge-tts.service"
render "$ROOT/deploy/systemd/echobridge-watchdog.service.in" "$UNIT_DIR/echobridge-watchdog.service"
install -m 644 "$ROOT/deploy/systemd/echobridge-watchdog.timer" "$UNIT_DIR/echobridge-watchdog.timer"

# The watchdog needs root for a system install; user units need no User= line.
if [[ "$MODE" == "system" ]]; then
  sed -i 's|^User=.*|User=root|' "$UNIT_DIR/echobridge-watchdog.service"
fi

if ((STOP_LEGACY)); then
  if "${SYSTEMCTL[@]}" is-active --quiet echobridge-gateway.service 2>/dev/null; then
    echo "systemd gateway already active; skipping --stop-legacy" >&2
  else
    stop_port() {
      local port="$1" pid=""
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        pid="$(ss -ltnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
        [[ -z "$pid" ]] && return 0
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
      done
      pid="$(ss -ltnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
      [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
      return 0
    }
    for port in 18787 18801 18802; do
      stop_port "$port"
    done
  fi
fi

"${SYSTEMCTL[@]}" daemon-reload
ENABLE_UNITS=(echobridge-gateway.service echobridge-watchdog.timer)
START_UNITS=()
if ((ASR_UNIT_ENABLE)); then
  ENABLE_UNITS=(echobridge-asr.service "${ENABLE_UNITS[@]}")
  START_UNITS+=(echobridge-asr.service)
else
  "${SYSTEMCTL[@]}" disable --now echobridge-asr.service >/dev/null 2>&1 || true
fi
if ((ASR_VLLM_UNIT_ENABLE)); then
  ENABLE_UNITS=(echobridge-asr-vllm.service "${ENABLE_UNITS[@]}")
  START_UNITS+=(echobridge-asr-vllm.service)
else
  "${SYSTEMCTL[@]}" disable --now echobridge-asr-vllm.service >/dev/null 2>&1 || true
fi
if ((TTS_UNIT_ENABLE)); then
  ENABLE_UNITS=(echobridge-tts.service "${ENABLE_UNITS[@]}")
  # systemctl queues one job per unit in argument order, and the vLLM ASR
  # unit's After=echobridge-tts.service only delays it when the TTS job is
  # already queued.  Keep TTS first so the ASR model never loads while the
  # TTS worker is still profiling memory.
  START_UNITS=(echobridge-tts.service "${START_UNITS[@]}")
else
  "${SYSTEMCTL[@]}" disable --now echobridge-tts.service >/dev/null 2>&1 || true
fi
"${SYSTEMCTL[@]}" enable "${ENABLE_UNITS[@]}" >/dev/null

if ((START)); then
  if ((${#START_UNITS[@]})); then
    echo "starting workers..."
    "${SYSTEMCTL[@]}" restart "${START_UNITS[@]}"
  fi
  echo "starting gateway (waits up to 300 s for model workers)..."
  "${SYSTEMCTL[@]}" restart echobridge-gateway.service
  "${SYSTEMCTL[@]}" restart echobridge-watchdog.timer
fi

echo
echo "EchoBridge installed ($MODE_NAME mode)"
echo "  unit dir: $UNIT_DIR"
echo "  env:      $ENV_FILE (mode 600, not printed)"
echo "  gateway:  http://127.0.0.1:$PORT"
echo "  status:   bash deploy/status.sh ${MODE:+--$MODE}"
if ((START)); then
  "${SYSTEMCTL[@]}" --no-pager --lines=0 status echobridge-gateway.service 2>&1 | sed -n '1,12p' || true
fi
