#!/usr/bin/env bash
# Expose only the loopback gateway to the tailnet over HTTPS (no Funnel).
#
# On WSL2 the Tailscale daemon usually lives on Windows, so run the equivalent
# PowerShell command from docs/DEPLOY.md instead.  This script is for a native
# Ubuntu host / a Linux Tailscale daemon.
set -euo pipefail

TS="${TAILSCALE_BIN:-tailscale}"
PORT="${ECHOBRIDGE_SERVE_PORT:-8443}"
GATEWAY_PORT="${ECHOBRIDGE_PORT:-18787}"

if ! command -v "$TS" >/dev/null 2>&1; then
  echo "tailscale not found; install it or use the Windows Serve instructions" >&2
  exit 1
fi

"$TS" serve --bg "--https=${PORT}" "http://127.0.0.1:${GATEWAY_PORT}"
echo
"$TS" serve status
echo
if "$TS" funnel status 2>/dev/null | grep -q .; then
  echo "WARNING: Funnel is configured. EchoBridge must not be exposed publicly." >&2
  exit 1
fi
echo "ok: tailnet-only Serve configured; Funnel is not enabled"
