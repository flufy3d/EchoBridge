# Keep a WSL session alive and start the EchoBridge user services.
# Called by Task Scheduler at logon.  The final `sleep infinity` is deliberate:
# WSL2 stops the distro when its last session exits, even if a user systemd
# manager has lingering enabled.  The outer loop restarts the anchor if WSL is
# terminated or the session dies.
param(
  [string]$Distro = "Ubuntu-26.04",
  [string]$LinuxUser = "flufy3d"
)

$ErrorActionPreference = "Stop"
$wsl = Join-Path $env:SystemRoot "System32\wsl.exe"
$script = @"
export XDG_RUNTIME_DIR=/run/user/`$(id -u)
# The gateway's Wants= pulls in only the worker units selected by the installed
# env (vLLM TTS and/or the legacy ASR worker).  In-process ASR/TTS need none.
systemctl --user start echobridge-gateway.service
systemctl --user start echobridge-watchdog.timer
exec sleep infinity
"@

while ($true) {
  & $wsl -d $Distro -u $LinuxUser --exec /bin/bash -lc $script
  Start-Sleep -Seconds 15
}
