# EchoBridge WSL GPU self-heal watchdog.
#
# Runs on Windows (not inside WSL) so it can reset a wedged WSL instance with
# `wsl.exe --shutdown`.  It only acts when TTS has been unavailable for several
# minutes and no voice session is active.  Normal TTS restarts take about
# 3 minutes because vLLM reloads the model; the 6-minute threshold avoids
# resetting WSL during those restarts.
#
# Register with deploy/windows/install-wsl-watchdog.ps1.
param(
    [string]$Distro = "Ubuntu-26.04",
    [string]$WslUser = "flufy3d",
    [int]$FailureMinutes = 6,
    [int]$CooldownMinutes = 30,
    [string]$GatewayUrl = "http://127.0.0.1:18787",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$stateDir = Join-Path $root "var"
$logFile = Join-Path $stateDir "wsl-watchdog.log"
$stateFile = Join-Path $stateDir "wsl-watchdog-state.json"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date).ToString("o"), $Message
    Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
}

function Save-State {
    $State | ConvertTo-Json -Compress | Set-Content -LiteralPath $stateFile -Encoding UTF8
}

$State = @{
    first_unhealthy = $null
    last_reset = $null
}
if (Test-Path -LiteralPath $stateFile) {
    try {
        $saved = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        if ($saved.first_unhealthy) { $State.first_unhealthy = [string]$saved.first_unhealthy }
        if ($saved.last_reset) { $State.last_reset = [string]$saved.last_reset }
    } catch {
        Write-Log "state file unreadable; starting fresh"
    }
}

$now = Get-Date
$ready = $null
$health = $null

try {
    $ready = Invoke-RestMethod -Uri "$GatewayUrl/readyz" -TimeoutSec 5
} catch {
    Write-Verbose "readyz failed: $($_.Exception.Message)"
}
try {
    $health = Invoke-RestMethod -Uri "$GatewayUrl/healthz" -TimeoutSec 5
} catch {
    Write-Verbose "healthz failed: $($_.Exception.Message)"
}

$gatewayOk = [bool]($ready -and $ready.ok)
$ttsOk = $false
if ($ready -and $ready.workers -and $ready.workers.tts) {
    $ttsOk = [bool]$ready.workers.tts.ok
}
$active = $null
if ($health) { $active = $health.active }
$activeSession = $null -ne $active -and "$active" -ne ""

if ($gatewayOk -and $ttsOk) {
    if ($State.first_unhealthy) {
        Write-Log "TTS healthy again; clearing failure timer"
        $State.first_unhealthy = $null
        Save-State
    }
    exit 0
}

if (-not $State.first_unhealthy) {
    $State.first_unhealthy = $now.ToString("o")
    Save-State
    Write-Log ("TTS unhealthy: gatewayOk={0} ttsOk={1} active={2}; starting {3}-minute timer" -f $gatewayOk, $ttsOk, $active, $FailureMinutes)
}

$first = [datetime]::Parse($State.first_unhealthy, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
$downFor = ($now - $first).TotalMinutes
$cooldownLeft = 0.0
if ($State.last_reset) {
    $lastReset = [datetime]::Parse($State.last_reset, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
    $cooldownLeft = $CooldownMinutes - ($now - $lastReset).TotalMinutes
}

Write-Log ("unhealthy for {0:N1} min (threshold={1}, gatewayOk={2}, ttsOk={3}, active={4}, cooldownLeft={5:N1})" -f $downFor, $FailureMinutes, $gatewayOk, $ttsOk, $active, $cooldownLeft)

if ($activeSession) {
    Write-Log "active voice session; deferring WSL reset"
    exit 0
}
if ($downFor -lt $FailureMinutes) {
    exit 0
}
if ($cooldownLeft -gt 0) {
    Write-Log "reset cooldown active; not resetting WSL"
    exit 0
}

Write-Log "TTS has been unavailable past threshold; resetting WSL ($Distro)"
if ($DryRun) {
    Write-Log "dry-run: would run wsl.exe --shutdown and start the services"
    exit 0
}

& wsl.exe --shutdown
Start-Sleep -Seconds 5

# A stale interactive session can start the legacy echobridge-asr.service on
# 18801; stop it before the vLLM ASR unit tries to bind the same port.
$startCommand = "systemctl --user stop echobridge-asr.service >/dev/null 2>&1 || true; pkill -f '[e]chobridge.asr_worker' >/dev/null 2>&1 || true; systemctl --user start echobridge-tts.service echobridge-asr-vllm.service echobridge-gateway.service"
$started = $false
for ($attempt = 1; $attempt -le 15; $attempt++) {
    $output = & wsl.exe -d $Distro --user $WslUser -e /bin/bash -lc $startCommand 2>&1
    if ($LASTEXITCODE -eq 0) {
        $started = $true
        break
    }
    Write-Log ("service start attempt {0} failed: {1}" -f $attempt, ($output -join " "))
    Start-Sleep -Seconds 5
}
if (-not $started) {
    Write-Log "could not start WSL services after reset"
}

$State.last_reset = $now.ToString("o")
$State.first_unhealthy = $null
Save-State

for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-RestMethod -Uri "$GatewayUrl/readyz" -TimeoutSec 3
        if ($r.ok -and $r.workers.tts.ok) {
            Write-Log "WSL reset recovered TTS"
            exit 0
        }
    } catch {
        # Still booting; keep waiting.
    }
}
Write-Log "WSL reset did not recover TTS within 5 minutes"
