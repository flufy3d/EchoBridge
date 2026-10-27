# Register a per-user logon task that keeps EchoBridge running in WSL2.
# No administrator rights are required.  Run from a normal PowerShell prompt:
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install-startup-task.ps1
#
# Linger must be enabled once so the user systemd manager survives after the
# last terminal closes:
#
#   wsl -d Ubuntu-26.04 -u flufy3d -- sudo loginctl enable-linger flufy3d
param(
  [string]$Distro = "Ubuntu-26.04",
  [string]$LinuxUser = "flufy3d",
  [string]$TaskName = "EchoBridge WSL"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$starter = Join-Path $root "deploy\windows\start-echobridge.ps1"
if (-not (Test-Path $starter)) {
  throw "starter script missing: $starter"
}

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$starter`""

$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -MultipleInstances IgnoreNew `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Start EchoBridge gateway, ASR and TTS in WSL2" `
  -Force | Out-Null

Write-Host "registered scheduled task: $TaskName"
Write-Host "test it now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "check linger: wsl -d $Distro -u $LinuxUser -- loginctl show-user $LinuxUser -p Linger"
