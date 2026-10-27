param(
  [string]$TaskName = "EchoBridge WSL"
)

$ErrorActionPreference = "Stop"
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "removed scheduled task: $TaskName"
