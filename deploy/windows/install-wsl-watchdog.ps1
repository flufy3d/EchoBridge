# Register the Windows-side EchoBridge WSL self-heal task.
#
# Run from PowerShell (normal user is enough):
#   powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\install-wsl-watchdog.ps1
#
# The task checks the gateway every 2 minutes.  If TTS stays unavailable for
# 6 minutes with no active voice session, it runs `wsl --shutdown`, starts the
# distro/services again, and waits up to 5 minutes for recovery.
param(
    [string]$TaskName = "EchoBridge WSL Watchdog",
    [string]$WatchdogScript = (Join-Path $PSScriptRoot "echobridge-wsl-watchdog.ps1")
)

$ErrorActionPreference = "Stop"
$scriptPath = [System.IO.Path]::GetFullPath($WatchdogScript)
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "watchdog script not found: $scriptPath"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ("-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"{0}`"" -f $scriptPath)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 2) -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId ("{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Reset WSL when EchoBridge TTS stays unavailable (WSL2 CUDA context loss)." -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2
$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host ("registered: {0} ({1}), last result={2}, next run={3}" -f $task.TaskName, $task.State, $info.LastTaskResult, $info.NextRunTime)
