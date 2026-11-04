# Windows WSL watchdog

EchoBridge runs its model workers inside WSL2.  When the WSL2 GPU/DXG context
is lost, a TTS process restart normally recovers in about 1.5 minutes (vLLM
has to reload the model).  This watchdog handles the rarer case where the
whole WSL GPU path is wedged and TTS cannot recover by itself, for example
after a Windows NVIDIA driver update while WSL is running.

Install it from a normal PowerShell window (no admin required):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\install-wsl-watchdog.ps1
```

It registers the scheduled task **EchoBridge WSL Watchdog** and starts it
immediately.  Every 2 minutes it reads the gateway `/readyz` and `/healthz`:

- healthy TTS: no action;
- TTS unavailable for 10 minutes, no active voice session: run
  `wsl.exe --shutdown`, stop any legacy `echobridge-asr.service`/orphan ASR
  process that could hold port 18801, start the TTS/ASR/gateway services, and
  wait up to 5 minutes for recovery;
- after a reset: 30-minute cooldown, so a failing host does not reset WSL in a
  loop.

The 10-minute threshold covers a cold start.  The WSL disk lives on an HDD
(`D:\WSL\Ubuntu-26.04\ext4.vhdx`), so after a Windows reboot or WSL reset the
venvs and models are read from disk: 7.5 minutes from logon to a ready
gateway on 2026-11-04, and 7 minutes from the watchdog's own reset to a ready
gateway.  A 6-minute threshold would have reset WSL in the middle of that.

The script logs only health state and actions (no tokens, audio, or base64) to:

```text
E:\Projects\EchoBridge\var\wsl-watchdog.log
```

Uninstall:

```powershell
Unregister-ScheduledTask -TaskName "EchoBridge WSL Watchdog" -Confirm:$false
```

The task runs in the interactive user session because WSL's `systemd --user`
services are per-user.  The WSL user has linger enabled, so the enabled
EchoBridge units start again after WSL boots; the watchdog also starts them
explicitly as a fallback.
