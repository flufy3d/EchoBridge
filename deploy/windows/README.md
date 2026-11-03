# Windows WSL watchdog

EchoBridge runs its model workers inside WSL2.  When the WSL2 GPU/DXG context
is lost, a TTS process restart normally recovers in about 3 minutes (vLLM has
to reload the model).  This watchdog handles the rarer case where the whole
WSL GPU path is wedged and TTS cannot recover by itself.

Install it from a normal PowerShell window (no admin required):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\install-wsl-watchdog.ps1
```

It registers the scheduled task **EchoBridge WSL Watchdog** and starts it
immediately.  Every 2 minutes it reads the gateway `/readyz` and `/healthz`:

- healthy TTS: no action;
- TTS unavailable for 6 minutes, no active voice session: run
  `wsl.exe --shutdown`, start the TTS/ASR/gateway services, and wait up to
  5 minutes for recovery;
- after a reset: 30-minute cooldown, so a failing host does not reset WSL in a
  loop.

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
