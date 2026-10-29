# EchoBridge

**An open-source, self-hosted realtime voice AI gateway.**

EchoBridge connects streaming ASR, LLM, and TTS services through a unified
Realtime API, so an existing Doubao-style client can talk to a locally hosted
voice stack with minimal changes.

> 🚧 **Status: Phase 8.** LLM thinking is off by default, the LLM connection is
> kept warm, TTS uses a 2/4/8/12/25-frame ramp (no post-first-packet hole), and
> Qwen3-ASR can run as a second vLLM process with a fixed 256 MiB KV cache.
> On an RTX 3070: ASR after endpoint **908 → 114 ms**, formal first audio
> **1185 → 768 ms**, TTS first-packet gap **500–880 → 0–16 ms**, whole-card VRAM
> **6695 MiB** with TTS+ASR+gateway. vLLM-ASR is opt-in; in-process ASR remains
> the fallback. Details: [`docs/PHASE8.md`](docs/PHASE8.md).

---

## Why EchoBridge?

Commercial realtime voice APIs are convenient but expensive, closed, and hard
to customize. EchoBridge aims to give you the same "just connect a WebSocket and
talk" experience while letting you choose — and self-host — every component in
the pipeline.

- **Bring your own models** — swap ASR, LLM, and TTS independently.
- **Run on your own hardware** — one consumer GPU is enough.
- **Keep your data** — audio and transcripts never leave your infrastructure.
- **Stay compatible** — Doubao Realtime WebSocket events, `/healthz`,
  `/readyz`, `/metrics`, and `/voices` remain stable.

## Key features

- **Doubao Realtime API compatibility** — existing clients can point at
  EchoBridge with minimal changes.
- **In-process Qwen3-ASR** — `ECHOBRIDGE_ASR_BACKEND=inprocess` loads
  Qwen3-ASR-0.6B directly in the gateway: no localhost `/v1/transcribe`, no
  Base64 context header, no second CUDA context. The legacy 18801 worker remains
  available as `qwen3`.
- **GPU-aware scheduling** — a process-wide priority turnstile gives ASR final
  requests precedence over speculative ASR and TTS; TTS releases the GPU between
  talker/codec steps so ASR can preempt.
- **Streaming Qwen3-TTS** — vLLM-Omni is the default real-time backend
  (first packet p50 300–357 ms, RTF <0.5 on the reference 3070). An experimental
  `inprocess` backend with streaming Code2Wav and GPU-resident clone prompts is
  included for stronger GPUs or future kernel work.
- **Voice cloning and custom characters** — a voice registry keeps reference
  audio, text, and clone prompts outside the repository.
- **Pluggable LLM providers** — DeepSeek, GLM, and any OpenAI-compatible API.
- **Production ops** — systemd units, watchdog, Tailscale Serve guidance,
  readiness that reports in-process model state and CUDA allocator memory.
- **No-playback benchmarks** — all test tools process PCM bytes and timings only;
  they never play audio on the host or device.

## Architecture

```text
Client ── WSS ──▶ Tailscale Serve (optional, tailnet only)
                     │ http://127.0.0.1:18787
                     ▼
              EchoBridge Gateway
                ├── Silero VAD (CPU ONNX)
                ├── Qwen3-ASR (in-process by default, CUDA)
                ├── LLM (DeepSeek / GLM / OpenAI-compatible)
                └── Qwen3-TTS (vLLM-Omni :18802 by default;
                               optional in-process experimental backend)
```

Default deployment is two resident processes: the gateway (including ASR and
VAD) and the vLLM-Omni TTS server. Setting
`ECHOBRIDGE_TTS_BACKEND=inprocess` makes it one process; setting
`ECHOBRIDGE_ASR_BACKEND=qwen3` restores the independent ASR worker.

## Performance snapshot

Host: RTX 3070 8 GiB, WSL2 Ubuntu 26.04, real ASR/TTS/DeepSeek turns, no
playback. Phase 8 numbers are a 6-turn smoke run with one tool hop; full
methodology and the Phase 7 30-turn baseline are in
[`docs/PHASE8.md`](docs/PHASE8.md) and [`docs/PHASE7.md`](docs/PHASE7.md).

| Metric | Phase 7 | Phase 8 (vLLM ASR) |
|---|---:|---:|
| Whole-card VRAM (`nvidia-smi`) | 6571 MiB | **6695 MiB** |
| Cold start to ready | ~3 min 15 s | **~2 min 52 s** |
| ASR after endpoint p50 | 908 ms | **114 ms** |
| Formal first audio p50 (after ASR) | 1185 ms | **768 ms** |
| TTS first packet p50 | 300–357 ms | **270–460 ms** |
| TTS first-packet gap | 500–880 ms | **0–16 ms p50 / 62 ms max** |
| TTS RTF p50 | 0.301–0.423 | **0.34–0.53** |
| ASR 4 s while TTS generates | — | **768 ms** (102 ms idle) |

The in-process ASR path still takes about 900 ms after endpoint. It remains the
fallback when the extra ~2.3 GiB for vLLM-ASR is not available. The experimental
all-in-one mode (`ASR_BACKEND=inprocess`, `TTS_BACKEND=inprocess`) uses about
**4517 MiB** and is ready in **~62 s**, but pure-PyTorch TTS RTF is ~4.

## Quick start

### Prerequisites

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- WSL2 with systemd (current production shape) or native Ubuntu
- NVIDIA GPU with CUDA; 8 GiB is enough for the default single-session profile
- Qwen3-ASR / Qwen3-TTS model files under `~/.cache/echobridge/models/`
- A reference voice under `~/.cache/echobridge/voices/mingyue/`

### Gateway environment

```bash
uv venv ~/.venvs/echobridge --python 3.12
uv pip install --python ~/.venvs/echobridge/bin/python -e '.[dev]'

# Optional: in-process ASR/TTS dependencies.
# The gateway still runs with ASR_BACKEND=qwen3 / TTS_BACKEND=vllm without them.
uv pip install --python ~/.venvs/echobridge/bin/python -e '.[inprocess]'
```

If the resolver tries to replace the CUDA torch build, install the in-process
packages with `--no-deps` first and let `uv pip check` guide the rest. The
current WSL environment uses `transformers==4.57.6`, `qwen-asr==0.0.6`, and
`qwen-tts==0.1.1`.

### Run

```bash
# 1. Download/verify models once
bash scripts/download_asr_model.sh

# 2. Start the real-time TTS backend (default vLLM-Omni profile)
ECHOBRIDGE_TTS_GPU_UTIL=0.30 bash scripts/tts_vllm.sh

# 3. Start the gateway (in-process ASR + VAD + LLM)
bash scripts/dev.sh
```

Or install the systemd deployment (recommended for 24/7 use):

```bash
bash deploy/install.sh --user --stop-legacy
bash deploy/status.sh --user
```

`deploy/install.sh` reads `ECHOBRIDGE_ASR_BACKEND` and
`ECHOBRIDGE_TTS_BACKEND` from the installed env file and only starts the worker
units that are actually selected. The `qwen3` ASR worker remains rendered for
rollback.

### Health

```bash
curl http://127.0.0.1:18787/healthz
curl http://127.0.0.1:18787/readyz
curl http://127.0.0.1:18787/metrics
curl -H "Authorization: Bearer $ECHOBRIDGE_TOKEN" \
  http://127.0.0.1:18787/voices
```

`/readyz` reports in-process ASR/TTS `loaded`, load time, CUDA allocator memory,
and the current GPU scheduler owner/waiters. `/metrics` adds
`echobridge_gpu_memory_allocated_bytes` and
`echobridge_gpu_memory_reserved_bytes` when an in-process model is loaded.

## Configuration

All runtime configuration is environment-only. See
[`config/echobridge.example.env`](config/echobridge.example.env) for the full
template.

| Variable | Deployment default | Notes |
|---|---|---|
| `ECHOBRIDGE_ASR_BACKEND` | `inprocess` | `inprocess` \| `qwen3` \| `mock` |
| `ECHOBRIDGE_ASR_MODEL` | `~/.cache/echobridge/models/Qwen3-ASR-0.6B` | In-process model path |
| `ECHOBRIDGE_ASR_URL` | `http://127.0.0.1:18801` | Used only by `qwen3` |
| `ECHOBRIDGE_TTS_BACKEND` | `vllm` | `vllm` \| `qwen3` \| `inprocess` \| `mock` |
| `ECHOBRIDGE_TTS_GPU_UTIL` | `0.30` | Single-session Stage 0 profile |
| `ECHOBRIDGE_VAD_BACKEND` | `silero` | `silero` \| `energy` |
| `ECHOBRIDGE_VAD_PRE_ROLL_MS` | `800` | Set `300` for the latency-first profile |
| `ECHOBRIDGE_VAD_SPECULATIVE_LONG_LEAD_MS` | `350` | Long-utterance ASR head start |
| `ECHOBRIDGE_LLM_BACKEND` | `deepseek` | `deepseek` \| `mock` |
| `ECHOBRIDGE_LLM_API_KEY` | — | Keep in the env file, never in git |
| `ECHOBRIDGE_BACKCHANNEL_ENABLED` | `1` | Cached filler before the formal reply |

The full WebSocket protocol and event compatibility guarantees are documented in
[`docs/CONTEXT.md`](docs/CONTEXT.md) and [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md).

## Testing and benchmarks

```bash
# Protocol/unit regression; no GPU model is loaded.
bash scripts/test.sh
# 94 passed

# Whole-gateway 30-turn benchmark (real ASR/TTS/LLM, no playback).
~/.venvs/echobridge/bin/python scripts/bench_gateway.py \
  --turns 30 --out var/bench/phase7_30.json

# Direct in-process ASR model latency.
~/.venvs/echobridge/bin/python scripts/bench_asr_inprocess.py \
  --repeat 3 --out var/bench/asr_inprocess.json

# vLLM-Omni TTS first packet / RTF.
~/.venvs/echobridge/bin/python scripts/bench_tts_vllm.py \
  --repeat 3 --out var/bench/tts_vllm.json

# Experimental in-process TTS.
~/.venvs/echobridge/bin/python scripts/bench_tts_inprocess.py \
  --backend inprocess --repeat 3 --out var/bench/tts_inprocess.json
```

None of these scripts plays audio. The `var/` directory (benchmark output,
tokens, env files) is intentionally git-ignored.

## Documentation

- [`docs/PHASE7.md`](docs/PHASE7.md) — single-process/CUDA-context refactor,
  memory and latency measurements, in-process TTS trade-off.
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — Phase 6 baseline and acceptance
  measurements.
- [`docs/DEPLOY.md`](docs/DEPLOY.md) — systemd, Windows WSL autostart,
  Tailscale Serve, rollback.
- [`docs/VOICES.md`](docs/VOICES.md) — voice registry and clone-prompt workflow.
- [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — protocol/device acceptance notes.
- [`docs/PHASE2.md`](docs/PHASE2.md), [`docs/PHASE3.md`](docs/PHASE3.md) —
  VAD/ASR and TTS implementation history.

## Roadmap

- [x] Doubao Realtime WebSocket protocol compatibility
- [x] Silero VAD, endpointing, pre-roll, speculative endpoint
- [x] Streaming Qwen3-ASR adapter and ASR worker
- [x] Streaming Qwen3-TTS with voice cloning (vLLM-Omni default)
- [x] LLM provider abstraction (DeepSeek / OpenAI-compatible)
- [x] Barge-in, cancellation, backchannel
- [x] Health/readiness/metrics, systemd watchdog, WSL autostart
- [x] In-process ASR + process-wide GPU scheduling
- [x] Single-session vLLM memory slimming
- [ ] ASR after-endpoint 350–500 ms target (model-throughput bound)
- [ ] Docker image and one-command self-hosted deployment
- [ ] Client SDKs and example apps

## 中文简介

EchoBridge 是一个轻量级、开源、自托管的实时 AI 语音网关，旨在通过兼容商业
Realtime API 的接口，将本地 ASR、TTS 与强大语言模型自由组合，实现低成本、
可定制、低延迟的实时语音交互。

Phase 7 之后，ASR 默认直接运行在网关进程内，与 vLLM TTS 共用一张 GPU；整卡
显存约 6.57 GiB（RTX 3070 8 GiB），正式首音 p95 约 1.56 s。完整的显存、延迟、
回退和实验性全进程 TTS 数据见 [`docs/PHASE7.md`](docs/PHASE7.md)。

适用于 AI 虚拟角色、智能音箱、Android 设备和自定义语音助手。

## Contributing

The project is at an early stage. Issues and ideas are welcome — feel free to
open one to discuss the design before the first code lands.

## License

To be announced (likely MIT or Apache-2.0). Stay tuned.
