# EchoBridge

**An open-source, self-hosted realtime voice AI gateway.**

A lightweight, model-agnostic voice gateway that connects streaming ASR, LLM, and TTS services through a unified Realtime API.

> 🚧 **Status: Phase 1.** The Doubao-compatible WebSocket gateway, mock ASR/LLM/TTS, protocol tests, and an X10A device probe are implemented. See [`docs/PHASE1.md`](docs/PHASE1.md) for run and test instructions; local ASR/TTS/DeepSeek land in Phases 2–4.

---

## Why EchoBridge?

Commercial realtime voice APIs are convenient but expensive, closed, and hard to customize. EchoBridge aims to give you the same "just connect a WebSocket and talk" experience while letting you choose — and self-host — every component in the pipeline.

- **Bring your own models** — swap ASR, LLM, and TTS independently.
- **Run on your own hardware** — one GPU box is enough; no per-minute API fees.
- **Keep your data** — audio and transcripts never leave your infrastructure.

## Key Features

- **Doubao Realtime API compatibility** — existing clients can point at EchoBridge with minimal changes
- **Streaming speech recognition and synthesis** — low-latency, chunk-by-chunk audio in and out
- **Qwen3-ASR and Qwen3-TTS integration** — strong multilingual performance out of the box
- **Pluggable LLM providers** — DeepSeek, GLM, and any OpenAI-compatible API
- **Voice cloning and custom character voices** — build a consistent persona for every assistant
- **Voice activity detection and interruption handling** — natural turn-taking, barge-in supported
- **WebSocket-based bidirectional audio streaming** — a single connection for the whole session
- **GPU-friendly self-hosted deployment** — designed to fit on a single consumer or datacenter GPU

## 中文简介

EchoBridge 是一个轻量级、开源、自托管的实时 AI 语音网关，旨在通过兼容商业 Realtime API 的接口，将本地 ASR、TTS 与强大语言模型自由组合，实现低成本、可定制、低延迟的实时语音交互。

适用于 AI 虚拟角色、智能音箱、Android 设备和自定义语音助手。

## Architecture (planned)

```
Client  ──WebSocket──▶  EchoBridge Gateway
                          ├── VAD / turn detection
                          ├── ASR  (Qwen3-ASR, ...)
                          ├── LLM  (DeepSeek / GLM / OpenAI-compatible)
                          └── TTS  (Qwen3-TTS, voice cloning)
```

## Roadmap

- [ ] Realtime WebSocket protocol compatible with the Doubao Realtime API
- [ ] Streaming ASR adapter (Qwen3-ASR)
- [ ] Streaming TTS adapter with voice cloning (Qwen3-TTS)
- [ ] LLM provider abstraction (DeepSeek / GLM / OpenAI-compatible)
- [ ] VAD and interruption handling
- [ ] Docker image and one-command self-hosted deployment
- [ ] Client SDKs and example apps

## Contributing

The project is at a very early stage. Issues and ideas are welcome — feel free to open one to discuss the design before the first code lands.

## License

To be announced (likely MIT or Apache-2.0). Stay tuned.
