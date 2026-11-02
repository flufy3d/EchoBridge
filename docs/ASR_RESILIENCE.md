# ASR 故障：探活、重启与“第一句话”错误语义

本文记录 2026-11-02 19:12 的 ASR 崩溃、现场证据，以及 EchoBridge 现在
对 ASR worker 的两层保护。日志只记录时间、状态码、延迟、item id 和字节数，
不包含音频、base64 或 token。

## 1. 故障时间线（WSL 主机时间）

| 时间 | 事件 |
|---|---|
| 17:45:20 | TTS stage0 `cuStreamSynchronize` → `CUDA_ERROR_UNKNOWN`；keepalive 500；17:45:41 重启 |
| 18:25:24 | TTS 同一位置再次崩溃；18:25:40 重启 |
| 19:12:32 | 用户第一句话 → ASR POST 500；`qwen3_omni_moe_thinker.py:467` 的 `chunk_num.sum()` 报 `cudaErrorUnknown`；19:12:33 ASR 进程退出 |
| 19:12:33–49 | 网关记录 item_7/8/9/10 失败：第一次 HTTP 500，后面 ConnectError；App 收到 `asr_error`，显示“没有听清” |
| 19:12:55 | systemd 重启 ASR；19:14:23 ready |
| 19:13:25 | 设备重连；19:13:28 item_11 仍 ConnectError |
| 19:15:28 | TTS 第三次 `cuStreamSynchronize` 崩溃；19:15:48 重启；19:17:29 ready |
| 19:20:55 | 用户测试会话成功：ASR、TTS 均正常 |

TTS 在 17:45、18:25 已经独立崩溃，当时 ASR 已稳定运行 27 小时。**19:15
的 TTS 崩溃不是 19:12 ASR 重启造成的**；它是 TTS 自己的 10 分钟 keepalive
周期上的第三次同型失败。`docs/TTS_RESILIENCE.md` 记录了 TTS 侧的兜底。

## 2. 现场证据

- TTS 的 CUDA 日志第一条失败 API 是 `cuStreamSynchronize` →
  `CUDA_ERROR_UNKNOWN`，`packed_ids_cpu.to(device)` 只是 Python 同步点。
- ASR 崩溃发生在普通 audio-tower forward `chunk_num.sum()`，不是显存分配；
  ASR 启动脚本的 `nvidia-smi` 余量检查通过，没有 `waiting for GPU memory`。
- Windows System 日志 19:05–19:20 没有 nvlddmkm / Display / WHEA / TDR；
  WSL dmesg 的 `dxg` ioctl 失败在 17:47、18:26、19:13、19:15 各有一簇。
- 结论：ASR 重启不是 TTS 崩溃的原因；两者更像是同一个 WSL/DXG CUDA
  context 不稳定环境的受害者。TTS 自身是否还有进程内泄漏，需要
  `CUDA_LAUNCH_BLOCKING=1` 的短窗口和显存采样进一步排除。

### 2.1 下一次 ASR 崩溃的 CUDA 日志

TTS 已有 `CUDA_LOG_FILE=stderr` 的 drop-in；ASR 也加一个，下一次崩溃时
能看到第一条 `[CUDA][E]`（具体 API + 错误码，而不是只有
`cudaErrorUnknown`）。drop-in 不会重启当前 ASR：

```bash
mkdir -p ~/.config/systemd/user/echobridge-asr-vllm.service.d
cat > ~/.config/systemd/user/echobridge-asr-vllm.service.d/cuda-log.conf <<'EOF'
[Service]
Environment=CUDA_LOG_FILE=stderr
EOF
systemctl --user daemon-reload

# 下次崩溃后：
journalctl --user -u echobridge-asr-vllm --since "-10 min" \
  | grep -E '\[CUDA\]|CUDA error|AcceleratorError|EngineDeadError'
```

## 3. ASR worker 探活

`deploy/asr-probe.sh` 由 `echobridge-asr-probe.timer` 每 5 分钟运行一次：

1. 如果 ASR unit 正在 `activating`/`reloading`，跳过；
2. 如果网关 `/healthz` 显示有活跃会话，跳过，避免和用户抢 GPU；
3. 生成 0.5 s、16 kHz、单声道静音 WAV（只在 probe 状态目录，不记录内容）；
4. POST 到 `/v1/audio/transcriptions`，HTTP 200 即健康（空文本也算成功）；
5. 连续 2 次失败后 `systemctl --user restart echobridge-asr-vllm`
   （system 模式用 `systemctl`），并清空失败计数。

日志只包含 `status`、`latency_ms`、失败次数，例如：

```text
asr probe: ok status=200 latency_ms=92
asr probe: failed status=000 latency_ms=1 (1/2)
asr probe: restarting echobridge-asr-vllm
```

安装脚本 `deploy/install.sh` 在 `ECHOBRIDGE_ASR_BACKEND=vllm` 时安装并
enable 该 timer；其它 ASR backend 会 disable 它。timer 不覆盖用户的
`ECHOBRIDGE_ASR_*` 设置。

探活不能消除“崩溃恰好发生在两次 probe 之间”的窗口；它负责尽早发现并
重建 worker。第一句话的体验由下一节的网关语义保证。

## 4. 网关错误语义

`src/echobridge/asr.py` 区分：

- `ASRUnavailableError`：ConnectError、timeout、408/429/5xx；
- `ASRError`：其它非 200、JSON 解析失败等不可重试错误；
- HTTP 200 + 空文本：正常的空结果。

`src/echobridge/session.py` 对 `ASRUnavailableError`：

1. 用同一段 PCM 重试，默认 3 次、总预算 3 s（`ECHOBRIDGE_ASR_RETRY_*`）；
2. 仍失败时发送
   `conversation.item.input_audio_transcription.failed`
   带 `error="asr_unavailable"`、`reason="asr_unavailable"`、
   `retryable=true`；
3. 不发送 `empty speech`，也不让 App 把“服务不可用”误判成“没听清”。

空文本仍走原来的 `error="empty speech"`，这才是用户真正需要重说的情况。
`reason`/`retryable` 是 additive 字段，旧客户端会忽略；App 侧建议把
`reason=asr_unavailable` 映射成“语音识别服务正在恢复，请稍后再试”，
`empty speech` 才显示“没有听清”。

`AsrHealth` 记录可用状态、最近成功时间、不可用/重试计数，并暴露为：

```text
/readyz  workers.asr.available / last_error
/metrics echobridge_asr_available
         echobridge_asr_unavailable_total
         echobridge_asr_error_total
         echobridge_asr_retry_total
         echobridge_asr_retry_success_total
         echobridge_asr_last_ok_timestamp_seconds
```

## 5. 验证

```bash
# 健康
curl -fsS http://127.0.0.1:18801/health
curl -fsS http://127.0.0.1:18802/health
~/.venvs/echobridge/bin/python var/probe_asr_vllm.py

# 探活（正常应输出 ok status=200）
bash deploy/asr-probe.sh

# 故障演练：先确认没有活跃会话
curl -fsS http://127.0.0.1:18787/healthz
systemctl --user stop echobridge-asr-vllm
bash deploy/asr-probe.sh   # failed (1/2)
bash deploy/asr-probe.sh   # failed (2/2) -> restart
# 等待 /health 200，再跑一次 var/probe_asr_vllm.py

# 单元测试
bash scripts/test.sh
```

预期：

- ASR 不可用时，网关日志出现 `asr unavailable item=... bytes=...`，
  App 收到 `asr_unavailable` / `retryable=true`，不会显示“没有听清”；
- ASR 恢复后同一段音频可以重新识别；
- 新日志中没有音频、base64、token；
- TTS 的 keepalive/supervisor/watchdog 行为不变。
