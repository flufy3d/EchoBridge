# Phase 8：端到端延迟收口

> 状态：**P0 全部落地；vLLM-ASR 已在 8 GiB 上跑通并设为可选后端。**
> 日期：2026-10-29
> 范围：RTX 3070 8 GiB / WSL2 Ubuntu 26.04；所有测量只处理 PCM 和计时，不播放音频。
> 原始数据：`var/bench/tts_before_ramp.jsonl`、`tts_after_ramp3.jsonl`、
> `phase8_kv_asr.json`（`var/` 不入库）。

## 1. 结论

| 环节 | Phase 7 | Phase 8 | 说明 |
|---|---:|---:|---|
| LLM 首个正文 | 780–900 ms | **约 550–600 ms** | 关闭 DeepSeek thinking + 连接保活/预热 |
| TTS 首包 | 300–357 ms | **270–460 ms** | 首包改为 160 ms 音频 |
| 首包后断音 | **500–880 ms** | **p50 0–16 ms / max 62 ms** | `codec_chunk_ramp: [2,4,8,12,25]` |
| ASR after endpoint | 908 ms | **约 114 ms**（vLLM-ASR） | 固定 256 MiB KV，CUDA Graph |
| 正式首音（endpoint→） | 2141 ms | **约 853 ms** | 6 轮真实链路，含 1 次工具 hop |
| 整卡显存（TTS+ASR+gateway） | 6571 MiB | **6695 MiB** | 省掉 gateway 内的 1.5 GiB ASR |

vLLM-ASR 的代价：ASR 与 TTS **同时解码**时，4 s 音频从约 102 ms 变成约 768 ms，
15 s 从约 305 ms 变成约 2289 ms。正常一轮结束时 TTS 已空闲，所以 after-endpoint
仍是约 100 ms；受影响的是 barge-in。会话层现在会在转写 barge-in 之前先取消旧
TTS，把并发窗口缩到最小。

## 2. P0 改动

### 2.1 关闭 DeepSeek thinking

`src/echobridge/llm_deepseek.py` 发送：

```json
{"thinking": {"type": "disabled"}}
```

`reasoning_effort=low` 只是降低思考深度，不会关闭思考。opencode 网关实测会透传
`thinking`；关闭后不再返回 `reasoning_content`，也避免 DeepSeek 在思考 + tools
模式下要求回传 reasoning 的兼容问题。可用 `ECHOBRIDGE_LLM_THINKING=enabled`
恢复。

### 2.2 LLM 连接保活与预热

- httpx `keepalive_expiry` 从默认 5 s 提高到 300 s（`ECHOBRIDGE_LLM_KEEPALIVE_S`）。
- VAD `started` 和 `session.create` 时后台 `GET /v1/models`，提前完成 TLS/HTTP
  握手；失败只记 debug 日志。
- 首次正文计时改为从请求发出算起，不再只算 SSE 解析阶段。

### 2.3 TTS 分块 ramp

`deploy/qwen3_tts_echobridge.yaml`：

```yaml
codec_chunk_ramp: [2, 4, 8, 12, 25]
```

首包 160 ms（不是 80 ms），后续 320/640/960/2000 ms。实测首包后断音：

```text
before: gap p50 500–880 ms
after:  gap p50 0–16 ms, max 62 ms
```

尝试过 `[1,2,4,8,16,25]`：8→16 的跳跃会让 1280 ms 分块解码时间超过之前缓存的
1200 ms 音频，出现 0.2–0.9 s 卡顿；`[2,4,8,12,25]` 没有这个断崖。in-process
TTS 同步支持 `ECHOBRIDGE_TTS_CODEC_CHUNK_RAMP`。

### 2.4 pacer 不再逐块补零

`AudioPacer` 现在把每次 HTTP 读的不足一帧的尾巴留到下一块拼接，只在响应结束时
补零。旧实现每 2 s 会插入约 15–20 ms 静音，听感是咔哒声。

## 3. vLLM-ASR

### 3.1 为什么需要固定 KV

vLLM 0.32 用“总显存 × `gpu_memory_utilization` − profiling 前后 free 差值”来
估算 KV。WSL2 进程内的 `cudaMemGetInfo` 看不到其他进程，读数会随 WDDM 挪动显存
而漂移；双 vLLM 进程时出现过 `Available KV cache memory: -2.4 ~ -4.6 GiB`，
而 nvidia-smi 实际还有约 3 GiB。

解决办法是直接给 KV 大小，跳过 profiling：

```bash
vllm serve ~/.cache/echobridge/models/Qwen3-ASR-0.6B \
  --host 127.0.0.1 --port 18801 \
  --served-model-name qwen3-asr --trust-remote-code \
  --gpu-memory-utilization 0.30 \
  --kv-cache-memory-bytes 268435456 \
  --max-model-len 512 --max-num-seqs 1
```

256 MiB 约 2336 tokens（112 KiB/token），可容纳 4.5 个 512-token 请求；VAD 最长
15 s 语音约 450 tokens。实测 ASR 进程约 2.25 GiB，跑过请求后约 2.37 GiB。

### 3.2 启动顺序

`deploy/worker-ready.sh` 作为 TTS/ASR 的 `ExecStartPost`，等 `/health` 通过后
才让 unit 进入 active；ASR 额外发两次静音预热请求。顺序：

```text
TTS ready → ASR ready → gateway
```

`install.sh` 显式把 TTS 放在 `systemctl start` 参数的第一位，因为
`systemctl start A B` 是按参数顺序提交 job 的，`After=` 不能补救反序。

### 3.3 实测

| 场景 | 结果 |
|---|---|
| 4 s 音频，TTS 空闲 | 100–117 ms |
| 15 s 音频，TTS 空闲 | 304–368 ms |
| 4 s 音频，TTS 同时生成 | 768 ms（max 1167 ms） |
| 15 s 音频，TTS 同时生成 | 2289 ms |
| ASR SIGKILL 后自动恢复 | 约 85 s（RestartSec 15 s） |
| ASR 热重启 | 约 62 s |
| 全停冷启动到全部就绪 | 约 172 s |

### 3.4 TTS worker 恢复后自动预热

`app.py` 每 10 s 探测一次已配置的 TTS worker；检测到 `down → up` 且当前没有
正在生成的回复时，调用 `rewarm()` 强制发一次真实合成请求。网关里的 PCM 缓存
和 `_warmed` 标记会跨 worker 重启保留，普通 `ensure_warm()` 会直接返回，所以
必须强制绕过缓存。

实测重启 TTS（114 s 恢复）：

```text
tts backend recovered after 110.9 s; rewarmed voice=mingyue_v1 in 782 ms
first real TTS request: 220 ms first packet (before: ~3.7 s)
```

### 3.5 回退

```bash
cp -p ~/.config/echobridge/backup-20261029/echobridge.env \
      ~/.config/echobridge/echobridge.env
bash deploy/install.sh --user
```

或只改 `ECHOBRIDGE_ASR_BACKEND=inprocess`；`install.sh` 会停用并禁用
`echobridge-asr-vllm.service`。不要同时保留两个 ASR 后端：会多占约 1.2 GiB，
并让 WDDM 把 TTS 的驻留显存挤到系统内存。

## 4. 并发与 barge-in

`Session._handle_vad_event(final)` 现在在转写之前先 `_cancel_active(...)`：

- 旧 TTS 立即停止生成，ASR 不再和 TTS 分时抢 GPU；
- 短噪声仍走 `speech_ms < vad_min_speech_ms` 提前返回，不会误停；
- stop command 仍只发一次 `response.canceled`；
- 旧的 `barge_ctx` / `asr_ms` 日志字段保持不变。

## 5. 仍未做

- 语义端点（Smart Turn v3）需要中文评测，暂未接入。
- fused 单 stage TTS 仍是实验项；当前 1.3 GiB 余量足够，不需要它为 ASR 让路。
- 更小的 CPU ASR / ONNX 是唯一能彻底消除 GPU 争抢的方向，但会失去 Qwen3-ASR
  的 context 热词。
