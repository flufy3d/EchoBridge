# Phase 2：Silero VAD + Qwen3-ASR-0.6B

状态：**主机侧完成；X10A 实录音与远场 WER 待用户配合**
日期：2026-10-26
范围：Silero VAD、500 ms endpointing、300 ms pre-roll、独立 Qwen3-ASR worker、ASR 事件与延迟测试。
设备侧：**未做任何试听/录音/发声操作**；X10A 已息屏。

## 1. 实现

### 1.1 VAD

- `SileroVAD`：Silero VAD ONNX，CPU，512 samples/32 ms 窗口。
- 端点：默认 **500 ms** 静音（可配 500–700）。
- 最小语音：**200 ms**；最大语音：**15 s**。
- pre-roll：**300 ms**。
- tail pad：**100 ms**（送 ASR 前裁掉多余尾部静音，降低 ASR 时长）。
- 起音：连续约 64 ms 高概率窗口。
- 播放期：有活跃 response 时不跑 VAD；设备半双工静音帧不会触发。
- **投机端点（speculative endpoint）**：
  - 在端点前 **150 ms**（默认 350 ms 静音时）提前启动一次 ASR；
  - 若用户继续说话，VAD 发 `resume`，session 丢弃投机结果；
  - 若正常端点到达，复用已完成的投机结果，降低「端点后 final」延迟。
  - 该机制不改变 App 协议；App 只看到 `transcription.started` → `completed`。
- 纯静音/白噪声测试：10 s 静音和 10 s 白噪声均未触发 VAD。
- 语音中 400 ms 停顿测试：`early` 正常触发，恢复说话时 `resume`，投机结果丢弃。

### 1.2 ASR worker

- 独立 venv：`~/.venvs/echobridge-asr`。
- 模型：`Qwen/Qwen3-ASR-0.6B`，BF16，transformers 后端，`cuda:0`。
- 服务：`127.0.0.1:18801`，`POST /v1/transcribe`，body 为 16 kHz mono PCM16。
- 语言固定为 `Chinese`（强制 text-only prompt；比 auto 快约 300 ms）。
- `max_new_tokens=192`，batch size 1。
- 模型缓存：`~/.cache/echobridge/models/Qwen3-ASR-0.6B`（仓库外）。
- 模型加载：**2.5–3.2 s**；加载后 VRAM 分配 **1506.5 MB**；整卡占用约 2.6 GB。
- 启动 warmup；gateway 启动时额外发一次 `/healthz` 预热 HTTP 连接，避免首轮多 ~100 ms。
- 日志只记录字节数、字符数、语言、延迟；不记录转写文本/音频。

### 1.3 Gateway 集成

- `ECHOBRIDGE_VAD_BACKEND=silero`
- `ECHOBRIDGE_ASR_BACKEND=qwen3`
- `ECHOBRIDGE_ASR_URL=http://127.0.0.1:18801`
- `scripts/dev.sh` 默认使用 silero/qwen3；mock 回归可显式设置：
  ```bash
  ECHOBRIDGE_VAD_BACKEND=energy ECHOBRIDGE_ASR_BACKEND=mock ./scripts/dev.sh
  ```
- `ASR started` 在 VAD 起音时发送；`completed` 在 final（含投机复用）；空结果/ASR 异常发 `failed`，不调用 LLM。
- 失败/超时：gateway 捕获后发 `transcription.failed`，App 回到 LISTENING。

## 2. 运行

```bash
# ASR venv（一次性）
uv venv ~/.venvs/echobridge-asr --python 3.12
uv pip install --python ~/.venvs/echobridge-asr/bin/python \
  qwen-asr 'huggingface_hub[cli]' fastapi 'uvicorn[standard]' httpx

# 模型（一次性，仓库外）
bash scripts/download_asr_model.sh

# 启动 ASR worker
bash scripts/asr_worker.sh

# 另一个终端启动 gateway
bash scripts/dev.sh
```

健康检查：

```bash
curl http://127.0.0.1:18801/healthz
curl http://127.0.0.1:18787/healthz
```

Phase 2 的 gateway 默认 `vad=silero, asr=qwen3`；`/healthz` 会返回这两个字段。

## 3. 基准（主机侧，公开音频）

公开样本：Qwen3-ASR 仓库 `asr_zh.wav`，4.2 s，参考文本「甚至出现交易几乎停滞的情况。」
模型：Qwen3-ASR-0.6B，BF16，Chinese，max_new_tokens=192。
所有测试只处理 PCM/返回文本，不播放音频。

### 3.1 直接 worker 延迟（完整 4.2 s，3 次）

| 音频长度 | 延迟范围 | 说明 |
|---|---|---|
| 1.0 s | 225–269 ms | 输出「甚至。」 |
| 1.5 s | 254–356 ms | 输出「甚至出现。」 |
| 2.0 s | 342–390 ms | 输出「甚至出现交易。」 |
| 2.5 s | 352–406 ms | 输出「甚至出现交易几乎。」 |
| 3.0 s | 462–504 ms | 输出「甚至出现交易几乎停滞的。」 |
| 4.2 s | 466–527 ms | 全句，CER 0 |

直接 worker 的 4.2 s 全句约 0.47–0.53 s；worker 内部生成是主要耗时。

### 3.2 VAD + ASR（无投机）

| 音频 | VAD speech | 送 ASR 长度 | 端点后 final p50 |
|---|---|---|---|
| 2.5 s | 1984 ms | 2.36 s | **435 ms** |
| 4.2 s | 3232 ms | 3.61 s | **524 ms** |

### 3.3 VAD + ASR（投机端点，默认）

4.2 s 公开样本，8 次：

```text
358.1, 381.9, 428.3, 489.9, 502.5, 530.6, 548.3, 569.4 ms
p50 = 496 ms
p95 = 569 ms（8 样本）
```

2.5 s 样本，3 次：

```text
303.8, 312.9, 376.0 ms
p50 = 313 ms
```

结论：

- **2–2.5 s 常见语音：端点后 final 稳定 < 500 ms。**
- **4.2 s 较长语音：p50 约 496 ms，p95 约 569 ms；受 GPU 时钟/生成方差影响，略超 500 ms。**
- 15 s 长语音直接推理约 1.7 s；投机端点可减少约 150 ms，但仍会超过 500 ms。v1 先接受该限制；Phase 6 若需要严格达标，可评估 vLLM ASR 可选后端或更激进的投机策略。

### 3.4 质量（模拟噪声）

| 场景 | CER | 说明 |
|---|---|---|
| clean | 0.0 | 公开样本 |
| TV white noise SNR 10 dB | 0.0 | 白噪声代理，不是真实电视 |
| far-field LPF 4 kHz + SNR 15 dB + -12 dB | 0.0 | 模拟远场，不是 X10A 实录音 |

**这些不是设备 WER/CER。** 真实安静/电视/1–3 m 远场录音待用户方便时在 X10A 上补测。

### 3.5 VAD 误触发

| 输入 | 结果 |
|---|---|
| 10 s 纯静音 | 无事件 |
| 10 s 白噪声 RMS≈3000 | 无事件 |
| 语音 + 400 ms 停顿 + 语音 | `early` 后 `resume`，投机结果丢弃 |

## 4. 已知限制

- 真实设备录音/WER/CER 未完成；当前只有公开样本 + 模拟噪声。
- 4 s 以上语音的端点后 final 可能超过 500 ms；15 s 明显超过。
- 投机端点在「350 ms 停顿后继续说」时会浪费一次 ASR；已发 `resume` 丢弃结果，但 worker 可能被占用约 0.5 s。
- 强制 `Chinese`；英文/中英混说需要 Phase 2 后续测试或改为 auto。
- 未实现 ASR delta（可选）；App 目前只显示 started/completed。
- transformers 后端在 0.6B 上单请求生成约 0.5 s；未上 vLLM，符合 v1 决策。
- gateway venv 因 `silero-vad` 依赖安装了 torch；生产可改为纯 ONNX Runtime 以减小体积。

## 5. 测试

```bash
bash scripts/test.sh
# 25 passed
```

新增：

- `tests/test_vad_energy.py`：pre-roll、tail trim、短语音；
- `tests/test_session.py` 继续覆盖 mock ASR 空结果/工具/cancel；
- 真实 Silero/Qwen3 集成通过 `scripts/bench_asr.py`、`scripts/bench_vad_asr.py`、`scripts/ws_client.py --wav` 手动验证。

## 6. 下一步

1. 用户方便时，在 X10A 上录安静/电视/1–3 m 远场样本，补 WER/CER 和端点延迟。
2. 根据真实录音调整 `silero_threshold`、endpoint、pre-roll、投机 lead。
3. Phase 3：Qwen3-TTS 0.6B Base + VoiceDesign 参考音色，继续不在卧室设备上发声。
