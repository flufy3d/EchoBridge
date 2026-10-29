# Phase 7：单进程 / 单一 CUDA Context 融合重构

> 状态：**Step 1（Gateway+VAD+ASR 进程内融合）完成；Step 2（单会话 vLLM 瘦身 +
> 投机去重）完成；Step 3（InProcessQwen3TTS）已实现并实测，但纯 PyTorch RTF
> 不达标，保留为实验后端。**
> 日期：2026-10-29
> 范围：RTX 3070 8 GiB / WSL2 Ubuntu 26.04；所有测试只处理 PCM 字节与计时，
> 未在主机或 X10A 上播放任何音频。
> 原始数据：`var/bench/phase7_30.json`、`var/bench/phase7_deepseek_12.json`、
> `var/bench/tts_slim.json`、`var/bench/tts_deployed.json`、
> `var/bench/tts_inprocess.json`、`var/bench/allinone_2.json`、
> `var/bench/asr_adaptive_12.json`、`var/bench/asr_preroll300_8.json`
> （`var/` 不入库）。

## 1. 结论先行

| 项 | 改造前 | 改造后（默认） | 结论 |
|---|---|---|---|
| 进程模型 | gateway + asr_worker + vLLM 三进程 | gateway（含 ASR）+ vLLM | 少一个常驻进程、一份 CUDA context |
| 整卡显存 | **7473 MiB** | **6576 MiB** | **-897 MiB**，剩余约 1.6 GiB |
| 仅 vLLM 显存 | 5443 MiB | **4429 MiB** | **-1014 MiB**（Stage 0/1 瘦身） |
| 冷启动就绪 | 约 4 min 45 s | 约 **3 min 15 s** | vLLM 仍需约 2 min 55 s；ASR 不再单独等 23 s |
| ASR after endpoint p50 | 858 ms | **908 ms**（30 轮真实 DeepSeek） | 未达 350–500 ms；p95 1176 ms 优于基线 1300 ms |
| Formal first audio p50 | 1231 ms | **1185 ms** | 略优 |
| Formal first audio p95 | 2474 ms | **1559 ms** | **-915 ms** |
| Endpoint→formal p95 | 3443 ms | **2721 ms** | **-722 ms** |
| TTS 首包 p50 / RTF p50 | 240–440 ms / 0.35–0.545 | **300–357 ms / 0.301–0.423** | 基本持平；瘦身未破坏实时性 |
| All-in-One（ASR+TTS 同进程） | — | 4515 MiB，首包约 600 ms，RTF≈4 | **不作为默认**，见 §5 |

**推荐默认**：`ECHOBRIDGE_ASR_BACKEND=inprocess` +
`ECHOBRIDGE_TTS_BACKEND=vllm` + 瘦身后的
`deploy/qwen3_tts_echobridge.yaml`。这就是 Step 1+2 的「Gateway+VAD+ASR 合一 +
瘦身版 vLLM」。

## 2. Step 1：Gateway + VAD + ASR 进程内融合

### 2.1 依赖与环境

- 主 venv `~/.venvs/echobridge` 补装 `qwen-asr`、`qwen-tts`、`transformers
  4.57.6`、`accelerate`、`librosa`、`soundfile`、`soynlp`、`nagisa`、
  `qwen-omni-utils`、`einops`、`sox`。
- `qwen-tts` 在 import 时会走到 25 Hz tokenizer，而该分支 import
  `torchaudio.compliance.kaldi`；主 venv 不安装 torchaudio（二进制与 torch
  CUDA 版本强绑定）。`echobridge/_qwen_compat.py` 提供 **import-only stub**，
  仅 12 Hz 路径使用，任何真正调用会显式报错。
- Silero VAD 仍是 **ONNX Runtime CPU**；它 import torch 只用于构造 CPU
  tensor，不创建 CUDA context。`torch.cuda.memory_allocated()` 只统计 ASR/TTS
  模型。
- ASR 模型：`Qwen3-ASR-0.6B`，BF16，`cuda:0`，
  `max_inference_batch_size=1`，`max_new_tokens=192`，语言固定 `Chinese`。
  加载后 `memory_allocated=1506.5 MB`，`reserved=1804 MB`。

### 2.2 新增后端

`src/echobridge/asr.py`：

- `InProcessQwen3ASR`，`ECHOBRIDGE_ASR_BACKEND=inprocess`。
- 直接接收 `pcm: bytes`，在 worker 线程中转 float32 并调用
  `Qwen3ASRModel.transcribe`；**没有** localhost `/v1/transcribe`、
  PCM body 复制和 Base64 `X-ASR-Context`。
- 推理通过 `GPUScheduler` 串行化；模型加载与 warmup 都在进程内完成。
- `/readyz` 的 `workers.asr` 直接返回 `loaded/load_s/vram_allocated_mb/
  vram_reserved_mb/device/supports_context`。

`src/echobridge/gpu.py`：

- 进程级优先级 turnstile：ASR final (30) > speculative ASR (20) > TTS (10)。
- TTS 每个 talker step / decode step 单独持锁，ASR 可以在 step 之间抢占。
- 协程被取消时，lease 会等 worker 线程真正退出后才释放，避免「取消了一个
  投机推理、又立刻在同一张卡上启动第二个推理」。
- `/readyz` 新增 `gpu.owner/waiting/memory`；`/metrics` 新增
  `echobridge_gpu_memory_allocated_bytes` /
  `echobridge_gpu_memory_reserved_bytes`（均为 additive）。

### 2.3 兼容与回退

- `ECHOBRIDGE_ASR_BACKEND=qwen3` 仍指向 18801 worker；`mock` 仍是测试默认。
- 原有 WebSocket 事件、`/healthz`、`/readyz`、`/metrics`、`/voices` 字段只增
  不减。
- `bash scripts/test.sh`：**94 passed**。

## 3. Step 2：单会话 vLLM 瘦身与投机去重

### 3.1 vLLM-Omni 配置

`deploy/qwen3_tts_echobridge.yaml` 的实测调整：

| 字段 | 原值 | 新值 | 说明 |
|---|---:|---:|---|
| Stage 0 `max_num_seqs` | 64 | **2** | 单设备单会话；2 给 async-chunk 留余量 |
| Stage 0 `gpu_memory_utilization` | 0.30 | **0.30** | 模型加载 1.91 GiB，0.30 是下限附近 |
| Stage 0 `max_model_len` | 4096 | **2048** | 4096 需要 0.44 GiB KV，但 0.30 只剩 0.33 GiB |
| Stage 1 `max_num_seqs` | 64 | **2** | 一条 codec 流 |
| Stage 1 `gpu_memory_utilization` | 0.30 | **0.18** | codec 权重 0.64 GiB |
| Stage 1 `max_num_batched_tokens` | 65536 | **8192** | 16×(72+25)=1552 tokens/window |
| Stage 1 `max_model_len` | 65536 | **8192** | 覆盖约 10 s 语音 + left context |

实测（无 ASR、无 gateway）：

```text
原配置   nvidia-smi 5443 MiB
新配置   nvidia-smi 4429 MiB   (-1014 MiB)
```

瘦身后 5 句 TTS（3 次/句，未播放；gateway + 进程内 ASR 已驻留，vLLM 已完成
greeting/backchannel warmup）：

```text
文本                                    音频    首包p50  首包max  RTF p50
你好呀，我在呢。                        2.72 s   331.8     359.7   0.301
今天也要开开心心的呀。                  2.08 s   352.1     379.1   0.423
现在是晚上十一点，该准备休息啦。        3.04 s   327.2     341.3   0.397
我看看哦，这个大概是一个蓝色的杯子。    4.24 s   357.0     374.3   0.330
23 乘以 47 等于 1081。                  3.92 s   300.0     308.9   0.364
```

首包 p50 全部 <400 ms，RTF p50 全部 <0.5。更早一次冷启动首请求（未 warmup）
曾出现 429 ms / RTF 0.62；网关的 greeting warmup 会吸收这条冷路径。

### 3.2 投机 ASR 去重

`src/echobridge/session.py`：

- `early` 的结果不再「等 1.5 s 超时就丢弃再跑一次」；只要 item 相同，就复用
  在途的 speculative 结果。
- early 段与 final 段在 tail-pad 后内容一致（最多差 endpoint lead 的静音），
  因此复用不会丢字。
- ASR backend 增加可选 `speculative=True` 关键字；旧 backend/测试 double 不
  支持时自动回退到三参数调用。
- 长语音的投机 lead 从 150 ms 提高到 **350 ms**（`speech_ms >= 2500` 时），
  短语音保持 150 ms，避免在自然停顿上频繁启动/丢弃 ASR。

### 3.3 ASR 延迟实测与瓶颈

30 轮真实 DeepSeek + 真实 ASR/TTS，单 WS 会话，无播放：

| 指标 | Phase 6 基线 | Phase 7 | 变化 |
|---|---:|---:|---:|
| ASR final（最后 speech 帧起）p50 | 1358 ms | 1408 ms | +50 ms |
| ASR after endpoint p50 | 858 ms | **908 ms** | +50 ms |
| ASR after endpoint p95 | 1300 ms | **1176 ms** | **-124 ms** |
| Formal first audio after ASR p50 | 1231 ms | **1185 ms** | -46 ms |
| Formal first audio after ASR p95 | 2474 ms | **1559 ms** | **-915 ms** |
| Endpoint→formal p50 | 2141 ms | **2103 ms** | -38 ms |
| Endpoint→formal p95 | 3443 ms | **2721 ms** | **-722 ms** |

**为什么没有达到 350–500 ms：**

1. 直接对 benchmark 的 4 s 语音做进程内 ASR（无 TTS、无 HTTP）实测
   848–1182 ms；模型生成本身就是主耗时，不是 HTTP/Base64。
2. 4 s 语音在 3070 上的 Qwen3-ASR-0.6B 需要约 0.85–1.1 s；投机 lead 350 ms
   只能把这部分提前，不能消除。
3. 把 `ECHOBRIDGE_VAD_PRE_ROLL_MS` 从 800 降到 300 后，同一 8 轮 benchmark
   的 ASR after endpoint 降到 **p50 652 ms / p95 972 ms**（-206 ms），但仍高于
   500 ms。该设置会削弱「没什么，」这类软起音的保留，属于质量/延迟权衡。
4. `torch.compile`（`ECHOBRIDGE_ASR_COMPILE=1`）在本机实测 840–920 ms，
   与未编译 848–890 ms 基本持平，未采用。
5. FP16 / FlashAttention-2 未纳入默认：前者需要重新验证识别质量，后者在
   torch 2.14 + CUDA 13 上没有可用 wheel。

**可选延迟优先配置**（质量换延迟）：

```bash
ECHOBRIDGE_VAD_PRE_ROLL_MS=300
ECHOBRIDGE_VAD_SPECULATIVE_LEAD_MS=150
ECHOBRIDGE_VAD_SPECULATIVE_LONG_LEAD_MS=350
```

实测 p50 652 ms / p95 972 ms（8 轮，TTS mock，仅用于隔离 ASR）。

## 4. Step 3：InProcessQwen3TTS（实验）

### 4.1 实现

`src/echobridge/tts_inprocess.py`，`ECHOBRIDGE_TTS_BACKEND=inprocess`：

- 与 `InProcessQwen3ASR` 同进程、同 CUDA context、同 caching allocator。
- 启动时把 `prompt.pt` 的 `ref_code` / `ref_spk_embedding` 常驻 GPU；不再每次
  发送 440 KB Base64 `reference.wav`。
- 直接驱动 `Qwen3TTSModel` 的 Talker 自回归循环，而不是等
  `generate_voice_clone()` 整句返回：
  - 首包 `initial_codec_chunk_frames=1`；
  - 之后 `codec_chunk_frames=25`，`codec_left_context_frames=72` 滑窗；
  - 每个 step 后检查 cancel，barge-in 最迟在一个 talker step 内停止。
- Talker attention 使用 SDPA；Code2Wav 是卷积解码器。`ECHOBRIDGE_TTS_COMPILE=1`
  会对无状态的 Code2Wav 解码器启用 `torch.compile(mode="reduce-overhead")`。Talker 的
  每步 Python 自回归循环 graph-break 严重，实测 compile 不改变 RTF，未纳入。
  默认关闭。
- 与 HF 参考实现的 codec token 前缀逐位一致（前 19 帧）；流式解码与整句解码
  时长完全一致，波形最大绝对差 0.018。

### 4.2 实测

5 句 × 3 次（每次清空网关缓存，未播放）：

```text
文本                                    音频    首包p50  首包max  RTF p50
你好呀，我在呢。                        2.08 s   394.3     407.2   3.865
今天也要开开心心的呀。                  2.08 s   370.0     399.2   3.586
现在是晚上十一点，该准备休息啦。        3.04 s   445.2     453.9   4.261
我看看哦，这个大概是一个蓝色的杯子。    3.60 s   423.5     465.7   3.977
23 乘以 47 等于 1081。                  4.16 s   412.2     432.2   4.137
```

- 首包 p50 **412 ms**，略高于 400 ms 门槛；首包 max 466 ms。
- RTF p50 **3.98**，远高于 0.5。纯 PyTorch 的 Talker + 15 路子码本自回归在
  3070 上仍是 3–4 倍实时，`torch.compile` 未能改变量级。
- 因此按验收条件**不设为默认**；保留 `inprocess` 分支与数据，供后续换更强
  GPU 或上游提供 CUDA Graph/融合 kernel 时复用。

### 4.3 全进程实测

实验启动方式：

```bash
ECHOBRIDGE_ASR_BACKEND=inprocess \
ECHOBRIDGE_TTS_BACKEND=inprocess \
ECHOBRIDGE_VAD_BACKEND=silero \
PYTHONPATH=src ~/.venvs/echobridge/bin/python -m echobridge
```

实测：

```text
nvidia-smi        4517 MiB / 8192 MiB
torch allocated   3579.9 MB
readyz            asr + tts 都 ok=true，同一 gpu.memory
冷启动就绪        约 62 s（含 ASR 5–23 s + TTS 2.6 s + greeting warmup）
greeting 首音     1.4–1.5 ms（缓存）
首包（ASR 后）    约 575–627 ms
打断             256–348 ms，0 条旧音频
```

内存和冷启动最好，但 RTF 不满足实时播放；仅作为实验模式。

## 5. 最终默认与回退

默认（推荐）：

```bash
ECHOBRIDGE_ASR_BACKEND=inprocess
ECHOBRIDGE_TTS_BACKEND=vllm
ECHOBRIDGE_TTS_GPU_UTIL=0.30
# deploy/qwen3_tts_echobridge.yaml 使用 §3.1 的瘦身值
```

回退：

```bash
# 回到独立 ASR worker
ECHOBRIDGE_ASR_BACKEND=qwen3
ECHOBRIDGE_ASR_URL=http://127.0.0.1:18801
bash scripts/asr_worker.sh

# 回到旧 vLLM 配置
ECHOBRIDGE_TTS_GPU_UTIL=0.40
# 并恢复 deploy/qwen3_tts_echobridge.yaml 的 max_num_seqs/max_model_len
```

`make_asr` / `make_tts` 仍支持 `mock` / `qwen3` / `vllm`；`inprocess` 是新增
可选值，不改变原有取值语义。

## 6. 复现

```bash
# 协议/单元回归（不加载 GPU 模型）
bash scripts/test.sh

# 直接 ASR 模型延迟（无 VAD/HTTP）
~/.venvs/echobridge/bin/python scripts/bench_asr_inprocess.py \
  --repeat 3 --out var/bench/asr_inprocess.json

# 进程内 TTS 首包/RTF（无播放）
~/.venvs/echobridge/bin/python scripts/bench_tts_inprocess.py \
  --backend inprocess --repeat 3 --out var/bench/tts_inprocess.json

# vLLM TTS（先启动瘦身服务）
ECHOBRIDGE_TTS_GPU_UTIL=0.30 bash scripts/tts_vllm.sh
~/.venvs/echobridge/bin/python scripts/bench_tts_vllm.py \
  --repeat 3 --out var/bench/tts_slim.json

# 30 轮真实网关（先启动 gateway；无播放）
~/.venvs/echobridge/bin/python scripts/bench_gateway.py \
  --turns 30 --out var/bench/phase7_30.json
```

## 7. 已知限制

- ASR after endpoint 未达 350–500 ms；模型本身在 4 s 语音上需要约
  0.85–1.1 s。要达标需要更快 ASR（vLLM/CTC/更小模型）或更激进的降 pre-roll。
- 进程内 ASR 与 gateway 共享 GIL；实测与独立 worker 的直接推理差异在方差内，
  但极端 CPU 竞争下仍可能比独立进程差。
- 投机 ASR 被 `resume` 取消时，CUDA 线程不能中断，会继续跑完当前推理；已通过
  取消后等待线程退出避免重叠，但会占用一次推理时间。
- `InProcessQwen3TTS` 的 RTF 3.98，不能用于实时整句播放；仅首包和取消路径
  可用。
- vLLM 瘦身值按 RTX 3070 8 GiB 单会话标定；换卡/换模型需要重新测
  `gpu_memory_utilization` 与 `max_model_len`。
