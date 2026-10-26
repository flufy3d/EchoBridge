# Phase 3：Qwen3-TTS 0.6B Base + 原创音色

状态：**主机侧技术完成；主观音色/10 分钟试听待用户**
日期：2026-10-27
范围：Qwen3-TTS-12Hz-0.6B-Base、VoiceDesign 原创参考音色、vLLM-Omni 流式 TTS、网关集成。
设备侧：**未做任何播放/试听**；X10A 保持息屏。

## 1. 结论

- **HF `qwen-tts` 0.6B Base 不可用于实时**：3070 上 GPU 利用率仅约 11%，2–4 s 音频需要 7–14 s（RTF 3.2–3.5）。
- **vLLM-Omni 0.6B Base 达到实时**：首包 p50 **262–379 ms**，RTF p50 **0.377–0.428**（5 条文本 × 3 次，ASR worker 同驻）。
- 已生成原创“鲸鱼娘”参考音色：3 个候选，6.48/6.80/6.88 s；选中 6.88 s 候选。**主观试听待用户**。
- 网关 TTS 已切到 vLLM 流式端点；HF worker 保留为参考/回退实验，不作为默认。

## 2. 音色设计

- 模型：`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`，BF16，离线生成后进程退出，不常驻。
- 参考文本：
  > 你好呀，我是溟月，一条住在屏幕里的小鲸鱼。今天也想陪你聊聊天，可以吗？
- instruct：
  > 年轻女性，二次元可爱声线，明亮、软糯、略带俏皮，语速中等偏快，句尾轻微上扬；自然清晰，不要过度夹子音/萝莉音，不要播音腔。
- 生成 3 个候选：

| 候选 | 时长 | peak | RMS |
|---|---|---|---|
| 1 | 6.48 s | 0.516 | 0.079 |
| 2 | 6.80 s | 0.477 | 0.060 |
| 3 | 6.88 s | 0.563 | 0.095 |

- 选择候选 3（时长最接近 7 s、无削波、RMS 较健康）为 `reference.wav`；候选文件保留在仓库外：
  `~/.cache/echobridge/voices/mingyue/`
- 参考音频、文本、prompt/meta 全部在仓库外；`.gitignore` 覆盖 `voices/`、`*.wav`、`*.pt`。
- **未播放、未试听、未克隆豆包音色。**

## 3. 运行时选择

### 3.1 HF `qwen-tts`（参考实现，不默认）

- 0.6B Base BF16，`generate_voice_clone()` 返回完整波形；仓库主线没有 token 级增量 PCM。
- 实测 8 字文本：7.5–10.4 s；11–18 字：7.0–14.3 s；RTF 3.2–3.5。
- GPU 利用率 11–12%，功耗约 54 W；瓶颈在子码本自回归/Python 调度，不是显存。
- 保留 `scripts/tts_worker.sh` + `src/echobridge/tts_worker.py` 作为参考和回退实验，不作为 v1 默认。

### 3.2 vLLM-Omni（默认）

- `vllm-omni serve Qwen3-TTS-12Hz-0.6B-Base --omni`，OpenAI 兼容 `/v1/audio/speech`，`stream=true` 返回 24 kHz PCM。
- Deploy 配置：`deploy/qwen3_tts_echobridge.yaml`，在官方配置基础上打开 stage 0 `enable_prefix_caching: true`（参考音频为共享前缀）。
- 启动脚本：`scripts/tts_vllm.sh`；默认 `gpu_memory_utilization=0.40`。
- 启动约 **109 s**（含 torch.compile/CUDA Graph warmup；缓存后仍需约 1–2 min）。
- 首请求冷启动约 1.9 s；网关启动时 `warmup()` 预合成固定 greeting 并缓存，实际 greeting 命中缓存。
- 网关 `VLLMTTSClient` 把参考音频作为 data URL（约 440 KB）随请求发送；vLLM 的 speech/speaker cache + prefix cache 复用。

## 4. TTS 基准（主机侧，无播放）

5 条文本，每条 3 次，ASR worker 同驻，vLLM `gpu_memory_utilization=0.40`：

| 文本 | 音频 | 首包 p50 | 首包 max | RTF p50 |
|---|---|---|---|---|
| 你好呀，我在呢。 | 2.00 s | 261.5 ms | 297.1 ms | 0.402 |
| 今天也要开开心心的呀。 | 2.40 s | 355.8 ms | 425.5 ms | 0.426 |
| 现在是晚上十一点，该准备休息啦。 | 3.52 s | 373.1 ms | 416.5 ms | 0.377 |
| 我看看哦，这个大概是一个蓝色的杯子。 | 3.36 s | 336.9 ms | 551.9 ms | 0.428 |
| 23 乘以 47 等于 1081。 | 3.44 s | 379.3 ms | 508.4 ms | 0.391 |

- 首包 p50 达标（<500 ms）；单条 max 551.9 ms，属于长句/冷缓存尾部。
- RTF p50 全部 <0.5。
- 冷启动首请求约 1.9 s；网关 warmup 只预合成 greeting，其他新文本首次仍可能有 400–900 ms 首包。
- 连续 10 分钟“无爆音”需要用户实际试听；本轮只验证了 PCM 流/网关 pacing 不报错。

### 4.1 网关端到端（mock LLM + ASR + vLLM TTS）

公开 4.2 s 语音、mock LLM、真实 ASR/TTS：

| run | 首个文本 | 首音（语音结束起） | ASR final → 首音 | 总时长 |
|---|---|---|---|---|
| 1 | 826.6 ms | 2100.7 ms | 1274.1 ms | 5553 ms |
| 2 | 823.3 ms | 977.5 ms | 154.2 ms | 5108 ms |
| 3 | 825.8 ms | 949.1 ms | 123.3 ms | 4113 ms |

- run 1 是新文本冷 TTS；run 2/3 命中网关 TTS 文本缓存。
- 真实 DeepSeek 首 content 约 1.1–2.2 s，Phase 4 必须在此基础上复测首音 p50/p95。
- 语音结束到首个 ASR 文本约 823 ms（500 ms endpoint + 约 300 ms ASR）。
- ASR final 到首音在缓存命中时 123–154 ms；新文本约 300–500 ms。

## 5. 显存与稳定性

- TTS vLLM 0.45 + ASR：整卡约 7.9 GB，仅剩约 100 MB，**不安全**。
- TTS vLLM 0.40 + ASR：整卡约 **7.27–7.4 GB**，剩余 **600–744 MB**；本轮基准稳定。
- 0.40 是当前 3070 8 GB 的折中；如果后续 OOM：
  1. 降 TTS 到 0.38/0.39 并观察 KV cache；
  2. 关闭桌面 GPU 占用；
  3. 降低 vLLM `max_model_len` / codec budget；
  4. 不盲目 Q4/Q5。
- vLLM-Omni 与 vLLM 0.32.0 有版本不匹配 warning；当前可运行，但生产需固定匹配版本。
- vLLM start ~109 s；ASR worker 在 TTS 已驻留时加载约 23 s（比单独 2.5 s 慢，受显存/CPU 竞争影响）。

## 6. 代码/部署

```bash
# 参考音色（一次性，仓库外；不播放）
~/.venvs/echobridge-tts/bin/python scripts/design_voice.py

# vLLM TTS 服务
ECHOBRIDGE_TTS_GPU_UTIL=0.40 bash scripts/tts_vllm.sh

# ASR worker
bash scripts/asr_worker.sh

# gateway（默认 tts_backend=vllm）
bash scripts/dev.sh
```

- 网关配置：
  - `ECHOBRIDGE_TTS_BACKEND=vllm`
  - `ECHOBRIDGE_TTS_URL=http://127.0.0.1:18802`
  - `ECHOBRIDGE_TTS_REF_AUDIO=~/.cache/echobridge/voices/mingyue/reference.wav`
  - `ECHOBRIDGE_TTS_REF_TEXT=~/.cache/echobridge/voices/mingyue/reference.txt`
- 文本清理/切分沿用 `src/echobridge/text.py`；首句优先短句。
- `VLLMTTSClient` 支持流式首包、odd-byte 边界、完整文本 PCM 缓存（最多 8 条）。
- `Qwen3TTSClient`/HF worker 保留为 `tts_backend=qwen3`，仅作参考。

## 7. 测试

```bash
bash scripts/test.sh
# 25 passed
```

- 单元/集成测试仍使用 mock TTS，不受 vLLM/GPU 依赖影响。
- TTS 基准：`scripts/bench_tts_vllm.py`（vLLM）、`scripts/bench_tts.py`（HF，参考）。
- 网关 cancel 测试在 vLLM TTS 下通过，无旧音频。
- 真机未播放；设备保持息屏。

## 8. 待用户确认/下一步

1. **主观试听** 3 个候选（`candidate_1/2/3.wav`），确认是否换成候选 1/2，或是否接受当前候选 3。
2. 10 分钟连续试听、长句/数字/中英混、爆音/断帧检查。
3. 若音色/质量不达标：先用 0.6B CustomVoice Serena/Vivian 验证；再评估 1.7B Base BF16（显存更紧，不阻塞 v1）。
4. Phase 4：DeepSeek 流式 + tools；在真实 LLM 下复测首音，目标 p50 <1.5 s / p95 <2.5 s。
5. 之后 Phase 5 设备端双后端；到时再在 X10A 试听。
