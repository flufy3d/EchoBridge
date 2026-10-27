# EchoBridge v1 实施计划

> 目标：把明月 Live（X10A）的豆包端到端实时语音替换为「本地 ASR + DeepSeek + 本地 TTS」的自托管网关，同时保持设备端豆包 Realtime 协议子集兼容，并保留豆包后端与自动回退。
>
> 状态：Phase 5 真机验收完成（VoiceClient 双后端、auto 回退、设置页、backchannel、10 次切换、视觉/工具/休眠/打断）；Windows Tailscale Serve 的 WSS 鉴权留待 Phase 6 在原生 Ubuntu 复测。Phase 0–4 结果见 `docs/PHASE1.md`–`docs/PHASE4.md`，Phase 5 见 `docs/PHASE5.md`。本文定义协议、边界、验收和阶段计划。
>
> 基线仓库：`E:\Projects\EchoBridge`（空仓库，仅 README）、`E:\Projects\mingyue-live`（M6，App 0.6.0）。

---

## 1. 结论先行

1. **设备端不改协议解析逻辑**。现有 `VoiceSession.java` 只使用豆包 Realtime 的一个子集，EchoBridge 必须按它实际读取的字段和状态机实现，而不是按官方文档“完整协议”实现。
2. **双后端是硬约束**。豆包后端原样保留；新增 EchoBridge 后端；两者实现同一个 `VoiceClient` 接口；设置页、试听、自动连接、休眠恢复、管理面板重连全部走 activeBackend。
3. **v1 的关键路径是“首音”**：DeepSeek 首内容实测 1.08–2.18 s，TTS 首包和网络必须控制在约 0.3–0.5 s 内，才能守住 p95 < 2.5 s。TTS 的“真流式”是当前最大技术风险（见 §10、§12）。
4. **半双工不做 AEC**。设备播放时上行发静音帧；用户抢话由设备本地 RMS 触发 `response.cancel`。网关收到 cancel 后必须立即取消 LLM/TTS、丢弃旧 generation、清理发送队列，并保证 `response.canceled` 之后不再出现旧音频。
5. **工具链不变**。设备仍然执行 `look_and_ask`、`set_emotion`、`play_animation`、`set_prop`、`get_time`、`go_to_sleep`、`end_conversation`；EchoBridge 过滤 `ask_expert`，因为主 LLM 已经是 DeepSeek。`look_and_ask` 的拍照和视觉请求仍由设备完成，网关只转发工具结果。
6. **默认仍是豆包**。EchoBridge 验收通过前，`voiceBackend` 默认 `doubao`；验收后可把默认改为 `auto`，但用户仍可手动切回豆包。

---

## 2. Phase 0 现状审计

### 2.1 设备端事实（以代码为准）

| 项 | 实测/代码事实 | 影响 |
|---|---|---|
| 设备 | Xiaomi X10A，MT8168，2 GB RAM，Android 9 GSI，1280×800，arm64-v8a | 模型不上设备 |
| App | `com.mingyue.live` 0.6.0，Java，OkHttp 3.14.9 | 不重写 App |
| 当前 URL | `VoiceSession.WS_URL` 硬编码 `wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue` | Phase 5 改为可配置 |
| 鉴权 | 豆包：`X-Api-Key: <key>`；User-Agent 自定义 | EchoBridge 用独立 token，不能复用豆包 key |
| 上行 | 16 kHz mono PCM16，20 ms / 320 samples / 640 B，base64，文本 JSON | 网关必须按 640 B 帧处理，兼容其他长度 |
| 下行 | 24 kHz mono PCM16，base64；App 按 960 B（20 ms）切片播放 | 网关按 20 ms/960 B 节奏发送 |
| 半双工 | 播放时上行发 640 B 静音帧；本地 RMS ≥9000 持续 150 ms 触发 cancel；冷却 600 ms | 网关不能依赖播放期上行做 VAD |
| 会话 | `session.created` 到达后才启动 AudioEngine；之后才 connected | `session.created` 必须尽早、且不能先发音频 |
| 空闲/无人/休眠 | 上层会 `disconnect()`，发送 `session.close`，最多等 1 s `session.closed` | 网关必须及时回 `session.closed` 并释放上游 |
| 事件解析 | 只解析本计划 §4 的事件；未知事件只记日志 | 可以带额外字段，App 会忽略 |
| 生成 ID | App 不使用 response/generation 字段 | 网关可加 `response_id`/`generation`，用于内部丢弃旧音频 |
| 工具 | App 发火山格式 tools；执行后发 `conversation.item.create`，`role=tool` | 网关做 OpenAI 格式转换与结果回填 |
| 试听 | `previewVoice` 新建 `VoiceSession`，用豆包 key 和豆包 voiceId | Phase 5 必须改走 activeBackend，否则继续烧豆包 |
| 记录 | SQLite 记录消息、工具、事件、指标；不存音频/base64/key | 不破坏 schema；只增加 backend 字段到 event detail |
| 设备实测 | ADB `100.83.39.115:5555` 可连；LAN IP `192.168.31.188`；`wlan0` 正常 | Phase 1 可真机测试 |

### 2.2 当前主机/网络/GPU 基线

| 项 | 实测 |
|---|---|
| 当前会话主机 | `360W-PC-03`（Tailscale work-pc），Windows；WSL2 Ubuntu 26.04 |
| GPU | RTX 3070，8192 MiB；空闲占用 656–661 MiB；剩余约 7363 MiB；0% util；49 °C |
| WSL2 GPU | `nvidia-smi` 可见同一张 3070 |
| WSL2 磁盘 | `/` ext4 1007 G，已用 56 G，可用 901 G |
| Python | WSL2 `/usr/bin/python3` = 3.14.4；Windows `py -3.12` 指向不存在的 `C:\Python312`；需安装/固定 Python 3.12 环境 |
| Tailscale 到 x10a | 直连 `188.156.106.197:41175`，`tailscale ping` 12 ms；一次路由切换期曾走 DERP 57–140 ms |
| ADB | TCP 5555 连接约 29 ms；设备在线 |
| 局域网 | 当前 PC 不能 ping 通 `192.168.31.188`；当前 PC 在 `10.20.0.0/24`，x10a 在家宽 `192.168.31.0/24`。生产主机若在同一 LAN，需另测 |
| Tailscale Serve | 当前 work-pc 的 443 已代理 `http://127.0.0.1:3001`；EchoBridge 需独立端口/路径 |
| 模型缓存 | 未发现 Qwen3-ASR/TTS 权重；HuggingFace/ModelScope 缓存基本为空 |
| EchoBridge 仓库 | 只有 README，无 `.gitignore`，无代码 |
| mingyue-live 仓库 | 干净，HEAD `4fdef03` |

### 2.3 DeepSeek Flash TTFT 实测

请求：`POST https://opencode.ai/inference/openai/v1/chat/completions`，`model=deepseek-v4.1-flash`，`reasoning_effort=low`，`stream=true`，`User-Agent: curl/8.0`。  
测量点：HTTP 请求开始 → 第一个 `delta.content`（不是 reasoning）。6 次独立请求，两个短提示词版本。

| # | 首个 reasoning (s) | 首个 content (s) | 完成 (s) | completion tokens | reasoning tokens |
|---|---|---|---|---|---|
| 1 | 0.847 | 1.106 | 1.150 | 118 | 90 |
| 2 | 0.786 | 1.079 | 1.144 | 129 | 90 |
| 3 | 0.691 | 1.227 | 1.272 | 129 | 90 |
| 4 | 0.889 | 1.423 | 1.463 | 169 | 148 |
| 5 | 0.900 | 1.292 | 1.329 | 175 | 157 |
| 6 | 0.782 | 2.181 | 2.221 | 180 | 163 |

- 首个 content：p50 ≈ **1.26 s**，p95（6 样本，最近秩）≈ **2.18 s**，均值 ≈ **1.38 s**。
- 首个 reasoning：0.69–0.90 s；reasoning tokens 90–163。即使 `reasoning_effort=low`，reasoning 仍占主要首延迟。
- `reasoning_effort=minimal` / `none` 均被上游拒绝（HTTP 400：“not supported by deepseek-v4.1-flash”），v1 不能靠关闭 reasoning 降延迟。
- 首音预算：DeepSeek 1.1–2.2 s + TTS 首包 + Tailscale/设备 0.05–0.15 s。若 TTS 首包 0.3–0.5 s，p95 会逼近 2.5 s。必须在 Phase 4 用真实 prompt/30 轮复测，并在 Phase 3 尽量压低 TTS 首包。
- 6 个样本不足以定义生产 p95；Phase 4/6 需要 30+ 次统计。

---

## 3. 总体架构

```text
┌────────────────────────────── X10A (Android 9) ──────────────────────────────┐
│ MainActivity                                                                  │
│   └── VoiceBackendController (activeBackend / auto fallback / single-flight)  │
│         ├── DoubaoBackend  = 现有 VoiceSession（协议行为不改）                 │
│         └── EchoBridgeBackend = 同一 VoiceClient 接口，新 WSS 客户端           │
│               └── AudioEngine（16k 上行 / 24k 下行 / 半双工 / 本地 RMS 抢话）  │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │ WSS (Tailscale only, token in header)
┌───────────────────────────────────▼───────────────────────────────────────────┐
│ EchoBridge Gateway (Python 3.12, FastAPI/uvicorn + websockets)                 │
│  auth / single-session manager / protocol adapter / turn state machine         │
│  VAD (Silero ONNX, CPU) / audio pacer / generation & cancel                    │
│  ├── LLM adapter  ── HTTPS ──> opencode DeepSeek Flash (stream)                │
│  ├── ASR adapter  ── localhost ──> ASR worker (Qwen3-ASR-0.6B, transformers)  │
│  └── TTS adapter  ── localhost ──> TTS worker (Qwen3-TTS-12Hz-0.6B-Base)      │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 进程/环境拆分

| 进程 | 环境 | 职责 | 显存（BF16 估算） |
|---|---|---|---|
| `echobridge-gateway` | Python 3.12 venv | WSS、协议、VAD、turn、LLM、pacer、会话 | 0（VAD CPU/ONNX） |
| `echobridge-asr` | 独立 Python 3.12 venv | Qwen3-ASR-0.6B transformers 分段推理 | 约 2–3 GB |
| `echobridge-tts` | 独立 Python 3.12 venv | Qwen3-TTS-12Hz-0.6B-Base，voice clone prompt | 约 2.5–3.5 GB |
| 离线 `voice-design` | 独立 venv，按需启动 | Qwen3-TTS-12Hz-1.7B-VoiceDesign 生成参考音色，生成后退出 | 约 4–5 GB，不常驻 |

- ASR/TTS 依赖容易冲突（`qwen-asr` 与 `qwen-tts` 各自锁定 transformers/torch 版本），**不装在同一 venv**。
- 三者都只监听 `127.0.0.1`；对外只有 gateway 的 WSS，经 Tailscale Serve 终止 TLS。
- 模型缓存放 ext4：`/var/lib/echobridge/models`（生产）、WSL2 的 `~/.cache/echobridge/models`；禁止 `/mnt/c`。
- 参考音频/克隆 prompt/日志/密钥全部在仓库外；`.gitignore` 覆盖 `models/`、`voices/`、`*.wav`、`*.pt`、`*.log`、`.env`、`var/`。

### 3.2 关键设计原则

1. **单设备单会话**：同一时刻只允许一个活跃设备会话。不同设备/不同 token 直接拒绝（WS close 1013）；同一设备的快速重连可在关闭旧会话后接管，避免重连风暴导致回退豆包。
2. **模型常热**：ASR/TTS worker 启动即加载并 warmup；会话关闭只取消请求，不卸载模型。
3. **generation 贯穿全链路**：每次 response 分配 `response_id` + `generation`；LLM/TTS/ASR/pacer/发送队列都带 generation；cancel 后旧 generation 一律丢弃。
4. **发送串行化**：所有下行帧走单一 writer；cancel 时清空未发送的旧队列，发送 `response.canceled`，此后旧 generation 的帧不允许再进入 writer。
5. **可观测但不泄密**：日志记录 session id、事件类型、耗时、字节数、generation；不记录 token、Authorization、完整 base64、完整音频、用户原文（可记录字数/哈希）。
6. **默认豆包**：EchoBridge 验收前不改变默认行为；所有新配置有默认值/迁移，不删除旧 key。

---

## 4. 协议适配层（以 VoiceSession 实际解析为准）

### 4.1 传输与鉴权

| 项 | 豆包后端（保持） | EchoBridge 后端（新增） |
|---|---|---|
| URL | `wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue` | 可配置，默认 `wss://work-pc.tailea818a.ts.net:8443/echobridge/ws`（示例） |
| 鉴权 | `X-Api-Key: <volc key>` | `Authorization: Bearer <echo token>`；不接受 query token |
| UA | 现有自定义 UA | 现有 UA 或 `Mingyue/0.6`；不得含 token |
| 建连超时 | 现有 30 s | 5 s 建连；8 s 内未收到 `session.created` 视为失败 |
| 关闭 | `session.close`，等 1 s `session.closed` | 同左；网关先取消上游，再回 `session.closed` |
| 第二连接 | 不适用 | 不同 token 拒绝；同设备重连可接管 |

EchoBridge 的 token 只用于网关；**绝不把豆包 key 发给 EchoBridge，也不把 EchoBridge token 发给豆包**。

### 4.2 上行事件（App → 网关）

| 事件 | App 实际发送的字段 | 网关处理 |
|---|---|---|
| `session.create` | `{type, event_id, session:{model, instructions, audio:{input:{format:{type:"pcm",rate:16000}}, output:{format:{type:"pcm_s16le",rate:24000}, voice}}, tools}, extension:{asr:{extra:{}}, tts:{extra:{}}, dialog:{extra:{enable_loudness_norm:true}}}}` | 解析 instructions/tools/voice；`model` 记录但不依赖；回 `session.created` |
| `input_audio_buffer.append` | `{type:"input_audio_buffer.append", audio:"<base64 640B>"}`，无 event_id | base64 解码；校验长度/采样率；送 VAD；播放期静音帧要识别 |
| `speech_text_buffer.append` | `{type, event_id, speech_id, text}` | 按 `speech_id` 聚合，不调用 LLM |
| `speech_text_buffer.commit` | `{type, event_id, speech_id, text}` | 合并 append+commit 文本；启动 TTS-only response |
| `conversation.item.create` | `{type, event_id, items:[{type:"message", role:"tool", call_id, content:[{type:"input_text", text}]}]}` | 取 call_id/text；可多条；回填工具结果，继续 LLM |
| `response.cancel` | `{type, event_id}` | 立即取消 LLM/TTS/工具等待；bump generation；发 `response.canceled` |
| `session.close` | `{type, event_id}` | 取消全部上游；回 `session.closed`；关闭 WS |

补充：
- App 不会发 `input_audio_buffer.commit`、`session.update`、`conversation.item.create` 的 user 消息；VAD/ASR 全由网关负责。
- App 在播放 TTS 时发的是 640 B 全零静音帧，不是真实麦克风。网关必须知道“当前有活跃 response”并避免把静音帧当作新的用户停顿/空 ASR。
- 麦克风手动静音时，AudioEngine 发全零帧且 RMS=0；网关不应产生 ASR 事件。
- App 每个上行 JSON 走单线程发送 executor；网关必须容忍突发/轻微乱序（同一 WS 内仍有序）。

### 4.3 下行事件（网关 → App）

App 只读这些字段；额外字段会被忽略。

| 事件 | App 读取字段/回退 | App 行为 | 网关约束 |
|---|---|---|---|
| `session.created` | `session.id`（可选） | 启动 AudioEngine，置 connected/LISTENING；发送 pending greeting | 必须在 `session.create` 后尽早发；此前不得发音频 |
| `session.updated` | 无 | 忽略 | 可不发 |
| `conversation.item.input_audio_transcription.started` | 无 | 若 SPEAKING：flush 播放并转 LISTENING；否则若在播放也 flush | VAD 起音即发；不要等 ASR |
| `conversation.item.input_audio_transcription.delta` | `delta`，回退 `text` | 更新 ASR 字幕 | 可选；v1 可只发 completed |
| `conversation.item.input_audio_transcription.completed` | `transcript`，回退 `text` | 清空 reply；置 THINKING；非空才 `onAsrFinal` | 只在非空时发；空 ASR 不要发 completed（见 §4.6） |
| `conversation.item.input_audio_transcription.failed` | `error`（仅日志） | 置 LISTENING“没有听清” | 空 ASR/ASR 失败用这个，回到聆听 |
| `response.output_text.delta` | `delta`，回退 `text` | 追加回复、字幕；必要时置 THINKING | 只送 content，不送 reasoning |
| `response.output_text.done` | `text`（可选） | 用 text 或累积文本收尾；`onReplyDone` 落库 | 建议在 audio.done 后发（保打断落库正确，见 §4.6） |
| `response.output_audio.started` | 无 | 置 SPEAKING（cancel 后 1 s 抑制期内忽略） | 必须在第一个 audio.delta 之前发 |
| `response.output_audio.delta` | `delta`，回退 `audio` | base64 解码，入队播放；必要时置 SPEAKING | 24 kHz PCM16；20 ms/960 B；base64；按节奏发 |
| `response.output_audio.done` | 无 | 有 pending tool → THINKING；否则 LISTENING | 在最后一块音频之后发 |
| `response.function_call_arguments.done` | `items[]`：`call_id`（回退 `id`）、`name`、`arguments` | THINKING；并行执行工具；把所有结果合并成一个 `conversation.item.create` | 多个调用必须放同一个 `items` 数组一次性发 |
| `response.done` | 无 | 清抑制；有 pending tool → THINKING；否则 LISTENING | 必须在 audio.done 之后；每个 response 恰好一次 |
| `response.canceled` | 无 | flush 播放；清 reply/pending tools；LISTENING | 收到 cancel 后 <300 ms 发；此后不得再发旧 generation 音频 |
| `response.backchannel` | `response_id`/`sound_id`/`delta`/`format` | 独立 AudioTrack 播放“嗯～”；正式音频 started 时 120 ms 淡出 | 仅 EchoBridge；独立于正式 response/pacer，每轮最多 1 条；不参与 text/audio 状态机 |
| `error` | `error.message`，回退 `error.toString()`，再回退顶层 `message` | **终止会话**：ERROR、关 socket、上层重试 | 只用于致命错误；不要用 error 表达“没听清” |
| `session.closed` | 无 | 关 socket、IDLE | 关闭前发；`disconnect()` 最多等 1 s |

### 4.4 正常一轮时序

```text
App                                  Gateway
 | session.create ------------------->|
 |<------------------ session.created|  (App start AudioEngine)
 | input_audio_buffer.append(20ms)...|
 |                                    | VAD onset
 |<-- transcription.started ---------|  (App flush playback)
 |<-- transcription.delta* ----------|  (optional)
 | input_audio_buffer.append...      |
 |                                    | endpoint 600ms silence
 |                                    | ASR final < 500ms
 |<-- transcription.completed --------|  (non-empty)
 |                                    | LLM stream
 |<-- response.output_text.delta* ----|
 |                                    | first sentence → TTS
 |<-- response.output_audio.started --|
 |<-- response.output_audio.delta* ---|  20ms paced
 |<-- response.output_text.done ------|  after audio (见 §4.6)
 |<-- response.output_audio.done -----|
 |<-- response.done ------------------|
```

### 4.5 工具调用时序

```text
LLM finish_reason=tool_calls
  → 网关只发一个 response.function_call_arguments.done:
    { "type": "...",
      "response_id": "resp_7",
      "items": [
        {"call_id":"call_a","name":"set_emotion","arguments":"{\"emotion\":\"happy\"}"},
        {"call_id":"call_b","name":"play_animation","arguments":"{\"name\":\"spin\"}"}
      ] }
  → App 并行执行（最多 2 线程），把结果合并成一个 conversation.item.create:
    { "type":"conversation.item.create",
      "items":[
        {"type":"message","role":"tool","call_id":"call_a","content":[{"type":"input_text","text":"ok"}]},
        {"type":"message","role":"tool","call_id":"call_b","content":[{"type":"input_text","text":"ok"}]}
      ] }
  → 网关等待所有 call_id 结果（超时 25s）
  → 把 assistant(tool_calls) + 所有 role=tool 消息加入上下文
  → 再次调用 DeepSeek，生成最终语音
```

- `ask_expert` 在转换为 OpenAI tools 时过滤；若模型仍幻觉出该调用，网关本地丢弃并追加一条 tool 结果“ask_expert 不可用，请直接回答”，不发给 App（避免 App 再烧 DeepSeek）。
- `look_and_ask` 原样转发；设备拍照、调用视觉模型、把文字结果回传。网关不接触图片。
- 工具结果缺失/超时：发 `response.canceled`（App 会清零 `pendingToolCalls`），不要让 App 卡在“等待工具结果”。不要在没有全部结果时开始新音频。
- 工具执行期间用户 cancel：立即 `response.canceled`；迟到的 tool result 按 generation 丢弃。

### 4.6 协议陷阱（必须按实际代码处理）

1. **空 ASR 不能发 completed**。`VoiceSession` 收到空 `transcript` 仍会置 THINKING，但不会触发 `onAsrFinal`；上层会卡在“思考中”。空 ASR/纯噪声必须发 `conversation.item.input_audio_transcription.failed`，App 才会回 LISTENING，且不调用 LLM。
2. **`response.output_text.done` 的时机影响“被打断”落库**。App 在 text.done 时立即 `onReplyDone` 并清 `replyPending`；若之后才被 cancel，`flushPendingReply(true)` 不会再把这条标为 interrupted。EchoBridge 应把 text.done 延迟到 audio.done 之后（有音频时），让用户打断时仍处于 pending，从而落库 `interrupted=1`。无音频（纯文本/工具）时可在文本结束时发。
3. **`response.output_audio.started` 早于第一个 delta**。App 依赖 started 置 SPEAKING；虽然 delta 也能置 SPEAKING，但用户要求明确，且便于调试/验收。
4. **`response.done` 必须在 audio.done 之后**。App 在 audio.done 可能已经回 LISTENING；response.done 再兜底。顺序反过来会让状态提前结束。
5. **cancel 后的旧音频必须真正消失**。App 在 cancel 后 1 s 内用 `audioSuppressUntilMs` 丢弃 delta，但 1 s 后不会。网关必须在 `response.canceled` 后保证旧 generation 不再产生任何 delta。
6. **`session.created` 是音频启动闸门**。早发音频会被 AudioEngine 启动时清空队列；晚发会让 App 一直 CONNECTING。建连后先回 created，再接受 greeting/ASR。
7. **`speech_text_buffer` 的 commit 文本是后半段**。App 把 greeting 拆成 append(前半) + commit(后半)，同一 `speech_id`；网关必须合并，不能只读 commit 或重复读。
8. **greeting 不要发 text.delta**。App 在本地已经显示/落库 greeting；网关再发文本会造成重复字幕/重复记录。greeting 只发 audio.started/delta/done + response.done。
9. **`response.canceled` 会清零 App 的 pendingToolCalls**。工具等待超时/取消时这是唯一安全的回退路径；不要只发 response.done。
10. **error 是致命的**。App 收到 error 会关 socket 并走重试/回退；ASR 空、LLM 超时、TTS 失败等可恢复问题不要用 error，改用 transcription.failed / response.canceled / 短 TTS 兜底。

---

## 5. EchoBridge 模块划分

```text
EchoBridge/
  pyproject.toml
  README.md
  .gitignore
  config/
    echobridge.example.toml
  src/echobridge/
    __main__.py
    config.py                # env/TOML，密钥只从 env/file 读
    logging.py               # 结构化日志 + 脱敏
    gateway/
      app.py                 # FastAPI/uvicorn，WSS endpoint
      auth.py                # Bearer token，constant-time compare
      session_manager.py     # 单会话、接管/拒绝、生命周期
      protocol.py            # 豆包子集 encode/decode、事件常量
      turn_manager.py        # 状态机、VAD、endpointing、generation
      pacer.py               # 20ms 24k PCM 发送、jitter/backpressure
      tools.py               # 火山→OpenAI tools、ask_expert 过滤、结果回填
    adapters/
      llm_deepseek.py        # OpenAI 兼容流式、reasoning 过滤、超时/429
      asr_client.py          # 调 ASR worker
      tts_client.py          # 调 TTS worker
    workers/
      asr_worker.py          # Qwen3-ASR-0.6B transformers
      tts_worker.py          # Qwen3-TTS-12Hz-0.6B-Base
      voice_design.py        # 离线 1.7B VoiceDesign（不常驻）
    audio/
      pcm.py                 # PCM16/base64/重采样/静音检测
      vad.py                 # Silero ONNX
      text.py                # Markdown/emoji/URL 清理、分句
  tests/
    test_protocol.py
    test_turn_manager.py
    test_pacer_cancel.py
    test_tools.py
    test_text.py
    integration/
      mock_backends.py
      ws_client.py
      test_greeting_turn_cancel.py
  deploy/
    systemd/*.service
    tailscale-serve.md
    install.sh
    dev.sh
  docs/
    PLAN.md
    BENCHMARK.md
    ACCEPTANCE.md
```

### 5.1 网关内部状态机

```text
IDLE
  └─ session.create → SESSION_READY (send session.created)
LISTENING
  └─ VAD onset → USER_SPEAKING (send transcription.started)
USER_SPEAKING
  ├─ cancel/close → IDLE/LISTENING
  └─ endpoint → ASR_RUNNING
ASR_RUNNING
  ├─ empty/failed → LISTENING (send transcription.failed)
  └─ final → LLM_STREAMING (send transcription.completed)
LLM_STREAMING
  ├─ content → TTS_STREAMING (text deltas + audio)
  ├─ tool_calls → TOOL_WAIT (send function_call_arguments.done)
  └─ empty → LISTENING (response.done)
TTS_STREAMING
  ├─ audio done → TEXT_DONE → LISTENING/TOOL_WAIT
  └─ cancel → LISTENING (canceled)
TOOL_WAIT
  ├─ all results → LLM_STREAMING (new response)
  └─ timeout/cancel → LISTENING (canceled)
```

- 每个 response 有独立 `asyncio.TaskGroup`/取消令牌；cancel/close 取消任务组并等待清理（有界超时）。
- ASR 任务与 response 任务分离：用户说话时 cancel 只取消 LLM/TTS，不取消 ASR（用户新语音仍需识别）。
- TTS worker 的合成任务也按 generation 取消；worker 内部要能中断当前 `model.generate`（至少丢弃结果，避免占用 GPU）。
- 发送 writer 是单协程；pacer 产生 20 ms 帧；writer 有有界队列；cancel 时清空旧 generation 并插入 canceled 事件。
- 背压：若客户端网络慢，pacer 不无限堆积；超过阈值（如 2 s 音频）暂停合成/丢弃最旧非关键帧；不能丢弃 canceled/done。

### 5.2 音频 pacer

- 输出格式：PCM16 mono 24 kHz，20 ms = 480 samples = 960 B，base64 后 1280 字符。
- 以 monotonic 时钟按 20 ms 发送；容忍事件循环抖动，使用 absolute deadline，不用 `sleep(0.02)` 累加。
- 首个 delta 前发 `response.output_audio.started`。
- 最后一个 delta 后发 `response.output_audio.done`，再 `response.output_text.done`（有音频时），再 `response.done`。
- jitter：设备 AudioTrack 有约 200 ms 缓冲；网关保持 1–2 帧预发送，不一次灌满。
- 背压：发送队列上限约 2 s；TTS 生成快于实时时阻塞/暂停生产，不要无限内存增长。
- cancel：停止 pacer；清空队列；bump generation；发 `response.canceled`；确保 writer 中不再有旧 generation 帧。
- 长句/连续 10 分钟：帧边界不切断样本；必要时 5–10 ms 淡入淡出，避免爆音。

---

## 6. VAD / ASR

### 6.1 VAD

| 参数 | 值 |
|---|---|
| 输入 | 16 kHz mono PCM16，20 ms 帧 |
| VAD | Silero VAD（ONNX，CPU；模型缓存放仓库外） |
| 窗口 | 512 samples（32 ms）聚合；阈值可配 |
| 起音 | 连续 2–3 个正窗口（约 64–96 ms） |
| 端点 | 500–700 ms 静音；v1 默认 600 ms |
| min speech | 200 ms |
| max speech | 15 s（强制端点） |
| pre-roll | 300 ms 环形缓冲 |
| 播放期 | 有活跃 response 时禁用新 turn 的 VAD；App 发的是静音帧，不能据此起音/端点 |
| 噪声 | 能量门限 + Silero 双条件；电视/远场需 Phase 2 调参 |

- VAD 起音即发 `transcription.started`，让 App flush 播放。
- 端点后把 pre-roll + speech 段送 ASR worker；ASR final 后发 `completed`。
- 段长 <200 ms 或 ASR 空/纯标点：发 `failed`，回 LISTENING，不调用 LLM，不产生 TTS。
- 用户在 ASR 运行中再次说话：v1 以当前段为准；若检测到强起音，丢弃当前 ASR 并重新开始（避免两个 turn 叠加）。
- 半双工：播放期没有真实上行，不做“语音打断”检测；只认 `response.cancel`。

### 6.2 ASR

- 模型：`Qwen/Qwen3-ASR-0.6B`，BF16，transformers 后端，`qwen-asr`。分段推理，不上 vLLM 流式（v1）。
- 输入：16 kHz mono；输出：中文/英文混合文本；语言自动或 `Chinese`，Phase 2 对比。
- `transcription.started` 在 VAD 起音时发；`completed` 在端点+推理完成后发；`failed` 用于空/错误。
- 目标：端点后 final < 500 ms（P50/P95 都记录）；ASR worker 推理本身目标 < 300 ms（3–5 s 语音）。
- 可选 delta：v1 默认关闭；若 Phase 2 有余量，可用“每 300–500 ms 对当前缓冲重跑”的伪增量，但不得影响 final 延迟/稳定性。
- worker API（v1）：`POST /v1/transcribe`，body 为原始 PCM16（`application/octet-stream`）+ header `X-Sample-Rate: 16000`；可选 header `X-ASR-Context: <base64 UTF-8>` 传热词/上下文增强；返回 `{text, language, latency_ms}`。本地 loopback，避免 base64 开销。
- 热词：`ECHOBRIDGE_ASR_HOTWORDS`（`|` 分隔）由网关拼成 `Technical terms: ...` 后随请求下发。Qwen3-ASR 的 `context` 是 prompt 偏置，不是带权重的热词表，列表要短，需用实录验证；网关对“整张热词表被原样吐出”的幻觉做兜底丢弃。
- 超时：5 s；失败返回 `failed`；worker 挂掉由 gateway health/watchdog 重启。
- 测试：设备实录安静/电视/远场 1–3 m；记录 WER/CER、端点后延迟；录音不入库/不入 repo。

---

## 7. TTS

### 7.1 模型与音色

| 项 | 决策 |
|---|---|
| 常驻模型 | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，BF16 |
| 离线设计 | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`，BF16；生成 5–10 s 参考音频后卸载 |
| 克隆 | 0.6B Base `create_voice_clone_prompt`，ICL 模式（`x_vector_only_mode=False`），缓存 prompt |
| 输出 | 24 kHz mono PCM16 |
| 参考音色 | 原创“鲸鱼娘/可爱二次元女声”；不克隆豆包 |
| 描述 | 年轻女性，二次元可爱声线，明亮、软糯、略带俏皮，语速中等偏快，句尾轻微上扬；自然清晰，不要过度夹子音/萝莉音，不要播音腔 |
| 兜底 | VoiceDesign 效果一般时，先用 0.6B CustomVoice 的 Serena/Vivian 验证链路，再换原创音色 |
| 不采用 | 1.7B + Q4/Q5；GGUF 量化；豆包音色克隆 |

参考音频/文本/prompt 缓存路径（仓库外）：

```text
/var/lib/echobridge/voices/mingyue/
  reference.wav        # 5–10 s, 24k, 原创
  reference.txt        # 参考文本（ICL 需要）
  prompt.pt            # create_voice_clone_prompt 缓存
  meta.json            # 描述、模型版本、hash、生成时间
```

### 7.2 文本清理与切分

- 清理：Markdown（`**`、`__`、反引号、标题、列表）、emoji、URL、多余空白、括号舞台说明；保留正常标点。
- 分句：按 `。！？!?；;\n`；首句若 >15 字，再按 `，,、` 切到 ≤12–15 字，优先保证首包。
- 首句短：LLM prompt 要求 1–2 句、≤60 字；TTS 层仍强制首句短。
- 数字/中英混：先不做激进替换；Phase 3 测“2026 年 10 月 26 日”“23×47”“Wi-Fi”等，必要时做中文读法规范化。
- 空文本/纯符号：不调用 TTS，直接 response.done。

### 7.3 合成与流式

- v1 适配器接口是流式：`synthesize_stream(text, voice_id, generation) -> AsyncIterator[bytes]`。
- 截至 Phase 0（2026-10）查看的 `qwen-tts` 主线代码，`generate_voice_clone()` 返回完整波形（即使 `non_streaming_mode=False` 也只是模拟流式文本输入），**没有暴露 token 级增量 PCM**。因此 v1 先实现“短句级流式”：
  1. 首句 8–15 字，单独合成；
  2. 合成完立即把 PCM 交给 pacer；
  3. 后续句子在 worker 中顺序合成，gateway 边收边发；
  4. 句子间用 5–10 ms 淡入淡出/交叉淡化，避免爆音。
- 首包目标 <500 ms：0.6B BF16 + 短首句 + warm model + 本地 loopback；Phase 3 实测。若达不到：
  - 先试 0.6B CustomVoice Serena/Vivian（更快、无 clone prompt）；
  - 再做 token 级解码 spike（HF `generate` streamer + 12Hz tokenizer 分段 decode + overlap/crossfade）；
  - 仍不达标再评估 1.7B Base BF16（只做实验，不阻塞 v1）。
- 不采用 1.7B 常驻 + Q4/Q5。

### 7.4 TTS worker API

- WebSocket `ws://127.0.0.1:.../v1/tts/stream`：
  - 请求：`{id, text, voice_id, language, generation}`
  - 响应：binary PCM16 24k 帧 + JSON `{type:"done", id, generation}`；错误 JSON。
- 取消：gateway 关闭连接或发 `{type:"cancel", generation}`；worker 必须尽快停止/丢弃当前生成，避免 GPU 被旧任务占用。
- 预热：启动时加载模型、构建/加载 clone prompt、合成一句短问候并丢弃；记录首包。
- 缓存：固定 greeting “你来啦，我在呢。” 可预合成 PCM 缓存（内存/仓库外），命中时首音几乎为 0；缓存 key 含 voice/model 版本。

---

## 8. LLM 与工具

### 8.1 DeepSeek 适配

| 项 | 值 |
|---|---|
| Endpoint | `POST https://opencode.ai/inference/openai/v1/chat/completions` |
| Headers | `Authorization: Bearer <gateway key>`, `Content-Type: application/json`, `User-Agent: curl/8.0` |
| Model | `deepseek-v4.1-flash` |
| reasoning | `reasoning_effort: "low"`；过滤 `reasoning_content` |
| stream | true |
| 首 token 超时 | 6 s（reasoning 首包也计入） |
| 首 content 超时 | 8 s |
| 总超时 | 30 s（不含工具执行等待） |
| 429 | 指数退避 1 s/2 s，最多 2 次；5xx 重试 1 次；4xx 不重试 |
| 空 content | 不送 TTS；若有 tool_calls 继续工具；否则 response.done |
| 上下文 | system + 最近 20 条消息/约 6k tokens；保留 system 和未完成的 tool 链 |
| 温度 | 0.6–0.7（Phase 4 调） |
| max_tokens | 300–400（语音短回复） |

- 只把 `delta.content` 送 TTS/文本；`reasoning_content` 只用于日志计数（可选），绝不下发。
- 工具调用增量按 `index` 聚合 `id/name/arguments`；`finish_reason=tool_calls` 后一次性发 `response.function_call_arguments.done`。
- 工具结果回填后再次请求 DeepSeek；`assistant` 消息必须带完整 `tool_calls`，`role=tool` 消息必须带对应 `tool_call_id`。
- 失败兜底：短 TTS“我这会儿有点卡住了，再说一遍好吗？”；连续失败/致命错误才发 `error`。

### 8.2 主对话 prompt

以 `session.create.instructions` 为底，追加“实时语音”约束，并声明优先于原文字数限制：

```text
【实时语音约束（优先于上文字数限制）】
1. 每次回复 1–2 句，口语化，不超过 60 个汉字。
2. 不要 Markdown、列表、标题、代码块、括号动作、表情符号、舞台说明。
3. 不要提“专家/工具/照片/API/模型”；不要解释推理过程。
4. 需要看用户眼前画面时调用 look_and_ask；不要猜。
5. 不要调用 ask_expert（该工具在本后端不可用）；复杂问题直接简短回答或说明。
6. 句末要顺口，避免长串数字、英文缩写和生僻符号。
```

- 工具描述沿用 App 的火山格式转换；不重写工具语义。
- 保留 `look_and_ask`、`set_emotion`、`play_animation`、`set_prop`、`get_time`、`go_to_sleep`、`end_conversation`。
- `ask_expert` 从 OpenAI tools 中删除；若模型仍输出，网关本地吞掉并继续。

### 8.3 工具结果与多调用

- 火山格式 → OpenAI：
  ```json
  {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
  ```
- 下行 → App：
  ```json
  {"type":"response.function_call_arguments.done",
   "response_id":"resp_7",
   "items":[{"call_id":"call_a","name":"set_emotion","arguments":"{\"emotion\":\"happy\"}"}]}
  ```
- App → 网关：
  ```json
  {"type":"conversation.item.create",
   "items":[{"type":"message","role":"tool","call_id":"call_a",
             "content":[{"type":"input_text","text":"ok"}]}]}
  ```
- 多个调用：同一 `items` 数组；App 并行执行后合并回传；网关按 call_id 匹配，全部到齐再继续。
- 工具等待 25 s；超时发 `response.canceled`，避免 App 卡 pending。
- 工具失败是正常结果（App 返回中文兜底），照常回填给 LLM，由 LLM 自然转述。

---

## 9. 设备端双后端改造（mingyue-live）

### 9.1 接口与类

```text
VoiceClient (interface)
  setListener(Listener)
  connect()
  sendAudio(pcm16, length, rms)       // AudioEngine 20ms 回调进入后端
  cancel()
  close(reason) / disconnect()
  preview(text)                        // mic muted + greeting，TTS-only
  isConnected()
  getState()
  sendGreeting(text)
  sendToolResult(callId, text)
  setMicMuted(muted)
  testMic(ms, callback)
  playTestTone()

DoubaoBackend   = 现有 VoiceSession，implements VoiceClient；协议/音频/半双工行为不改
EchoBridgeBackend = 新类，implements VoiceClient；同协议子集，不同 URL/token/voice
VoiceBackendController = 单飞切换、activeBackend、auto fallback、防双连接
```

- 为了不破坏豆包路径，`VoiceSession` 只做“实现接口 + 构造参数化 URL/token + 把现有 `handleMicFrame` 暴露为 `sendAudio`”的最小改动；不重写其内部状态机、音频处理、cancel、工具逻辑。
- `AudioEngine` 可以由后端持有，也可以由 controller 注入；`VoiceBackendController` 保证同一时刻只有一个 AudioEngine 活动。无论哪种，麦克风 20 ms 回调最终进入当前 `VoiceClient.sendAudio(pcm16, length, rms)`；后端内部决定半双工静音、本地 RMS 抢话、导播。这样 `sendAudio` 是真正的接口方法，而不是只在某个后端内部存在。
- `EchoBridgeBackend` 可以复用 `AudioEngine` 和 `VoiceSession.Listener` 语义；协议解析/发送独立实现，避免豆包分支被新逻辑污染。
- `MainActivity` 只依赖 `VoiceClient`，不再直接 `new VoiceSession`。
- 试听：`VoiceClient.preview(text)` 在 activeBackend 上创建临时连接（mic muted、只发 greeting），同一时间只允许一个预览 + 主会话关闭；`VoiceBackendController` 负责暂停/恢复主会话。
- 工具执行器仍绑定 `ToolRegistry`，两个后端共用；`ask_expert` 过滤在网关，App 工具定义不改。

### 9.2 配置模型（两套独立，互不覆盖）

```text
voiceBackendMode: doubao | echobridge | auto      # 用户选择；默认 doubao
activeBackend: doubao | echobridge                # 持久化的具体选择；auto 下记录首选
forceDoubao: boolean                              # 强制豆包，优先于 auto
runtimeBackend: doubao | echobridge               # 仅内存，auto 回退时变化

voice.doubao.wsUrl       = wss://openspeech.bytedance.com/...
voice.doubao.apiKey      = <device-local, 不进日志>
voice.doubao.voiceId     = zh_female_... (现有默认)
voice.doubao.model       = 1.2.6.1

voice.echobridge.wsUrl   = wss://<host>.<tailnet>.ts.net:8443/echobridge/ws
voice.echobridge.token   = <device-local, 不进日志>
voice.echobridge.localVoiceId = mingyue_v1
```

- 现有 `voice_id` 继续作为豆包 voiceId；新增 EchoBridge 的 `local_voice_id`，不要把豆包 voiceId 传给 EchoBridge。
- BuildConfig 提供默认值（`VOLC_WS_URL`、`VOLC_API_KEY`、`ECHOBRIDGE_WS_URL`、`ECHOBRIDGE_TOKEN`、`VOICE_BACKEND_MODE`）；SettingsStore 覆盖；密钥输入框只写不读回显。
- 迁移：老用户没有新 key 时，`voiceBackendMode=doubao`，豆包行为不变；不改动 SQLite schema。
- 日志：记录 backend/mode/effective，不记录 token/key/URL query。

### 9.3 选择与回退语义

| 模式 | 行为 |
|---|---|
| `doubao` | 始终豆包；EchoBridge 配置保留但不用 |
| `echobridge` | 只连 EchoBridge；失败报错，不自动烧豆包（除非用户开 auto） |
| `auto` + `forceDoubao=false` | 先试 EchoBridge；失败/超时/中途断开 → 本次连接临时回退豆包，UI 提示“已临时回退豆包”；不修改保存的 `activeBackend` |
| `auto` + `forceDoubao=true` | 永远豆包；EchoBridge 配置保留 |

- 自动回退触发条件：
  - 建连失败 / 5 s 超时；
  - 8 s 内没有 `session.created`；
  - 会话中途 WS 断开/心跳失败；
  - 网关返回致命 `error`。
- 回退流程（单飞）：
  1. `BackendController` 进入 `SWITCHING`，拒绝新连接；
  2. `old.close("fallback")`，最多等 1 s `session.closed`；强制 flush AudioEngine，清空播放/发送队列；
  3. 创建 DoubaoBackend，使用豆包独立配置；
  4. 若 greeting 尚未成功/被丢弃，重置 `firstGreetingPending=true`，连接成功后补发；
  5. UI 显示“已临时回退豆包”；记录 event（backend/fallback/reason），不写敏感信息。
- 下一次连接（下一次唤醒/重连）仍按 auto 先试 EchoBridge；可加 30–60 s 运行时冷却，避免每次重连都等 5 s。冷却不持久化。
- 防止双连接/幽灵音频：
  - `SWITCHING` 期间只允许一个 close/open；
  - 旧 socket 的 generation 全部失效；
  - AudioEngine 在 close 时 flush + stop，再启动新后端；
  - 预览与主会话互斥；
  - 10 次连续切换测试必须无重叠音频、无残留 socket。

### 9.4 设置页/试听/重连/休眠

- 设置页新增“语音后端”：豆包 / EchoBridge / 自动；豆包与 EchoBridge 各自配置卡片；强制豆包开关。
- 设置页试听必须走 activeBackend：
  - `doubao` → 豆包 voiceId；
  - `echobridge` → localVoiceId；
  - `auto` → 按 effectiveBackend；若 EchoBridge 失败，试听是否回退豆包需产品确认（见 §12）。默认建议：试听不回退，提示“EchoBridge 不可用”，避免在设置页静默烧豆包。
- 启动自动连接、休眠恢复、`reconnectVoice` 管理动作、`selfTest` 语音相关路径全部走 `VoiceBackendController`。
- `openVoiceSession` 的日志 model 改为动态：`doubao-1.2.6.1` 或 `echobridge-local-v1`；SQLite 表结构不变，event detail 增加 `backend`。
- 保持现有 `previewVoice` 的 UI 行为（主会话暂停/恢复），但底层客户端由控制器创建。

### 9.5 设备端验收（Phase 5）

- 豆包 → EchoBridge → 豆包连续 10 次切换：无双连接、无幽灵音频、配置不丢。
- 预览：主会话暂停/恢复；不出现两个 TTS 同时播放。
- 重启 App：activeBackend/两套配置保持；默认仍可回豆包。
- 断网、EchoBridge 挂掉、豆包 key 失效：分别验证错误提示/回退/不崩溃。
- 休眠/无人断连：EchoBridge 收到 `session.close`，上游 LLM/TTS 任务取消，模型仍热。
- SQLite：对话、工具、事件、打断记录不回归；不记录音频/base64/key。

---

## 10. 验收指标

| 指标 | 目标 | 测量点 |
|---|---|---|
| 首音 p50 | < 1.5 s | 用户语音结束（VAD endpoint）→ 设备收到第一个 audio delta/开始播放 |
| 首音 p95 | < 2.5 s | 同上，30+ 轮 |
| ASR final | < 500 ms | 最后一个 speech 帧 → `transcription.completed` |
| TTS 首包 | < 500 ms | 文本块送入 worker → 第一个 PCM chunk |
| TTS RTF | < 0.5 | 生成音频时长 / 墙钟时间 |
| 连续 10 分钟 TTS | 无爆音/断帧/累积延迟 | 长文本循环，设备扬声器 |
| 打断 | < 300 ms | 设备发 `response.cancel` → 收到 `response.canceled`；之后无旧音频 |
| 30 轮 | 无状态错乱 | 普通聊天/工具/视觉/打断/工具失败/空 ASR |
| greeting | 正常 | 连接后主动问候，不重复/不双声 |
| 工具 | 正常 | 单/多工具、失败、超时、cancel |
| 视觉问答 | 正常 | `look_and_ask` → 设备拍照 → 视觉结果 → TTS |
| 无人/休眠断连 | 正常 | WS 关闭、上游取消、无残留音频/费用 |
| 显存 | 1 h 稳定 | ASR+TTS 常驻，无增长 |
| 24 h | 无泄漏/崩溃 | RSS/VRAM/句柄/队列 |
| 回退 | 可用 | EchoBridge 挂掉 → 豆包；配置不丢 |
| 语音质量 | 主观达标 | 原创鲸鱼娘音色；不看豆包相似度 |

---

## 11. 阶段计划

### Phase 0：现状审计与基线（本次）

- 读两个仓库 README、`docs/ARCHITECTURE.md`、`VoiceSession.java`、`AudioEngine.java`、`ToolRegistry.java`、`SettingsStore.java`、`MainActivity.java`、`ExpertClient.java`。
- 输出本 `docs/PLAN.md`：协议字段表、时序、模块、风险、验收、阶段。
- 测 DeepSeek Flash TTFT、Tailscale 直连/DERP、3070 显存、设备/ADB 状态。
- 不改实现代码；提交 `docs/PLAN.md`。
- 退出标准：计划评审通过；阻塞问题确认。

### Phase 1：协议网关 + Mock 后端

- 实现 WSS server、token auth、单会话、`session.create/created`、20 ms 音频接收、greeting、mock ASR/LLM/TTS、mock tool_call、`response.cancel`、`session.close`。
- 写协议测试客户端；用 mock 后端做事件顺序/取消/超时测试。
- X10A 真机跑通：连接、打招呼、一轮对话、打断、休眠断连。
- 退出标准：真机无崩溃；cancel 后无残留；`session.close` 1 s 内完成；日志无敏感信息。
- 提交：`phase 1: protocol gateway + mock backend`（含测试结果）。

### Phase 2：ASR

- Silero VAD + endpointing（600 ms 静音、200 ms min、15 s max、300 ms pre-roll）。
- Qwen3-ASR-0.6B transformers 分段推理；16k；started/completed/failed；可选 delta。
- 设备实录音：安静/电视/远场 1–3 m；记录 WER/CER、端点后延迟。
- 退出标准：ASR final p95 < 500 ms；空 ASR 回 LISTENING；不调用 LLM。
- 提交：`phase 2: Silero VAD + Qwen3-ASR`。

### Phase 3：TTS

- 离线 1.7B VoiceDesign 生成鲸鱼娘参考音频；常驻 0.6B Base 流式克隆；缓存 prompt。
- 24k 输出；20 ms pacing；按句/逗号切分；首句短；清理 Markdown/emoji/URL。
- 测首包、RTF、长句/数字/中英混、连续 10 分钟无爆音；主观音色评分。
- 若 0.6B 不达标：先试 0.6B CustomVoice Serena/Vivian；再做 1.7B Base BF16 实验，不阻塞 v1。
- 退出标准：TTS 首包 p95 < 500 ms（或明确记录偏差并给出修复计划）；RTF < 0.5；10 分钟无爆音。
- 提交：`phase 3: local Qwen3-TTS`。

### Phase 4：LLM + 工具

- DeepSeek Flash 流式；OpenAI tools；过滤 ask_expert；tool_call → App → tool result → 继续。
- reasoning_content 过滤；首 token/总超时；429 重试；空 content 不送 TTS。
- 主对话 prompt 追加实时语音约束。
- 30 轮：普通聊天、工具、视觉问答、打断、工具失败、空 ASR。
- 退出标准：30 轮无状态错乱；首音 p50/p95 达标或记录；工具多调用正确。
- 提交：`phase 4: DeepSeek streaming + tools`。

### Phase 5：设备端集成 + 回退

- `VoiceClient` 接口、DoubaoBackend、EchoBridgeBackend、VoiceBackendController。
- `voiceBackend/voiceWsUrl/voiceToken/localVoiceId` 配置；设置页；试听；启动/休眠/重连走 activeBackend。
- EchoBridge 失败/超时/中途断开回退豆包；强制豆包开关；10 次切换测试。
- 真机验收；SQLite/工具/视觉不回归。
- 退出标准：双后端切换无幽灵音频；默认仍可切回豆包；设置页试听不烧豆包（EchoBridge 模式）。
- 提交：`phase 5: dual backend + fallback`。

### Phase 6：部署 + 压测 + 验收

- WSL2/原生 Ubuntu 部署脚本；systemd 或任务计划；Tailscale Serve；预加载/warmup；watchdog；单会话；监控。
- 1 h/24 h 稳定性；A/B 30 条语音；输出 `docs/BENCHMARK.md` 和真机验收记录。
- 退出标准：所有 §10 指标达标或有明确豁免；EchoBridge 挂掉可回退；无公网暴露。
- 提交：`phase 6: deployment + benchmark`。

---

## 12. 风险与待确认

### 12.1 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| TTS 官方 Python API 无 token 级流式 | 首包 <500 ms 可能不达标 | 短句级流式 + 短首句 + warmup；必要时 token 级 decode spike；CustomVoice 对照 |
| DeepSeek 首 content 1.1–2.2 s | p95 首音逼近 2.5 s | 真实 prompt 复测；首句短；TTS 低首包；减少网络跳数；必要时产品接受 p95 偏差 |
| 8 GB 显存 | ASR+TTS+桌面 OOM | BF16、独立进程、1.7B 离线卸载、监控；OOM 先降 TTS 0.6B/ASR 分段/关桌面 GPU，不盲目量化 |
| 当前主机可能不是生产 24/7 主机 | 基准/部署不一致 | 确认目标主机；Phase 2/3 在目标 GPU 上跑 |
| 半双工无 AEC | 播放期无法真打断 | 设备本地 RMS cancel；网关只认 response.cancel；不做 AEC |
| 远场/电视噪声 | VAD 误触发/WER 高 | 双条件 VAD、600 ms 端点、pre-roll；Phase 2 调参/录音 |
| 工具等待/超时 | App pending 卡死 | 25 s 超时发 response.canceled；不提前开新音频 |
| 双后端切换 | 双连接/幽灵音频/配置丢失 | 单飞控制器、close→open、generation 失效、10 次测试 |
| Tailscale Serve 端口冲突 | WSS 不可用 | 使用独立 8443/路径；仅 tailnet；不 Funnel |
| 设备 WebView 66 | 设置页 JS 不兼容 | 只用 ES2017；不引入新框架/CDN |
| 模型/参考音频许可 | 不能分发 | Qwen Apache-2.0；参考音频原创；模型/音频/日志不入 repo |
| EchoBridge 成本/可用性 | 回退豆包仍计费 | auto 冷却、强制豆包开关、监控回退次数 |

### 12.2 真正阻塞的问题（需确认）

1. **生产主机到底是哪台？** 当前会话在 `360W-PC-03`（work-pc，RTX 3070，WSL2），但 x10a 的 LAN `192.168.31.188` 从这台机器不可达，只能走 Tailscale 直连（12 ms）。请确认 Phase 2/3 的模型基准和最终 24/7 部署是这台机器，还是 `home-pc`/另一台同 LAN 的 3070 主机。目标主机决定 GPU 占用、模型缓存路径、systemd/网络方案和验收数据。
2. **TTS “流式”的定义。** 官方 `qwen-tts` 0.6B Base 当前返回完整波形，没有 token 级增量 PCM；v1 先做“短句级流式”（首句 8–15 字，合成完立即发）。若首包 <500 ms 达不到，是否接受短句级作为 v1，还是必须在本期做 token 级解码改造？这直接决定 Phase 3 的范围和风险。
3. **试听在 EchoBridge 不可用时是否允许回退豆包？** 需求一方面要求“试听必须走新后端，否则继续烧豆包”，另一方面要求 EchoBridge 失败回退豆包。建议：试听不回退，提示“EchoBridge 不可用”；主对话按 auto 回退。请确认。
4. **设备实录音样本的使用许可。** Phase 2 需要安静/电视/远场 1–3 m 的真实录音来算 WER/CER；这些音频可能包含人声/室内环境。是否允许在设备现场录制并只保存在仓库外（不入库、不上传）？还是只用合成/公开音频？
5. **第二连接策略。** 需求允许“拒绝或排队”。建议：不同设备/不同 token 拒绝；同一设备快速重连在关闭旧会话后接管。请确认是否接受“同设备接管”，否则严格拒绝可能导致重连时短暂回退豆包。

---

## 13. 提交与验证纪律

- 每阶段结束：跑测试 → `git status` → `git add` → `git commit`；commit message 包含 phase 和验证结果；不跨阶段混提交。
- 子代理可用于仓库审计、协议核对、ASR/TTS benchmark、测试脚本、部署脚本、文档；子代理不直接提交主分支，主代理 review 后合并。
- 不提交：API key、token、模型权重、参考音频、设备录音、完整日志、base64、`.env`、`local.properties`。
- 模型/缓存/参考音频/日志放仓库外；`.gitignore` 在 Phase 1 建立。
- 所有服务只绑 `127.0.0.1` 或 Tailscale；禁止 Funnel/公网端口转发。
- 遇到 8 GB OOM：先降 TTS 到 0.6B、ASR 分段、关闭桌面 GPU 占用；不要盲目 GGUF Q4/Q5。

---

## 14. 参考资料

- 明月 Live 协议实测：`mingyue-live/docs/ARCHITECTURE.md` §豆包协议要点、`VoiceSession.java`。
- Qwen3-ASR-0.6B：`https://huggingface.co/Qwen/Qwen3-ASR-0.6B`（transformers/vLLM，0.6B/1.7B，流式仅 vLLM）。
- Qwen3-TTS-12Hz-0.6B-Base：`https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`（`qwen-tts`，`create_voice_clone_prompt`，24 kHz）。
- Qwen3-TTS-12Hz-1.7B-VoiceDesign：`https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`。
- Qwen3-TTS-12Hz-0.6B-CustomVoice：`https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`（Serena/Vivian 兜底）。
- DeepSeek/opencode：`https://opencode.ai/inference/openai/v1/chat/completions`（`User-Agent: curl/8.0`）。
