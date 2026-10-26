# Phase 1：协议网关 + Mock 后端

状态：**完成（待评审）**

日期：2026-10-26
范围：EchoBridge 网关、mock ASR/LLM/TTS、协议测试客户端、X10A 真机网关/音频/取消/断连验证。

## 1. 已实现

- FastAPI + uvicorn WebSocket 服务，路径 `/echobridge/ws`。
- Bearer token 鉴权；浏览器探针支持 `Sec-WebSocket-Protocol: echobridge, token.<token>`，token 不进入 URL 查询/访问日志。
- 单设备单会话：
  - 不同 device id 返回 1013 busy；
  - 同一 device id 快速重连时关闭旧会话后接管。
- 豆包子集下行事件：
  - `session.created` / `session.updated`（可发）
  - ASR `started` / `delta` / `completed` / `failed`
  - `response.output_text.delta` / `done`
  - `response.output_audio.started` / `delta` / `done`
  - `response.function_call_arguments.done`（`items[]`）
  - `response.done` / `response.canceled`
  - `error` / `session.closed`
- 上行事件：
  - `session.create`
  - `input_audio_buffer.append`（16 kHz PCM16，20 ms/640 B base64）
  - `speech_text_buffer.append` / `commit`（按 `speech_id` 合并，TTS-only）
  - `conversation.item.create`（工具结果）
  - `response.cancel`
  - `session.close`
- Mock 行为：
  - 能量 VAD：起音 40 ms，端点 600 ms，min speech 200 ms，max 15 s；
  - mock ASR 文本可通过查询参数 `asr_text` 指定（仅 dev/test）；
  - mock LLM 流式文本；
  - mock TTS 生成 24 kHz 音调序列（非语音，仅验证链路/节奏/播放）；
  - 工具触发：`笑一个` → `set_emotion`；`转个圈` → `play_animation`；`几点` → `get_time`；`笑一个，转个圈` → 两个工具批量；
  - 工具调用后等待 `conversation.item.create`，再生成最终语音；
  - 25 s 工具超时 → `response.canceled`，避免 App pending 卡死。
- 音频 pacer：
  - 24 kHz PCM16，20 ms/960 B；
  - `started` 在第一个 delta 前；
  - `audio.done` 后 `text.done`，再 `response.done`；
  - 有界队列/背压；
  - cancel 后旧 generation 不再发送。
- 测试：
  - 23 个 pytest（协议、音频、工具、session、单会话、WebSocket 集成）；
  - 独立协议测试客户端 `scripts/ws_client.py`。

## 2. 代码结构

```text
src/echobridge/
  app.py            FastAPI + WSS + /healthz + /probe
  __main__.py       python -m echobridge
  config.py         env 配置
  protocol.py       豆包子集编解码
  session.py        会话/状态机/工具/取消
  manager.py        单设备单会话/接管
  vad.py            Phase 1 能量 VAD
  llm.py            mock LLM
  tts.py            mock TTS
  pacer.py          20 ms 音频发送
  audio.py          PCM/base64/音调/WAV
  tools.py          火山 → OpenAI tools + ask_expert 过滤
tests/
  test_protocol.py
  test_audio.py
  test_tools.py
  test_session.py
  test_ws_integration.py
  test_single_session.py
scripts/
  dev.sh
  test.sh
  ws_client.py
  windows_tcp_proxy.py   # 仅 WSL2 + Windows Tailscale 开发用
```

## 3. 本地运行（WSL2 开发）

```bash
# 一次性
uv venv ~/.venvs/echobridge --python 3.12
uv pip install --python ~/.venvs/echobridge/bin/python -e '.[dev]'

# 启动网关；首次会生成 var/dev-token（gitignored）
bash scripts/dev.sh

# 测试
bash scripts/test.sh
```

默认绑定 `0.0.0.0:18787`，使用 `18787` 是为了避开当前 Windows 主机上已占用的 `8787`。

### 3.1 Windows + WSL2 + Tailscale Serve

WSL2 的 `127.0.0.1` 不保证被 Windows Tailscale 守护进程访问。开发时用 TCP 桥：

```powershell
# Windows -> WSL2
python E:\Projects\EchoBridge\scripts\windows_tcp_proxy.py --port 18787

# Tailscale Serve（仅 tailnet）
tailscale serve --bg --https=8443 http://127.0.0.1:18787
```

WSS 地址：

```text
wss://work-pc.tailea818a.ts.net:8443/echobridge/ws
```

> 生产环境（原生 Ubuntu）不需要 `windows_tcp_proxy.py`；gateway 直接监听 `127.0.0.1`，由 Tailscale Serve 代理。

### 3.2 真机探针

X10A 目前不能解析 MagicDNS 主机名，因此 Phase 1 探针走 Tailscale IP 的 HTTP（Tailscale 本身加密）：

```powershell
# 第二个桥，只绑定 Tailscale IP；端口 18788
python E:\Projects\EchoBridge\scripts\windows_tcp_proxy.py `
  --listen 100.83.182.104 --port 18788 `
  --target <WSL2-IP> --target-port 18787
```

在 X10A 浏览器打开：

```text
http://100.83.182.104:18788/probe#token=<var/dev-token 内容>
```

探针支持：连接、打招呼、模拟 1 s 说话 + 0.8 s 静音、打断、关闭会话、工具调用自动回执。

## 4. 测试结果

### 4.1 自动化

```text
23 passed in 4.32s
```

覆盖：

- `session.created` 后 AudioEngine 启动语义（协议层）；
- greeting 只发音频，不发 `output_text.*`；
- ASR started/delta/completed 顺序；
- 空 ASR → `transcription.failed`，不调用 LLM；
- 文本/音频/`text.done`/`response.done` 顺序；
- `audio.done` 在 `text.done` 前（保证打断落库）；
- cancel 后旧 generation 无 audio/response.done；
- 单/多工具调用、工具结果回传、最终语音；
- token 拒绝、healthz、probe、单设备接管、不同设备 1013。

### 4.2 真机（X10A，2026-10-26）

| 项目 | 结果 | 证据 |
|---|---|---|
| 连接 | 通过 | `session.created sess_a22254aad88e`，页面状态“会话已建立” |
| 打招呼 | 通过 | `resp_1`，29 个 `audio.delta`，`audio.done`，`response.done status=completed` |
| 一轮对话 | 通过 | ASR `started`/`delta '你好呀'`/`completed '你好呀，溟月'`；`resp_2` 文本 8 个 delta、音频、`text.done '你好呀，我在呢。'`、`response.done` |
| 打断 | 通过 | 播放中点击“打断”：页面 `sent response.cancel` → `local audio flushed` → `response.canceled`；此后无旧 `audio.delta`；服务端记录 `response canceled response=resp_1 reason=client` |
| 工具调用 | 通过 | ASR `'溟月，笑一个'` → `response.function_call_arguments.done set_emotion` → 页面回 `conversation.item.create` → 最终文本“好呀，看我笑一个。” + 音频 + `response.done` |
| 休眠/关闭 | 通过 | 页面 `session.closed` + `websocket closed code=1000 reason=bye`；`/healthz` 的 `active=null` |
| 设备音频输出 | 通过 | 播放期间 `dumpsys audio` 显示 `state:started` 的 OpenSL ES AudioPlayer（WebView 探针），确认 AudioContext 输出到 STREAM_MUSIC |

> 真机 Phase 1 使用探针页面验证网关、Tailscale 路径、设备扬声器/音频时钟和事件顺序；**Java `VoiceSession` 的接入按计划在 Phase 5 完成**，避免在 Phase 1 混入设备端双后端改造。PC 协议客户端 `ws_client.py` 使用与 `VoiceSession.java` 相同的 `session.create`、greeting 拆分、640 B 上行、工具结果格式。

## 5. 已知限制 / 下阶段

- Mock TTS 是音调，不是语音；Phase 3 替换为 Qwen3-TTS。
- Phase 1 VAD 是能量门限，不是 Silero；Phase 2 替换。
- Mock ASR 文本通过 `asr_text` 查询参数指定；仅用于测试，Phase 2 移除。
- Mock LLM 是规则回复；Phase 4 替换为 DeepSeek。
- 真实 App 接入、设置页、`voiceBackend/voiceWsUrl/voiceToken`、回退豆包在 Phase 5。
- WSS 已通过 Tailscale Serve 验证；X10A 当前 MagicDNS 不解析，Phase 5 需要固定 `wss` 主机名/DNS 或使用 Tailscale 加密的直连 `ws://100.x`。
- `18787/18788` 是避开本机 8787 占用的开发端口；生产可用 8787 或 443/8443。

## 6. 安全

- 不提交 token/key/音频/日志；`var/`、`*.wav`、`*.log` 已 gitignore。
- token 不进入 URL query；浏览器探针用 WebSocket subprotocol，App 用 `Authorization: Bearer`。
- 日志只记录 session id、事件类型、字符数/字节数、generation；不记录 base64/音频全文/token。
- Tailscale Serve 仅 tailnet；不使用 Funnel/公网端口转发。
