# Phase 5：设备端双后端 + 回退 + 自然 backchannel

状态：**主机 + 真机验收完成；WSS 经 Windows Tailscale Serve 已验证（2026-10-27）；Phase 6 已完成部署与短基准，1 h/24 h 长稳豁免见 `docs/BENCHMARK.md` 与 `docs/ACCEPTANCE.md`**
日期：2026-10-27
范围：VoiceClient 接口、Doubao/EchoBridge 双后端、独立配置、自动回退、设置页、试听、10 次切换验收、智能 backchannel。

## 1. 设备端架构

```text
MainActivity
  └── VoiceBackendController (VoiceClient)
        ├── VoiceSession        = 豆包后端（原实现，协议行为不变）
        └── EchoBridgeBackend   = 同一实现 + EchoBridge 配置（Bearer/localVoiceId/backchannel）
              └── AudioEngine（每个后端独立，切换时先 close 再 open）
```

新增文件：

- `voice/VoiceClient.java`：统一接口与 Listener/State/ToolExecutor/TestCallback。
- `voice/VoiceBackendConfig.java`：不可变后端配置；豆包/ EchoBridge 两套默认值。
- `voice/EchoBridgeBackend.java`：EchoBridge 薄封装。
- `voice/VoiceBackendController.java`：单飞切换、auto 回退、5 s 连接超时、45 s 运行时冷却。
- `voice/BackchannelPlayer.java`：独立 AudioTrack，播放“嗯～”等短音，支持 120 ms 淡出与立即停止。

`VoiceSession` 只做了参数化（URL/鉴权头/model/voice/显示名）和 backchannel 事件支持；豆包的协议解析、半双工、cancel、工具链未改。

## 2. 配置

`SettingsStore` 新增（均有默认值，迁移不删除旧 key）：

```text
voice_backend_mode         doubao | echobridge | auto     默认 doubao
active_voice_backend       doubao | echobridge            持久化首选，回退不修改
force_doubao                bool                           默认 false

doubao_ws_url              默认 BuildConfig.VOLC_WS_URL
doubao_api_key             默认 BuildConfig.VOLC_API_KEY（设置页只写不读回显）
doubao_model               默认 1.2.6.1
voice_id                   豆包 voiceId（沿用原 key）

echobridge_ws_url          默认 BuildConfig.ECHOBRIDGE_WS_URL
echobridge_token           默认 BuildConfig.ECHOBRIDGE_TOKEN（只写不回显）
echobridge_voice_id        默认 mingyue_v1（别名 localVoiceId）
```

- 设置页：语音后端选择、强制豆包、豆包 URL/Key/Model/音色、EchoBridge URL/Token/本地音色、试听。
- 页面状态显示“当前生效：豆包/EchoBridge（已临时回退豆包）· 保存模式：…”。
- Debug 构建允许 `ws://`（Tailscale/LAN 测试）；release 仍禁止 cleartext，必须 `wss://`。
- 音量安全：`applyMediaVolume()` 只允许降低当前媒体音量，绝不提高。

## 3. 回退语义

| 模式 | 行为 |
|---|---|
| doubao | 只连豆包 |
| echobridge | 只连 EchoBridge；失败报错，不自动烧豆包 |
| auto（forceDoubao=false） | 先 EchoBridge；5 s 未连接/中途断开/致命 error → 本次连接临时切豆包 |
| auto（forceDoubao=true） | 永远豆包 |

- 回退只改内存 `runtimeBackend`，不写 `voice_backend_mode` / `active_voice_backend`。
- 记录：`onNotice("已临时回退豆包")` → toast + `LiveStatus.notice` + SQLite `connect` 事件（backend/mode）。
- 冷却 45 s 仅内存；下一轮重连/唤醒到期后仍先试 EchoBridge。
- 切换顺序：`old.setListener(null)` → `old.disconnect()`（flush+stop AudioEngine，最多等 1 s `session.closed`）→ 创建新后端 → connect。旧 generation 全部失效。
- 预览：主会话先关闭；`VoiceBackendController.createPreview()` 按 effective backend 创建临时客户端；EchoBridge 预览失败不回退豆包，只提示。
- `reconnectVoice` 管理动作清冷却并 `rebuild()`，重新按 auto 首选 EchoBridge。

## 4. Backchannel（自然确认音）

目标：让首音慢时听起来自然，不把确认音当正式答复，也不增加正式回答延迟。

### 4.1 主机侧

- `ECHOBRIDGE_BACKCHANNEL_ENABLED=1`（默认）。
- `ECHOBRIDGE_BACKCHANNEL_DELAY_MS=700`（可配置）。
- `ECHOBRIDGE_BACKCHANNEL_PHRASES="嗯～|唔～|我想想～"`（`|` 分隔）。
- 启动时 TTS warmup 预合成 greeting 和全部短音；`app.py` 改为全后端共享一个 TTS client，缓存不再随会话丢失。
- ASR final 后 700 ms，若正式 TTS 首个 chunk 尚未产生且本轮未发过短音：
  - 从预缓存取一条（排除上一轮用过的）；
  - 发独立事件 `response.backchannel`（带 `response_id/sound_id/format/delta`），**不进 pacer、不参与正式 response 的 audio/文本状态机**；
  - 每轮最多 1 条；工具 hop 后不重复。
- 正式音频一旦出首个 TTS chunk，后续不再发；已发出的短音由设备淡出。

### 4.2 设备侧

- `EchoBridgeBackend` 配置启用 `BackchannelPlayer`；豆包后端禁用。
- 收到 `response.backchannel`：若正式音频尚未开始、且不在 cancel 抑制期，独立 AudioTrack 播放（0.8 增益）。
- `response.output_audio.started` 到达：立即 120 ms 淡出短音；正式音频仍由原 AudioEngine 播放，不被短音阻塞。
- 用户 cancel/`response.canceled`/断开/新一轮起音：立即 `stopNow()`，并 3 s 抑制迟到短音。
- `response.done` 且本轮没有正式音频时，80 ms 淡出短音。
- 指标分开上报：
  - `backchannel_first_audio_ms`：ASR final 收到 → 短音开始播放。
  - `reply_first_audio_ms`：ASR final 收到 → 正式 audio.started。
  - `MainActivity` 写 SQLite `latency` 事件，`detail.metric` 区分；`LiveStatus` 分别保存。
- 主机日志分开记录 `backchannel sent ... after_asr_ms` 与 `tts first chunk formal ... after_asr_ms`。

## 5. 验证

### 5.1 主机

```bash
bash scripts/test.sh
# 29 passed
```

新增 `tests/test_backchannel.py`：

- 慢正式 TTS（800 ms）→ 50 ms 阈值时先发 backchannel，且早于 `response.output_audio.started`。
- 正式音频快（1 ms）→ 不发送 backchannel。
- 连续两轮 → sound_id 不重复。
- 工具 hop 一轮 → 最多 1 条 backchannel。

真实 vLLM TTS + 真实 ASR + DeepSeek 工具轮：

```text
backchannel sent response=resp_1 sound=我想想～ after_asr_ms=700.3
tts first chunk formal response=resp_1 after_asr_ms=2735.4 backchannel=1
response.backchannel count=1
```

（该轮 DeepSeek 先流 text 再 tool_call，正式首 chunk 2.7 s；短音在 0.7 s，未进入正式 pacer。）

### 5.2 设备构建

```bash
JAVA_HOME="E:\Program Files\Java\jdk-17" ./gradlew :app:assembleDebug
# BUILD SUCCESSFUL
```

APK：`app/build/outputs/apk/debug/app-debug.apk`。

### 5.3 真机验收（2026-10-27，X10A / Android 9 / Phh-Treble）

设备日常验收通过 `adb reverse tcp:18787 tcp:18787` 访问主机网关（`ws://127.0.0.1:18787/echobridge/ws`），避免把网关暴露到 LAN/公网；debug APK 允许 cleartext，release 仍只允许 WSS。2026-10-27 另外用 `wss://work-pc.tailea818a.ts.net:8443/echobridge/ws` 在 X10A 上完成了一次真实 WSS 握手与 `session.created` 验证（X10A 的 MagicDNS 由 `x10a-tailscale` Magisk 模块的 dnsmasq + netd 方案提供，系统 DNS 已能解析 `work-pc.tailea818a.ts.net`；详见 5.6）。

- [x] 默认 doubao：连接、greeting、真实 ASR、get_time 工具、TTS 正常。
- [x] EchoBridge：连接、greeting、真实一轮对话、get_time、set_emotion + play_animation 多工具、look_and_ask 视觉（识别正确）。
- [x] Backchannel：真实设备上听到“嗯～/唔～/我想想～”，设备侧首音 749–926 ms；正式回答首音仍单独统计。
- [x] 语音打断：提高音量触发本地 barge-in；说“停一下/不用了”被识别为 stop command，旧回答停止且不再生成新回答（ASR 可能截成“一下，不用讲了”，已兼容）。
- [x] 点击停止：TTS 播放中点击屏幕，立即 `response.cancel` → `response.canceled`。
- [x] 休眠：说“晚安”后等整句回答播完再进入休眠/断开（不再截断）；tap 唤醒并重连成功。
- [x] 自动回退：auto 模式连 EchoBridge，杀掉网关后本次连接回退豆包；界面提示“已临时回退豆包”，`voice_backend_mode=auto`、`activeBackend=echobridge` 未变，`runtimeBackend=doubao`；回退后豆包对话正常。
- [x] 10 次豆包 ↔ EchoBridge 切换：10/10 成功；日志顺序均为 `audio stopped` → 下一次 `audio started`，无并存 socket/幽灵音频；两套配置保留。
- [x] 设置页试听：EchoBridge 生效时走本地音色，豆包生效时走豆包音色；试听结束主会话自动恢复，无双声。
- [x] SQLite：`PRAGMA integrity_check=ok`；消息/工具/事件正常增长；`latency` 事件分别记录 `backchannel_first_audio_ms` 与 `reply_first_audio_ms`；未发现 token/Bearer/base64 事件。
- [x] 音量：全程保持当前系统音量 4/15；应用只允许降低、不提高。

实测延迟（设备侧，ASR final → 首音）：

```text
backchannel_first_audio_ms: 749, 752, 753, 777, 785, 842, 853, 926 ms
reply_first_audio_ms 普通轮: 1.3–2.2 s
reply_first_audio_ms 工具轮: 2.2–6.7 s（含 get_time / look_and_ask / 工具 hop）
豆包 fallback 轮: 1.5–1.6 s
```

一次 DeepSeek 工具 hop 出现 17.8 s 的 provider 长尾；backchannel 在 0.7 s 播出后，正式答案 17.8 s 才到。复测同场景 2.2 s，属上游偶发长尾，已记录。

### 5.4 真机阶段发现并修复

1. **AudioTrack 初始化失败**：Android 9/MTK 拒绝 `AudioTrack.Builder + MODE_STATIC`；改为与 AudioEngine 相同的 legacy `MODE_STREAM` 构造，backchannel 正常出声。
2. **Backchannel 缓存被 LRU 挤掉**：正式句子缓存会驱逐预热的短音；新增独立 pinned `_backchannel_cache`，普通 LRU 再多也不影响短音。
3. **Barge-in 阈值过高**：原 9000 在低音量下无法触发；改为回采自适应阈值 `max(900, echoEMA × 1.8)`，350 ms 预热，实测 977/1490 RMS 可触发。
4. **Stop 片段**：半双工截掉“停一”，ASR 只剩“一下，不用讲了”；`is_stop_command` 兼容前导碎片和句尾 stop 短语，stop 只发 `response.done(status=stopped)`，不生成新回答、不播 backchannel。
5. **点击停止**：TTS 播放中点击屏幕立即取消；比提高音量更适合办公室/低音量场景。
6. **晚安截断**：`go_to_sleep` 原本 2.5 s 后休眠，会截断正在生成的回答；改为等待本轮 SPEAKING/THINKING 结束 + 900 ms 余量再休眠（60 s 上限）。`end_conversation` 同样处理。
7. **本地时区**：`get_time` 原先硬编码 Asia/Shanghai；改为设备本地时区，X10A 为 Europe/Budapest，回答 17:xx 正确。
8. **豆包工具重复调用防护**：曾出现一次 provider 重复 `function_call_arguments.done` 导致 get_time 循环；加入按 `call_id` 缓存结果、最多 3 次重复后 `response.cancel` 的保护；纯豆包和回退复测均正常。

### 5.5 仍未完成/注意

- **WSS 经 Windows Tailscale Serve 已验证**（见 5.6）；X10A 的系统级 MagicDNS 已由 `x10a-tailscale` Magisk 模块的 dnsmasq + netd 方案提供，WSS 日常可用；`adb reverse` + `ws://127.0.0.1:18787` 仍保留为备用。
- 默认值仍为 `doubao`；X10A 当前保存为 `echobridge` 用于 WSS 验收；auto 回退只在本次连接生效，未改保存值。
- 17.8 s 长尾是 provider 侧偶发，不是链路固定延迟；Phase 6 30+ 条 A/B 再统计。

### 5.6 WSS 403 的复现、根因与修复（2026-10-27）

**结论：Tailscale Serve 没有剥离 `Authorization` / `Sec-WebSocket-Protocol`；403 是 EchoBridge 自己的鉴权拒绝。** 修复前的手工探测使用了与运行中网关不一致的 token；设备当时保存 `voice_backend_mode=doubao`，并没有发起 EchoBridge 握手。网关日志里的 `Address(host='100.83.182.104', port=0)` 是 work-pc 本机（Serve 代理源），X10A 的源地址是 `100.83.39.115`。

#### 复现

1. 最小 WebSocket 回显服务器，只记录 header 名称/存在/长度；`token.<...>` 子协议替换为 `token.<redacted>`：

```bash
# WSL2
python scripts/ws_header_echo.py --host 0.0.0.0 --port 19443 --once
# Windows：WSL2 端口桥（TARGET 用 wsl hostname -I）
python scripts/windows_tcp_proxy.py --port 19443 --target <WSL2-IP> --target-port 19443
# Windows Tailscale Serve（仅 tailnet；验证完删除）
tailscale serve --bg --https=9443 http://127.0.0.1:19443
```

```bash
python scripts/ws_handshake_probe.py \
  --url wss://work-pc.tailea818a.ts.net:9443/ \
  --auth both --no-session-create --expect 101
```

回显结果：

```text
authorization_present=true authorization_len=39
subprotocol_len=50 subprotocol_items=["echobridge","token.<redacted>"]
tailscale_user_present=true x_forwarded_for_present=true
```

2. 用运行中网关的 token 经 8443 握手：

```bash
python scripts/ws_handshake_probe.py --auth bearer --expect 101
# result=101
# session.created id=sess_...

python scripts/ws_handshake_probe.py --auth subprotocol --expect 101
# result=101 subprotocol='echobridge'

python scripts/ws_handshake_probe.py --auth bearer --token wrong --expect 403
# result=403 Forbidden
```

网关新诊断（示例，token 值不会出现）：

```text
accepted websocket from Address(host='100.83.182.104', port=0) path=/echobridge/ws auth=bearer auth_len=39 subprotocol_len=0 ... tailscale_user=yes xff=yes
rejecting unauthorized websocket from ... auth=none auth_len=0 ... tailscale_user=yes xff=yes
rejecting unauthorized websocket from ... auth=bearer auth_len=12 ... tailscale_user=yes xff=yes
```

#### 根因

- `_extract_token` 只接受 `Authorization: Bearer` 或 `Sec-WebSocket-Protocol: ..., token.<token>`；没有或不匹配就 `ws.close(4401)`。Starlette 在 `accept()` 前关闭会表现为 HTTP 403，这是预期行为。
- 修复前的手工客户端没有携带运行中网关的 32 字符 `var/dev-token`（例如使用了占位值或旧 token），所以 403；同一 token 经 Serve 的 `Authorization` 和子协议两种方式均可 101。
- X10A 当时 `voice_backend_mode=doubao`、`active_voice_backend=doubao`，`echobridge_ws_url=ws://127.0.0.1:18787/...`，且没有 `adb reverse`；它没有发起 EchoBridge 握手。设备保存的 EchoBridge token 与 `var/dev-token` 一致（32 字符），不是 token 文件不一致。
- 若把豆包后端的 `doubao_ws_url` 指到 `/echobridge/ws`，设备会发 `X-Api-Key`；EchoBridge 按设计只认 Bearer/子协议，因此必然 403。设备端已加 fail-fast，不再把豆包 key 发给 EchoBridge。

#### 修复

- 服务端 `_auth_diagnostics`：接受/拒绝日志记录 `auth` 来源、`auth_len`、`subprotocol_len`、`origin/ua/device/tailscale_user/xff` 是否存在；不记录 header 值、token、Origin 值或 Tailscale 身份值。
- `scripts/ws_handshake_probe.py`：Bearer/子协议/两者/无凭据四种方式，支持 `--expect 101|403`，只打印 token 来源和长度。
- `scripts/ws_header_echo.py`：Serve 回显测试工具。
- 测试：`tests/test_ws_integration.py` 增加 bearer/子协议/两者/缺失/错误 token 矩阵，以及“日志不含 token/身份值”的断言；`37 passed`。
- 设备端：
  - HTTP 401/403 → 明确提示“EchoBridge 拒绝连接/鉴权失败（HTTP 403）...”，不再只显示 `Expected HTTP 101...`。
  - 豆包后端 + EchoBridge URL → 本地 fail-fast，提示切换 `echobridge`/`auto`，不发送 `X-Api-Key`。
  - App 不再内置 DNS 覆盖；X10A 的系统级 MagicDNS 由 `x10a-tailscale` Magisk 模块的 dnsmasq + netd 方案提供（模块外改动）。
  - `app/build.gradle`：URL 含 `token=` 直接构建失败；URL 已设置但 token 为空时构建警告。
- 没有修改 Tailscale Serve 配置；继续使用 Bearer；token 不进 URL query；没有 Funnel/公网端口转发；没有关闭鉴权。

#### 验证

| 客户端 | 路径 | 凭据 | 结果 |
|---|---|---|---|
| work-pc `100.83.182.104` | Serve WSS 8443 | Bearer 正确 | 101 + `session.created` |
| work-pc | Serve WSS 8443 | 子协议正确 | 101 + `session.created` |
| work-pc | Serve WSS 8443 | 两者 | 101 + `session.created` |
| work-pc | Serve WSS 8443 | 无/错误 | 403 |
| X10A `100.83.39.115` | Serve WSS 8443 | Bearer 正确 | 101 + `session.created` `sess_ff0c1190b8a0`，greeting 正常 |
| X10A | Serve WSS 8443 | Bearer 错误 | 403；设备显示明确鉴权错误 |
| X10A | doubao 后端 + EchoBridge URL | 未发送 X-Api-Key | 本地“配置冲突”，网关无请求 |

X10A 的 MagicDNS 原先不解析（`CorpDNS=false`；Android 应用走 netd，不读 `/etc/resolv.conf`）。WSS 真机验证时曾临时给 `/system/etc/hosts` 加 `100.83.182.104 work-pc.tailea818a.ts.net` 的 bind mount，验证后已卸载。随后在 `x10a-tailscale` Magisk 模块中加入 dnsmasq + netd 方案：dnsmasq 监听 `127.0.0.1:53`，`*.ts.net` 转发到 `100.100.100.100`，其余转发到当前 Wi-Fi DNS，并把 netid 100 的 resolver 指向 `127.0.0.1`。X10A 的系统 `getaddrinfo` 已能解析 `work-pc.tailea818a.ts.net`，无需 App 覆盖即可用 `wss://work-pc.tailea818a.ts.net:8443/echobridge/ws` 完成 `session.created`；普通公网 DNS 仍正常。模块不可用时仍可走 `adb reverse`。

#### 回滚

- EchoBridge：`git revert <commit>`；无数据库/配置迁移，回滚后只是少掉诊断日志和探针脚本。
- 设备：`git revert <commit>` 并重新构建安装；或安装上一版 APK。若只想回滚设备配置，把 `voice_backend_mode` 改回 `doubao`、`echobridge_ws_url` 改回 `ws://127.0.0.1:18787/echobridge/ws`。
- Serve 未改动，无需回滚。

## 6. 已知限制

- Backchannel 只使用 warmup 时 `ECHOBRIDGE_TTS_VOICE_ID` 的缓存；设备改 localVoiceId 后若未重新 warmup，短音自动跳过（不现场合成，避免增加延迟）。
- 默认值仍为 `doubao`；X10A 当前保存为 `echobridge`；是否把默认改为 auto 由产品验收后决定。
- 低音量下语音打断需要稍微提高音量（实测 900–1500 RMS）；点击屏幕停止始终可用。
- WSS 经 Windows Tailscale Serve 已验证；X10A 的系统级 MagicDNS 由 `x10a-tailscale` Magisk 模块的 dnsmasq + netd 方案提供，WSS 可直接使用；`adb reverse` / Tailscale 加密的 `ws://100.x`（debug）仍作为后备。不要用公网端口/Funnel 绕过。
- 17.8 s 工具 hop 长尾属 provider 偶发；Phase 6 用 30+ 条 A/B 统计。
