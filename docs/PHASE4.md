# Phase 4：DeepSeek 流式 + 工具

状态：**主机侧完成；首音 p50 未达标（1.84 s vs 1.5 s），p95 达标**
日期：2026-10-27
范围：DeepSeek Flash 流式、OpenAI tools、工具结果回填、reasoning 过滤、30 轮状态、工具/视觉转发。
设备侧：**未播放/未录音**；X10A 保持息屏。

## 1. 实现

- `DeepSeekLLM`：OpenAI 兼容流式客户端。
  - `POST https://opencode.ai/inference/openai/v1/chat/completions`
  - `model=deepseek-v4.1-flash`，`reasoning_effort=low`
  - `Authorization: Bearer <gateway key>`；`User-Agent: curl/8.0`
  - `reasoning_content` 完全丢弃，只把 `content` 送给文本/TTS。
  - SSE 工具调用按 `index` 聚合 `id/name/arguments`。
  - 首 content 超时 8 s；总超时 30 s；429/5xx 指数退避重试；4xx 不重试。
  - 空 content 不送 TTS；无 tool 时直接 `response.done`。
- 工具：
  - 火山 `{type,name,description,parameters}` → OpenAI `{type:"function",function:{...}}`。
  - `ask_expert` 从 tools 中过滤；若模型幻觉调用，本地吞掉并补一条不可用 tool 结果。
  - `look_and_ask`、`set_emotion`、`play_animation`、`set_prop`、`get_time`、`go_to_sleep`、`end_conversation` 保留。
  - App 回 `conversation.item.create` 后，把 assistant(tool_calls) + role=tool 加入上下文，再发下一次 LLM；新 response 使用新 `response_id`。
  - 多工具调用一次 `items[]` 下发；App 合并回传；测试覆盖 2 个工具。
  - 25 s 工具超时 → `response.canceled`，清理 App pending。
- 主 prompt：以 `session.create.instructions` 为底，追加“实时语音约束”：1–2 句、≤60 字、口语化、无 Markdown/列表/括号动作、不提专家/工具/照片、需要看画面时调用 `look_and_ask`、不要 `ask_expert`。
- 会话状态：
  - `_run_deepseek_turn` 最多 3 个 tool hop。
  - 文本先流式送 App + TTS，音频结束后 `text.done`，最后 `response.done`。
  - tool-call response 先发 function event，等 App 结果，发 `response.done(status=tool_calls)`，再用新 generation 继续生成最终语音。
  - 30 轮测试后 history 截断，保留 tool 链顺序。

## 2. 实测（主机侧，无播放）

### 2.1 5 轮真实 ASR + DeepSeek + vLLM TTS

同一个 WS 会话，公开 4.2 s 语音，真实 ASR/TTS，DeepSeek：

| turn | 首个文本（语音结束起） | 首音 | 总时长 |
|---|---|---|---|
| 1 | 1675.8 ms | 1978.1 ms | 7195.6 ms |
| 2 | 1362.6 ms | 1641.4 ms | 8135.4 ms |
| 3 | 1577.2 ms | 1836.3 ms | 8300.1 ms |
| 4 | 1480.8 ms | 1674.4 ms | 7724.9 ms |
| 5 | 1714.8 ms | 1936.0 ms | 8258.0 ms |

- 首音 p50 **1.84 s**，p95 **1.98 s**（5 轮）。
- **p95 达标（<2.5 s）；p50 未达标（目标 <1.5 s）**。
- 首个文本 1.36–1.71 s（含 500 ms endpoint + ASR + DeepSeek 首 content）。
- DeepSeek 首 content 在同一会话后续轮次约 0.2–0.5 s（provider prefix cache/上下文复用）；第一轮约 0.8–0.9 s。
- TTS 首包约 0.3 s；ASR final 约 0.3 s；endpoint 0.5 s。

### 2.2 30 轮状态

临时 gateway：mock ASR/energy VAD/mock TTS + 真实 DeepSeek，同一会话 30 轮：

```text
1 1.94 ok
2 2.18 ok
...
29 2.75 ok
30 3.26 ok
DONE
```

- 30/30 完成，无状态错乱/空响应/异常。
- 该测试不包含 TTS/VAD/ASR 模型延迟；只验证 DeepSeek 上下文、turn 状态、history 截断。

### 2.3 工具

| 场景 | 结果 |
|---|---|
| `set_emotion` 单工具 | DeepSeek 调用 → App 回 ok → 新 response 最终语音 |
| `set_emotion` + `play_animation` 多工具 | 一个 `function_call_arguments.done` 含 2 个 items；App 合并回传；最终语音 |
| `look_and_ask` | 工具转发；App 返回结果；DeepSeek 继续生成（未接真实相机，只验证转发） |
| `ask_expert` | 从 OpenAI tools 过滤；单元测试覆盖 |
| tool 超时 | 代码回 `response.canceled`；未做 25 s 长测 |

工具调用后，App 的第一条 tool-call response 为 `response.done status=tool_calls`；最终回答使用 `resp_2` 及新的 `response.done status=completed`。客户端测试已按此顺序等待。

### 2.4 reasoning 过滤

- DeepSeek 可能同时返回 `reasoning_content`；gateway 只转发 `delta.content`。
- 日志只记录首 content 时间、tool 数、总耗时；不记录推理内容/完整用户文本。

## 3. 验收判定

| 指标 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 首音 p50 | <1.5 s | 1.84 s | **未达标** |
| 首音 p95 | <2.5 s | 1.98 s | 达标（5 轮样本） |
| 30 轮 | 无状态错乱 | 30/30 | 达标 |
| 工具单/多 | 正常 | 通过 | 达标 |
| 视觉工具转发 | 正常 | 通过（假结果） | 达标 |
| reasoning 过滤 | 不朗读 | 已过滤 | 达标 |
| 空 content | 不送 TTS | 代码分支 | 未专项触发 |
| 429/首 token 超时 | 重试/兜底 | 代码分支 | 未专项故障注入 |
| 真机 | Phase 5 | 未做 | 待设备 |

## 4. 首音 p50 未达标的原因

- endpoint 500 ms + ASR final ≈300 ms + DeepSeek 首 content 0.5–0.9 s + TTS 首包 ≈0.3 s = **1.6–2.0 s**。
- 目标 1.5 s 需要在当前链路上再省 ≈0.35 s。可选优化：
  1. TTS 预计算自定义 voice / 更短首句，首包再降 100–150 ms；
  2. ASR 提前到 300 ms 静音投机（已部分做），再省 ~100 ms；
  3. DeepSeek 首 content 预热/前缀缓存（同会话已明显生效）；
  4. **短确认音**（如缓存的“嗯，我想想”）在 ASR final 后立即播放，把 perceived 首音压到 <1 s，但真实回答仍按正常链路到达；
  5. 接受 p50 1.8 s，只保留 p95 <2.5 s。
- 不建议未经确认把“确认音”当作正式答复；需要产品确认。

## 5. 配置

```env
ECHOBRIDGE_LLM_BACKEND=deepseek
ECHOBRIDGE_LLM_URL=https://opencode.ai/inference/openai/v1/chat/completions
ECHOBRIDGE_LLM_MODEL=deepseek-v4.1-flash
ECHOBRIDGE_LLM_API_KEY=...        # 只放 var/llm.env 或 /etc/echobridge，不入库
ECHOBRIDGE_LLM_MAX_TOKENS=400
ECHOBRIDGE_LLM_TEMPERATURE=0.6
ECHOBRIDGE_LLM_FIRST_TOKEN_TIMEOUT_S=8
ECHOBRIDGE_LLM_TOTAL_TIMEOUT_S=30
```

- `var/llm.env` 已加入 `.gitignore`；本机测试从 `mingyue-live/local.properties` 生成，权限 600。
- 日志/提交中无 API key。

## 6. 测试

```bash
bash scripts/test.sh
# 25 passed
```

新增手动验证：

- 5 轮真实 ASR + DeepSeek + vLLM TTS（`/tmp/e2e_5turns.py`，不播放）；
- 30 轮 mock ASR/TTS + DeepSeek；
- 单/多工具、vision 工具、工具结果续答；
- 真机未参与。

## 7. 下一步

1. 产品确认首音 p50 1.84 s 是否接受，或引入确认音/进一步预计算。
2. Phase 5：设备端 `VoiceClient` 双后端、设置页、回退豆包；等待用户醒后做真机/试听。
3. Phase 6：部署、24h、A/B。
