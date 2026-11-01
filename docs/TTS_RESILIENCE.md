# TTS 故障：文字保留、历史连续与 WSL2 崩溃根因

本文记录 2026-11-01 10:34 的实测故障、根因证据，以及 EchoBridge 现在的
三层兜底。日志只引用时间、stage、错误类型和字符数，不包含 token、参考音频
或 base64。

## 1. 故障时间线（本机时间）

| 时间 | 事件 |
|---|---|
| 10-31 20:51 | TTS 最后一次正常合成；之后空闲约 13.7 h |
| 11-01 09:55:52 | **ASR** stage 也报 `torch.AcceleratorError: CUDA error: unknown error`，位置在 `qwen3_omni_moe_thinker.py:467` 的普通 forward，不在 pin_memory 行 |
| 09:56:15 | ASR 进程整体退出，systemd `Restart=always` 立即拉起 |
| 10:34:21 | 用户问天气；TTS stage-0 在 `prompt_embeds_builder.py:1083` 的 `torch.tensor(..., pin_memory=True)` 报同一个 `cudaErrorUnknown` |
| 10:34:26 | vLLM-Omni API server 返回 500；**API server 进程没有退出**，`/health` 持续 503 |
| 10:35:07 | 用户说“小鲸鱼”；上一轮没进 history，模型再次调用 `get_weather`，再次 500 |
| 10:35:42 | watchdog 连续 3 次失败后重启 TTS |
| 10:37:50 | 新 worker 恢复，首个请求 200 |

网关侧的后果：`_stream_speech` 在第二次合成时抛错，`session.py` 只发了
`response.done(status="failed")`，没有 `response.output_text.done`，也没有
把 assistant/tool 消息写入 history。

## 2. 根因

### 2.1 直接触发点：WSL2 上的 pinned memory

vLLM 在 WSL2 上默认关闭 `pin_memory`（`CudaPlatform.is_pin_memory_available()`
读取 `VLLM_WSL2_ENABLE_PIN_MEMORY`，默认 0）。但 vLLM-Omni 的 Qwen3-TTS
prompt builder 绕过了这个判断：

```python
use_async_h2d = device.type == "cuda"
packed_ids_cpu = torch.tensor(flat_ids, dtype=torch.long, pin_memory=use_async_h2d)
```

WSL2 的 pinned pool 很小；PyTorch 的 pinned caching allocator 又会保留已释放
的块。长驻 worker 每个请求都按不同文本长度申请 pinned 块，运行数天后可能
耗尽该池，错误以 `cudaErrorUnknown`（而不是 `cudaErrorMemoryAllocation`）
冒出来。崩溃行正好是下一次 pinned 分配。

### 2.2 平台层：CUDA context 已经不可用

ASR 在 09:55 的崩溃发生在普通 forward（`qwen3_omni_moe_thinker.py:467`），
说明不是同一个 `pin_memory` 行；两个独立的 CUDA 进程在同一时间段先后报
`cudaErrorUnknown`。实测环境为 RTX 3070 + WSL2，Windows 驱动 610.57.04 /
CUDA UMD 13.4，两个 worker 已连续运行 2 天以上。这符合 WSL2/DXG GPU 透传在
长时间空闲、宿主电源状态变化或驱动恢复后 context 失效的特征。context 一旦
丢失，进程内无法恢复；只能让进程退出并重建。

因此：
- 关掉 `pin_memory` 能移除 TTS 特有的触发点，降低再次触发的概率；
- 但它不能救活已经失效的 context，必须有进程级重启兜底。

### 2.3 放大器：API server 不退出 + watchdog 3×30 s

ASR 的 engine 死后整个进程退出，systemd 立即重启；TTS 的 vLLM-Omni API
server 却继续存活并返回 500，watchdog 要等 90 s 才重启，模型再加载约
2 min，形成 3.5 min 空窗。

## 3. 现在的兜底

### 3.1 网关层：先保文字，再报原因

`_stream_speech` 捕获 `TtsUnavailableError` 后：

1. 用最多 `ECHOBRIDGE_TTS_FAILURE_DRAIN_S`（默认 2 s）把 LLM 流剩余部分
   读完，收集完整的 `full` 和可能排在后面的 `tool_calls`；
2. 通过 `response.output_text.done` 发出已生成的完整文字；
3. 抛出异常，由 `_run_deepseek_turn` 写 history 并发送
   `response.done(status="failed", reason="tts_unavailable")`。

`reason` 是新增的可选字段；成功响应不携带它，旧客户端会忽略。`status`
取值集合没有变化，失败仍是 `failed`，不会伪装成 `stopped`。

如果 TTS 在工具执行前失败，但 `tool_calls` 已经拿到，网关仍会发送
`response.function_call_arguments.done`、等待 app 的 tool result，并把
assistant（文字 + tool_calls）和 tool 消息写入 history。下一轮模型知道
“已经查过”，不会重放 `get_weather`。

### 3.2 TTS worker：engine 死后主动退出

`scripts/tts_vllm.sh` 不再是裸 `exec`，而是：

- 启动 vLLM-Omni 子进程；
- 等 `/health` 第一次 200（启动期 503 不计失败）；
- 之后每 `ECHOBRIDGE_TTS_ENGINE_PROBE_S`（默认 5 s）检查一次；连续
  `ECHOBRIDGE_TTS_ENGINE_FAILURES`（默认 2）次 503 就 `SIGTERM` 子进程，
  让系统立即重启，而不是让 API server 一直返回 500。

`ECHOBRIDGE_TTS_ENGINE_SUPERVISOR=0` 可退回旧行为。

### 3.3 空闲 keepalive

`ECHOBRIDGE_TTS_KEEPALIVE_S`（默认 600，0 关闭）由 gateway 的后台任务执行：
只在没有活跃 response、且距上次合成超过间隔时，合成一句 `我在呢。` 并丢弃
PCM。不要改成单字 `嗯。`：实测 Qwen3-TTS 对该输入可能不输出 codec EOS，
最终撞上 192-token 上限（`Qwen3TTSCodecLimitError`），keepalive 会误报失败。
它不经过 pacer，也不带 `ResponseContext`，因此不会污染 first_audio /
backchannel 指标。失败只在状态从成功转为失败时记一次 warning，恢复时记一次
info。

如果 worker 在半夜已经坏了，keepalive 会在下一次探测时发现并让 supervisor
重启，而不是等用户早上第一句话才发现。

### 3.4 watchdog

`deploy/watchdog.sh` 默认 `ECHOBRIDGE_WATCHDOG_FAILURES=2`，30 s 定时器下
即 60 s 触发。它现在只是最后一道兜底：engine 死亡由 supervisor 在约 10 s
内处理。

## 4. 根因补丁与代价

`src/echobridge/vllm_omni_patch.py` 会在 TTS 启动时幂等地修改
`~/.cache/echobridge/src/vllm-omni`：

- `prompt_embeds_builder.py`：`use_async_h2d` 改为
  `device.type == "cuda" and is_pin_memory_available()`；
- `device_copy.py`：`to_device_nonblocking`、`index_to_device`、
  `DeviceStager` 同样遵守 vLLM 的 WSL2 判断。

在 RTX 3070 / WSL2 上对 96 个 text id 做 500 次 A/B（含
`torch.cuda.synchronize()`）：

| 方式 | median | p95 | mean |
|---|---:|---:|---:|
| `pin_memory=True` | 0.106 ms | 0.134 ms | 0.112 ms |
| `pin_memory=False` | 0.102 ms | 0.132 ms | 0.108 ms |

差异约 0.004 ms，低于测量噪声；这些张量只有几十到几百个 id，不是音频或
hidden states。因此首字延迟没有可感知代价。原生 Linux 上
`is_pin_memory_available()` 返回 True，行为不变。

补丁是 best-effort：upstream 改动导致匹配不到时只打印 warning，TTS 仍会
启动；需要严格失败时设 `ECHOBRIDGE_TTS_STRICT_PATCH=1`。

## 5. 部署与验证

```bash
# 应用源码改动后
systemctl --user restart echobridge-gateway echobridge-tts

# 确认补丁
PYTHONPATH=src ~/.venvs/echobridge-tts-vllm/bin/python \
  -m echobridge.vllm_omni_patch --check ~/.cache/echobridge/src/vllm-omni

# 确认 supervisor / keepalive
journalctl --user -u echobridge-tts -n 30
journalctl --user -u echobridge-gateway -n 50 | grep -E 'keepalive|tts unavailable'

# 故障演练后应看到：
#   response.output_text.done（文字保留）
#   response.done status=failed reason=tts_unavailable
#   history 中有 assistant(tool_calls) + tool
```

## 6. 已知边界

- `cudaErrorUnknown` 的根因在 WSL2/DXG/驱动层，EchoBridge 不能在进程内修复
  已经丢失的 context；只能尽早发现并重建 worker。
- 如果宿主 GPU 驱动彻底挂死，systemd 重启也可能反复失败；此时需要
  `wsl --shutdown` 或重启 Windows。keepalive 会在第一次失败时记录一条
  `tts keepalive failed`，watchdog 日志会显示反复重启。
- 工具执行仍由 app 负责。若 TTS 失败后 app 不返回 tool result，网关会写一条
  “工具调用未完成，请勿重复调用” 的占位 tool 消息，保证下一轮 OpenAI
  tool-call 链合法。
