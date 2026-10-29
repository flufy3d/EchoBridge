# Phase 6 基准与验收测量

> **Phase 7 后续**：单进程 ASR + 瘦身 vLLM 的显存/延迟复测见
> [`docs/PHASE7.md`](PHASE7.md)。整卡显存 7473 → 6576 MiB；ASR after endpoint
> p50 908 ms、p95 1176 ms；正式首音 p95 2474 → 1559 ms。ASR 350–500 ms 目标
> 受 0.6B 模型本身吞吐限制，未完全达成，PHASE7 给出实测瓶颈与延迟优先配置。
>
> 状态：**30/30 轮主机受控基准完成；感知首音达标；正式首音与 ASR final 有明确豁免**。
> 日期：2026-10-27 22:24–22:30（CET）
> 范围：真实 Silero VAD + Qwen3-ASR + DeepSeek Flash + Qwen3-TTS（vLLM-Omni），单 WS 会话 30 轮，无播放。
> 原始数据：`var/bench/phase6.json`、`var/bench/tts_vllm.json`、`var/bench/asr_direct.json`（`var/` 不入库）。

## 1. 环境

| 项 | 值 |
|---|---|
| 主机 | `360W-PC-03`，Windows 11 + WSL2 Ubuntu 26.04 |
| CPU / RAM | Intel i7-5820K（6C/12T）/ 15 GiB |
| GPU | RTX 3070 8192 MiB，驱动 617.14，WSL2 CUDA |
| 网关 | EchoBridge `0.1.0`，Python 3.12.15 |
| ASR | Qwen3-ASR-0.6B，BF16，`127.0.0.1:18801` |
| TTS | Qwen3-TTS-12Hz-0.6B-Base，vLLM-Omni，`127.0.0.1:18802` |
| LLM | `deepseek-v4.1-flash`，`reasoning_effort=low`，流式 |
| VAD | Silero，endpoint 500 ms，pre-roll 300 ms，投机端点 150 ms |
| 对外 | Windows Tailscale Serve 8443 → `127.0.0.1:18787`，仅 tailnet；基准直连 localhost |

部署形态：`systemctl --user` 三个服务 + watchdog timer；Windows 登录任务 `EchoBridge WSL`；`Linger=yes`。详见 `docs/DEPLOY.md`。

## 2. 方法

- 同一个 WSS 会话先 greeting，再连续 30 轮；每轮等 `response.done` 后再发下一轮。
- 输入为 4 条 16 kHz 单声道 PCM16：`phase5_question` 1.771 s，`phase5_turn/turn3/turn4` 各截到 4.000 s；循环 30 次。
- 每轮以 20 ms/640 B 实时发送，尾部 800 ms 静音（500 ms endpoint + 余量）。
- tool call 由脚本自动回 `ok`；本轮仅第 1 轮触发一次 `get_time` hop。
- 不播放音频；只测网关到客户端的 PCM delta 到达时间。
- 统计使用 nearest-rank：p95 = 排序后第 `ceil(0.95n)` 个样本。

计时点：

| 名称 | 定义 |
|---|---|
| ASR final | 最后一个 speech 帧 → `transcription.completed` |
| ASR after endpoint | ASR final − 500 ms（估算；实际 endpoint 由 VAD 决定） |
| Formal first audio | `transcription.completed` → 第一个正式 `response.output_audio.delta` |
| Perceived first audio | `transcription.completed` → 最早出现的 backchannel 或正式 audio |
| Endpoint → audio | ASR after endpoint + 对应 audio 延迟（估算） |
| Cancel | `response.output_audio.started` 后发 `response.cancel` → `response.canceled` |

### 2.1 A/B 定义

本次 A/B 是**同一 30 轮的配对对比**：

- **A = 感知首音**：有 backchannel 时取短音到达时间，否则取正式首音；
- **B = 正式首音**：正式回复第一个 PCM delta。

不是“EchoBridge vs 豆包”的受控 A/B；豆包真机数据见 §5，属于历史混合样本，不能当作同批配对。

## 3. 30 轮结果

```text
turns: 30 / 30 ok, 0 failed
```

| 指标（ms） | n | mean | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| ASR final（最后一个 speech 帧起） | 30 | 1407.8 | 1358.2 | 1732.5 | 1799.5 | 1913.6 |
| ASR after endpoint（估算） | 30 | 907.8 | 858.2 | 1232.5 | 1299.5 | 1413.6 |
| 首个文本（ASR final 起） | 30 | 902.7 | 817.1 | 1118.6 | 1429.1 | 1905.2 |
| **B 正式首音（ASR final 起）** | 30 | 1461.7 | **1231.1** | 2159.7 | **2474.4** | 3869.9 |
| **A 感知首音（backchannel，ASR final 起）** | 30 | 707.1 | **706.5** | 711.1 | **711.5** | 711.6 |
| 整轮（最后一个 speech 帧 → done） | 30 | 9694.3 | 9275.6 | 11194.4 | 12995.9 | 13209.8 |

从 **VAD endpoint** 起算（估算）：

| 指标（ms） | p50 | p95 | max |
|---|---:|---:|---:|
| B 正式首音 | 2141.1 | 3443.4 | 4961.9 |
| A 感知首音 | 1560.9 | 2008.3 | 2117.1 |
| 首个文本 | 1727.8 | 2394.1 | 2733.4 |

A/B 配对差（B − A，ASR final 起）：p50 **524.6 ms**，p95 **1762.9 ms**。

### 3.1 Greeting / 打断 / 其他

| 项 | 结果 |
|---|---|
| Greeting 首音 | **0.9 ms**（greeting 已 pin 在 TTS 缓存；本轮重启后复测） |
| Greeting 总时长 | 1.8–3.4 s（缓存命中时首音几乎为 0） |
| 打断 5 次 | 3.6 / 4.8 / 7.9 / 5.4 / 5.3 ms，**0 条旧音频** |
| 工具 hop | 第 1 轮 `get_time`，最终回答正常完成 |
| 失败轮 | 0 |
| 单次最长正式首音 | 3869.9 ms（第 28 轮，provider/LLM 长尾；p95 不受影响） |

### 3.2 TTS 单独测量（vLLM-Omni，3 次/句）

| 文本 | 字数 | 音频 | 首包 p50 | 首包 max | RTF p50 |
|---|---:|---:|---:|---:|---:|
| 你好呀，我在呢。 | 8 | 2.00 s | 322.1 ms | 497.1 ms | 0.500 |
| 今天也要开开心心的呀。 | 11 | 2.80 s | 239.6 ms | 545.0 ms | 0.350 |
| 现在是晚上十一点，该准备休息啦。 | 16 | 2.80 s | 386.6 ms | 585.6 ms | 0.504 |
| 我看看哦，这个大概是一个蓝色的杯子。 | 18 | 3.12 s | 317.6 ms | 429.6 ms | 0.470 |
| 23 乘以 47 等于 1081。 | 17 | 3.28 s | 440.5 ms | 738.8 ms | 0.545 |

- 5 句首包 p50 均在 500 ms 内；长句/数字句的 max 会超过 500 ms。
- RTF p50 0.35–0.545；3/5 ≤0.5，2/5 略超（0.504/0.545）。
- 网关实际只把首句（8–15 字）单独合成，因此线上首包更接近 240–390 ms。

## 4. §10 指标判定

| 指标 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 首音 p50 | <1.5 s | 主机 endpoint→正式 **2.14 s**；主机 ASR final→正式 1.23 s；A 感知 1.56 s | **正式未达标（豁免）**；感知差 61 ms |
| 首音 p95 | <2.5 s | 主机 endpoint→正式 **3.44 s**；A 感知 **2.01 s**；ASR final→正式 **2.47 s** | 感知达标；正式按 endpoint 超（豁免） |
| ASR final | <500 ms | endpoint 后 p50 **858 ms** / p95 **1300 ms**；直接 worker 702–845 ms | **未达标（豁免）** |
| TTS 首包 | <500 ms | p50 240–440 ms；max 430–739 ms | 基本达标；长句 max 超 |
| TTS RTF | <0.5 | p50 0.35–0.545 | 3/5 达标，2/5 轻微超 |
| 连续 10 min TTS | 无爆音/断帧/累积延迟 | 本次未重跑；Phase 3 长文本通过 | **豁免**（按日常使用） |
| 打断 | <300 ms | 3.6–7.9 ms，0 旧音频 | **达标** |
| 30 轮 | 无状态错乱 | 30/30 | **达标** |
| greeting | 正常/不重复 | 缓存命中首音 0.9 ms，无双声 | **达标** |
| 工具 | 单/多/失败/超时/cancel | 本轮 1 次 get_time；Phase 5 单/多/视觉/失败通过 | **达标（沿用 Phase 5）** |
| 视觉问答 | 正常 | Phase 5 真机通过；本次未重复 | **达标（沿用 Phase 5）** |
| 无人/休眠断连 | 正常 | Phase 5 真机通过；本次 session.close / 服务重启无残留 | **达标（沿用 Phase 5）** |
| 显存 1 h | 无增长 | 本次 30 轮约 7.5 GiB / 8 GiB，无 OOM；未做 1 h | **豁免** |
| 24 h | 无泄漏/崩溃 | 未做；systemd + watchdog + 日常使用替代 | **豁免** |
| 回退 | EchoBridge 挂掉→豆包 | Phase 5 真机通过；本次未重拔线 | **达标（沿用 Phase 5）** |
| 语音质量 | 主观达标 | Phase 5 真机通过；日常使用继续听 | **达标（主观）** |
| 无公网暴露 | 仅 tailnet | `ss` 仅 127.0.0.1；Serve `(tailnet only)`；Funnel 空 | **达标** |

## 5. 设备历史 30 条（补充，不是受控 A/B）

从 X10A 的 `mingyue.db` 导出最近 30 条 `latency` 事件（192 个 session，119 豆包 / 73 EchoBridge；包含工具、视觉、预览和不同网络时段）：

| 数据 | n | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| EchoBridge backchannel 首音 | 30 | 801.4 | 792 | 864 | 926 |
| EchoBridge 正式首音 | 30 | 2679.0 | 2058 | 6703 | 9165 |
| 豆包正式首音（参考） | 30 | 378.6 | 60 | 1101 | 5806 |

- 设备侧**感知首音** p50 792 ms、p95 864 ms，表现好于主机（短音缓存命中）。
- 设备侧**正式首音** p50 2.06 s、p95 6.70 s，明显受工具 hop、视觉、provider 长尾和混合 session 影响；不能直接与主机受控 30 轮比较。
- 豆包数字仅为历史参考：其中可能包含 greeting/试听/缓存路径，样本未按轮配对。
- 结论：感知体验达标；正式答复长尾作为明确豁免，后续按日常使用继续定位（优先看工具 hop 与 provider 长尾）。

## 6. 部署与稳定性验证

| 项 | 结果 |
|---|---|
| 单元 | gateway / ASR / TTS active，watchdog timer active |
| `/healthz` | 200，版本/uptime/RSS/active session 正常 |
| `/readyz` | 200，ASR `ok=true`，TTS `healthy` |
| `/metrics` | `echobridge_up 1`、active session、RSS 正常 |
| watchdog | 手动停网关 → 3 次失败 → 自动 restart → `/healthz` 200；冷启动 300 s 宽限避免误杀加载中的 TTS |
| WSL 冷启动 | `wsl --terminate` 后任务 15 s 内重拉；ASR/TTS 约 4 min 45 s 就绪；TTS `NRestarts=0` |
| WSS（Tailscale Serve） | `wss://work-pc.tailea818a.ts.net:8443/echobridge/ws`，Bearer 101 + `session.created` |
| 无公网 | `ss -ltnp` 仅 127.0.0.1；Serve tailnet-only；Funnel 空 |
| Windows 自启动 | 计划任务 `EchoBridge WSL` 保持 `sleep infinity` 会话；`wsl --terminate` 后冷启动恢复，`/readyz` 200 |
| 测试 | `44 passed` |

## 7. 豁免与后续

1. **1 h / 24 h 长稳不做专项压测**：用户明确要求小精力收尾；由 systemd `Restart=always`、30 s watchdog、日常使用观察替代。
2. **ASR final <500 ms 未达**：Qwen3-ASR 与 vLLM TTS 共卡时，1.9–4 s 语音的 worker 延迟约 0.7–1.4 s；Phase 2 隔离测试 2–2.5 s 语音为 313–569 ms。后续可评估 ASR vLLM 后端、降低 ASR 上下文、或错峰 GPU。
3. **设备正式答复 p50/p95**：历史样本受工具/视觉/长尾影响；感知首音已达标。后续优先优化工具 hop 和 provider 长尾，不盲目改协议。
4. **TTS 10 min / RTF 边缘**：首句短、流式发送已掩盖大部分 RTF；2/5 长句 RTF 0.504/0.545，暂不优化。
5. **真机受控 A/B**：本次用设备历史数据补充，未做 30 条逐轮配对；Phase 5 已有 10 次切换与真实对话验收。

## 8. 复现

```bash
# 单元/协议回归
bash scripts/test.sh

# 部署状态
bash deploy/status.sh --user

# 30 轮真实基准（会占用 GPU/DeepSeek，约 5 min）
~/.venvs/echobridge/bin/python scripts/bench_gateway.py \
  --turns 30 --out var/bench/phase6.json

# TTS 首包 / RTF
~/.venvs/echobridge/bin/python scripts/bench_tts_vllm.py \
  --repeat 3 --out var/bench/tts_vllm.json

# 受控的 WSS 握手
~/.venvs/echobridge/bin/python scripts/ws_handshake_probe.py \
  --url wss://work-pc.tailea818a.ts.net:8443/echobridge/ws \
  --auth bearer --expect 101
```
