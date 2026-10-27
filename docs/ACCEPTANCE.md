# Phase 6 验收记录

> 日期：2026-10-27
> 主机：`360W-PC-03`（Windows 11 + WSL2 Ubuntu 26.04，RTX 3070 8 GB）
> 设备：X10A，`100.83.39.115:5555`
> 提交：`phase 6: deployment + benchmark`
> 结论：**通过（带明确豁免）**。部署、WSS、watchdog、单会话、监控、30 轮真实基准、打断与回退链路已验收；1 h/24 h 长稳、ASR <500 ms、设备正式答复长尾列为豁免，按日常使用继续观察。

## 1. 部署

| 项 | 证据 | 结果 |
|---|---|---|
| WSL2 systemd | `systemctl --user is-active echobridge-gateway/asr/tts`，三个 active | 通过 |
| 单元启用 | `default.target.wants` 下三个 service + watchdog timer | 通过 |
| Linger | `loginctl show-user flufy3d -p Linger` → `Linger=yes` | 通过 |
| Windows 登录任务 | `EchoBridge WSL` 注册，State=`Running`（保持 WSL 会话的 `sleep infinity`）；`wsl --terminate` 后重新启动任务，服务自动恢复 | 通过 |
| 冷启动 | WSL 重启后 ASR/TTS 加载约 4 min 45 s；`ExecStartPre` 等待；watchdog 300 s 宽限；TTS `NRestarts=0` | 通过 |
| WSL 会话锚 | `wsl --terminate` 后任务循环 15 s 内重新拉起；`/readyz` 200 | 通过 |
| 配置权限 | `~/.config/echobridge/echobridge.env` mode 600；日志不打印 token/key | 通过 |
| 绑定 | `ss -ltnp` 仅 `127.0.0.1:18787/18801/18802` | 通过 |
| 预加载 | 网关启动 `ExecStartPre` 等 worker；ASR/TTS warmup；greeting/backchannel 预合成 | 通过 |
| 单会话 | `tests/test_single_session.py` 等 44 passed；不同 device 1013 | 通过 |

## 2. 对外入口

| 项 | 证据 | 结果 |
|---|---|---|
| Tailscale Serve | `https://work-pc.tailea818a.ts.net:8443 (tailnet only)` → `127.0.0.1:18787` | 通过 |
| WSS Bearer | `ws_handshake_probe.py --auth bearer --expect 101` → 101 + `session.created` | 通过 |
| WSS 无凭据/错 token | 403（Phase 5 矩阵，回归仍由测试覆盖） | 通过 |
| Funnel | `tailscale funnel status` 为空 | 通过 |
| 公网端口 | 无端口转发/UFW 放行；只依赖 Tailscale | 通过 |

## 3. 健康与监控

| 端点/工具 | 实测 | 结果 |
|---|---|---|
| `GET /healthz` | 200，`ok=true`，service/uptime/RSS/active | 通过 |
| `GET /readyz` | 200，ASR `ok=true`，TTS `healthy` | 通过 |
| `GET /metrics` | `echobridge_up 1`，uptime/RSS/active session | 通过 |
| `deploy/status.sh --user` | units/health/ready/metrics/GPU 一屏输出 | 通过 |
| watchdog | 停网关 → 3 次失败 → restart → `/healthz` 200 | 通过 |
| 日志脱敏 | 无 token/Authorization/base64/用户原文 | 通过 |

## 4. 功能回归

| 场景 | 证据 | 结果 |
|---|---|---|
| greeting | 缓存 pin 后首音 0.9 ms；无 text delta/双声 | 通过 |
| 普通对话 | 30/30 轮完成 | 通过 |
| 工具 | 第 1 轮 `get_time` hop 完成；Phase 5 单/多/视觉/失败 | 通过 |
| 打断 | 5/5，cancel 3.6–7.9 ms，旧音频 0 条 | 通过 |
| 空 ASR/短语音 | benchmark 初版短样本触发 `transcription.failed` 并快速回 LISTENING（未计入 30 轮） | 通过 |
| session.close | 正常回 `session.closed`；服务重启无残留 | 通过 |
| 豆包回退 | Phase 5 真机通过；本次代码未改回退路径 | 沿用 |
| 视觉问答 | Phase 5 真机通过（look_and_ask） | 沿用 |
| 休眠/唤醒 | Phase 5 真机通过 | 沿用 |
| 语音质量 | Phase 5 真机+日常使用主观通过 | 沿用 |

## 5. 基准

完整数据见 `docs/BENCHMARK.md`。

- 30/30 轮真实 ASR + DeepSeek + vLLM TTS，0 失败。
- A 感知首音（backchannel）p50 **706.5 ms** / p95 **711.5 ms**（ASR final 起）。
- B 正式首音 p50 **1231.1 ms** / p95 **2474.4 ms**（ASR final 起），p95 <2.5 s。
- 若从 VAD endpoint 起算：A p50 **1560.9 ms** / p95 **2008.3 ms**；B p50 **2141.1 ms** / p95 **3443.4 ms**。
- ASR after endpoint p50 **858.2 ms** / p95 **1299.5 ms**；直接 worker 702–845 ms。
- TTS 首包 p50 240–440 ms；RTF p50 0.35–0.545。
- 设备历史 30 条：backchannel p50 792 ms / p95 864 ms；正式 p50 2058 ms / p95 6703 ms（混合工具/视觉/长尾）。

## 6. 回退演练

| 层级 | 操作 | 结果 |
|---|---|---|
| 设备 | `auto`：EchoBridge 5 s 超时/中途断开 → 临时豆包；不改保存的 active backend | Phase 5 真机通过 |
| 设备手动 | 设置页改 `doubao` 或强制豆包 | Phase 5 通过 |
| 主机 | `bash deploy/uninstall.sh --user` 停用并移除 unit，保留 env | 本次脚本就绪；未在验收中实际移除（避免中断） |
| 手工开发 | `bash scripts/dev.sh` | 保留可用 |
| 代码 | `git revert <phase-6-commit>`；无数据库迁移 | 可用 |

## 7. 明确豁免

1. **1 h/24 h 长稳**：用户明确要求不做长测；由 systemd、30 s watchdog、日常使用替代。
2. **ASR final <500 ms**：共卡时 0.7–1.4 s；Phase 2 隔离测试 2–2.5 s 语音为 313–569 ms。后续可评估 ASR vLLM/上下文裁剪/GPU 错峰。
3. **设备正式答复 p50/p95**：历史混合样本含工具、视觉、provider 长尾；感知首音已达标。后续优先优化工具 hop 与长尾。
4. **TTS 10 min 无爆音专项**：本次未重跑；Phase 3 已做长文本/连续合成，日常使用观察。
5. **受控 30 条真机 A/B**：本次以主机受控 30 轮 + 设备历史 30 条补充；未做逐轮配对。

## 8. 验收签署

- [x] 部署可重复（WSL2 `--user` / 原生 `--system` 脚本）
- [x] 仅 tailnet 可达，无公网暴露
- [x] EchoBridge 挂掉可回退（Phase 5 沿用 + 本次 watchdog/健康验证）
- [x] 监控、watchdog、单会话、warmup 就绪
- [x] 30 轮基准与打断数据落盘
- [x] `docs/BENCHMARK.md`、`docs/DEPLOY.md`、本记录完成
- [ ] 1 h/24 h 长稳（豁免，日常使用）
- [ ] ASR <500 ms / 设备正式首音 p50（豁免，后续优化）

签署：Phase 6 收尾通过，进入日常使用与按问题迭代。
