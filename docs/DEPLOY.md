# EchoBridge 部署运维

> 适用：Windows 11 + WSL2 Ubuntu（当前生产形态）或原生 Ubuntu 24.04/26.04。
> 原则：所有模型与网关只监听 `127.0.0.1`；对外只通过 Tailscale Serve 的 tailnet-only HTTPS/WSS；禁止 Funnel、公网端口转发和 `0.0.0.0` 暴露。

## 1. 进程拓扑

```text
X10A ── wss://work-pc.<tailnet>.ts.net:8443 ── Tailscale Serve（仅 tailnet）
                                                     │ http://127.0.0.1:18787
                                                     ▼
                                            echobridge-gateway
                                              ├── Qwen3-ASR（in-process 或 :18801 vLLM）
                                              └── 127.0.0.1:18802  TTS worker (vLLM-Omni)
```

`ECHOBRIDGE_ASR_BACKEND` 有三种选择：

- `inprocess`（默认/回退）：ASR 模型直接加载在 gateway 进程内。
- `vllm`：独立的 vLLM-ASR 服务，监听 18801，固定 256 MiB KV；after-endpoint
  从约 900 ms 降到约 114 ms，代价是约 2.3 GiB 显存，以及和 TTS 同时解码时
  ASR 会慢约 7 倍。部署为 `echobridge-asr-vllm.service`。
- `qwen3`：旧的 18801 HF worker（`scripts/asr_worker.sh`）。

| 服务 | 端口 | 启动方式 | 说明 |
|---|---|---|---|
| `echobridge-gateway` | 18787 | `python -m echobridge` | WSS、VAD、turn、LLM、pacer |
| `echobridge-asr-vllm`（可选） | 18801 | `scripts/asr_vllm.sh` | Qwen3-ASR-0.6B，vLLM 0.32 + CUDA Graph |
| `echobridge-asr`（旧回退） | 18801 | `scripts/asr_worker.sh` | Qwen3-ASR-0.6B，独立 venv |
| `echobridge-tts` | 18802 | `scripts/tts_vllm.sh` | Qwen3-TTS-0.6B，vLLM-Omni |
| `echobridge-watchdog.timer` | — | `deploy/watchdog.sh` | 每 30 s 检查进程与 worker |

gateway 还会每 10 s 探测 TTS worker，检测到重启后自动 `rewarm()`（强制一次真实
合成，绕过网关内的 PCM 缓存）。`ECHOBRIDGE_TTS_RECOVERY_PROBE_S=0` 可关闭。

模型、参考音频、日志、token、API key 全部在仓库外（`~/.cache/echobridge`、`/etc/echobridge` 或 `~/.config/echobridge`），不提交。

## 2. 前置条件

```bash
# WSL2 已启用 systemd
cat /etc/wsl.conf          # [boot] systemd=true
ps -p 1 -o comm=           # systemd

# Python 环境：主 venv 现在需要 qwen-asr/qwen-tts（进程内后端）
ls ~/.venvs/echobridge ~/.venvs/echobridge-tts-vllm
# 可选的 qwen3 回退环境
ls ~/.venvs/echobridge-asr

# 模型与音色
ls ~/.cache/echobridge/models/Qwen3-ASR-0.6B
ls ~/.cache/echobridge/models/Qwen3-TTS-12Hz-0.6B-Base
ls ~/.cache/echobridge/voices/mingyue/reference.wav

# 一次性允许用户服务在最后一个终端关闭后继续运行（WSL2）
loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger   # Linger=yes
```

> WSL2 的 24/7 运行依赖 Windows 登录任务（§6）：任务会保持一个 `sleep infinity` 的 WSL 会话，否则 WSL2 在最后一个会话退出后会停掉整个发行版。Windows 重启后需要登录一次（或用管理员把任务改为开机启动）。

## 3. 安装

### 3.1 WSL2（无 sudo，当前主机）

```bash
cd /mnt/e/Projects/EchoBridge
bash deploy/install.sh --user --stop-legacy
```

- 单元安装到 `~/.config/systemd/user/`。
- 配置安装到 `~/.config/echobridge/echobridge.env`（600）。
- 首次安装会复用 `var/dev-token`，**不会更换 X10A 已保存的 token**；并从 `var/llm.env` 导入 LLM key（不打印）。
- `--stop-legacy` 只停止当前监听 18787/18801/18802 的手工进程；如果 systemd 网关已运行则跳过。
- `ECHOBRIDGE_ASR_BACKEND=inprocess` 时不会启动任何 ASR worker；网关直接加载
  ASR 模型。`vllm` 会启用 `echobridge-asr-vllm`，`qwen3` 会启用旧
  `echobridge-asr`。`ECHOBRIDGE_TTS_BACKEND=inprocess` 时不会启动
  `echobridge-tts`。切换后端后重新运行 `deploy/install.sh` 才会启用/停用对应
  worker；**不要同时启用两个 ASR worker**。
- 安装后按 env 启动 worker（如 vLLM），再起网关；网关 `ExecStartPre` 最多等
  300 s 让 TTS 就绪。

### 3.2 原生 Ubuntu（systemd 系统服务）

```bash
cd /opt/EchoBridge            # 仓库外也可，但建议固定路径
sudo bash deploy/install.sh --system --stop-legacy
```

- 单元安装到 `/etc/systemd/system/`，以 `SUDO_USER` 身份运行。
- 配置安装到 `/etc/echobridge/echobridge.env`（600，属主为服务用户）。
- 若已有生产配置：`sudo bash deploy/install.sh --system --env-file /path/to/echobridge.env`。

### 3.3 常用参数

```text
--user / --system       安装模式（WSL 默认 user，原生默认 system）
--no-start              只安装并 enable，不启动
--stop-legacy           启动前停止手工进程
--env-file PATH         复制已有 env 文件
--python PATH           网关 venv python
--port PORT             网关端口（默认 18787）
```

## 4. 配置

模板：`config/echobridge.example.env`。生产关键项：

```env
ECHOBRIDGE_DEV=0
ECHOBRIDGE_TOKEN=<32+ 字符随机串，复用设备已保存值>
ECHOBRIDGE_BIND=127.0.0.1
ECHOBRIDGE_PORT=18787
ECHOBRIDGE_ENABLE_PROBE=0

ECHOBRIDGE_VAD_BACKEND=silero
# 默认：ASR 在 gateway 进程内，和 vLLM 共用一张卡但不再多一个 CUDA context。
ECHOBRIDGE_ASR_BACKEND=inprocess
ECHOBRIDGE_ASR_MODEL=~/.cache/echobridge/models/Qwen3-ASR-0.6B
ECHOBRIDGE_ASR_DEVICE=cuda:0
# 回退到独立 worker 时改成 qwen3 并保留 URL：
# ECHOBRIDGE_ASR_BACKEND=qwen3
ECHOBRIDGE_ASR_URL=http://127.0.0.1:18801
ECHOBRIDGE_TTS_BACKEND=vllm
ECHOBRIDGE_TTS_URL=http://127.0.0.1:18802
ECHOBRIDGE_LLM_BACKEND=deepseek
ECHOBRIDGE_LLM_API_KEY=<只放 env>
# 单会话瘦身：Stage 0 0.30 + Stage 1 0.18；旧值 0.40。
ECHOBRIDGE_TTS_GPU_UTIL=0.30
ECHOBRIDGE_TTS_VOICE_DIR=~/.cache/echobridge/voices/mingyue
# 可选；留空使用 <voice_dir>/voices.json
ECHOBRIDGE_TTS_VOICE_REGISTRY=
ECHOBRIDGE_WATCHDOG_GRACE_S=300
```

多音色 registry、默认音色兼容和新增音色步骤见 `docs/VOICES.md`。

- `ECHOBRIDGE_BIND` 必须是 `127.0.0.1`。Tailscale Serve 在 Windows 侧代理 WSL 的 localhost；不要为了“能连上”改成 `0.0.0.0`。
- 不要在 URL 中放 token；客户端用 `Authorization: Bearer` 或 `Sec-WebSocket-Protocol: token.<token>`。
- 修改 env 后：`systemctl --user restart echobridge-gateway`（系统模式去掉 `--user`）。

## 5. Tailscale Serve（唯一对外入口）

### 5.1 当前 Windows 主机

Tailscale 守护进程在 Windows，Serve 把 tailnet 的 8443 代理到 WSL 的 18787：

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" serve --bg --https=8443 http://127.0.0.1:18787
& "C:\Program Files\Tailscale\tailscale.exe" serve status
```

预期只出现：

```text
https://work-pc.tailea818a.ts.net:8443 (tailnet only)
|-- / proxy http://127.0.0.1:18787
```

- 不要执行 `tailscale funnel ...`；`serve status` 中不应出现 `(Funnel on)`。
- 设备配置：`wss://work-pc.tailea818a.ts.net:8443/echobridge/ws`，Bearer token 与 env 中一致。
- 验证：在 WSL 或 Windows 上运行 `python scripts/ws_handshake_probe.py --url wss://work-pc.tailea818a.ts.net:8443/echobridge/ws --auth bearer --expect 101`。

### 5.2 原生 Ubuntu

```bash
bash deploy/tailscale-serve.sh
```

脚本使用 `tailscale serve --bg --https=8443 http://127.0.0.1:18787`，并在检测到 Funnel 时返回失败。

## 6. Windows 登录自启动（WSL2）

```powershell
powershell -ExecutionPolicy Bypass -File E:\Projects\EchoBridge\deploy\windows\install-startup-task.ps1
```

- 创建当前用户登录时运行的计划任务 `EchoBridge WSL`：启动网关、ASR、TTS、watchdog，并用 `sleep infinity` 保持 WSL 会话；外层循环在 WSL 被终止后 15 s 重拉。
- 任务会长期处于 `Running`（不是一次性任务）；这是预期行为，用来防止 WSL2 在最后一个终端关闭后关停发行版。
- 先确保 §2 的 `Linger=yes`。
- 立即测试：`Start-ScheduledTask -TaskName "EchoBridge WSL"`。
- 查看：`Get-ScheduledTask -TaskName "EchoBridge WSL" | Select State`；`Running` 正常。
- 验证冷启动：`wsl --terminate Ubuntu-26.04`，再启动任务，等待 ASR/TTS 模型加载（约 3–5 min），`/readyz` 应为 200。
- 卸载：`powershell -ExecutionPolicy Bypass -File deploy\windows\uninstall-startup-task.ps1`。

## 7. 健康检查、watchdog、监控

| 端点 | 用途 |
|---|---|
| `GET /healthz` | 网关存活；返回版本、uptime、RSS、当前会话 |
| `GET /readyz` | 网关 + 已配置的 ASR/TTS worker 就绪；任一未就绪返回 503 |
| `GET /metrics` | Prometheus 文本：`echobridge_up`、uptime、RSS、active session |
| `deploy/status.sh` | 一屏查看 units、health、ready、metrics、GPU、Serve |

```bash
bash deploy/status.sh --user        # WSL2
bash deploy/status.sh --system      # 原生
```

Watchdog 每 30 s 检查：

- 网关 `/healthz`（连续 3 次失败 → 重启网关）；进程内 ASR 的就绪状态在
  `/readyz` 的 `workers.asr` 中直接可见；
- 当 `ECHOBRIDGE_ASR_BACKEND=vllm` 时检查 `echobridge-asr-vllm` 的 `/health`；
  当 `qwen3` 时检查 `echobridge-asr` 的 `/healthz`；
- TTS `/health`（vLLM）或 `/healthz`（HF）；
- 冷启动宽限 `ECHOBRIDGE_WATCHDOG_GRACE_S`（默认 300 s）：`Type=simple` 的 worker 刚启动就显示 active，但模型可能还在加载；宽限期内不重启，避免把正在加载的 TTS 反复杀掉；
- 进程崩溃由 systemd `Restart=always` 兜底；watchdog 只处理“进程在但无响应”。

维护时先停 watchdog，避免手工 stop 的 unit 被拉起来：

```bash
systemctl --user stop echobridge-watchdog.timer
# ... 维护 ...
systemctl --user start echobridge-watchdog.timer
```

日志：

```bash
journalctl --user -u echobridge-gateway -f   # 含 inprocess ASR 日志
journalctl --user -u echobridge-tts -f
journalctl --user -u echobridge-asr-vllm -f   # ASR_BACKEND=vllm
journalctl --user -u echobridge-asr -f        # qwen3 回退模式
journalctl --user -u echobridge-watchdog -n 50
```

日志只记录 session id、事件、耗时、字节数、字符数；不记录 token、Authorization、base64、完整音频或用户原文。

## 8. 启停与升级

```bash
# WSL2（ASR_BACKEND=vllm）
systemctl --user stop echobridge-gateway echobridge-asr-vllm echobridge-tts
systemctl --user start echobridge-tts
systemctl --user start echobridge-asr-vllm
systemctl --user start echobridge-gateway

# inprocess ASR：没有 ASR worker
systemctl --user start echobridge-tts echobridge-gateway

# qwen3 回退：把 echobridge-asr-vllm 换成 echobridge-asr
systemctl --user start echobridge-asr echobridge-tts echobridge-gateway

# 原生
sudo systemctl restart echobridge-gateway
```

升级代码后：

```bash
git pull
systemctl --user stop echobridge-gateway
systemctl --user restart echobridge-tts
systemctl --user restart echobridge-asr-vllm
systemctl --user start echobridge-gateway
```

只改网关代码时可不重启模型 worker，模型保持常热。ASR/TTS 的 `Restart=always`
会兜底崩溃；ASR 用固定 KV，热重启约 62 s。

## 9. 回退

### 9.1 设备回退豆包

设备端 `auto` 模式在以下情况自动临时回退豆包，不修改保存的 `active_voice_backend`：

- 5 s 建连超时 / 8 s 无 `session.created`；
- 会话中途 WS 断开；
- 网关返回致命 `error`。

手动回退：设置页把“语音后端”改为 `doubao` 或打开“强制豆包”。

### 9.2 主机回退

```bash
# 停止并移除 unit，保留 env（token 不丢）
bash deploy/uninstall.sh --user

# 需要回到手工开发模式
bash scripts/dev.sh
```

系统模式：`sudo bash deploy/uninstall.sh --system`。

### 9.3 ASR 后端回退

从 vLLM-ASR 回到 in-process：

```bash
sed -i 's/^ECHOBRIDGE_ASR_BACKEND=.*/ECHOBRIDGE_ASR_BACKEND=inprocess/' \
  ~/.config/echobridge/echobridge.env
bash deploy/install.sh --user
```

`install.sh` 会停用并禁用 `echobridge-asr-vllm`。**不要只改 env 不重装**：旧
unit 仍可能被 gateway 的 `Wants=` 拉起，导致两份 ASR 同时占卡（约多 1.2 GiB）。

如果保留了实验前的 env 备份：

```bash
cp -p ~/.config/echobridge/backup-20261029/echobridge.env \
      ~/.config/echobridge/echobridge.env
bash deploy/install.sh --user
```

### 9.4 代码回退

```bash
git revert <phase-6-commit>
```

Phase 6 没有数据库迁移；回退只影响部署文件和健康端点。

## 10. 不允许的公网暴露

部署后自查：

```bash
ss -ltnp | grep -E '18787|18801|18802'      # 只应看到 127.0.0.1
tailscale serve status                       # 只应看到 (tailnet only)
tailscale funnel status                      # 应为空
```

- 不要开启路由器端口转发、UPnP、公网反向代理或 Tailscale Funnel。
- 不要把 `ECHOBRIDGE_BIND` 设为 `0.0.0.0`。
- 不要把 token 放进 URL query 或提交到仓库。

## 11. 已知限制

- WSL2 在 Windows 休眠/关机后不会自行恢复；登录任务保证登录后启动，并保持一个 `sleep infinity` 会话；冷启动 ASR/TTS 约 3–5 min，期间网关 `ExecStartPre` 等待，watchdog 宽限 300 s。
- GPU 由桌面/其他进程共享；TTS 默认 `0.40` 显存利用率，出现 OOM 时先降低该值或关闭桌面 GPU 占用，不要盲目量化。
- 单设备单会话是设计约束；不同 `device_id` 的第二连接返回 1013。
- 1 h/24 h 长稳未做专项压测；由 systemd/watchdog + 日常使用观察替代（见 `docs/BENCHMARK.md`）。
