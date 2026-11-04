# WSL2 CUDA context 丢失与 Windows 驱动 NVRM 池泄漏

本文记录 2026-11-04 对“TTS/ASR 每 1–3 小时一次 `cudaErrorUnknown`”的调查。
`docs/TTS_RESILIENCE.md` 和 `docs/ASR_RESILIENCE.md` 记录的是崩溃后的兜底；
本文记录的是崩溃来源。日志只引用时间、计数和字节数。

## 1. 现象

- 2026-11-01 起约每天 10 次，每次由 systemd 在 83–113 s 内恢复。
- 11-01 至 11-04 的 34 次事件中，TTS 与 ASR **总是成对**失败：WSL 里所有已存在
  的 CUDA context 同时失效，下一次 GPU 调用返回 `CUDA_ERROR_UNKNOWN`
  （`cuStreamSynchronize`、`cuEventQuery`），随后 `Sticky error detected`。
- 丢失发生在空闲期间。用“最后一次成功的 GPU 请求”和“第一次失败”可以把每次
  事件夹在 3–10 min 的窗口里，窗口宽度只取决于 ASR 探活（5 min）和 TTS
  keepalive（10 min）的间隔。
- 事件之后新建的进程立即正常：11-01 09:57 新起的 ASR 在 10:34 仍然可用，
  而 10-29 起常驻的 TTS 已经失效。
- Windows 侧没有 nvlddmkm/Display 4101、WER LiveKernelEvent 或
  LiveKernelReports；30 s 一次的 systemd 定时器没有间断（VM 没有暂停）；
  WSL dmesg 只有进程启动时常见的 `dxgkio_escape: Ioctl failed: -22`。
- 10-29 19:52 至 10-31 20:51，同一组进程处理了 500+ 个请求，零失败。驱动
  （617.14，9/24 安装）、WSL（3.0.3，10/13 安装）和 Windows
  （19045.6456）在这之后都没有变化。

## 2. 证据：NVIDIA 驱动 NVRM 非分页池泄漏

2026-11-04 16:10，Windows 已连续运行 41.0 天（9/24 起）：

| 指标 | 值 |
|---|---:|
| `\Memory\Pool Nonpaged Bytes` | 3,303 MiB |
| 池标签 `NVRM`（NVIDIA Resource Manager） | 1,958 MiB |
| `NVRM` 未释放分配（allocs − frees） | 7,271,168 |
| 第二大非分页标签 `VdMm` | 67 MiB |

逐项测量 `NVRM` 未释放分配的增量（16:15–16:30）：

| 操作 | 增量 |
|---|---:|
| 空闲 60 s（Windows 侧监控每 ~6 s 调一次 `nvidia-smi.exe`） | 0 |
| Windows `nvidia-smi.exe` × 10 | 0 |
| WSL 里经 interop 调 `/mnt/c/Windows/System32/nvidia-smi.exe` × 5 | 0 |
| WSL 里新建并销毁一个 CUDA context | 0 |
| ASR 转写请求、TTS keepalive 合成 | 0（约 +158 临时，几秒内释放） |
| **WSL 里的 Linux `nvidia-smi` × 1** | **+21（+7,520 B），不释放** |
| **WSL 里的 Linux `nvidia-smi` × 5** | **+105（+37,600 B）** |
| **一个 vLLM 进程的平台检测**（`current_platform` + 3 个 NVML 查询） | **+72，不释放** |

驱动 621.14（11-04 更新）没有修复，只是变少了：WSL 里的 Linux `nvidia-smi`
每次泄漏 13 个对象。

Linux 侧 NVML（`/usr/lib/wsl/lib/libnvidia-ml.so.1`）每打开一次会话，就在宿主
驱动里留下对象，与 microsoft/WSL#12756 一致。`wsl --shutdown` 不会释放它们：
泄漏在 Windows 内核里，只有重启 Windows（或重启显卡设备）才能清空。

当前的增长很慢：稳态请求不泄漏，只有进程启动（每次崩溃恢复要重启 5 个 vLLM
进程，加上 ASR 启动脚本的 `nvidia-smi`）和人工在 WSL 里运行 `nvidia-smi`
才会增加。727 万这个量级是 41 天开发期间累积的。

## 3. 推断与验证状态

**已否定（2026-11-04 19:40）**：重启 Windows 并更新驱动后，`NVRM` 只有
55–58 MiB，但 18:03–18:28 仍然连续丢失 3 次 context（窗口分别为
18:03:12–18:07:07、18:12:30–18:13:19、18:27:08–18:27:50）。泄漏是真实的，
第 4 节的修复仍然保留，但它不是 context 丢失的原因。后续调查记录在
`docs/TTS_RESILIENCE.md`。

当时支持这个推断的事实：

- 失败的位置在宿主 GPU 栈，而不是 EchoBridge 或 vLLM；
- 开机约 38 天后（11-01）突然开始，之后不再消失；
- 11-03 14:25 `wsl --shutdown` 后仍然复发（泄漏不随 WSL 重启释放）；
- 公开报告中同一泄漏在长时间开机后导致 GPU 故障。

验证方法：重启 Windows，确认 `NVRM` 回到 MiB 级，然后观察 2–3 天的
`cudaErrorUnknown` 次数。

2026-11-04 16:45 重启 Windows，17:09:21 驱动更新到 621.14。重启后半小时，
`NVRM` 为 55 MiB / 15,897 个对象，非分页池总量 472 MiB（重启前分别是
1,958 MiB / 7,271,584 个和 3,303 MiB）。17:09–17:12 的那次 context 丢失是
安装驱动重置 GPU 造成的（UserPnp 20003，`DEVPKEY_Device_InstallDate`
17:09:21），不计入统计。驱动和泄漏是同时清掉的，如果之后不再丢失，单靠这次
实验无法区分是哪一个起了作用。

```bash
# WSL 里统计每天的 context 丢失（TTS + ASR）
journalctl --user -u echobridge-tts -u echobridge-asr-vllm --since "-3 days" \
  | grep -E "CUDA_ERROR_UNKNOWN|CUDA error: unknown" \
  | awk '{print substr($1,1,10)}' | sort | uniq -c
```

## 4. 已做的改动

- `scripts/asr_vllm.sh`：显存余量检查在 WSL2 下改用
  `/mnt/c/Windows/System32/nvidia-smi.exe`（输出是 CRLF，已去掉 `\r`）。
  `ECHOBRIDGE_NVIDIA_SMI` 可以指定其它路径；原生 Linux 上仍用 `nvidia-smi`。
- `deploy/status.sh`：同样改用 Windows 版。
- `deploy/windows/nvrm-pool.ps1`：只读查看 `NVRM` 占用，不需要管理员。

WSL 环境（不在仓库里，2026-11-04 手工完成）：

- `/usr/local/bin/nvidia-smi` 是包装脚本：调用 Windows 版、去掉 `\r`、保留
  退出码；Windows 版不可用时退回 Linux 版。它在 PATH 里排在
  `/usr/lib/wsl/lib` 之前，登录 shell、`watch nvidia-smi` 和 systemd 服务都会
  命中它，实测 5 次调用 `NVRM` +0。需要看 WSL 进程列表时，直接运行
  `/usr/lib/wsl/lib/nvidia-smi`。
- 已卸载 `nvtop`（snap），它同样通过 Linux NVML 采样。

vLLM 进程启动时的 NVML 调用（平台检测）无法简单关掉：NVML 初始化失败时
vLLM 只在 Jetson 上回退到非 NVML 平台。它每次启动泄漏约 72 个对象，只要
崩溃不再频繁发生，这部分可以忽略。

## 5. 检查与预防

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\nvrm-pool.ps1
# NVRM non-paged pool: 1,958 MiB in 7,271,168 objects (Windows uptime 41.0 days)
```

- 不要在 WSL 里运行 `/usr/lib/wsl/lib/nvidia-smi`、`nvtop`、`gpustat` 或循环
  调用 NVML 的脚本；普通的 `nvidia-smi` 已经由包装脚本转到 Windows 版。
- `NVRM` 超过几百 MiB 时安排一次 Windows 重启。
