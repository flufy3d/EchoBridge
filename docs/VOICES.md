# EchoBridge 多音色（Voice Registry）

状态：Phase 7。默认仍为 `mingyue_v1`，旧设备配置无需修改。

## 1. 三个确认音色

| voice id | 名字 | 描述 | 参考音频 |
|---|---|---|---|
| `mingyue_v1` | 月月 | 温暖自然，默认声线 | `candidate_3.wav` |
| `mingyue_xixi_v1` | 汐汐 | 明亮活泼，音调偏高 | `candidate_1.wav` |
| `mingyue_nuonuo_v1` | 糯糯 | 软糯温柔，音量偏轻 | `candidate_2.wav` |

参考音频、参考文本、prompt 缓存全部在仓库外（默认
`~/.cache/echobridge/voices/mingyue/`），不提交。

## 2. Registry 位置

优先级：

1. `ECHOBRIDGE_TTS_VOICE_REGISTRY`（显式文件路径）；
2. `<ECHOBRIDGE_TTS_VOICE_DIR>/voices.json`；
3. 没有 registry 文件时，若 `candidate_1/2/3.wav` 与 `reference.txt` 都存在，
   自动注册上表三个音色；否则只注册 `ECHOBRIDGE_TTS_VOICE_ID` 的单默认音色。

相关配置：

```env
ECHOBRIDGE_TTS_VOICE_ID=mingyue_v1
ECHOBRIDGE_TTS_VOICE_DIR=~/.cache/echobridge/voices/mingyue
# 可选；留空即使用 <voice_dir>/voices.json
ECHOBRIDGE_TTS_VOICE_REGISTRY=
ECHOBRIDGE_TTS_REF_AUDIO=~/.cache/echobridge/voices/mingyue/reference.wav
ECHOBRIDGE_TTS_REF_TEXT=~/.cache/echobridge/voices/mingyue/reference.txt
```

## 3. `voices.json` 格式

```json
{
  "default": "mingyue_v1",
  "voices": [
    {
      "id": "mingyue_v1",
      "name": "月月",
      "description": "温暖自然，默认声线",
      "ref_audio": "candidate_3.wav",
      "ref_text_file": "reference.txt"
    }
  ]
}
```

- `id`：`^[a-z0-9_]{4,80}$`，必须唯一。
- `ref_audio` / `ref_text_file`：相对路径按 registry 文件所在目录解析；支持 `~`。
- `ref_text` 也可以直接内联，优先于 `ref_text_file`。
- `default` 必须指向一个可用音色。

校验失败时不会让网关崩溃：记录 warning，保留能用的音色；默认音色不可用时
回退到 `ECHOBRIDGE_TTS_VOICE_ID` + `ECHOBRIDGE_TTS_REF_AUDIO` /
`ECHOBRIDGE_TTS_REF_TEXT`。坏 JSON、重复 ID、缺参考文件都按此处理。

模板：`config/voices.example.json`。

## 4. 会话与未知音色

设备在 `session.create` 的 `session.audio.output.voice` 传 voice id：

- 已知 id：使用该音色，`voice_fallback=false`。
- 未知 id：使用默认音色，`voice_fallback=true`。
- 空字符串或未传：使用默认音色，`voice_fallback=false`。

`session.created` / `session.updated` 会回显：

```json
{
  "session": {
    "voice": "mingyue_xixi_v1",
    "voice_name": "汐汐",
    "voice_fallback": false
  }
}
```

backchannel 与正式 TTS 都使用解析后的 voice id。

`GET /voices` 使用与 WebSocket 相同的 Bearer token，返回：

```json
{
  "default": "mingyue_v1",
  "voices": [
    {"id": "mingyue_v1", "name": "月月", "description": "温暖自然，默认声线", "default": true}
  ]
}
```

响应不包含参考音频路径、token 或内部 URL。

## 5. TTS 缓存与预热

- `VLLMTTSClient` 为每个 voice 使用各自的 `ref_audio` / `ref_text`；未知 voice
  使用默认参考。
- `Qwen3TTSClient` 继续把 `voice_id` 发给 worker，由 worker 选择 prompt。
- 缓存 key 是 `(voice, text)`，不同音色不会互相命中。
- 启动时默认音色的 greeting/backchannel 同步预热；其他音色用后台任务预热，
  不阻塞网关启动。会话选择其他音色时调用 `ensure_warm(voice)`。
- `close()` 取消后台预热任务。
- `MockTTS` 按 voice hash 偏移五声音阶，测试里可直接听/比对出不同音色。

## 6. HF worker（`tts_worker.py`）

- 启动时加载 registry；默认音色的 prompt 同步加载，其他音色首次请求时 lazy 加载。
- prompt 文件：
  - 默认音色：`prompt.pt` / `prompt.meta.json`（兼容旧缓存）；
  - 其他音色：`prompt.<voice_id>.pt` / `prompt.<voice_id>.meta.json`。
- `_ref_hash` 包含该音色的参考音频、参考文本和模型路径。
- `POST /v1/synthesize` 读取 `voice_id`；未知音色回退默认；响应头
  `X-Voice-Id` 返回实际使用的音色。
- `GET /healthz` 增加 `default_voice` 和
  `voices:[{id,name,default,prompt_cached}]`。

## 7. 生成/重建 registry

```bash
# 生成三个候选、reference.wav、reference.txt、voices.json（需要 VoiceDesign 模型）
~/.venvs/echobridge-tts/bin/python scripts/design_voice.py

# 只扫描 candidate_*.wav 重建 voices.json（不需要模型）
python scripts/build_voice_registry.py
python scripts/build_voice_registry.py --default-candidate 1
```

`design_voice.py` 默认把 candidate 3 作为 `mingyue_v1`；`--default-candidate`
可改为 1/2/3，并把该候选复制为兼容旧配置的 `reference.wav`。

## 8. 新增音色

1. 把参考音频放到 voice 目录（仓库外）。
2. 在 `voices.json` 的 `voices` 数组增加一项，确保 id 唯一、参考文件存在。
3. 如需新音色使用 HF worker，首次请求会自动生成
   `prompt.<voice_id>.pt`；也可以提前预热。
4. 重启网关和 TTS worker，让两者重新加载 registry：
   `systemctl --user restart echobridge-tts echobridge-gateway`。
5. 设备端通过 `GET /voices` 获取列表后即可选择。
