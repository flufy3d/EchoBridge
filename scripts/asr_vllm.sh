#!/usr/bin/env bash
# Qwen3-ASR served by vLLM 0.32 (CUDA-graph decoder) for EchoBridge.
#
# Measured on an RTX 3070 (8 GiB, WSL2) next to the two-stage vLLM-Omni TTS:
# the server holds ~2.3 GiB (1.53 GiB weights + 0.25 GiB KV + context and
# activations), 4 s of audio transcribes in ~105 ms and 15 s in ~300-340 ms
# including the HTTP upload.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM="${ECHOBRIDGE_VLLM_BIN:-$HOME/.venvs/echobridge-tts-vllm/bin/vllm}"
MODEL="${ECHOBRIDGE_ASR_MODEL:-$HOME/.cache/echobridge/models/Qwen3-ASR-0.6B}"
HOST="${ECHOBRIDGE_ASR_HOST:-127.0.0.1}"
PORT="${ECHOBRIDGE_ASR_PORT:-18801}"
GPU_UTIL="${ECHOBRIDGE_ASR_GPU_UTIL:-0.30}"
KV_BYTES="${ECHOBRIDGE_ASR_KV_CACHE_BYTES:-268435456}"
MAX_LEN="${ECHOBRIDGE_ASR_MAX_MODEL_LEN:-512}"
MAX_SEQS="${ECHOBRIDGE_ASR_MAX_NUM_SEQS:-1}"
SERVED="${ECHOBRIDGE_ASR_VLLM_MODEL:-qwen3-asr}"
MIN_FREE_MB="${ECHOBRIDGE_ASR_MIN_FREE_MB:-2600}"
GPU_WAIT_S="${ECHOBRIDGE_ASR_GPU_WAIT_S:-180}"

# EnvironmentFile values are not shell-expanded, so expand a leading ~ here.
MODEL="${MODEL/#\~/$HOME}"
VLLM="${VLLM/#\~/$HOME}"

if [[ ! -x "$VLLM" ]]; then
  echo "vLLM not found: $VLLM" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "ASR model not found: $MODEL" >&2
  exit 1
fi

# Never start a second server on the port.  A manual `vllm serve` plus this
# unit means two copies of the model; WSL2 does not fail that allocation, it
# silently evicts the idle TTS worker's VRAM to system memory, and the next
# TTS request pays ~0.5 s to page it back in.
if ss -Hltn "sport = :$PORT" 2>/dev/null | grep -q .; then
  echo "port $PORT is already in use; not starting a second ASR server" >&2
  exit 1
fi

# Wait for real headroom, and fail (systemd retries) instead of starting into
# a full card.  Use nvidia-smi: it reports the whole device, while a WSL2
# process's cudaMemGetInfo() ignores every other process (it shows ~6.9 GiB
# free with the TTS worker resident and nvidia-smi at ~2.5 GiB).
if ((MIN_FREE_MB > 0)); then
  deadline=$((SECONDS + GPU_WAIT_S))
  while :; do
    FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && ((FREE_MB >= MIN_FREE_MB)); then
      break
    fi
    if ((SECONDS >= deadline)); then
      echo "GPU memory still short after ${GPU_WAIT_S}s: free=${FREE_MB:-?}MiB need=${MIN_FREE_MB}MiB" >&2
      exit 1
    fi
    echo "waiting for GPU memory: free=${FREE_MB:-?}MiB need=${MIN_FREE_MB}MiB"
    sleep 5
  done
fi

echo "EchoBridge vLLM ASR server"
echo "  model:  ${MODEL}"
echo "  bind:   ${HOST}:${PORT}"
echo "  gpu:    kv=${KV_BYTES}B util=${GPU_UTIL} max_len=${MAX_LEN} max_seqs=${MAX_SEQS}"

# The KV cache size is fixed rather than profiled.  vLLM 0.32 sizes it as
#   total * gpu_memory_utilization - (free at init - free after profiling)
# and on WSL2 those free readings are a per-process WDDM view that shifts
# whenever memory is shuffled between processes, which produced "available KV
# cache memory" of -2.4 to -4.6 GiB with ~3 GiB really free.
# --kv-cache-memory-bytes skips that measurement; gpu_memory_utilization then
# only gates vLLM's own startup check.  256 MiB holds 2,336 tokens (112 KiB per
# token), 4.5 requests of 512 tokens.
exec "$VLLM" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS"
