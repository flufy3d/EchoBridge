#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECHOBRIDGE_ASR_PYTHON:-$HOME/.venvs/echobridge-asr/bin/python}"
MODEL_DIR="${ECHOBRIDGE_ASR_MODEL:-$HOME/.cache/echobridge/models/Qwen3-ASR-0.6B}"
export HF_HOME="${HF_HOME:-$HOME/.cache/echobridge/hf}"
mkdir -p "$(dirname "$MODEL_DIR")"
HF_BIN="${ECHOBRIDGE_HF_BIN:-$HOME/.venvs/echobridge-asr/bin/hf}"
"$HF_BIN" download Qwen/Qwen3-ASR-0.6B --local-dir "$MODEL_DIR"
echo "model ready: $MODEL_DIR"
