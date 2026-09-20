#!/usr/bin/env bash
# Local RX 6950 XT (gfx1030) launcher for sd-server.
# HIP graphs crash this card mid-UNet; keep them off.
# Do not pass --linear-scale 0.0078125 here: on SD 1.4 it produced all-white images.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build/bin/sd-server"
MODEL="${SD_MODEL:-${HOME}/ComfyUI/models/checkpoints/sd-v1-4.ckpt}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"

exec "$BIN" \
  -m "$MODEL" \
  --listen-port "$PORT" \
  --lora-model-dir "$LORA_DIR" \
  "$@"
