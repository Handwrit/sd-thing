#!/usr/bin/env bash
# Local RX 6950 XT launcher using ggml Vulkan (RADV NAVI21), not HIP/ROCm.
# HIP on gfx1030 NaN'd, orb'd, and page-faulted; Vulkan Euler seed 42 matched CPU.
# GPU VAE decode was bad on HIP; keep VAE on CPU here too.
# --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
MODEL="${SD_MODEL:-${HOME}/ComfyUI/models/checkpoints/sd-v1-4.ckpt}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"

exec "$BIN" \
  -m "$MODEL" \
  --listen-port "$PORT" \
  --lora-model-dir "$LORA_DIR" \
  --rng cpu \
  --vae-on-cpu \
  "$@"
