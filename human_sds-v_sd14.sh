#!/usr/bin/env bash

#human edit.
#added   --diffusion-fa   --offload-to-cpu  --max-vram 13   --disable-prefetch 

# Local RX 6950 XT Vulkan launcher for SD 1.4.
# Same shape as human_sds-v_f2devq8.sh: build_vulkan/bin/sd-server, port 1493,
# index.html, ComfyUI loras, CPU RNG, VAE on CPU.
# The checkpoint is one file and fits in about 2.6 GiB, so this does not use
# Flux's --offload-to-cpu, --diffusion-fa, --max-vram, or --disable-prefetch.
# HIP on gfx1030 NaN'd, orb'd, and page-faulted; Vulkan Euler matched CPU.
# --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
MODEL="${SD_MODEL:-${HOME}/ComfyUI/models/checkpoints/sd-v1-4.ckpt}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"
OUTPUT_DIR="${SD_OUTPUT_DIR:-${ROOT}/output}"

exec "$BIN" \
  -m "$MODEL" \
  --listen-port "$PORT" \
  --lora-model-dir "$LORA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --serve-html-path "${ROOT}/index.html" \
  --rng cpu \
  --vae-on-cpu \
  --diffusion-fa \
  --offload-to-cpu \
  --max-vram 13 \
  --disable-prefetch \
  "$@"
