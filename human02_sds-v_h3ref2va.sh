#!/usr/bin/env bash

# Local RX 6950 XT Vulkan launcher for MiniMax-H3 Ref2VA.
# Same runtime flags as human02_sds-v_f2devq8.sh.
# docs/minimax_h3.md asks for --offload-to-cpu and --audio-vae.
# This launcher omits both on purpose.
# Vision weights are inside the Qwen3-VL GGUF, so --llm_vision is not passed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
DIFFUSION="${SD_DIFFUSION_MODEL:-${HOME}/ComfyUI/models/diffusion_models/minimax_h3_ref2va_pruned-Q4_K_M.gguf}"
VAE="${SD_VAE:-${HOME}/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors}"
LLM="${SD_LLM:-${HOME}/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"
OUTPUT_DIR="${SD_OUTPUT_DIR:-${ROOT}/output}"

exec "$BIN" \
  --diffusion-model "$DIFFUSION" \
  --vae "$VAE" \
  --llm "$LLM" \
  --listen-port "$PORT" \
  --lora-model-dir "$LORA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --serve-html-path "${ROOT}/index.html" \
  --rng cpu \
  --vae-on-cpu \
  --diffusion-fa \
  --max-vram 12 \
  --disable-prefetch \
  "$@"
