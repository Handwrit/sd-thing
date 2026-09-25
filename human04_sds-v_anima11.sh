#!/usr/bin/env bash
# Local RX 6950 XT Vulkan launcher for Anima Aesthetic v1.1.
# HIP on gfx1030 NaN'd / page-faulted; this uses build_vulkan/bin/sd-server.
# GPU VAE was bad on HIP; keep VAE on CPU here too.
# --diffusion-fa and --offload-to-cpu match docs/anima.md.
# Anima uses the Qwen-Image/Wan VAE and Qwen3-0.6B (1024-d llm_adapter), not Flux AE / Qwen3-4B.
# --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
DIFFUSION="${SD_DIFFUSION_MODEL:-${HOME}/ComfyUI/models/diffusion_models/anima-aesthetic-v1.1.safetensors}"
VAE="${SD_VAE:-${HOME}/ComfyUI/models/vae/qwen_image_vae.safetensors}"
LLM="${SD_LLM:-${HOME}/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"
OUTPUT_DIR="${SD_OUTPUT_DIR:-${ROOT}/output}"

ARGS=(
  --diffusion-model "$DIFFUSION"
  --vae "$VAE"
  --llm "$LLM"
  --listen-port "$PORT"
  --lora-model-dir "$LORA_DIR"
  --output-dir "$OUTPUT_DIR"
  --serve-html-path "${ROOT}/index.html"
  --rng cpu
#  --vae-on-cpu
  --diffusion-fa
# --offload-to-cpu
  --max-vram 12    # I don't think this works.
  --disable-prefetch
)

exec "$BIN" "${ARGS[@]}" "$@"
