#!/usr/bin/env bash
# Local RX 6950 XT Vulkan launcher for Qwen Image 2.1 Q8_0.
# HIP on gfx1030 NaN'd / page-faulted; this uses build_vulkan/bin/sd-server.
# docs/qwen_image_2.1.md: Qwen3-VL-8B, the 2.1 VAE (not qwen_image_vae), CFG 6 / euler in the request.
# --offload-to-cpu is what the docs pass. It stays commented here, same as the Anima launcher.
# --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.
# --llm_vision is the GGUF mmproj, required for edits.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
DIFFUSION="${SD_DIFFUSION_MODEL:-${HOME}/ComfyUI/models/diffusion_models/qwen_image_2.1-Q8_0.gguf}"
VAE="${SD_VAE:-${HOME}/ComfyUI/models/vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors}"
LLM="${SD_LLM:-${HOME}/ComfyUI/models/text_encoders/Qwen3VL-8B-Instruct-Q8_0.gguf}"
LLM_VISION="${SD_LLM_VISION:-${HOME}/ComfyUI/models/text_encoders/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"
OUTPUT_DIR="${SD_OUTPUT_DIR:-${ROOT}/output}"

ARGS=(
  --diffusion-model "$DIFFUSION"
  --vae "$VAE"
  --llm "$LLM"
  --llm_vision "$LLM_VISION"
  --listen-port "$PORT"
  --lora-model-dir "$LORA_DIR"
  --output-dir "$OUTPUT_DIR"
  --serve-html-path "${ROOT}/index.html"
  --rng cpu
#  --vae-on-cpu
  --diffusion-fa
# --offload-to-cpu
  --max-vram 12
  --disable-prefetch
)

exec "$BIN" "${ARGS[@]}" "$@"
