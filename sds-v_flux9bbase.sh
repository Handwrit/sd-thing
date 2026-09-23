#!/usr/bin/env bash
# Local RX 6950 XT Vulkan launcher for FLUX.2 Klein Base 9B.
# HIP on gfx1030 NaN'd / page-faulted; this uses build_vulkan/bin/sd-server.
# GPU VAE was bad on HIP; keep VAE on CPU here too.
# Matches docs/flux2.md Klein Base 9B: --diffusion-model, Flux.2 AE, Qwen3-8B,
# --offload-to-cpu --diffusion-fa. Base wants cfg ~4 / ~20 steps in the request,
# not here. --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.
# Official example is flux-2-klein-base-9b.safetensors; local copy is the fp8.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
DIFFUSION_FULL="${HOME}/ComfyUI/models/diffusion_models/flux-2-klein-base-9b.safetensors"
DIFFUSION_FP8="${HOME}/ComfyUI/models/diffusion_models/flux-2-klein-base-9b-fp8.safetensors"
if [[ -n "${SD_DIFFUSION_MODEL:-}" ]]; then
  DIFFUSION="$SD_DIFFUSION_MODEL"
elif [[ -f "$DIFFUSION_FULL" ]]; then
  DIFFUSION="$DIFFUSION_FULL"
else
  DIFFUSION="$DIFFUSION_FP8"
fi
# Flux.2 AE (docs/flux2.md flux2_ae.safetensors). Not Flux.1 ae.safetensors.
VAE="${SD_VAE:-${HOME}/ComfyUI/models/vae/flux2-vae.safetensors}"
LLM="${SD_LLM:-${HOME}/ComfyUI/models/text_encoders/qwen_3_8b.safetensors}"
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
  --offload-to-cpu \
  "$@"
