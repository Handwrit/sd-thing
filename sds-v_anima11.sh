#!/usr/bin/env bash
# Local RX 6950 XT Vulkan launcher for Anima Aesthetic v1.1.
# HIP on gfx1030 NaN'd / page-faulted; this uses build_vulkan/bin/sd-server.
# GPU VAE was bad on HIP; keep VAE on CPU here too.
# --diffusion-fa and --offload-to-cpu match docs/anima.md (4B F16 + DiT is tight on 16 GB).
# --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
DIFFUSION="${SD_DIFFUSION_MODEL:-${HOME}/ComfyUI/models/diffusion_models/anima-aesthetic-v1.1.safetensors}"
VAE="${SD_VAE:-${HOME}/ComfyUI/models/vae/ae.safetensors}"
LLM="${SD_LLM:-${HOME}/Qwen3-4B-Instruct-2507-F16.gguf}"
LORA_DIR="${SD_LORA_DIR:-${HOME}/ComfyUI/models/loras}"
PORT="${SD_LISTEN_PORT:-1493}"

exec "$BIN" \
  --diffusion-model "$DIFFUSION" \
  --vae "$VAE" \
  --llm "$LLM" \
  --listen-port "$PORT" \
  --lora-model-dir "$LORA_DIR" \
  --rng cpu \
  --vae-on-cpu \
  --diffusion-fa \
  --offload-to-cpu \
  "$@"
