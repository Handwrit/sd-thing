#!/usr/bin/env bash

#human edit. changed q6 to q8. changed max-vram from 8 to 12. 13-14 might be possible with all apps closed.
#ok so now I am trying the q8 MISTRAL llm.

# Local RX 6950 XT Vulkan launcher for FLUX.2-dev Q6_K.
# HIP on gfx1030 NaN'd / page-faulted; this uses build_vulkan/bin/sd-server.
# GPU VAE was bad on HIP; keep VAE on CPU here too.
# Matches docs/flux2.md Flux.2-dev: --diffusion-model, Flux.2 AE, Mistral-Small 24B,
# --offload-to-cpu --diffusion-fa. Cap Vulkan at 8 GiB and skip next-segment
# prefetch. Dev wants cfg ~1 / euler in the request, not here.
# --rng cpu matches ComfyUI noise (README); default cuda is WebUI RNG.
# full_encoder_small_decoder.safetensors is the alt VAE; override with SD_VAE if you use it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build_vulkan/bin/sd-server"
DIFFUSION="${SD_DIFFUSION_MODEL:-${HOME}/ComfyUI/models/diffusion_models/flux2-dev-Q8_0.gguf}"
VAE_SMALL="${HOME}/ComfyUI/models/vae/full_encoder_small_decoder.safetensors"
VAE_FULL="${HOME}/ComfyUI/models/vae/flux2-vae.safetensors"
if [[ -n "${SD_VAE:-}" ]]; then
  VAE="$SD_VAE"
elif [[ -f "$VAE_SMALL" ]]; then
  VAE="$VAE_SMALL"
else
  VAE="$VAE_FULL"
fi
# was Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf
LLM="${SD_LLM:-${HOME}/ComfyUI/models/text_encoders/Mistral-Small-3.2-24B-Instruct-2506-UD-Q8_K_XL.gguf}"
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
