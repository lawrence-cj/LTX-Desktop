#!/bin/bash
# Setup environment for running LTX Desktop with Sana Video pipeline.
#
# Usage (on GPU cluster):
#   source setup_sana_models.sh
#   cd backend && uv run python ltx2_server.py
#
# SSH tunnel (on Mac):
#   ssh -L 8000:localhost:8000 user@host
#
# Frontend (on Mac):
#   LTX_BACKEND_URL=http://localhost:8000 pnpm dev
#
# No symlinks needed — model paths are configured via environment variables.

# ─── Model Paths (edit if your files are elsewhere) ──────────────────────
LTX2_DIR="/home/junsongc/junsongc/code/diffusion/LTX-2/output/pretrained_models/ltx2"
SANA_DIR="/home/junsongc/junsongc/code/diffusion/Sana/output/open_source/Sana_video/safetensors/sana_ltxvae_sft"
DESKTOP_DIR="/home/junsongc/junsongc/code/diffusion/LTX-Desktop"

# ─── Desktop data directory & server config ──────────────────────────────
export LTX_APP_DATA_DIR="${DESKTOP_DIR}/app_data"
export LTX_HOST="0.0.0.0"    # Listen on all interfaces (for remote access)
export LTX_PORT="8000"        # Fixed port (default: random)
export LTX_AUTH_TOKEN=""      # Empty = no auth (allow remote Electron frontend)
export LTX_ADMIN_TOKEN=""     # Empty = no admin auth
mkdir -p "${LTX_APP_DATA_DIR}/models"
mkdir -p "${LTX_APP_DATA_DIR}/outputs"

# ─── LTX model paths (override Desktop defaults via env vars) ────────────
# These tell resolve_model_path() to use your files directly,
# bypassing the default "ltx-2.3-*" file names.
export LTX_CHECKPOINT_PATH="${LTX2_DIR}/ltx-2-19b-dev.safetensors"
export LTX_UPSAMPLER_PATH="${LTX2_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
export LTX_TEXT_ENCODER_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"
export LTX_DISTILLED_LORA_PATH="${LTX2_DIR}/ltx-2-19b-distilled-lora-384.safetensors"

# ─── Pipeline backend: Sana + LTX refiner ────────────────────────────────
export PIPELINE_BACKEND="sana"
export SANA_MODEL_PATH="${SANA_DIR}"

# ─── Sana generation options ─────────────────────────────────────────────
export SANA_ENABLE_REFINE="true"      # LTX refiner Stage 2 (3 steps)
export SANA_ENABLE_UPSAMPLE="false"   # spatial 2x upsampler (set true for higher res)
export SANA_GUIDANCE_SCALE="6.0"
export SANA_NUM_STEPS="50"
export SANA_MOTION_SCORE="30"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          LTX Desktop — Sana Video Configuration         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Pipeline:   Sana Video → LTX Refiner                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Sana Model: ${SANA_DIR##*/}"
echo "║ LTX Ckpt:   ${LTX_CHECKPOINT_PATH##*/}"
echo "║ Upsampler:  ${LTX_UPSAMPLER_PATH##*/}"
echo "║ Text Enc:   ${LTX_TEXT_ENCODER_PATH##*/}"
echo "║ Refine:     ${SANA_ENABLE_REFINE}  Upsample: ${SANA_ENABLE_UPSAMPLE}"
echo "║ Auth:     disabled (no token)                              ║"
echo "║ Host:     ${LTX_HOST}:${LTX_PORT}                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Run:  cd backend && uv run python ltx2_server.py        ║"
echo "╚══════════════════════════════════════════════════════════╝"
