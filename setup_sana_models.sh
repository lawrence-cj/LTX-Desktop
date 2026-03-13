#!/bin/bash
# Setup environment for running LTX Desktop with Sana Video pipeline.
#
# Model strategy:
#   - IC-LoRA / A2V / Retake → LTX 2.3 (22B) default models
#   - Sana + LTX refiner     → LTX 2.0 (19B) via SANA_LTX_* env vars
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

# ─── Global LTX model paths ──────────────────────────────────────────────
# These are the DEFAULT models used by IC-LoRA, A2V, Retake, and other
# non-Sana features.  Leave unset to use the built-in LTX 2.3 defaults,
# or set them to point to your local LTX 2.3 model files.
#
# export LTX_CHECKPOINT_PATH="..."     # default: ltx-2.3-22b-distilled.safetensors
# export LTX_UPSAMPLER_PATH="..."      # default: ltx-2.3-spatial-upscaler-x2-1.0.safetensors
# export LTX_TEXT_ENCODER_PATH="..."   # default: gemma-3-12b-it-qat-q4_0-unquantized

# ─── Sana-specific LTX 2.0 refiner models ────────────────────────────────
# These ONLY affect the Sana pipeline's LTX Stage-2 refiner.
# Other pipelines (IC-LoRA, A2V, Retake) always use the global defaults above.
export SANA_LTX_CHECKPOINT_PATH="${LTX2_DIR}/ltx-2-19b-dev.safetensors"
export SANA_LTX_UPSAMPLER_PATH="${LTX2_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
export SANA_LTX_DISTILLED_LORA_PATH="${LTX2_DIR}/ltx-2-19b-distilled-lora-384.safetensors"
export SANA_LTX_GEMMA_ROOT="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"

# ─── Pipeline backend: Sana + LTX refiner ────────────────────────────────
export PIPELINE_BACKEND="sana"      # "ltx" or "sana": sana + ltx refiner
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
echo "║ Pipeline:   Sana Video → LTX 2.0 Refiner               ║"
echo "║ Other:      IC-LoRA / A2V / Retake → LTX 2.3 defaults  ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Sana Model: ${SANA_DIR##*/}"
echo "║ Sana LTX:   ${SANA_LTX_CHECKPOINT_PATH##*/}"
echo "║ Sana LoRA:  ${SANA_LTX_DISTILLED_LORA_PATH##*/}"
echo "║ Sana Gemma: ${SANA_LTX_GEMMA_ROOT##*/}"
echo "║ Refine:     ${SANA_ENABLE_REFINE}  Upsample: ${SANA_ENABLE_UPSAMPLE}"
echo "║ Host:       ${LTX_HOST}:${LTX_PORT}"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Run:  cd backend && uv run python ltx2_server.py        ║"
echo "╚══════════════════════════════════════════════════════════╝"
