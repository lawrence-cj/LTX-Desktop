#!/bin/bash
# Run Video Agent test. Use project-local venv.
#
# Usage (on GPU node after `gg`):
#   bash run_agent_test.sh              # full test (GPU required)
#   bash run_agent_test.sh --director-only  # director only (no GPU)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23"
GEMMA_DIR="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"

cd "$PROJECT_DIR"
source ltx-desktop/bin/activate

echo "=== Environment ==="
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null

export LTX_APP_DATA_DIR="/tmp/ltx-agent-test-$$"
export LTX_CHECKPOINT_PATH="$MODEL_DIR/ltx-2.3-22b-distilled.safetensors"
export LTX_UPSAMPLER_PATH="$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
export LTX_TEXT_ENCODER_PATH="$GEMMA_DIR"
export USE_SAGE_ATTENTION=0

echo "=== Running Test ==="
python test_video_agent.py "$@"
